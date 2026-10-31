"""Blind-check generated MCQs before they go into training.

Each generated question is answered again in a fresh call that does NOT see
the intended answer or explanation. A question is kept only if the blind
answer matches, the checker finds exactly one defensible option, and it flags
no defect (factual error, ambiguity, needs a figure, answer given away,
Hindi not matching English).

Usage:
  python scripts/verify_mcq.py [--chapter kebo108] [--limit 20] [--workers 12]

Reads  data/synthetic/mcq/<chapter>.jsonl
Writes data/synthetic/mcq_verify/<chapter>.jsonl  (one verdict per question id; resumable)
"""
import argparse
import glob
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Literal

from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import runlog  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(ROOT, ".env"), override=True)
MODEL = os.environ.get("VERIFY_MODEL", "gemini-3.8-flash")


class Verdict(BaseModel):
    answer: int = Field(description="1-4: the option you judge correct")
    exactly_one_correct: bool = Field(description="True if exactly one option is defensibly correct")
    defects: list[Literal["factual_error", "ambiguous", "needs_figure", "answer_in_stem", "trivial", "hindi_mismatch", "outside_syllabus"]] = Field(
        description="Defects found; empty list if none")
    note: str = Field(description="One short sentence justifying the answer or naming the defect")


PROMPT = """You are checking a practice question written for NEET-UG students. Answer it yourself, then judge its quality.

NCERT passage it was written from (Class {klass} {subject}, "{title}"):
{passage}

QUESTION (English):
{q_en}
{opts_en}

QUESTION (Hindi):
{q_hi}
{opts_hi}

Pick the correct option. Then check: is exactly one option defensibly correct? Any factual error against NCERT, ambiguity, need for a figure that is not given, answer given away in the stem, trivially easy, Hindi not saying the same as English, or content outside the NCERT syllabus?"""

_lock = threading.Lock()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chapter")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()
    client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
    chunks = {json.loads(l)["id"]: json.loads(l) for l in open(os.path.join(ROOT, "data", "ncert", "chunks.jsonl"), encoding="utf-8")}
    src_dir = os.path.join(ROOT, "data", "synthetic", "mcq")
    out_dir = os.path.join(ROOT, "data", "synthetic", "mcq_verify")
    os.makedirs(out_dir, exist_ok=True)
    files = sorted(glob.glob(os.path.join(src_dir, "*.jsonl")))
    if args.chapter:
        files = [f for f in files if os.path.basename(f)[:-6] in args.chapter.split(",")]
    todo = []
    for f in files:
        ch = os.path.basename(f)[:-6]
        out = os.path.join(out_dir, f"{ch}.jsonl")
        done = {json.loads(l)["id"] for l in open(out, encoding="utf-8")} if os.path.exists(out) else set()
        todo += [(out, r) for r in (json.loads(l) for l in open(f, encoding="utf-8")) if r["id"] not in done and r.get("valid_shape")]
    if args.limit:
        todo = todo[: args.limit]
    print(f"verifying {len(todo)} questions with {MODEL}")
    tot = {"input_tokens": 0, "output_tokens": 0}
    stats = {"kept": 0, "rejected": 0, "failed": 0}

    def work(item):
        out, r = item
        c = chunks[r["source_chunk"]]
        prompt = PROMPT.format(
            klass=c["class"], subject=c["subject"], title=c["title"], passage=c["text"],
            q_en=r["question_en"], opts_en="\n".join(f"({i + 1}) {o}" for i, o in enumerate(r["options_en"])),
            q_hi=r["question_hi"], opts_hi="\n".join(f"({i + 1}) {o}" for i, o in enumerate(r["options_hi"])),
        )
        for attempt in range(3):
            try:
                resp = client.models.generate_content(
                    model=MODEL, contents=prompt,
                    config=types.GenerateContentConfig(response_mime_type="application/json", response_schema=Verdict, temperature=0,
                                                       thinking_config=types.ThinkingConfig(thinking_level=os.environ.get("VERIFY_THINKING", "low"))),
                )
                v = Verdict.model_validate_json(resp.text)
                u = resp.usage_metadata
                return out, r, v, (u.prompt_token_count or 0, (u.candidates_token_count or 0) + (u.thoughts_token_count or 0))
            except Exception as e:  # noqa: BLE001
                last = e
                time.sleep(3 * (attempt + 1))
        raise RuntimeError(f"{r['id']}: {str(last)[:150]}")

    t0 = time.time()
    with ThreadPoolExecutor(args.workers) as ex:
        futs = [ex.submit(work, it) for it in todo]
        for k, f in enumerate(as_completed(futs), 1):
            try:
                out, r, v, (ti, to) = f.result()
            except Exception as e:  # noqa: BLE001
                stats["failed"] += 1
                print("  FAILED", e)
                continue
            keep = v.answer == r["answer"] and v.exactly_one_correct and not v.defects
            stats["kept" if keep else "rejected"] += 1
            tot["input_tokens"] += ti
            tot["output_tokens"] += to
            with _lock, open(out, "a", encoding="utf-8") as fh:
                fh.write(json.dumps({"id": r["id"], "keep": keep, "intended": r["answer"], **v.model_dump(), "model": MODEL}, ensure_ascii=False) + "\n")
            if k % 200 == 0:
                print(f"  {k}/{len(todo)} {stats}", flush=True)
                # Partial record, so a run stopped by a time limit still shows its spend.
                runlog.log("verify_mcq_progress", model=MODEL, done=k, **stats, input_tokens=tot["input_tokens"], output_tokens=tot["output_tokens"])
    cost = runlog.cost_usd(MODEL, tot["input_tokens"], tot["output_tokens"])
    print(f"done: {stats}  tokens in {tot['input_tokens']:,} out {tot['output_tokens']:,}  ~${cost}")
    runlog.log("verify_mcq", model=MODEL, **stats, input_tokens=tot["input_tokens"], output_tokens=tot["output_tokens"],
               seconds=round(time.time() - t0, 1))


if __name__ == "__main__":
    main()
