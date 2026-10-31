"""Link each NEET question to the NCERT passage that grounds it.

Usage:
  python scripts/link_ncert.py --paper 2025_en_45 [--limit 20] [--workers 4]

Two steps per question:
  1. Retrieve: embed "stem + options" (RETRIEVAL_QUERY), cosine-rank the NCERT
     passage embeddings restricted to the question's subject (Botany/Zoology
     -> Biology), keep the top K.
  2. Select: ask gemini-3.8-flash (configurable) which candidate actually
     grounds the question. It returns a chunk id, a confidence, and a short
     note on *why* in its own words. It is told not to quote the passage; the
     passage text is already ours by id, and verbatim NCERT output would trip
     the recitation filter.

Writes data/links/<paper>.jsonl with one record per question:
  {id, chunk_id, chapter_code, chapter, title, class, subject, printed_page,
   confidence, rationale, candidates: [chunk ids], model}
Re-runs skip questions already linked unless --force.
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Literal, Optional

import numpy as np
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import runlog  # noqa: E402
from embed_ncert import DIMS, EMBED_MODEL, embed_texts  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(ROOT, ".env"), override=True)

SELECT_MODEL = os.environ.get("LINK_MODEL", "gemini-3.8-flash")
TOP_K = 8
SUBJECT_MAP = {"Botany": "Biology", "Zoology": "Biology", "Biology": "Biology", "Physics": "Physics", "Chemistry": "Chemistry"}


class Selection(BaseModel):
    chunk_id: Optional[str] = Field(description="id of the single best grounding passage, or null if none of the candidates grounds the question")
    confidence: Literal["high", "medium", "low"]
    rationale: str = Field(description="One or two sentences, in your own words, on what in the passage the question relies on. Do not quote the passage.")
    needs_other_chapter: Optional[str] = Field(default=None, description="If no candidate fits, name the NCERT chapter you would expect to ground it")


PROMPT = """A NEET-UG question and {k} candidate passages from NCERT textbooks are given below.
Pick the ONE passage a student would need to read to answer the question. Prefer the passage that states the specific fact, law, formula, structure, or process the question tests, over a passage that merely shares the topic.
If no candidate genuinely grounds the question, return chunk_id = null and name the chapter you would expect.
Write the rationale in your own words; do not quote or closely paraphrase the passage text.

QUESTION ({subject}):
{question}

CANDIDATES:
{candidates}
"""


def load_index():
    base = os.path.join(ROOT, "data", "ncert")
    chunks = {c["id"]: c for c in (json.loads(l) for l in open(os.path.join(base, "chunks.jsonl"), encoding="utf-8"))}
    ids = json.load(open(os.path.join(base, "embeddings_ids.json"), encoding="utf-8"))
    emb = np.load(os.path.join(base, "embeddings.npy"))
    subj = np.array([chunks[i]["subject"] for i in ids])
    return chunks, ids, emb, subj


def question_text(q: dict) -> str:
    opts = "\n".join(f"({o['key']}) {o['text_en']}" for o in q["options"])
    figs = ""
    if q.get("figures"):
        figs = "\n[figure: " + "; ".join(f["description_en"] for f in q["figures"]) + "]"
    return f"{q['text_en']}{figs}\n{opts}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--paper", required=True, help="e.g. 2025_en_45")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    qpath = os.path.join(ROOT, "data", "questions", f"{args.paper}.jsonl")
    questions = [json.loads(l) for l in open(qpath, encoding="utf-8")]
    if args.limit:
        questions = questions[: args.limit]
    out_dir = os.path.join(ROOT, "data", "links")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{args.paper}.jsonl")
    done = {}
    if os.path.exists(out_path) and not args.force:
        done = {json.loads(l)["id"]: json.loads(l) for l in open(out_path, encoding="utf-8")}
    todo = [q for q in questions if q["id"] not in done]

    chunks, ids, emb, subj = load_index()
    client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
    t_start = time.time()
    tok_in = tok_out = 0

    # Step 1: embed all queries in batches.
    texts = [question_text(q) for q in todo]
    qvecs = embed_texts(client, texts, "RETRIEVAL_QUERY")  # batched, retries on 429, L2-normalised

    titles_by_code = {chunks[i]["chapter_code"]: chunks[i]["title"] for i in ids}

    def chapter_mask(named: str, subject: str) -> np.ndarray | None:
        """Match a chapter the model named in free text to one of ours, within the subject."""
        named_l = named.lower()
        best, best_n = None, 0
        for code, title in titles_by_code.items():
            if chunks[next(i for i in ids if chunks[i]["chapter_code"] == code)]["subject"] != subject:
                continue
            words = [w for w in title.lower().replace(",", " ").split() if len(w) > 3]
            n = sum(w in named_l for w in words)
            if n > best_n:
                best, best_n = code, n
        if best is None or best_n == 0:
            return None
        return np.array([chunks[i]["chapter_code"] == best for i in ids])

    def ask(prompt: str) -> tuple[Selection, dict]:
        last = None
        for attempt in range(4):
            try:
                resp = client.models.generate_content(
                    model=SELECT_MODEL, contents=prompt,
                    config=types.GenerateContentConfig(response_mime_type="application/json", response_schema=Selection, temperature=0,
                                                       thinking_config=types.ThinkingConfig(thinking_level="low")),
                )
                u = resp.usage_metadata
                return Selection.model_validate_json(resp.text), {"input_tokens": u.prompt_token_count, "output_tokens": (u.candidates_token_count or 0) + (u.thoughts_token_count or 0)}  # thinking is billed as output
            except Exception as e:  # noqa: BLE001
                last = e
                time.sleep(2 * (attempt + 1))
        raise RuntimeError(str(last)[:200])

    def candidates_for(sims: np.ndarray, mask: np.ndarray) -> list[int]:
        s = sims.copy()
        s[~mask] = -1
        return [int(j) for j in np.argsort(-s)[:TOP_K]]

    def build_prompt(q: dict, i: int, top: list[int]) -> str:
        cand_text = "\n\n".join(
            f"[{ids[j]}] Class {chunks[ids[j]]['class']} {chunks[ids[j]]['subject']}, Ch. {chunks[ids[j]]['chapter']} {chunks[ids[j]]['title']}, p. {chunks[ids[j]]['printed_page']}\n{chunks[ids[j]]['text']}"
            for j in top
        )
        return PROMPT.format(k=len(top), subject=q["subject"], question=texts[i], candidates=cand_text)

    # Step 2: select with the LLM; on a miss, retry within the chapter it names.
    def select(i: int) -> dict:
        q = todo[i]
        subject = SUBJECT_MAP[q["subject"]]
        sims = emb @ qvecs[i]
        top = candidates_for(sims, subj == subject)
        usage = {"input_tokens": 0, "output_tokens": 0}
        passes = []
        try:
            sel, u = ask(build_prompt(q, i, top))
            usage = {k: usage[k] + (u[k] or 0) for k in usage}
            passes.append("subject")
            if not sel.chunk_id and sel.needs_other_chapter:
                m = chapter_mask(sel.needs_other_chapter, subject)
                if m is not None and m.any():
                    top2 = candidates_for(sims, m)
                    sel2, u2 = ask(build_prompt(q, i, top2))
                    usage = {k: usage[k] + (u2[k] or 0) for k in usage}
                    passes.append("chapter")
                    if sel2.chunk_id:
                        sel, top = sel2, top2
        except RuntimeError as e:
            return {"id": q["id"], "qno": q["qno"], "subject": q["subject"], "chunk_id": None, "error": str(e),
                    "candidates": [ids[j] for j in top]}
        rec = {"id": q["id"], "qno": q["qno"], "subject": q["subject"], "chunk_id": sel.chunk_id,
               "confidence": sel.confidence, "rationale": sel.rationale,
               "needs_other_chapter": sel.needs_other_chapter, "passes": passes,
               "candidates": [ids[j] for j in top], "retrieval_scores": [round(float(sims[j]), 3) for j in top],
               "model": SELECT_MODEL, "usage": usage}
        if sel.chunk_id and sel.chunk_id in chunks:
            c = chunks[sel.chunk_id]
            rec.update({"level": "passage", "chapter_code": c["chapter_code"], "class": c["class"], "chapter": c["chapter"],
                        "title": c["title"], "printed_page": c["printed_page"] or c["pdf_page"]})
        else:
            if sel.chunk_id:
                rec["rationale"] = "(model returned an id not in candidates) " + rec["rationale"]
            rec["chunk_id"] = None
            # No single passage states it (typical for numericals): keep the chapter the model named.
            m = chapter_mask(sel.needs_other_chapter, subject) if sel.needs_other_chapter else None
            if m is not None and m.any():
                c = chunks[ids[int(np.argmax(m))]]
                rec.update({"level": "chapter", "chapter_code": c["chapter_code"], "class": c["class"],
                            "chapter": c["chapter"], "title": c["title"], "printed_page": None})
            else:
                rec["level"] = "none"
        return rec

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(select, range(len(todo))))

    # Rewrite the file: previously linked records (unless --force) plus this run's, in qno order.
    merged = {**done, **{r["id"]: r for r in results}}
    with open(out_path, "w", encoding="utf-8") as f:
        for rid in sorted(merged, key=lambda i: int(i.rsplit("-", 1)[-1])):
            f.write(json.dumps(merged[rid], ensure_ascii=False) + "\n")
    for r in results:
        u = r.get("usage") or {}
        tok_in += u.get("input_tokens") or 0
        tok_out += u.get("output_tokens") or 0
        lvl = r.get("level", "none")
        tag = f"{r.get('confidence','-'):6}" if lvl == "passage" else f"{lvl.upper():6}"
        if lvl == "passage":
            where = f"C{r['class']} ch{r['chapter']:02d} p{r['printed_page']} {r['title']}"
        elif lvl == "chapter":
            where = f"C{r['class']} ch{r['chapter']:02d} {r['title']}"
        else:
            where = r.get("needs_other_chapter") or r.get("error") or ""
        print(f"q{r['qno']:03d} {r['subject'][:4]} {tag} {where}", flush=True)

    levels = {l: sum(1 for r in results if r.get("level") == l) for l in ("passage", "chapter", "none")}
    conf = {c: sum(1 for r in results if r.get("confidence") == c and r.get("level") == "passage") for c in ("high", "medium", "low")}
    print(f"\nlevels {levels}  passage confidence {conf}  tokens in={tok_in} out={tok_out}")
    if results:
        runlog.log("link_ncert", paper=args.paper, model=SELECT_MODEL, questions=len(results), levels=levels,
                   confidence=conf, input_tokens=tok_in, output_tokens=tok_out, seconds=round(time.time() - t_start, 1))


if __name__ == "__main__":
    main()
