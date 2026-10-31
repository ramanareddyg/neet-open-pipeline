"""Re-locate figure boxes on pages that were transcribed in bands or crops.

Until 5 Oct 2026 the band and crop rungs of extract_pages.py stored figure
boxes relative to the band/crop the model saw, while assembly cuts figures
from the full page, so those figure crops show the wrong region. This pass
shows the model the full page and asks only *where* each figure of a given
question is (no transcription, so the recitation filter does not apply), then
patches box_2d in data/extracted/.../pNNN.json.

Usage:
  python scripts/relocate_figures.py [--year 2024 --lang en] [--dry-run]

Affected = any page with `split_into` set, plus any question fixed by
fix_qnos (notes contain fixed_by_qno), whose boxes came from a crop or band.
Patched questions get `figures_relocated: <model>`; re-runs skip them.
Then re-run assemble_questions.py for the paper.
"""
import argparse
import glob
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Literal

from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import runlog  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(ROOT, ".env"), override=True)
MODELS = ["gemini-3.8-flash", "gemini-2.5-flash"]


class FigBox(BaseModel):
    belongs_to: Literal["question", "option_1", "option_2", "option_3", "option_4"]
    box_2d: list[int] = Field(description="[ymin, xmin, ymax, xmax] on a 0-1000 scale of this full page")
    description_en: str


class FigBoxes(BaseModel):
    figures: list[FigBox]


PROMPT = """This is a page of a printed exam. Locate the figure(s) that belong to {target}.
Known figures for it (from an earlier pass; positions are unreliable):
{known}
Return one entry per figure, in the same order, with a bounding box [ymin, xmin, ymax, xmax] on a 0-1000 scale of THIS FULL PAGE.
The box must cover the whole drawing, graph, structure, circuit or table image including its labels, axis titles and subscripts, but not the question text or option text around it.
A figure that is an option (e.g. four graphs as the four options) belongs to option_1..option_4.
Do not transcribe any text."""


def affected(page: dict) -> list[int]:
    """Indices of questions in this page JSON whose figure boxes need relocating."""
    banded = bool(page.get("split_into"))
    return [i for i, q in enumerate(page["questions"])
            if q.get("figures") and not q.get("figures_relocated")
            and (banded or "fixed_by_qno" in (q.get("notes") or ""))]


def relocate(client, png: str, q: dict) -> tuple[list[dict] | None, dict]:
    with open(png, "rb") as f:
        img = f.read()
    target = (f"question {q['qno']}" if not q.get("continuation")
              else f"the part of question {q['qno']} that continues at the top of this page")
    known = "\n".join(f"- {fg['belongs_to']}: {fg['description_en']}" for fg in q["figures"])
    usage = {"input_tokens": 0, "output_tokens": 0}
    for model in MODELS:
        try:
            r = client.models.generate_content(
                model=model,
                contents=[types.Part.from_bytes(data=img, mime_type="image/png"), PROMPT.format(target=target, known=known)],
                config=types.GenerateContentConfig(response_mime_type="application/json", response_schema=FigBoxes, temperature=0),
            )
            u = r.usage_metadata
            usage = {"input_tokens": usage["input_tokens"] + (u.prompt_token_count or 0),
                     "output_tokens": usage["output_tokens"] + (u.candidates_token_count or 0) + (u.thoughts_token_count or 0)}
            if not r.text:
                continue
            got = [fb for fb in FigBoxes.model_validate_json(r.text).figures
                   if len(fb.box_2d) == 4 and all(0 <= v <= 1000 for v in fb.box_2d)
                   and fb.box_2d[2] > fb.box_2d[0] and fb.box_2d[3] > fb.box_2d[1]]
            if got:
                return [fb.model_dump() for fb in got], {**usage, "model": model}
        except Exception as e:  # noqa: BLE001
            print(f"  {os.path.basename(png)} q{q['qno']}: {model} {type(e).__name__} {str(e)[:120]}")
            time.sleep(2)
    return None, usage


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int)
    ap.add_argument("--lang", choices=["en", "hi"])
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    pattern = os.path.join(ROOT, "data", "extracted", str(args.year or "*"), args.lang or "*", "*", "p*.json")
    jobs = []
    for path in sorted(glob.glob(pattern)):
        page = json.load(open(path, encoding="utf-8"))
        for i in affected(page):
            jobs.append((path, i))
    print(f"{len(jobs)} questions with figures to relocate")
    if args.dry_run or not jobs:
        return
    client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
    pages = {p: json.load(open(p, encoding="utf-8")) for p in {j[0] for j in jobs}}

    def work(job):
        path, i = job
        page = pages[path]
        q = page["questions"][i]
        figs, usage = relocate(client, os.path.join(ROOT, page["source_png"]), q)
        return path, i, figs, usage

    t0 = time.time()
    tot = {"input_tokens": 0, "output_tokens": 0}
    ok = changed_count = failed = 0
    with ThreadPoolExecutor(args.workers) as ex:
        for path, i, figs, usage in ex.map(work, jobs):
            tot = {k: tot[k] + usage.get(k, 0) for k in tot}
            q = pages[path]["questions"][i]
            if figs is None:
                failed += 1
                print(f"  {os.path.relpath(path, ROOT)} q{q['qno']}: not relocated")
                continue
            old = q["figures"]
            if len(figs) == len(old):
                for fo, fn in zip(old, figs):
                    fo["box_2d_before_relocation"] = fo["box_2d"]
                    fo["box_2d"] = fn["box_2d"]
            else:
                changed_count += 1
                q["figures_before_relocation"] = old
                q["figures"] = figs
            q["figures_relocated"] = usage.get("model")
            ok += 1
    for path, page in pages.items():
        with open(path, "w", encoding="utf-8") as f:
            json.dump(page, f, ensure_ascii=False, indent=1)
    print(f"relocated {ok} (figure count changed on {changed_count}), failed {failed}; tokens in {tot['input_tokens']:,} out {tot['output_tokens']:,}")
    runlog.log("relocate_figures", model=MODELS[0], questions=ok, count_changed=changed_count, failed=failed,
               input_tokens=tot["input_tokens"], output_tokens=tot["output_tokens"], seconds=round(time.time() - t0, 1))


if __name__ == "__main__":
    main()
