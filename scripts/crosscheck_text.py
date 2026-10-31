"""Compare vision transcriptions against a paper's embedded PDF text layer.

Usage: python scripts/crosscheck_text.py --year 2020 --lang en

Only meaningful for papers whose PDF carries real text (2020, 2021 EN, 2022).
For each question, the stem and each option are normalised and searched in
the normalised text of the source pages; a token-level fuzzy ratio is
reported. Prints the distribution and the worst cases, and appends a summary
to the run log so the number lands in the story.
"""
import argparse
import difflib
import glob
import json
import os
import re
import sys

import fitz
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import runlog  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def norm(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"\$[^$]*\$", " ", s)            # drop LaTeX spans: PDF text has glyph soup there
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def best_window_ratio(needle: str, hay: str) -> float:
    """Best SequenceMatcher ratio of needle against any window of hay with similar length."""
    if not needle:
        return 1.0
    n = len(needle)
    if n < 12:
        return 1.0 if needle in hay else 0.0
    best = 0.0
    step = max(8, n // 4)
    for i in range(0, max(1, len(hay) - n + 1), step):
        r = difflib.SequenceMatcher(None, needle, hay[i:i + n + 10]).quick_ratio()
        if r > best:
            r = difflib.SequenceMatcher(None, needle, hay[i:i + n + 10]).ratio()
            best = max(best, r)
            if best > 0.97:
                break
    return best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--lang", default="en")
    args = ap.parse_args()
    papers = yaml.safe_load(open(os.path.join(ROOT, "data", "papers.yaml"), encoding="utf-8"))["papers"]
    entry = next(p for p in papers if p["year"] == args.year and p["lang"] == args.lang)
    doc = fitz.open(os.path.join(ROOT, entry["file"]))
    page_text = {i + 1: norm(p.get_text("text")) for i, p in enumerate(doc)}
    if sum(len(t) for t in page_text.values()) < 2000:
        sys.exit("this PDF has no usable text layer")

    qpath = glob.glob(os.path.join(ROOT, "data", "questions", f"{args.year}_{args.lang}_*.jsonl"))[0]
    rows = [json.loads(l) for l in open(qpath, encoding="utf-8")]
    scores = []
    for q in rows:
        hay = " ".join(page_text.get(p, "") for p in q["source_pages"])
        stem = best_window_ratio(norm(q["text_en"])[:300], hay)
        opts = [best_window_ratio(norm(o["text_en"])[:120], hay) for o in q["options"] if norm(o["text_en"])]
        scores.append((q["qno"], stem, min(opts) if opts else 1.0, sum(opts) / len(opts) if opts else 1.0))

    stems = [s for _, s, _, _ in scores]
    opt_means = [m for _, _, _, m in scores]
    def pct(xs, t): return 100 * sum(x >= t for x in xs) / len(xs)
    print(f"{args.year} {args.lang}: {len(rows)} questions vs PDF text layer")
    print(f"  stems   >=0.95: {pct(stems,0.95):.1f}%   >=0.90: {pct(stems,0.90):.1f}%   mean {sum(stems)/len(stems):.3f}")
    print(f"  options >=0.95: {pct(opt_means,0.95):.1f}%   >=0.90: {pct(opt_means,0.90):.1f}%   mean {sum(opt_means)/len(opt_means):.3f}")
    worst = sorted(scores, key=lambda s: min(s[1], s[2]))[:8]
    print("  worst (qno, stem, min option, mean option):", [(a, round(b, 2), round(c, 2), round(d, 2)) for a, b, c, d in worst])
    runlog.log("crosscheck_text", year=args.year, lang=args.lang, questions=len(rows),
               stem_ge_095=round(pct(stems, 0.95), 1), stem_mean=round(sum(stems) / len(stems), 3),
               options_ge_095=round(pct(opt_means, 0.95), 1), options_mean=round(sum(opt_means) / len(opt_means), 3))


if __name__ == "__main__":
    main()
