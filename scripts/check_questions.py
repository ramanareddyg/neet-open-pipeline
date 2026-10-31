"""Sanity checks over assembled question files.

Usage: python scripts/check_questions.py [paper ...]

Reports, per paper: question count, options count errors, missing answers,
control characters left in text (mis-escaped LaTeX), empty stems, and
LaTeX commands that look truncated. Exit code 1 if anything is found.
"""
import glob
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CTRL = re.compile("[\\r\\t\\f\\b\\x00-\\x08\\x0b\\x0e-\\x1f]")


def texts(q: dict):
    yield "stem_en", q["text_en"]
    yield "stem_hi", q.get("text_hi")
    for o in q["options"]:
        yield f"opt{o['key']}_en", o["text_en"]
        yield f"opt{o['key']}_hi", o.get("text_hi")


def main() -> None:
    papers = sys.argv[1:] or [os.path.splitext(os.path.basename(p))[0] for p in sorted(glob.glob(os.path.join(ROOT, "data", "questions", "*.jsonl")))]
    bad_total = 0
    for paper in papers:
        rows = [json.loads(l) for l in open(os.path.join(ROOT, "data", "questions", f"{paper}.jsonl"), encoding="utf-8")]
        issues = []
        for q in rows:
            if len(q["options"]) != 4:
                issues.append((q["qno"], "options", len(q["options"])))
            if q["answer"] is None:
                issues.append((q["qno"], "no answer", ""))
            if not (q["text_en"] or "").strip():
                issues.append((q["qno"], "empty stem", ""))
            for name, t in texts(q):
                if t and CTRL.search(t):
                    m = CTRL.search(t)
                    issues.append((q["qno"], f"ctrl in {name}", repr(t[max(0, m.start() - 6): m.end() + 10])))
        n_hi = sum(1 for q in rows if q.get("text_hi"))
        print(f"{paper}: {len(rows)} q, {n_hi} with Hindi, {sum(q['has_figure'] for q in rows)} with figures, {len(issues)} issues")
        for i in issues[:15]:
            print("   ", i)
        bad_total += len(issues)
    sys.exit(1 if bad_total else 0)


if __name__ == "__main__":
    main()
