"""Iteratively repair a paper: find broken/missing questions, fix them by number, re-assemble.

Usage: python scripts/fix_paper.py --year 2022 --lang en [--rounds 2]

A question is "broken" if it is missing, has an empty stem, or does not have
exactly four options. Each round runs extract_pages.py --fix-qnos on that
list, then assemble_questions.py, and stops when nothing is broken or the
list stops shrinking. Prints the final residue so it can go on the verify list.
"""
import argparse
import glob
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXPECTED = {2020: 180, 2021: 200, 2022: 200, 2023: 200, 2024: 200, 2025: 180}
ENV = {**os.environ, "PYTHONIOENCODING": "utf-8"}


def run(args: list[str]) -> str:
    r = subprocess.run([sys.executable] + args, cwd=ROOT, env=ENV, capture_output=True, text=True, encoding="utf-8", errors="replace")
    return (r.stdout or "") + (r.stderr or "")


def broken(year: int, lang: str) -> list[int]:
    path = glob.glob(os.path.join(ROOT, "data", "questions", f"{year}_{lang}_*.jsonl"))
    if not path:
        return list(range(1, EXPECTED[year] + 1))
    rows = {}
    for l in open(path[0], encoding="utf-8"):
        q = json.loads(l)
        rows[q["qno"]] = q
    out = []
    for n in range(1, EXPECTED[year] + 1):
        q = rows.get(n)
        if q is None or len(q["options"]) != 4 or not (q["text_en"] or "").strip():
            out.append(n)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--lang", choices=["en", "hi"], required=True)
    ap.add_argument("--rounds", type=int, default=2)
    args = ap.parse_args()

    print(run(["scripts/assemble_questions.py", "--year", str(args.year), "--lang", args.lang]).splitlines()[0])
    prev = None
    for rnd in range(1, args.rounds + 1):
        bad = broken(args.year, args.lang)
        print(f"round {rnd}: {len(bad)} broken/missing -> {bad}", flush=True)
        if not bad or (prev is not None and len(bad) >= len(prev)):
            break
        prev = bad
        out = run(["scripts/extract_pages.py", "--year", str(args.year), "--lang", args.lang, "--fix-qnos", ",".join(map(str, bad))])
        print("\n".join(l for l in out.splitlines() if l.startswith("q")), flush=True)
        print(run(["scripts/assemble_questions.py", "--year", str(args.year), "--lang", args.lang]).splitlines()[0])
    final = broken(args.year, args.lang)
    print(f"RESIDUE {args.year} {args.lang}: {len(final)} -> {final}")


if __name__ == "__main__":
    main()
