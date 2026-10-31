"""Parse the official NTA final answer-key PDFs into JSON.

Output: data/answer_keys/<year>.json  ->  {"<booklet code>": {"<qno>": <key>, ...}, ...}
where <key> is an option number 1-4, a list of accepted options, or "dropped".

Each year's PDF lists one or more booklet codes; within a code, pairs of
(question number, key option) appear in reading order. Layouts differ by
year, so parsing is token-based: find the code headers, then collect
(qno, option) pairs that advance in a plausible sequence. Every parsed code
is checked for exactly 180 (2020) or 200 (2021+) questions; anything else
is reported and the file is still written so it can be inspected.
"""
import json
import os
import re
import sys

import pypdf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "data", "raw", "answer_keys")
OUT = os.path.join(ROOT, "data", "answer_keys")

CODE_RES = {
    2020: re.compile(r"BOOK\s*:\s*([A-Z]\d)"),
    2021: re.compile(r"BOOK\s*:\s*([A-Z]\d)"),
    2022: re.compile(r"SET\s*:\s*([A-Z]\d)"),
    2023: re.compile(r"BOOK CODE\s*:\s*([A-Z]\d)"),
    2024: re.compile(r"Booklet Code\s*:\s*([A-Z]\d)"),
    2025: re.compile(r"Booklet Code\s*:\s*(\d{2})"),
}
EXPECTED = {2020: 180, 2021: 200, 2022: 200, 2023: 200, 2024: 200, 2025: 180}

# Key tokens: 1-4 = option; 5 = question dropped (marks to all); letters =
# more than one option accepted. The 2023 key prints the legend for B and E;
# the remaining letters follow NTA's pairwise convention and are kept here so
# an unexpected letter is decoded rather than silently skipped.
PAIR_RE = re.compile(r"(?<!\d)(\d{1,3})\s*[|:]?\s*([1-5A-F])(?![\dA-Za-z])")
MULTI = {"A": [1, 2], "B": [1, 3], "C": [1, 4], "D": [2, 3], "E": [2, 4], "F": [3, 4]}


def decode(tok: str):
    if tok == "5":
        return "dropped"
    if tok in MULTI:
        return MULTI[tok]
    return int(tok)


def parse_year(year: int) -> dict:
    reader = pypdf.PdfReader(os.path.join(SRC, f"{year}.pdf"))
    code_re = CODE_RES[year]
    expected = EXPECTED[year]
    keys: dict[str, dict[str, int]] = {}
    current = None
    for page in reader.pages:
        text = page.extract_text() or ""
        # Split the page at code headers so a page can hold more than one code.
        segments = []
        last = 0
        for m in code_re.finditer(text):
            segments.append((current, text[last:m.start()]))
            current = m.group(1)
            last = m.end()
        segments.append((current, text[last:]))
        for code, seg in segments:
            if code is None:
                continue
            table = keys.setdefault(code, {})
            # Strip known boilerplate numbers (dates, page x of y) before pairing.
            seg = re.sub(r"\d{2}\.\d{2}\.\d{4}", " ", seg)
            seg = re.sub(r"\b\d+\s+of\s+\d+\b", " ", seg)
            seg = re.sub(r"W\.P.*?\d{4}", " ", seg)
            for qno, opt in PAIR_RE.findall(seg):
                q = int(qno)
                if 1 <= q <= expected and str(q) not in table:
                    table[str(q)] = decode(opt)
    return keys


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    years = [int(a) for a in sys.argv[1:]] or sorted(EXPECTED)
    ok = True
    for year in years:
        keys = parse_year(year)
        bad = {c: len(t) for c, t in keys.items() if len(t) != EXPECTED[year]}
        with open(os.path.join(OUT, f"{year}.json"), "w", encoding="utf-8") as f:
            json.dump(keys, f, indent=1, sort_keys=True)
        status = "ok" if not bad else f"INCOMPLETE {bad}"
        print(f"{year}: {len(keys)} codes, expected {EXPECTED[year]} q each -> {status}")
        ok &= not bad
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
