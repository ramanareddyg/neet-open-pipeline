"""Assemble per-page extractions into one question file per paper.

Usage:
  python scripts/assemble_questions.py --year 2025 --lang en

Reads  data/extracted/<year>/<lang>/<code>/p*.json
Writes data/questions/<year>_<lang>_<code>.jsonl   (one question per line)
       data/figures/<year>/<lang>/<code>/q<NNN>[_optN][_k].png   (cropped figures)

Per question: qno, subject, section, text_en/text_hi, options, answer
(from the official key), figures (paths + descriptions), flags, source
page(s). Questions split over a page boundary are stitched. Validation
prints anything that needs eyes: wrong question count, missing options,
duplicate numbers, key mismatches.
"""
import argparse
import glob
import json
import os
import re
import sys

import yaml
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# NEET structure by year. 2020/2025: 180 questions, 45 per subject, no
# sections. 2021-2024: 200 questions, 50 per subject = Section A (35) +
# Section B (15, attempt any 10).
STRUCTURE = {
    180: [("Physics", 1, 45), ("Chemistry", 46, 90), ("Botany", 91, 135), ("Zoology", 136, 180)],
    200: [("Physics", 1, 50), ("Chemistry", 51, 100), ("Botany", 101, 150), ("Zoology", 151, 200)],
}
EXPECTED = {2020: 180, 2021: 200, 2022: 200, 2023: 200, 2024: 200, 2025: 180}
# Years whose booklet orders subjects differently. NEET 2020 prints Biology
# first (1-90, not split into Botany/Zoology), then Chemistry, then Physics.
YEAR_STRUCTURE = {
    2020: [("Biology", 1, 90), ("Chemistry", 91, 135), ("Physics", 136, 180)],
}


def subject_for(qno: int, total: int, year: int | None = None) -> tuple[str, str | None]:
    for subj, lo, hi in YEAR_STRUCTURE.get(year) or STRUCTURE[total]:
        if lo <= qno <= hi:
            if total == 200:
                return subj, "A" if qno - lo < 35 else "B"
            return subj, None
    return "Unknown", None


# The model sometimes emits "\rho" etc. as a JSON escape, so the decoded text
# holds a control character followed by the rest of the command. Map each
# control char back to the LaTeX commands it could have started.
_CTRL_CMDS = {
    "\r": ["rho", "right", "rangle", "rceil", "rfloor", "rightarrow", "rightleftharpoons", "rightharpoonup", "ref", "rm"],
    "\t": ["theta", "times", "text", "tan", "tau", "to", "tilde", "textbf", "textit", "textgreater", "textless", "textrm", "therefore", "top", "triangle", "tfrac"],
    "\f": ["frac", "forall", "flat", "fbox"],
    "\n": ["nu", "neq", "nabla", "not", "ne", "neg", "newline", "nonumber", "nolimits"],
    "\b": ["beta", "bar", "begin", "bigcup", "bigcap", "bullet", "binom", "boldsymbol", "bigg", "big", "bmod", "bot"],
}
_CTRL_LETTER = {"\r": "r", "\t": "t", "\f": "f", "\n": "n", "\b": "b"}
# The control char stands for backslash + first letter; the rest of the command follows it.
_CTRL_RE = re.compile(
    "|".join(
        f"(?P<c{i}>{re.escape(c)})(?=({'|'.join(sorted({cmd[1:] for cmd in cmds}, key=len, reverse=True))})(?![A-Za-z]))"
        for i, (c, cmds) in enumerate(_CTRL_CMDS.items())
    )
)


def repair_latex(s: str | None) -> str | None:
    if not s:
        return s
    return _CTRL_RE.sub(lambda m: "\\" + _CTRL_LETTER[m.group(0)], s)


def crop_figure(page_png: str, box: list[int], out_path: str) -> None:
    im = Image.open(page_png)
    w, h = im.size
    y0, x0, y1, x1 = box
    # Model boxes run tight on small structures; pad generously, more so on the
    # right where subscripts and trailing groups get clipped.
    pad = 14
    crop = im.crop((max(0, x0 * w // 1000 - pad), max(0, y0 * h // 1000 - pad),
                    min(w, x1 * w // 1000 + pad * 2), min(h, y1 * h // 1000 + pad)))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    crop.save(out_path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--lang", choices=["en", "hi"], required=True)
    args = ap.parse_args()

    with open(os.path.join(ROOT, "data", "papers.yaml"), encoding="utf-8") as f:
        papers = yaml.safe_load(f)["papers"]
    entry = next(p for p in papers if p["year"] == args.year and p["lang"] == args.lang)
    code = str(entry["code"])
    total = EXPECTED[args.year]

    ex_dir = os.path.join(ROOT, "data", "extracted", str(args.year), args.lang, code)
    page_files = sorted(glob.glob(os.path.join(ex_dir, "p*.json")))
    if not page_files:
        sys.exit(f"nothing extracted in {ex_dir}")

    with open(os.path.join(ROOT, "data", "answer_keys", f"{args.year}.json"), encoding="utf-8") as f:
        key = json.load(f)[code]

    questions: dict[int, dict] = {}
    pending = None  # question flagged incomplete, awaiting continuation
    problems = []
    last_qno = 0
    for pf in page_files:
        with open(pf, encoding="utf-8") as f:
            page = json.load(f)
        pno = page["page"]
        page_png = os.path.join(ROOT, page["source_png"])
        if str(page.get("split_into", "")).startswith("boxes"):
            # Per-crop transcription can misread printed numbers. The page's
            # questions must continue from the previous page; if they don't form
            # a contiguous run from last_qno+1, renumber in reading order.
            qs = [q for q in page["questions"] if not q.get("continuation")]
            got = [q["qno"] for q in qs]
            want = list(range(last_qno + 1, last_qno + 1 + len(qs)))
            # Gaps are missing questions, not misnumbering; only a non-ascending or
            # out-of-range sequence indicates the crops' numbers were misread.
            ascending = all(a < b for a, b in zip(got, got[1:]))
            plausible = ascending and got and got[0] > last_qno and got[-1] <= total
            if qs and not plausible:
                box_pos = {b["qno"]: b["box_2d"] for b in page.get("boxes", [])}
                qs.sort(key=lambda q: (box_pos.get(q["qno"], [0, 0])[1] >= 500, box_pos.get(q["qno"], [0, 0])[0]))
                for q, new in zip(qs, want):
                    if q["qno"] != new:
                        q["notes"] = f"renumbered_from:{q['qno']}" + (f"; {q['notes']}" if q.get("notes") else "")
                        q["qno"] = new
                problems.append(f"p{pno:03d}: box-split page renumbered {got} -> {want} (verify)")
        for q in page["questions"]:
            qno = q["qno"]
            # A "continuation" that carries the *next* question's number is the
            # next question misflagged (common on crops), not a tail: treat it
            # as a new question rather than stitching it onto the pending one.
            if q.get("continuation") and pending is not None and qno not in (0, None, pending["qno"]) and qno == pending["qno"] + 1:
                q["continuation"] = False
                q["notes"] = "continuation_flag_ignored_next_qno" + (f"; {q['notes']}" if q.get("notes") else "")
            if q.get("continuation") and pending is not None:
                # Stitch onto the pending question regardless of the qno the
                # model guessed for the fragment; fragments often lack a number.
                tgt = pending
                if q["text_en"].strip():
                    tgt["text_en"] = (tgt["text_en"].rstrip() + "\n" + q["text_en"].strip()).strip()
                if q.get("text_hi") and q["text_hi"].strip():
                    tgt["text_hi"] = ((tgt.get("text_hi") or "").rstrip() + "\n" + q["text_hi"].strip()).strip()
                have = {o["key"] for o in tgt["options"]}
                tgt["options"] += [o for o in q["options"] if o["key"] not in have]
                tgt["_figs"] += [(page_png, fg) for fg in q.get("figures", [])]
                tgt["pages"].append(pno)
                pending = None
                continue
            if not (1 <= qno <= total):
                problems.append(f"p{pno:03d}: dropped fragment with qno {qno}")
                continue
            if qno in questions:
                problems.append(f"duplicate qno {qno} on p{pno:03d} (first on p{questions[qno]['pages'][0]:03d})")
                continue
            rec = {
                "qno": qno,
                "text_en": q["text_en"],
                "text_hi": q.get("text_hi"),
                "options": q["options"],
                "section_header": q.get("section_header"),
                "notes": q.get("notes"),
                "pages": [pno],
                "_figs": [(page_png, fg) for fg in q.get("figures", [])],
            }
            questions[qno] = rec
            last_qno = max(last_qno, qno)
            pending = rec if q.get("incomplete") else None

    # Finalise: subjects, answers, figures.
    out_rows = []
    fig_dir = os.path.join(ROOT, "data", "figures", str(args.year), args.lang, code)  # per language: en and bilingual booklets can share a code
    for qno in sorted(questions):
        rec = questions[qno]
        subj, sec = subject_for(qno, total, args.year)
        figs = []
        for i, (png, fg) in enumerate(rec.pop("_figs")):
            if len(fg.get("box_2d") or []) != 4:
                problems.append(f"q{qno}: figure box malformed {fg.get('box_2d')}; skipped")
                continue
            tag = "" if fg["belongs_to"] == "question" else "_" + fg["belongs_to"].replace("option_", "opt")
            fname = f"q{qno:03d}{tag}{'' if i == 0 else f'_{i}'}.png"
            crop_figure(png, fg["box_2d"], os.path.join(fig_dir, fname))
            figs.append({
                "path": f"data/figures/{args.year}/{args.lang}/{code}/{fname}",
                "belongs_to": fg["belongs_to"],
                "description_en": fg["description_en"],
            })
        if args.lang == "en":
            # English-only booklet: any Hindi the model produced is its own
            # translation, not what is printed. Keep only transcribed text.
            rec["text_hi"] = None
            for o in rec["options"]:
                o["text_hi"] = None
        opts = sorted(rec["options"], key=lambda o: o["key"])
        if [o["key"] for o in opts] != ["1", "2", "3", "4"]:
            problems.append(f"q{qno}: options {[o['key'] for o in opts]}")
        ans = key.get(str(qno))
        if ans is None:
            problems.append(f"q{qno}: no key entry")
        out_rows.append({
            "id": f"neet-{args.year}-{code}-{qno:03d}",
            "year": args.year,
            "booklet_code": code,
            "qno": qno,
            "subject": subj,
            "section": sec,
            "text_en": repair_latex(rec["text_en"]),
            "text_hi": repair_latex(rec["text_hi"]),
            "options": [{"key": o["key"], "text_en": repair_latex(o["text_en"]), "text_hi": repair_latex(o.get("text_hi"))} for o in opts],
            "answer": ans,
            "figures": figs,
            "has_figure": bool(figs),
            "section_header": rec["section_header"],
            "notes": rec["notes"],
            "source_pages": rec["pages"],
        })

    missing = [q for q in range(1, total + 1) if q not in questions]
    if missing:
        problems.append(f"missing qnos: {missing}")
    extra = [q for q in questions if q > total or q < 1]
    if extra:
        problems.append(f"out-of-range qnos: {extra}")

    os.makedirs(os.path.join(ROOT, "data", "questions"), exist_ok=True)
    out = os.path.join(ROOT, "data", "questions", f"{args.year}_{args.lang}_{code}.jsonl")
    with open(out, "w", encoding="utf-8") as f:
        for row in out_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    n_fig = sum(len(r["figures"]) for r in out_rows)
    n_hi = sum(1 for r in out_rows if r["text_hi"])
    print(f"{args.year} {args.lang} {code}: {len(out_rows)}/{total} questions, {n_fig} figures, {n_hi} with Hindi -> {os.path.relpath(out, ROOT)}")
    for p in problems:
        print("  PROBLEM:", p)
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
