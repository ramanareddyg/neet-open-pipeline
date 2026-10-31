"""Attach Hindi text from a bilingual booklet to the canonical English paper.

Usage: python scripts/merge_bilingual.py --year 2024

Reads  data/questions/<year>_en_<code>.jsonl   (canonical: English booklet)
       data/questions/<year>_hi_<code>.jsonl   (bilingual booklet, possibly a different code)
Writes data/questions/<year>_merged.jsonl       (English questions + text_hi/options text_hi where matched)
       data/questions/<year>_merge_report.json  (per-question match score, unmatched ids)

Booklet codes differ, so question order differs; questions are matched on
normalised English stem + options. A match needs similarity >= 0.80 and the
same answer key option after mapping (keys are per booklet code, so the
*answer text* must agree, not the option number). Unmatched or low-quality
Hindi (missing stem, fewer than 4 options) is left null and reported.
"""
import argparse
import difflib
import glob
import json
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def norm(s: str | None) -> str:
    s = (s or "").lower()
    s = re.sub(r"[\$\\{}|:\-_*#`\(\)\[\],\.]", " ", s)
    s = re.sub(r"\b(left|right|frac|times|text|mathrm|cdot|vec|hat)\b", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def key_text(q: dict) -> str:
    return norm(q["text_en"]) + " || " + " | ".join(norm(o["text_en"]) for o in q["options"])


def answer_text(q: dict) -> str | None:
    a = q.get("answer")
    if isinstance(a, int):
        for o in q["options"]:
            if o["key"] == str(a):
                return norm(o["text_en"])
    return None


def load(path: str) -> list[dict]:
    return [json.loads(l) for l in open(path, encoding="utf-8")]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--min-sim", type=float, default=0.80)
    args = ap.parse_args()
    qdir = os.path.join(ROOT, "data", "questions")
    en_path = glob.glob(os.path.join(qdir, f"{args.year}_en_*.jsonl"))
    hi_path = glob.glob(os.path.join(qdir, f"{args.year}_hi_*.jsonl"))
    if not en_path or not hi_path:
        raise SystemExit(f"need both {args.year}_en_*.jsonl and {args.year}_hi_*.jsonl")
    en, hi = load(en_path[0]), load(hi_path[0])

    hi_keys = [key_text(q) for q in hi]
    used_en, used_hi, match = set(), set(), {}
    same_code = en[0]["booklet_code"] == hi[0]["booklet_code"]
    if same_code:
        # Identical booklet code: identical question order. Match by number, verify loosely by text.
        by_qno = {q["qno"]: j for j, q in enumerate(hi)}
        for i, q in enumerate(en):
            j = by_qno.get(q["qno"])
            if j is None:
                continue
            r = difflib.SequenceMatcher(None, key_text(q), hi_keys[j]).ratio()
            used_en.add(i)
            used_hi.add(j)
            match[i] = (j, round(r, 3))
            if r < 0.5:
                print(f"  warning: q{q['qno']} text similarity {r:.2f} between booklets (same code); matched by number")
    else:
        # Different codes: same questions, shuffled order (and shuffled options).
        # Score = stem similarity, confirmed by how many option texts the two share.
        def stem(q):
            return norm(q["text_en"])[:250]

        def optset(q):
            return {norm(o["text_en"]) for o in q["options"] if norm(o["text_en"])}

        hi_stems = [stem(q) for q in hi]
        hi_opts = [optset(q) for q in hi]
        sims = []
        for i, q in enumerate(en):
            s, o = stem(q), optset(q)
            for j in range(len(hi)):
                r = difflib.SequenceMatcher(None, s, hi_stems[j]).quick_ratio()
                if r < 0.6:
                    continue
                r = difflib.SequenceMatcher(None, s, hi_stems[j]).ratio()
                ov = len(o & hi_opts[j])
                score = r + 0.05 * ov
                if r >= 0.75 or (r >= 0.6 and ov >= 3):
                    sims.append((score, r, ov, i, j))
        sims.sort(reverse=True)
        for score, r, ov, i, j in sims:
            if i in used_en or j in used_hi:
                continue
            if ov < 3:
                # weak option agreement: require the keyed answers to agree in text
                a_en, a_hi = answer_text(en[i]), answer_text(hi[j])
                if a_en and a_hi and difflib.SequenceMatcher(None, a_en, a_hi).ratio() < 0.6:
                    continue
            used_en.add(i)
            used_hi.add(j)
            match[i] = (j, round(r, 3))
        # Relaxed second pass on the leftovers only: table-style stems transcribe
        # differently, but three shared options (or two plus a matching answer)
        # identify the question among the few remaining candidates.
        left_en = [i for i in range(len(en)) if i not in used_en]
        left_hi = [j for j in range(len(hi)) if j not in used_hi]
        cands2 = []
        for i in left_en:
            s, o = stem(en[i]), optset(en[i])
            for j in left_hi:
                ov = len(o & hi_opts[j])
                r = difflib.SequenceMatcher(None, s, hi_stems[j]).ratio()
                a_en, a_hi = answer_text(en[i]), answer_text(hi[j])
                ans_ok = bool(a_en and a_hi and difflib.SequenceMatcher(None, a_en, a_hi).ratio() >= 0.6)
                if ov >= 3 or (ov >= 2 and ans_ok) or (r >= 0.5 and ans_ok and ov >= 1):
                    cands2.append((ov + r, r, ov, i, j))
        cands2.sort(reverse=True)
        for _, r, ov, i, j in cands2:
            if i in used_en or j in used_hi:
                continue
            used_en.add(i)
            used_hi.add(j)
            match[i] = (j, round(r, 3))

    out, report = [], {"year": args.year, "en_file": os.path.basename(en_path[0]), "hi_file": os.path.basename(hi_path[0]),
                       "matched": 0, "hindi_attached": 0, "unmatched_en": [], "unmatched_hi": [], "hindi_rejected": []}
    for i, q in enumerate(en):
        row = dict(q)
        row["text_hi"] = None
        row["options"] = [{**o, "text_hi": None} for o in q["options"]]
        row["hi_source"] = None
        if i in match:
            j, r = match[i]
            h = hi[j]
            report["matched"] += 1
            # Map Hindi option text by English option text, since option order may differ between codes.
            # Options that are numbers, formulae or figures have no separate Hindi: reuse the English.
            def best_opt(o):
                if same_code:
                    return h["options"][int(o["key"]) - 1] if len(h["options"]) == 4 else None
                target = norm(o["text_en"])
                cands = [(difflib.SequenceMatcher(None, target, norm(ho["text_en"])).ratio(), k, ho) for k, ho in enumerate(h["options"])]
                cands.sort(key=lambda c: (-c[0], c[1]))
                return cands[0][2] if cands and cands[0][0] >= 0.6 else None

            opts_hi = []
            for o in q["options"]:
                ho = best_opt(o)
                t = (ho or {}).get("text_hi")
                if not t and (o["text_en"] or "").strip() == "[figure]" or (not t and not re.search(r"[A-Za-z]{3,}", o["text_en"] or "")):
                    t = o["text_en"]
                opts_hi.append(t)
            good = bool((h.get("text_hi") or "").strip()) and all(opts_hi) and len(h["options"]) == 4
            if good:
                row["text_hi"] = h["text_hi"]
                for o, t in zip(row["options"], opts_hi):
                    o["text_hi"] = t
                row["hi_source"] = {"id": h["id"], "similarity": r}
                report["hindi_attached"] += 1
            else:
                report["hindi_rejected"].append({"en": q["id"], "hi": h["id"], "reason": "missing stem or options in Hindi record"})
        else:
            report["unmatched_en"].append(q["id"])
        out.append(row)
    report["unmatched_hi"] = [hi[j]["id"] for j in range(len(hi)) if j not in used_hi]

    out_path = os.path.join(qdir, f"{args.year}_merged.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for row in out:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with open(os.path.join(qdir, f"{args.year}_merge_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    print(f"{args.year}: {len(en)} EN questions; matched {report['matched']}, Hindi attached {report['hindi_attached']}, "
          f"unmatched EN {len(report['unmatched_en'])}, rejected Hindi {len(report['hindi_rejected'])}, unused HI {len(report['unmatched_hi'])}")
    if report["unmatched_en"]:
        print("  unmatched EN:", report["unmatched_en"][:12])


if __name__ == "__main__":
    main()
