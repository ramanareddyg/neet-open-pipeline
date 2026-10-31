"""Build the fixed evaluation set (v1).

Usage: python scripts/build_eval.py

Writes data/eval/eval_v1.jsonl, one item per (question, language):
  {eval_id, source, qid, lang, subject, question, options[4], answer, images[], has_figure, style}

Contents:
  - NEET 2025 (booklet 45), all 180 questions, in English and in Hindi
    (Hindi from the bilingual booklet; figures from the English booklet).
  - 150 generated NCERT MCQs that passed the blind check, 50 per subject,
    stratified by style, in English and Hindi. These ids are recorded in
    data/eval/heldout_mcq_ids.json and must be excluded from training.

The selection is seeded and deterministic: re-running gives the same file.
"""
import glob
import json
import os
import random
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
N_PER_SUBJECT = 50


def load(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def main() -> None:
    out = []
    # 1. NEET 2025, both languages.
    for q in load(os.path.join(ROOT, "data", "questions", "2025_merged.jsonl")):
        subj = "Biology" if q["subject"] in ("Botany", "Zoology") else q["subject"]
        imgs = [f["path"] for f in q["figures"]]
        for lang in ("en", "hi"):
            text = q["text_en"] if lang == "en" else q["text_hi"]
            opts = [o["text_en"] if lang == "en" else o["text_hi"] for o in q["options"]]
            if not text or not all(opts) or len(opts) != 4:
                continue
            out.append({
                "eval_id": f"{q['id']}-{lang}", "source": "neet2025", "qid": q["id"], "lang": lang,
                "subject": subj, "subject_detail": q["subject"], "question": text, "options": opts,
                "answer": q["answer"], "images": imgs, "has_figure": bool(imgs), "style": None,
            })

    # 2. Held-out generated MCQs.
    verdict = {}
    for p in glob.glob(os.path.join(ROOT, "data", "synthetic", "mcq_verify", "*.jsonl")):
        for v in load(p):
            verdict[v["id"]] = v
    kept = [r for p in sorted(glob.glob(os.path.join(ROOT, "data", "synthetic", "mcq", "*.jsonl")))
            for r in load(p) if verdict.get(r["id"], {}).get("keep")]
    rng = random.Random(20251005)
    held = []
    for subj in ("Physics", "Chemistry", "Biology"):
        by_style = defaultdict(list)
        for r in kept:
            if r["subject"] == subj:
                by_style[r["style"]].append(r)
        for lst in by_style.values():
            rng.shuffle(lst)
        # round-robin across styles so every style is represented
        picked, styles = [], sorted(by_style)
        while len(picked) < N_PER_SUBJECT and any(by_style.values()):
            for s in styles:
                if by_style[s] and len(picked) < N_PER_SUBJECT:
                    picked.append(by_style[s].pop())
        held += picked
    for r in held:
        for lang in ("en", "hi"):
            out.append({
                "eval_id": f"{r['id']}-{lang}", "source": "ncert_mcq", "qid": r["id"], "lang": lang,
                "subject": r["subject"], "subject_detail": r["subject"],
                "question": r["question_en"] if lang == "en" else r["question_hi"],
                "options": r["options_en"] if lang == "en" else r["options_hi"],
                "answer": r["answer"], "images": [], "has_figure": False, "style": r["style"],
            })

    os.makedirs(os.path.join(ROOT, "data", "eval"), exist_ok=True)
    with open(os.path.join(ROOT, "data", "eval", "eval_v1.jsonl"), "w", encoding="utf-8") as f:
        for row in out:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with open(os.path.join(ROOT, "data", "eval", "heldout_mcq_ids.json"), "w", encoding="utf-8") as f:
        json.dump(sorted(r["id"] for r in held), f, indent=0)
    by = defaultdict(int)
    for row in out:
        by[(row["source"], row["lang"])] += 1
    print(f"{len(out)} eval items:", dict(by), f"| figure items: {sum(r['has_figure'] for r in out)}")


if __name__ == "__main__":
    main()
