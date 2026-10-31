"""Build the small development probe used *during* training (v2 onwards).

Usage: python scripts/build_dev_probe.py

Writes data/eval/dev_probe_v1.jsonl (same item format as eval_v1.jsonl) and
data/eval/dev_probe_ids.json (question ids to exclude from training).

40 questions, each in English and Hindi = 80 items, text-only (no figures,
so the in-training check is fast):
  - 20 real NEET 2023 questions (stratified: 7 Physics, 6 Chemistry, 7 Biology)
    2023 is an English-only booklet, so the Hindi item is the English
    question with the Hindi instruction, as in training.
  - 20 generated NCERT MCQs that passed the blind check, not in the eval
    hold-out (stratified by subject).
Neither the dev items nor the eval items may appear in training; the dev set
exists so checkpoints can be compared without touching the eval set.
"""
import glob
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_common as ec  # noqa: E402

ROOT = ec.ROOT


def load(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def main() -> None:
    rng = random.Random(7)
    out, ids = [], []
    pyq = [q for q in load(os.path.join(ROOT, "data", "questions", "2023_en_E1.jsonl"))
           if not q["figures"] and isinstance(q["answer"], int) and len(q["options"]) == 4]
    for subj, n in (("Physics", 7), ("Chemistry", 6), ("Botany", 4), ("Zoology", 3)):
        pool = [q for q in pyq if q["subject"] == subj]
        rng.shuffle(pool)
        for q in pool[:n]:
            ids.append(q["id"])
            for lang in ("en", "hi"):
                out.append({"eval_id": f"{q['id']}-{lang}", "source": "dev_pyq2023", "qid": q["id"], "lang": lang,
                            "subject": "Biology" if subj in ("Botany", "Zoology") else subj, "question": q["text_en"],
                            "options": [o["text_en"] for o in q["options"]], "answer": q["answer"], "images": [],
                            "has_figure": False, "style": None})
    held = set(json.load(open(os.path.join(ROOT, "data", "eval", "heldout_mcq_ids.json"), encoding="utf-8")))
    verdict = {v["id"]: v for p in glob.glob(os.path.join(ROOT, "data", "synthetic", "mcq_verify", "*.jsonl")) for v in load(p)}
    mcq = [r for p in sorted(glob.glob(os.path.join(ROOT, "data", "synthetic", "mcq", "*.jsonl"))) for r in load(p)
           if verdict.get(r["id"], {}).get("keep") and r["id"] not in held]
    for subj, n in (("Physics", 7), ("Chemistry", 6), ("Biology", 7)):
        pool = [r for r in mcq if r["subject"] == subj]
        rng.shuffle(pool)
        for r in pool[:n]:
            ids.append(r["id"])
            for lang in ("en", "hi"):
                out.append({"eval_id": f"{r['id']}-{lang}", "source": "dev_mcq", "qid": r["id"], "lang": lang, "subject": subj,
                            "question": r["question_en"] if lang == "en" else r["question_hi"],
                            "options": r["options_en"] if lang == "en" else r["options_hi"], "answer": r["answer"],
                            "images": [], "has_figure": False, "style": r["style"]})
    with open(os.path.join(ROOT, "data", "eval", "dev_probe_v1.jsonl"), "w", encoding="utf-8") as f:
        for row in out:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    json.dump(sorted(ids), open(os.path.join(ROOT, "data", "eval", "dev_probe_ids.json"), "w"), indent=0)
    print(f"dev probe: {len(out)} items from {len(ids)} questions")


if __name__ == "__main__":
    main()
