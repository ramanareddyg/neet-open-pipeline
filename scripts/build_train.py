"""Assemble the supervised fine-tuning set (v1) for Gemma 4 E4B.

Usage: python scripts/build_train.py

Writes data/train/train_v1.jsonl, one chat example per line:
  {id, source, lang, with_passage, images: [paths], messages: [{role, content}]}
and data/train/train_v1_stats.json.

Every user turn uses exactly the eval prompt (eval_common.prompt_text), so
training and evaluation match; every assistant turn is a worked explanation
ending in "Answer: N".

Sources
  - PYQ explanations 2020-24 (data/synthetic/explain), only where the final
    answer matches the official key. Each question yields an English example
    and a Hindi example (Hindi question where the bilingual booklet has it,
    otherwise the English question with the Hindi instruction, answered in
    Hindi). Real papers are up-weighted x2: the second copy differs in
    whether the NCERT passage is shown.
  - Generated NCERT MCQs that passed the blind check, minus the 150 held out
    for eval (data/eval/heldout_mcq_ids.json), in English and Hindi.
    Assertion-reason items are rebalanced: "both true, R explains A" is cut
    to half of A/R items (it was 81%; real NEET ~50%), and a third of A/R
    items get shuffled option order, as NTA prints about a third of them.
  - ~25% of examples carry the NCERT passage before the question, so the
    model also learns to use a retrieved passage; the rest match the eval.

The 2025 paper never appears (held-out eval).
"""
import glob
import json
import os
import random
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_common as ec  # noqa: E402

ROOT = ec.ROOT
SEED = 20251005
PASSAGE_SHARE = 0.25
PAPERS = {2020: "2020_en_E1", 2021: "2021_en_M1", 2022: "2022_en_R6", 2023: "2023_en_E1", 2024: "2024_en_T1"}
MERGED = {2021: "2021_merged", 2024: "2024_merged"}


def load(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def passage_block(chunk: dict | None, lang: str) -> str:
    if not chunk:
        return ""
    head = (f"NCERT reference (Class {chunk['class']} {chunk['subject']}, \"{chunk['title']}\"):"
            if lang == "en" else f"NCERT संदर्भ (कक्षा {chunk['class']} {chunk['subject']}, \"{chunk['title']}\"):")
    return f"{head}\n{chunk['text']}\n\n"


def example(eid, source, lang, question, options, images, answer_text, chunk, with_passage):
    item = {"lang": lang, "question": question, "options": options, "images": images}
    user = (passage_block(chunk, lang) if with_passage else "") + ec.prompt_text(item)
    return {"id": eid, "source": source, "lang": lang, "with_passage": bool(with_passage and chunk), "images": images,
            "messages": [{"role": "user", "content": user}, {"role": "assistant", "content": answer_text}]}


def main() -> None:
    rng = random.Random(SEED)
    chunks = {c["id"]: c for c in load(os.path.join(ROOT, "data", "ncert", "chunks.jsonl"))}
    out = []
    stats = Counter()

    # ---- PYQ explanations ----
    for year, paper in PAPERS.items():
        qs = {q["id"]: q for q in load(os.path.join(ROOT, "data", "questions", f"{paper}.jsonl"))}
        hi_rows = {}
        if year in MERGED:
            hi_rows = {q["id"]: q for q in load(os.path.join(ROOT, "data", "questions", f"{MERGED[year]}.jsonl"))}
        for r in load(os.path.join(ROOT, "data", "synthetic", "explain", f"{paper}.jsonl")):
            if not r.get("final_answer_matches_key"):
                stats["pyq_skipped_no_key_match"] += 1
                continue
            q = qs[r["id"]]
            imgs = [f["path"] for f in q["figures"]]
            chunk = chunks.get(r.get("ncert_chunk_id") or "")
            ans = r["chosen_option"]
            opts_en = [o["text_en"] for o in q["options"]]
            h = hi_rows.get(r["id"])
            have_hi_q = bool(h and h.get("text_hi") and all(o.get("text_hi") for o in h["options"]))
            q_hi = h["text_hi"] if have_hi_q else q["text_en"]
            opts_hi = [o["text_hi"] for o in h["options"]] if have_hi_q else opts_en
            for copy, with_p in enumerate((False, True)):  # x2 up-weight; copies differ in passage
                if with_p and not chunk:
                    with_p = False
                out.append(example(f"{r['id']}-en-{copy}", "pyq", "en", q["text_en"], opts_en, imgs,
                                   f"{r['explanation_en'].strip()}\n\nAnswer: {ans}", chunk, with_p))
                out.append(example(f"{r['id']}-hi-{copy}", "pyq_hi_q" if have_hi_q else "pyq_en_q_hi_answer", "hi", q_hi, opts_hi, imgs,
                                   f"{r['explanation_hi'].strip()}\n\nAnswer: {ans}", chunk, with_p))
            stats["pyq_questions"] += 1
            stats["pyq_with_hindi_question"] += have_hi_q

    # ---- Generated MCQs ----
    held = set(json.load(open(os.path.join(ROOT, "data", "eval", "heldout_mcq_ids.json"), encoding="utf-8")))
    verdict = {v["id"]: v for p in glob.glob(os.path.join(ROOT, "data", "synthetic", "mcq_verify", "*.jsonl")) for v in load(p)}
    mcq = [r for p in sorted(glob.glob(os.path.join(ROOT, "data", "synthetic", "mcq", "*.jsonl"))) for r in load(p)
           if verdict.get(r["id"], {}).get("keep") and r["id"] not in held]
    stats["mcq_kept_not_heldout"] = len(mcq)

    # Assertion-reason rebalance.
    ar = [r for r in mcq if r["style"] == "assertion_reason"]
    ar1 = [r for r in ar if r["answer"] == 1]
    ar_other = [r for r in ar if r["answer"] != 1]
    rng.shuffle(ar1)
    drop = {r["id"] for r in ar1[len(ar_other):]}  # keep as many answer-1 as all other answers combined
    mcq = [r for r in mcq if r["id"] not in drop]
    stats["ar_dropped_answer1"] = len(drop)
    for r in mcq:
        if r["style"] == "assertion_reason" and rng.random() < 1 / 3:
            perm = list(range(4))
            while perm == [0, 1, 2, 3]:
                rng.shuffle(perm)
            r["options_en"] = [r["options_en"][k] for k in perm]
            r["options_hi"] = [r["options_hi"][k] for k in perm]
            r["answer"] = perm.index(r["answer"] - 1) + 1
            stats["ar_option_order_shuffled"] += 1
    stats["ar_answer_dist_after"] = dict(Counter(r["answer"] for r in mcq if r["style"] == "assertion_reason"))

    for r in mcq:
        chunk = chunks.get(r["source_chunk"])
        for lang in ("en", "hi"):
            with_p = rng.random() < PASSAGE_SHARE
            q = r["question_en"] if lang == "en" else r["question_hi"]
            opts = r["options_en"] if lang == "en" else r["options_hi"]
            expl = r["explanation_en"] if lang == "en" else r["explanation_hi"]
            out.append(example(f"{r['id']}-{lang}", "ncert_mcq", lang, q, opts, [], f"{expl.strip()}\n\nAnswer: {r['answer']}", chunk, with_p))

    rng.shuffle(out)
    os.makedirs(os.path.join(ROOT, "data", "train"), exist_ok=True)
    with open(os.path.join(ROOT, "data", "train", "train_v1.jsonl"), "w", encoding="utf-8") as f:
        for row in out:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    stats["examples"] = len(out)
    stats["by_source"] = dict(Counter(r["source"] for r in out))
    stats["by_lang"] = dict(Counter(r["lang"] for r in out))
    stats["with_passage"] = sum(r["with_passage"] for r in out)
    stats["with_images"] = sum(bool(r["images"]) for r in out)
    stats["answer_dist"] = dict(Counter(r["messages"][1]["content"].rsplit("Answer: ", 1)[1].strip() for r in out))
    stats["approx_chars"] = sum(len(r["messages"][0]["content"]) + len(r["messages"][1]["content"]) for r in out)
    with open(os.path.join(ROOT, "data", "train", "train_v1_stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=1)
    print(json.dumps(stats, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
