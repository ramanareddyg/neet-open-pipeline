"""v3 training prompts: every past-paper and generated question, in the setting we ship.

Usage:
  python scripts/build_v3_prompts.py            # step 1: prompts + retrieval queries -> data/train/v3_prompts.jsonl
  python scripts/build_v3_prompts.py --render   # step 3: after modal_retrievers.py::retrieve_prompts wrote
                                                #         data/train/v3_passages.json, render the final user text
                                                #         -> data/train/v3_prompts_rendered.jsonl and
                                                #            data/eval/dev_probe_v3.jsonl

Differences from train_v1 (build_train.py):
  - one prompt per (question, language), no duplicates, no teacher answers yet;
  - the gold answer is the official key (past papers) or the verified key (MCQs);
  - every prompt will carry the top-1 NCERT passage retrieved by the offline
    embedder (Qwen3-Embedding-0.6B), exactly as the product would;
  - dev-probe questions and the 150 held-out MCQs are excluded; NEET 2025 never appears.
Fields: id, split (train|dev), source, lang, qid, subject, question, options, images, gold,
        query_en (English question text used for retrieval), chunk_oracle (reader's link, for stats).
"""
import glob
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_common as ec  # noqa: E402
from build_train import MERGED, PAPERS, load, passage_block  # noqa: E402

ROOT = ec.ROOT
OUT = os.path.join(ROOT, "data", "train", "v3_prompts.jsonl")


def single_key(ans) -> int | None:
    ks = [int(x) for x in re.findall(r"[1-4]", str(ans))]
    return ks[0] if len(ks) == 1 else None


def build() -> None:
    dev = set(json.load(open(os.path.join(ROOT, "data", "eval", "dev_probe_ids.json"), encoding="utf-8")))
    held = set(json.load(open(os.path.join(ROOT, "data", "eval", "heldout_mcq_ids.json"), encoding="utf-8")))
    out, skipped = [], {"no_single_key": 0, "dev": 0, "heldout": 0}
    for year, paper in PAPERS.items():
        links = {l["id"]: l["chunk_id"] for l in load(os.path.join(ROOT, "data", "links", f"{paper}.jsonl"))}
        hi_rows = {q["id"]: q for q in load(os.path.join(ROOT, "data", "questions", f"{MERGED[year]}.jsonl"))} if year in MERGED else {}
        for q in load(os.path.join(ROOT, "data", "questions", f"{paper}.jsonl")):
            if q["id"] in dev:
                skipped["dev"] += 1
                continue
            gold = single_key(q.get("answer"))
            if gold is None:
                skipped["no_single_key"] += 1
                continue
            imgs = [f["path"] for f in q.get("figures", [])]
            opts_en = [o["text_en"] for o in q["options"]]
            query = f"{q['text_en']}\n" + "\n".join(f"({i + 1}) {o}" for i, o in enumerate(opts_en))
            base = {"split": "train", "qid": q["id"], "subject": q["subject"], "images": imgs, "gold": gold,
                    "query_en": query, "chunk_oracle": links.get(q["id"])}
            out.append({"id": f"{q['id']}-en", "source": "pyq", "lang": "en", "question": q["text_en"], "options": opts_en, **base})
            h = hi_rows.get(q["id"])
            have_hi = bool(h and h.get("text_hi") and all(o.get("text_hi") for o in h["options"]))
            out.append({"id": f"{q['id']}-hi", "source": "pyq_hi_q" if have_hi else "pyq_en_q_hi_answer", "lang": "hi",
                        "question": h["text_hi"] if have_hi else q["text_en"],
                        "options": [o["text_hi"] for o in h["options"]] if have_hi else opts_en, **base})
    verdict = {v["id"]: v for p in glob.glob(os.path.join(ROOT, "data", "synthetic", "mcq_verify", "*.jsonl")) for v in load(p)}
    for p in sorted(glob.glob(os.path.join(ROOT, "data", "synthetic", "mcq", "*.jsonl"))):
        for r in load(p):
            if not verdict.get(r["id"], {}).get("keep"):
                continue
            if r["id"] in held:
                skipped["heldout"] += 1
                continue
            if r["id"] in dev:
                skipped["dev"] += 1
                continue
            query = f"{r['question_en']}\n" + "\n".join(f"({i + 1}) {o}" for i, o in enumerate(r["options_en"]))
            base = {"split": "train", "qid": r["id"], "subject": r["subject"], "images": [], "gold": int(r["answer"]),
                    "query_en": query, "chunk_oracle": r["source_chunk"], "style": r.get("style")}
            out.append({"id": f"{r['id']}-en", "source": "ncert_mcq", "lang": "en", "question": r["question_en"], "options": r["options_en"], **base})
            out.append({"id": f"{r['id']}-hi", "source": "ncert_mcq", "lang": "hi", "question": r["question_hi"], "options": r["options_hi"], **base})
    # dev probe: same shape, split=dev, so it gets passages too (the English text of each item is its own query;
    # Hindi dev items reuse the English item's query via qid)
    dev_items = [json.loads(l) for l in open(os.path.join(ROOT, "data", "eval", "dev_probe_v1.jsonl"), encoding="utf-8")]
    en_q = {it["qid"]: f"{it['question']}\n" + "\n".join(f"({i + 1}) {o}" for i, o in enumerate(it["options"])) for it in dev_items if it["lang"] == "en"}
    for it in dev_items:
        out.append({"id": it["eval_id"], "split": "dev", "source": it["source"], "lang": it["lang"], "qid": it["qid"], "subject": it["subject"],
                    "question": it["question"], "options": it["options"], "images": it["images"], "gold": it["answer"],
                    "query_en": en_q.get(it["qid"], f"{it['question']}"), "chunk_oracle": None})
    with open(OUT, "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    from collections import Counter
    print(f"{len(out)} prompts -> {OUT}; by source {dict(Counter(r['source'] for r in out))}; by split {dict(Counter(r['split'] for r in out))}; skipped {skipped}")


def render() -> None:
    chunks = {c["id"]: c for c in load(os.path.join(ROOT, "data", "ncert", "chunks.jsonl"))}
    pas = json.load(open(os.path.join(ROOT, "data", "train", "v3_passages.json"), encoding="utf-8"))
    rows = load(OUT)
    n_train = n_dev = 0
    with open(OUT.replace(".jsonl", "_rendered.jsonl"), "w", encoding="utf-8") as ft, \
            open(os.path.join(ROOT, "data", "eval", "dev_probe_v3.jsonl"), "w", encoding="utf-8") as fd:
        for r in rows:
            p = pas.get(r["qid"])
            chunk = chunks.get(p["chunk"]) if p else None
            item = {"lang": r["lang"], "question": r["question"], "options": r["options"], "images": r["images"]}
            r["passage_chunk"] = p["chunk"] if p else None
            r["user_text"] = passage_block(chunk, r["lang"]) + ec.prompt_text(item)
            if r["split"] == "dev":
                fd.write(json.dumps(r, ensure_ascii=False) + "\n")
                n_dev += 1
            else:
                ft.write(json.dumps(r, ensure_ascii=False) + "\n")
                n_train += 1
    hit = sum(1 for r in rows if r["chunk_oracle"] and pas.get(r["qid"]) and pas[r["qid"]]["chunk"] == r["chunk_oracle"])
    have = sum(1 for r in rows if r["chunk_oracle"])
    print(f"rendered {n_train} train prompts and {n_dev} dev prompts; retrieved passage == reader's link on {hit}/{have}")


if __name__ == "__main__":
    render() if "--render" in sys.argv else build()
