"""Can an offline embedder replace the Gemini embedding for NCERT retrieval?

Usage: python scripts/eval_retrievers.py [model ...]

For each sentence-transformers model: embed the 1,923 NCERT chunks (same text
as embed_ncert.py) and the 330 eval questions (English text, and the Hindi
text of the Hindi items), restrict to the question's subject, and report how
often the top-1 / top-3 passage equals the one a reader chose (300 questions
have one). Reference: gemini-embedding-001 top-1 = 188/300 with English queries.

Writes data/eval/retriever_eval.json and, per model, a passage map
data/eval/eval_v1_passages_<slug>.json in the same shape as eval_v1_passages.json
(retrieved = top-1 from the English query for both languages, plus retrieved_hi
= top-1 from the Hindi query) so modal_eval.py can use it with --passage-file.

Runs on CPU; the largest model (bge-m3, 568M) takes 20-30 min.
"""
import json
import os
import re
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_common as ec  # noqa: E402
from link_ncert import SUBJECT_MAP  # noqa: E402

DEFAULT = ["intfloat/multilingual-e5-base", "google/embeddinggemma-300m", "BAAI/bge-m3"]
PREFIX = {"intfloat/multilingual-e5-base": ("query: ", "passage: ")}  # e5 needs these; others use built-in prompts or none


def main() -> None:
    from sentence_transformers import SentenceTransformer

    models = sys.argv[1:] or DEFAULT
    chunks = [json.loads(l) for l in open(os.path.join(ec.ROOT, "data", "ncert", "chunks.jsonl"), encoding="utf-8")]
    ctexts = [f"{c['subject']} Class {c['class']} — {c['title']}\n\n{c['text']}" for c in chunks]
    csubj = np.array([c["subject"] for c in chunks])
    cids = [c["id"] for c in chunks]
    items = ec.load_eval()
    sel = json.load(open(os.path.join(ec.ROOT, "data", "eval", "eval_v1_passages.json"), encoding="utf-8"))
    by_qid = {}
    for it in items:
        q = by_qid.setdefault(it["qid"], {"subject": it["subject"], "oracle": sel[it["eval_id"]]["oracle"], "gemini": sel[it["eval_id"]]["retrieved"]})
        q[it["lang"]] = f"{it['question']}\n" + "\n".join(f"({i + 1}) {o}" for i, o in enumerate(it["options"]))
    qids = list(by_qid)
    results = json.load(open(os.path.join(ec.ROOT, "data", "eval", "retriever_eval.json"), encoding="utf-8")) \
        if os.path.exists(os.path.join(ec.ROOT, "data", "eval", "retriever_eval.json")) else {}

    def topk(qvecs, cvecs, subjects, k=3):
        out = []
        for v, s in zip(qvecs, subjects):
            sims = np.where(csubj == SUBJECT_MAP[s], cvecs @ v, -1.0)
            idx = np.argsort(-sims)[:k]
            out.append(([cids[i] for i in idx], float(sims[idx[0]])))
        return out

    for name in models:
        t0 = time.time()
        try:
            model = SentenceTransformer(name, device="cpu")
        except Exception as e:  # noqa: BLE001
            print(f"{name}: could not load ({str(e)[:120]})", flush=True)
            results[name] = {"error": str(e)[:200]}
            continue
        model.max_seq_length = min(1024, model.max_seq_length or 1024)  # e5 (XLM-R) tops out at 512
        qp, dp = PREFIX.get(name, ("", ""))
        kw_q = {"prompt_name": "query"} if "embeddinggemma" in name else {}
        kw_d = {"prompt_name": "document"} if "embeddinggemma" in name else {}
        cvecs = model.encode([dp + t for t in ctexts], batch_size=8, normalize_embeddings=True, show_progress_bar=False, **kw_d)
        t_chunks = time.time() - t0
        res = {"chunks_seconds": round(t_chunks), "dims": int(cvecs.shape[1])}
        pmap = {}
        for lang in ("en", "hi"):
            qs = [q for q in qids if lang in by_qid[q]]
            qvecs = model.encode([qp + by_qid[q][lang] for q in qs], batch_size=8, normalize_embeddings=True, show_progress_bar=False, **kw_q)
            tk = topk(qvecs, cvecs, [by_qid[q]["subject"] for q in qs])
            have = [(q, t) for q, t in zip(qs, tk) if by_qid[q]["oracle"]]
            res[f"{lang}_queries"] = len(qs)
            res[f"{lang}_hit1"] = sum(t[0][0] == by_qid[q]["oracle"] for q, t in have)
            res[f"{lang}_hit3"] = sum(by_qid[q]["oracle"] in t[0] for q, t in have)
            res[f"{lang}_have_oracle"] = len(have)
            res[f"{lang}_agree_with_gemini_top1"] = sum(t[0][0] == by_qid[q]["gemini"] for q, t in zip(qs, tk))
            for q, t in zip(qs, tk):
                pmap.setdefault(q, {})[f"top1_{lang}"] = t[0][0]
                pmap[q][f"score_{lang}"] = round(t[1], 3)
        res["total_seconds"] = round(time.time() - t0)
        results[name] = res
        print(f"{name}: EN top-1 {res['en_hit1']}/{res['en_have_oracle']}, top-3 {res['en_hit3']}; "
              f"HI top-1 {res['hi_hit1']}/{res['hi_have_oracle']}; {res['total_seconds']}s", flush=True)
        slug = re.sub(r"[^A-Za-z0-9]+", "-", name).strip("-").lower()
        out = {}
        for it in items:
            p = pmap[it["qid"]]
            out[it["eval_id"]] = {"retrieved": p["top1_en"], "retrieved_score": p["score_en"], "oracle": by_qid[it["qid"]]["oracle"],
                                  "retrieved_hi": p.get("top1_hi"), "retrieved_hi_score": p.get("score_hi")}
        json.dump(out, open(os.path.join(ec.ROOT, "data", "eval", f"eval_v1_passages_{slug}.json"), "w", encoding="utf-8"), indent=0)
        json.dump(results, open(os.path.join(ec.ROOT, "data", "eval", "retriever_eval.json"), "w", encoding="utf-8"), indent=1)
    print(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
