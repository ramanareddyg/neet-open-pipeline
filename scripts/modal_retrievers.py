"""Offline-embedder retrieval test on a Modal GPU (the CPU version, eval_retrievers.py, takes hours).

Usage:
  modal run scripts/modal_retrievers.py
  modal run scripts/modal_retrievers.py --models "BAAI/bge-m3,Qwen/Qwen3-Embedding-0.6B"

Encodes the NCERT chunks and the eval questions (English and Hindi) with each
sentence-transformers model on one L4, then scores top-1 / top-3 against the
reader-chosen passage locally and writes the same files as eval_retrievers.py:
data/eval/retriever_eval.json and data/eval/eval_v1_passages_<slug>.json.
Gated models (google/embeddinggemma-300m) need the Modal secret `huggingface`
with HF_TOKEN; they are skipped if it is absent.
"""
import json
import os
import re
import sys
import time

import modal

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

app = modal.App("neet-open-retrievers")
# vLLM's image already carries a working torch/transformers; pip-resolving them on debian_slim failed.
image = (modal.Image.from_registry("vllm/vllm-openai:latest")
         .entrypoint([])
         .run_commands("ln -sf \"$(which python3)\" /usr/local/bin/python")
         .run_commands("python -m pip install --no-cache-dir -U transformers sentence-transformers")  # EmbeddingGemma 2 (6 Oct 2026) needs the newest transformers
         .env({"HF_XET_HIGH_PERFORMANCE": "1"}))
hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)
PREFIX = {"intfloat/multilingual-e5-base": ("query: ", "passage: "), "intfloat/multilingual-e5-large": ("query: ", "passage: ")}
DEFAULT = "intfloat/multilingual-e5-base,intfloat/multilingual-e5-large,BAAI/bge-m3,Qwen/Qwen3-Embedding-0.6B,google/embeddinggemma-300m"
_secrets = []
try:
    _secrets = [modal.Secret.from_name("huggingface")]
except Exception:  # noqa: BLE001
    pass


@app.function(image=image, gpu="L4", timeout=3600, volumes={"/root/.cache/huggingface": hf_cache}, secrets=_secrets)
def encode(name: str, docs: list[str], queries: dict[str, list[str]]) -> dict:
    import numpy as np
    from sentence_transformers import SentenceTransformer

    t0 = time.time()
    model = SentenceTransformer(name, device="cuda", trust_remote_code=True)
    model.max_seq_length = min(1024, model.max_seq_length or 1024)
    qp, dp = PREFIX.get(name, ("", ""))
    kw_q = {"prompt_name": "query"} if "embeddinggemma" in name or "Qwen3-Embedding" in name else {}
    kw_d = {"prompt_name": "document"} if "embeddinggemma" in name else {}
    out = {"dims": None, "seconds": None, "docs": None, "queries": {}}
    d = model.encode([dp + t for t in docs], batch_size=32, normalize_embeddings=True, show_progress_bar=False, **kw_d)
    out["docs"] = d.astype(np.float16).tolist()
    out["dims"] = int(d.shape[1])
    for lang, qs in queries.items():
        q = model.encode([qp + t for t in qs], batch_size=32, normalize_embeddings=True, show_progress_bar=False, **kw_q)
        out["queries"][lang] = q.astype(np.float16).tolist()
    hf_cache.commit()
    out["seconds"] = round(time.time() - t0, 1)
    return out


@app.local_entrypoint()
def main(models: str = DEFAULT):
    import numpy as np

    import eval_common as ec
    from link_ncert import SUBJECT_MAP

    chunks = [json.loads(l) for l in open(os.path.join(ec.ROOT, "data", "ncert", "chunks.jsonl"), encoding="utf-8")]
    docs = [f"{c['subject']} Class {c['class']} — {c['title']}\n\n{c['text']}" for c in chunks]
    csubj = np.array([c["subject"] for c in chunks])
    cids = [c["id"] for c in chunks]
    items = ec.load_eval()
    sel = json.load(open(os.path.join(ec.ROOT, "data", "eval", "eval_v1_passages.json"), encoding="utf-8"))
    by_qid = {}
    for it in items:
        q = by_qid.setdefault(it["qid"], {"subject": it["subject"], "oracle": sel[it["eval_id"]]["oracle"], "gemini": sel[it["eval_id"]]["retrieved"]})
        q[it["lang"]] = f"{it['question']}\n" + "\n".join(f"({i + 1}) {o}" for i, o in enumerate(it["options"]))
    qids = {lang: [q for q in by_qid if lang in by_qid[q]] for lang in ("en", "hi")}
    queries = {lang: [by_qid[q][lang] for q in qs] for lang, qs in qids.items()}
    res_path = os.path.join(ec.ROOT, "data", "eval", "retriever_eval.json")
    results = json.load(open(res_path, encoding="utf-8")) if os.path.exists(res_path) else {}

    names = [m.strip() for m in models.split(",") if m.strip()]
    for name, enc in zip(names, encode.map(names, [docs] * len(names), [queries] * len(names), return_exceptions=True)):
        if isinstance(enc, Exception):
            print(f"{name}: failed ({str(enc)[:150]})")
            results[name] = {"error": str(enc)[:200]}
            continue
        cvecs = np.asarray(enc["docs"], dtype=np.float32)
        res = {"dims": enc["dims"], "gpu_seconds": enc["seconds"]}
        pmap = {}
        for lang, qs in qids.items():
            qvecs = np.asarray(enc["queries"][lang], dtype=np.float32)
            hit1 = hit3 = have = agree = 0
            for q, v in zip(qs, qvecs):
                sims = np.where(csubj == SUBJECT_MAP[by_qid[q]["subject"]], cvecs @ v, -1.0)
                idx = np.argsort(-sims)[:3]
                top = [cids[i] for i in idx]
                pmap.setdefault(q, {})[f"top1_{lang}"] = top[0]
                pmap[q][f"score_{lang}"] = round(float(sims[idx[0]]), 3)
                agree += top[0] == by_qid[q]["gemini"]
                if by_qid[q]["oracle"]:
                    have += 1
                    hit1 += top[0] == by_qid[q]["oracle"]
                    hit3 += by_qid[q]["oracle"] in top
            res.update({f"{lang}_queries": len(qs), f"{lang}_hit1": hit1, f"{lang}_hit3": hit3, f"{lang}_have_oracle": have,
                        f"{lang}_agree_with_gemini_top1": agree})
        results[name] = res
        print(f"{name}: EN top-1 {res['en_hit1']}/{res['en_have_oracle']}, top-3 {res['en_hit3']}; HI top-1 {res['hi_hit1']}/{res['hi_have_oracle']}, "
              f"top-3 {res['hi_hit3']}; {res['dims']}d, {res['gpu_seconds']}s on L4")
        slug = re.sub(r"[^A-Za-z0-9]+", "-", name).strip("-").lower()
        out = {it["eval_id"]: {"retrieved": pmap[it["qid"]]["top1_en"], "retrieved_score": pmap[it["qid"]]["score_en"],
                               "oracle": by_qid[it["qid"]]["oracle"], "retrieved_hi": pmap[it["qid"]].get("top1_hi"),
                               "retrieved_hi_score": pmap[it["qid"]].get("score_hi")} for it in items}
        json.dump(out, open(os.path.join(ec.ROOT, "data", "eval", f"eval_v1_passages_{slug}.json"), "w", encoding="utf-8"), indent=0)
        json.dump(results, open(res_path, "w", encoding="utf-8"), indent=1)
    print("reference: gemini-embedding-001 EN top-1 188/300")


@app.local_entrypoint()
def retrieve_prompts(model: str = "Qwen/Qwen3-Embedding-0.6B", prompts: str = "data/train/v3_prompts.jsonl",
                     out: str = "data/train/v3_passages.json"):
    """Top-1 NCERT passage for every training prompt (one query per qid, English text, subject-restricted).

    modal run scripts/modal_retrievers.py::retrieve_prompts
    """
    import numpy as np

    import eval_common as ec
    from link_ncert import SUBJECT_MAP

    chunks = [json.loads(l) for l in open(os.path.join(ec.ROOT, "data", "ncert", "chunks.jsonl"), encoding="utf-8")]
    docs = [f"{c['subject']} Class {c['class']} — {c['title']}\n\n{c['text']}" for c in chunks]
    csubj = np.array([c["subject"] for c in chunks])
    cids = [c["id"] for c in chunks]
    rows = [json.loads(l) for l in open(os.path.join(ec.ROOT, prompts), encoding="utf-8")]
    by_qid = {}
    for r in rows:
        by_qid.setdefault(r["qid"], (r["query_en"], r["subject"]))
    qids = list(by_qid)
    print(f"{len(rows)} prompts, {len(qids)} unique questions; encoding with {model}")
    enc = encode.remote(model, docs, {"q": [by_qid[q][0] for q in qids]})
    cvecs = np.asarray(enc["docs"], dtype=np.float32)
    qvecs = np.asarray(enc["queries"]["q"], dtype=np.float32)
    res = {}
    for q, v in zip(qids, qvecs):
        sims = np.where(csubj == SUBJECT_MAP[by_qid[q][1]], cvecs @ v, -1.0)
        j = int(sims.argmax())
        res[q] = {"chunk": cids[j], "score": round(float(sims[j]), 3)}
    json.dump(res, open(os.path.join(ec.ROOT, out), "w", encoding="utf-8"), indent=0)
    print(f"wrote {len(res)} passages to {out} in {enc['seconds']}s GPU; median score {np.median([r['score'] for r in res.values()]):.3f}")
