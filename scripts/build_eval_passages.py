"""Pick an NCERT passage for every eval question, two ways, for the retrieval test.

Usage: python scripts/build_eval_passages.py

Writes data/eval/eval_v1_passages.json: {eval_id: {"retrieved": chunk_id, "retrieved_score": cos,
                                                   "oracle": chunk_id | null}}
  retrieved  what a product could do: embed the English question + options
             (gemini-embedding-001, RETRIEVAL_QUERY, same index as link_ncert),
             restrict to the question's subject, take the top passage. Hindi
             items use the English text of the same question (NCERT chunks are
             English; a product would translate or use the bilingual question).
  oracle     the passage a reader chose: data/links/2025_en_45.jsonl for NEET
             2025 (150 of 180 have one), the source passage for generated MCQs
             (an upper bound only: the question was written from that passage).

Costs a few cents of embedding calls (330 unique questions).
"""
import json
import os
import sys
import time

import numpy as np
from dotenv import load_dotenv
from google import genai

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_common as ec  # noqa: E402
import runlog  # noqa: E402
from embed_ncert import embed_texts  # noqa: E402
from link_ncert import SUBJECT_MAP, load_index  # noqa: E402

load_dotenv(os.path.join(ec.ROOT, ".env"), override=True)
import glob  # noqa: E402


def main() -> None:
    items = ec.load_eval()
    en = {it["qid"]: it for it in items if it["lang"] == "en"}
    chunks, ids, emb, subj = load_index()
    texts = [f"{it['question']}\n" + "\n".join(f"({i + 1}) {o}" for i, o in enumerate(it["options"])) for it in en.values()]
    client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
    t0 = time.time()
    q = embed_texts(client, texts, "RETRIEVAL_QUERY")
    runlog.log("embed_eval_queries", model="gemini-embedding-001", queries=len(texts), seconds=round(time.time() - t0, 1),
               input_tokens=sum(len(t) for t in texts) // 4)

    links = {l["id"]: l["chunk_id"] for l in map(json.loads, open(os.path.join(ec.ROOT, "data", "links", "2025_en_45.jsonl"), encoding="utf-8"))}
    mcq = {r["id"]: r["source_chunk"] for p in glob.glob(os.path.join(ec.ROOT, "data", "synthetic", "mcq", "*.jsonl"))
           for r in map(json.loads, open(p, encoding="utf-8"))}
    by_qid = {}
    for (qid, it), vec in zip(en.items(), q):
        mask = subj == SUBJECT_MAP[it["subject"]]
        sims = np.where(mask, emb @ vec, -1.0)
        j = int(sims.argmax())
        by_qid[qid] = {"retrieved": ids[j], "retrieved_score": round(float(sims[j]), 3),
                       "oracle": links.get(qid) if it["source"] == "neet2025" else mcq.get(qid)}
    out = {it["eval_id"]: by_qid[it["qid"]] for it in items}
    with open(os.path.join(ec.ROOT, "data", "eval", "eval_v1_passages.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=0)
    hit = sum(bool(v["oracle"]) and v["retrieved"] == v["oracle"] for v in by_qid.values())
    have = sum(bool(v["oracle"]) for v in by_qid.values())
    print(f"{len(by_qid)} questions; retrieval top-1 equals the reader's passage on {hit}/{have} that have one; "
          f"median score {np.median([v['retrieved_score'] for v in by_qid.values()]):.3f}")


if __name__ == "__main__":
    main()
