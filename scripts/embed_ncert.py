"""Embed NCERT passages for retrieval.

Usage: python scripts/embed_ncert.py

Reads  data/ncert/chunks.jsonl
Writes data/ncert/embeddings.npy        (float32, one row per chunk, L2-normalised; gitignored)
       data/ncert/embeddings_ids.json   (chunk ids in row order)

Uses gemini-embedding-001 with task_type RETRIEVAL_DOCUMENT and 768 output
dimensions (ample for 1.9k passages; keeps the file small). Query-time
embedding in link_ncert.py uses RETRIEVAL_QUERY with the same model/dims.
"""
import json
import os
import sys
import time

import numpy as np
from dotenv import load_dotenv
from google import genai
from google.genai import types

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import runlog  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(ROOT, ".env"), override=True)

EMBED_MODEL = "gemini-embedding-001"
DIMS = 768
BATCH = 50


def embed_texts(client: genai.Client, texts: list[str], task_type: str) -> np.ndarray:
    out = []
    for i in range(0, len(texts), BATCH):
        batch = texts[i:i + BATCH]
        for attempt in range(7):
            try:
                resp = client.models.embed_content(
                    model=EMBED_MODEL,
                    contents=batch,
                    config=types.EmbedContentConfig(task_type=task_type, output_dimensionality=DIMS),
                )
                out.extend(e.values for e in resp.embeddings)
                break
            except Exception as e:  # noqa: BLE001
                if attempt == 6:
                    raise
                time.sleep(10 * (attempt + 1))  # 429s are per-minute quotas
        print(f"  embedded {min(i + BATCH, len(texts))}/{len(texts)}", flush=True)
    arr = np.asarray(out, dtype=np.float32)
    arr /= np.linalg.norm(arr, axis=1, keepdims=True)
    return arr


def main() -> None:
    src = os.path.join(ROOT, "data", "ncert", "chunks.jsonl")
    chunks = [json.loads(l) for l in open(src, encoding="utf-8")]
    # Prefix with chapter context so short passages still carry their topic.
    texts = [f"{c['subject']} Class {c['class']} — {c['title']}\n\n{c['text']}" for c in chunks]
    client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
    t0 = time.time()
    arr = embed_texts(client, texts, "RETRIEVAL_DOCUMENT")
    np.save(os.path.join(ROOT, "data", "ncert", "embeddings.npy"), arr)
    with open(os.path.join(ROOT, "data", "ncert", "embeddings_ids.json"), "w", encoding="utf-8") as f:
        json.dump([c["id"] for c in chunks], f)
    words = sum(c["words"] for c in chunks)
    runlog.log("embed_ncert", model=EMBED_MODEL, dims=DIMS, chunks=len(chunks), words=words,
               seconds=round(time.time() - t0, 1))
    print(f"{arr.shape} embeddings in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
