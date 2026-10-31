"""Generate answers for the v3 prompts on Modal (deployed + spawned, survives this machine going offline).

Usage:
  modal deploy scripts/modal_v3_sample.py
  python scripts/modal_v3_sample.py upload                       # v3_prompts_rendered.jsonl + figures -> neet-data volume
  python scripts/modal_v3_sample.py spawn --name e4b-v3-greedy --model google/gemma-4-E4B-it --gpu H100
  python scripts/modal_v3_sample.py spawn --name 31b-v3 --model google/gemma-4-31B-it --gpu H200 --only data/train/v3_31b_ids.json
  python scripts/modal_v3_sample.py status <call-id>
  python scripts/modal_v3_sample.py fetch --name e4b-v3-greedy    # /ckpt/samples/<name>.jsonl -> data/train/samples/

Every prompt is the shipping setting: thinking on, retrieved NCERT passage in
the user text, greedy decoding, max 8,192 new tokens. Output rows:
  {id, source, lang, gold, text, output_tokens, finish}
with `text` holding the raw model output (thought block + answer).
"""
import argparse
import json
import os
import sys
import time

import modal

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

app = modal.App("neet-open-v3-sample")
image = (modal.Image.from_registry("vllm/vllm-openai:latest").entrypoint([])
         .run_commands("ln -sf \"$(which python3)\" /usr/local/bin/python").env({"HF_XET_HIGH_PERFORMANCE": "1"}))
hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)
data_vol = modal.Volume.from_name("neet-data", create_if_missing=True)
ckpt_vol = modal.Volume.from_name("neet-ckpt", create_if_missing=True)
VOLS = {"/root/.cache/huggingface": hf_cache, "/data": data_vol, "/ckpt": ckpt_vol}
MAX_IMAGES = 12


def _run(name, model_id, only_ids, max_tokens, temperature, max_model_len, think):
    import base64

    from vllm import LLM, SamplingParams

    t0 = time.time()
    rows = [json.loads(l) for l in open("/data/v3_prompts_rendered.jsonl", encoding="utf-8")]
    only = set(only_ids) if only_ids else None
    rows = [r for r in rows if (only is None or r["id"] in only) and len(r["images"]) <= MAX_IMAGES]
    llm = LLM(model=model_id, max_model_len=max_model_len, limit_mm_per_prompt={"image": MAX_IMAGES}, gpu_memory_utilization=0.92)
    hf_cache.commit()
    t_load = time.time() - t0

    def conv(r):
        content = []
        for p in r["images"]:
            b64 = base64.b64encode(open(f"/data/{p}", "rb").read()).decode()
            content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}})
        content.append({"type": "text", "text": r["user_text"]})
        return [{"role": "user", "content": content}]

    os.makedirs("/ckpt/samples", exist_ok=True)
    out_path = f"/ckpt/samples/{name}.jsonl"
    n_tok = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for i in range(0, len(rows), 2000):  # chunks, so partial results land on the volume
            grp = rows[i:i + 2000]
            outs = llm.chat([conv(r) for r in grp], SamplingParams(temperature=temperature, top_p=1.0 if temperature == 0 else 0.95,
                                                                  max_tokens=max_tokens, seed=20251006),
                            chat_template_kwargs={"enable_thinking": True} if think else None)
            for r, o in zip(grp, outs):
                c = o.outputs[0]
                n_tok += len(c.token_ids)
                f.write(json.dumps({"id": r["id"], "source": r["source"], "lang": r["lang"], "gold": r["gold"], "text": c.text,
                                    "output_tokens": len(c.token_ids), "finish": c.finish_reason}, ensure_ascii=False) + "\n")
            f.flush()
            ckpt_vol.commit()
            print(f"{name}: {min(i + 2000, len(rows))}/{len(rows)} done, {time.time() - t0:.0f}s", flush=True)
    return {"name": name, "model": model_id, "prompts": len(rows), "output_tokens": n_tok,
            "load_seconds": round(t_load, 1), "total_seconds": round(time.time() - t0, 1)}


@app.function(image=image, gpu="H100", timeout=6 * 3600, volumes=VOLS)
def run_h100(name: str, model_id: str, only_ids: list = None, max_tokens: int = 8192, temperature: float = 0.0,
             max_model_len: int = 16384, think: bool = True) -> dict:
    return _run(name, model_id, only_ids, max_tokens, temperature, max_model_len, think)


@app.function(image=image, gpu="H200", timeout=6 * 3600, volumes=VOLS)
def run_h200(name: str, model_id: str, only_ids: list = None, max_tokens: int = 8192, temperature: float = 0.0,
             max_model_len: int = 16384, think: bool = True) -> dict:
    return _run(name, model_id, only_ids, max_tokens, temperature, max_model_len, think)


def main() -> None:
    import eval_common as ec

    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["upload", "spawn", "status", "fetch"])
    ap.add_argument("call_id", nargs="?")
    ap.add_argument("--name")
    ap.add_argument("--model", default="google/gemma-4-E4B-it")
    ap.add_argument("--gpu", default="H100", choices=["H100", "H200"])
    ap.add_argument("--only", help="JSON list of prompt ids")
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--temperature", type=float, default=0.0)
    a = ap.parse_args()
    if a.cmd == "upload":
        path = os.path.join(ec.ROOT, "data", "train", "v3_prompts_rendered.jsonl")
        rows = [json.loads(l) for l in open(path, encoding="utf-8")]
        figs = sorted({p for r in rows for p in r["images"]})
        with data_vol.batch_upload(force=True) as up:
            up.put_file(path, "/v3_prompts_rendered.jsonl")
            for p in figs:
                up.put_file(os.path.join(ec.ROOT, p), f"/{p}")
        print(f"uploaded {len(rows)} prompts and {len(figs)} figures")
    elif a.cmd == "spawn":
        only = json.load(open(a.only, encoding="utf-8")) if a.only else None
        fn = modal.Function.from_name("neet-open-v3-sample", "run_h200" if a.gpu == "H200" else "run_h100")
        call = fn.spawn(a.name, a.model, only_ids=only, max_tokens=a.max_tokens, temperature=a.temperature)
        print(f"spawned {a.name} on {a.gpu}: call id {call.object_id}")
    elif a.cmd == "status":
        fc = modal.FunctionCall.from_id(a.call_id)
        try:
            print(json.dumps(fc.get(timeout=0), indent=1))
        except TimeoutError:
            print("still running")
    elif a.cmd == "fetch":
        dst = os.path.join(ec.ROOT, "data", "train", "samples", f"{a.name}.jsonl")
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        n = 0
        with open(dst, "wb") as f:
            for chunk in ckpt_vol.read_file(f"samples/{a.name}.jsonl"):
                f.write(chunk)
                n += len(chunk)
        print(f"fetched {n / 1e6:.1f} MB -> {dst}")


if __name__ == "__main__":
    main()
