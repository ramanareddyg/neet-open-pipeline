"""Sample the model's own reasoning on training prompts (rejection-sampling data for v2).

Deployed + spawned so it does not depend on this machine staying online:
  modal deploy scripts/modal_sample.py
  python scripts/modal_sample.py spawn --name e4b-v2 --model google/gemma-4-E4B-it --n-pyq 4 --n-mcq 2
  python scripts/modal_sample.py spawn --name 31b-rescue --model google/gemma-4-31B-it --gpu H200 --only data/train/rescue_ids.json --n-pyq 4 --n-mcq 4
  python scripts/modal_sample.py status <call-id>
  modal volume get neet-ckpt samples/e4b-v2.jsonl data/train/samples/

Reads the prompts of /data/train_v1.jsonl (user turns only; the assistant
turns are ignored), minus the dev-probe questions, and writes
/ckpt/samples/<name>.jsonl:
  {id, source, lang, gold, samples: [{text, output_tokens, finish}]}
Answers are parsed and filtered locally (build_train_v2.py) with the same
parser as the eval.
"""
import json
import os
import sys
import time

import modal

app = modal.App("neet-open-sample")
image = (
    modal.Image.from_registry("vllm/vllm-openai:latest")
    .entrypoint([])
    .run_commands("ln -sf \"$(which python3)\" /usr/local/bin/python")
    .env({"HF_XET_HIGH_PERFORMANCE": "1"})
)
hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)
data_vol = modal.Volume.from_name("neet-data", create_if_missing=True)
ckpt_vol = modal.Volume.from_name("neet-ckpt", create_if_missing=True)
VOLS = {"/root/.cache/huggingface": hf_cache, "/data": data_vol, "/ckpt": ckpt_vol}


def _sample(name, model_id, n_pyq, n_mcq, temperature, max_tokens, exclude_qids, only_ids, seed):
    import base64

    from vllm import LLM, SamplingParams

    t0 = time.time()
    rows = [json.loads(l) for l in open("/data/train_v1.jsonl", encoding="utf-8")]
    ex = set(exclude_qids)
    only = set(only_ids) if only_ids else None

    def qid(r):
        return r["id"].rsplit("-", 2)[0] if r["source"].startswith("pyq") else r["id"].rsplit("-", 1)[0]

    rows = [r for r in rows if qid(r) not in ex and (only is None or r["id"] in only)]
    MAX_IMAGES = 12  # 2024 q86 has 9 figures; prompts above the limit are skipped, not fatal
    rows = [r for r in rows if len(r["images"]) <= MAX_IMAGES]
    llm = LLM(model=model_id, max_model_len=16384, limit_mm_per_prompt={"image": MAX_IMAGES}, gpu_memory_utilization=0.92)
    hf_cache.commit()
    t_load = time.time() - t0

    def conv(r):
        content = []
        for p in r["images"]:
            b64 = base64.b64encode(open(f"/data/{p}", "rb").read()).decode()
            content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}})
        content.append({"type": "text", "text": r["messages"][0]["content"]})
        return [{"role": "user", "content": content}]

    out_path = f"/ckpt/samples/{name}.jsonl"
    os.makedirs("/ckpt/samples", exist_ok=True)
    n_gen = n_tok = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for group, n in (("pyq", n_pyq), ("ncert_mcq", n_mcq)):
            grp = [r for r in rows if (r["source"].startswith("pyq") if group == "pyq" else r["source"] == "ncert_mcq")]
            if not grp or n == 0:
                continue
            outs = llm.chat([conv(r) for r in grp], SamplingParams(n=n, temperature=temperature, top_p=0.95,
                                                                  max_tokens=max_tokens, seed=seed))
            for r, o in zip(grp, outs):
                gold = int(r["messages"][1]["content"].rsplit("Answer: ", 1)[1].strip()[0])
                samples = [{"text": c.text, "output_tokens": len(c.token_ids), "finish": c.finish_reason} for c in o.outputs]
                n_gen += len(samples)
                n_tok += sum(s["output_tokens"] for s in samples)
                f.write(json.dumps({"id": r["id"], "source": r["source"], "lang": r["lang"], "gold": gold,
                                    "samples": samples}, ensure_ascii=False) + "\n")
            ckpt_vol.commit()
    ckpt_vol.commit()
    return {"name": name, "model": model_id, "prompts": len(rows), "generations": n_gen, "output_tokens": n_tok,
            "load_seconds": round(t_load, 1), "total_seconds": round(time.time() - t0, 1)}


@app.function(image=image, gpu="H100", timeout=6 * 3600, volumes=VOLS)
def sample_h100(name: str, model_id: str, n_pyq: int = 4, n_mcq: int = 2, temperature: float = 0.8, max_tokens: int = 3072,
                exclude_qids: list = (), only_ids: list = (), seed: int = 1) -> dict:
    return _sample(name, model_id, n_pyq, n_mcq, temperature, max_tokens, exclude_qids, only_ids, seed)


@app.function(image=image, gpu="H200", timeout=6 * 3600, volumes=VOLS)
def sample_h200(name: str, model_id: str, n_pyq: int = 4, n_mcq: int = 2, temperature: float = 0.8, max_tokens: int = 3072,
                exclude_qids: list = (), only_ids: list = (), seed: int = 1) -> dict:
    return _sample(name, model_id, n_pyq, n_mcq, temperature, max_tokens, exclude_qids, only_ids, seed)


if __name__ == "__main__":
    import argparse

    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("spawn")
    s.add_argument("--name", required=True)
    s.add_argument("--model", default="google/gemma-4-E4B-it")
    s.add_argument("--gpu", default="H100", choices=["H100", "H200"])
    s.add_argument("--n-pyq", type=int, default=4)
    s.add_argument("--n-mcq", type=int, default=2)
    s.add_argument("--only", help="JSON list of training example ids to sample (rescue pass)")
    st = sub.add_parser("status")
    st.add_argument("call_id")
    a = ap.parse_args()
    if a.cmd == "status":
        try:
            print(json.dumps(modal.FunctionCall.from_id(a.call_id).get(timeout=0), indent=1))
        except TimeoutError:
            print("still running")
        sys.exit(0)
    exclude = json.load(open(os.path.join(ROOT, "data", "eval", "dev_probe_ids.json")))
    only = json.load(open(a.only)) if a.only else []
    fn = modal.Function.from_name("neet-open-sample", f"sample_{a.gpu.lower()}")
    call = fn.spawn(a.name, a.model, n_pyq=a.n_pyq, n_mcq=a.n_mcq, exclude_qids=exclude, only_ids=only)
    print(f"spawned {a.name} on {a.gpu}: call id {call.object_id}")
