"""Run the eval set through an open model on Modal with vLLM.

Usage (from the repo root):
  modal run scripts/modal_eval.py --model google/gemma-4-E4B-it
  modal run scripts/modal_eval.py --model google/gemma-4-31B-it --limit 20
  modal run scripts/modal_eval.py --model google/gemma-4-E4B-it --think   # Gemma 4 thinking mode, run saved as <model>#think
  modal run scripts/modal_eval.py --model google/gemma-4-E4B-it --passage retrieved   # NCERT passage in the prompt (or --passage oracle)

All 660 prompts go to one GPU container as a single vLLM batch (images
inline as data URIs), greedy decoding. Responses land in
data/eval/runs/<model>.jsonl with tokens and timings; GPU seconds and an
estimated cost go to the run log. Weights are cached in the Modal volume
`hf-cache`, so only the first run per model downloads them.
"""
import json
import os
import sys
import time

import modal

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

GPU = os.environ.get("EVAL_GPU", "H100")  # 31B at 16k context needs EVAL_GPU=H200 (KV cache does not fit on H100)
GPU_USD_PER_HOUR = {"H100": 3.95, "H200": 4.54, "A100-80GB": 2.50, "L40S": 1.95}  # Modal list prices; check modal.com/pricing
MAX_TOKENS = int(os.environ.get("EVAL_MAX_TOKENS", "3072"))
MAX_MODEL_LEN = int(os.environ.get("EVAL_MAX_MODEL_LEN", "16384"))  # thinking reruns: EVAL_MAX_TOKENS=16384 EVAL_MAX_MODEL_LEN=32768

app = modal.App("neet-open-eval")
# vLLM's own image: it carries the CUDA toolkit that FlashInfer's JIT kernels
# need at startup (a pip install on debian_slim fails with "Could not find nvcc").
image = (
    modal.Image.from_registry("vllm/vllm-openai:latest")
    .entrypoint([])
    # Modal detects the image's Python via `python`; the vLLM image only ships python3.
    .run_commands("which python3 && python3 --version && ln -sf \"$(which python3)\" /usr/local/bin/python && python --version")
    .env({"HF_XET_HIGH_PERFORMANCE": "1"})
)
hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)
ckpt_vol = modal.Volume.from_name("neet-ckpt", create_if_missing=True)  # fine-tuned models: --model /ckpt/<run>/merged


@app.function(image=image, gpu=GPU, volumes={"/root/.cache/huggingface": hf_cache, "/ckpt": ckpt_vol}, timeout=3 * 3600)
def generate(model_id: str, conversations: list[list[dict]], max_tokens: int, think: bool = False, lora: str = "") -> dict:
    import time as _t

    from vllm import LLM, SamplingParams

    t0 = _t.time()
    lora_kw = {"enable_lora": True, "max_lora_rank": 32} if lora else {}  # checkpoints: --lora /ckpt/<run>/adapter-step<N>
    llm = LLM(model=model_id, max_model_len=MAX_MODEL_LEN, limit_mm_per_prompt={"image": 12},  # NEET 2025 q23 has 8 option graphs, 2024 q86 has 9
              gpu_memory_utilization=0.92, trust_remote_code=True, **lora_kw)
    t_load = _t.time() - t0
    hf_cache.commit()
    # Gemma 4 thinking is off by default in its chat template; enable_thinking=True injects <|think|>.
    lora_req = None
    if lora:
        from vllm.lora.request import LoRARequest
        lora_req = LoRARequest("adapter", 1, lora)
    outs = llm.chat(conversations, SamplingParams(temperature=0.0, max_tokens=max_tokens),
                    chat_template_kwargs={"enable_thinking": True} if think else None, lora_request=lora_req)
    t_gen = _t.time() - t0 - t_load
    return {
        "load_seconds": round(t_load, 1),
        "gen_seconds": round(t_gen, 1),
        "results": [{"text": o.outputs[0].text, "input_tokens": len(o.prompt_token_ids),
                     "output_tokens": len(o.outputs[0].token_ids), "finish": o.outputs[0].finish_reason} for o in outs],
    }


@app.local_entrypoint()
def main(model: str = "google/gemma-4-E4B-it", limit: int = 0, think: bool = False, passage: str = "", passage_file: str = "",
         redo_truncated: bool = False, eval_file: str = "", max_tokens_cap: int = 0, lora: str = ""):
    import eval_common as ec
    import runlog

    label = f"{model}#think" if think else model  # separate run file; the scorer parses the final "Answer: N" as for every model
    if eval_file:  # e.g. val_v3.jsonl (items carry user_text already rendered with the passage)
        ec.EVAL_PATH = os.path.join(ec.ROOT, "data", "eval", eval_file)
        label += "#" + eval_file.replace(".jsonl", "")
    passages = {}
    cap_label = f"#cap{max_tokens_cap}" if max_tokens_cap else ""
    if lora:  # label the run by the adapter, e.g. /ckpt/neet-e4b-v3/adapter-step50
        label = lora + ("#think" if think else "")
    if passage:  # "retrieved" or "oracle": prepend an NCERT passage (data/eval/eval_v1_passages.json, build_eval_passages.py)
        from build_train import passage_block
        chunks = {c["id"]: c for c in map(json.loads, open(os.path.join(ec.ROOT, "data", "ncert", "chunks.jsonl"), encoding="utf-8"))}
        pf = os.path.join(ec.ROOT, "data", "eval", passage_file or "eval_v1_passages.json")  # e.g. eval_v1_passages_baai-bge-m3.json from eval_retrievers.py
        sel = json.load(open(pf, encoding="utf-8"))
        passages = {eid: passage_block(chunks.get(v[passage] or ""), eid.rsplit("-", 1)[-1]) for eid, v in sel.items()}
        label += f"#passage-{passage}" + (f"-{passage_file.replace('eval_v1_passages_', '').replace('.json', '')}" if passage_file else "")
    max_tokens = max(MAX_TOKENS, 8192) if think else MAX_TOKENS
    if max_tokens_cap:  # thinking-budget test: hard cap on thought + answer tokens, labelled separately
        max_tokens = max_tokens_cap
    label += cap_label
    if redo_truncated:
        # Re-run only the saved answers that hit the token cap (finish == "length"), replacing those rows in place.
        old = [json.loads(l) for l in open(ec.run_path(label), encoding="utf-8") if l.strip()]
        redo = {r["eval_id"] for r in old if r["finish"] == "length"}
        items = [it for it in ec.load_eval() if it["eval_id"] in redo]
        print(f"{label}: redoing {len(items)} truncated items with max_tokens {max_tokens}")
    else:
        items = [it for it in ec.load_eval() if it["eval_id"] not in ec.done_ids(label)]
    if limit:
        items = items[:limit]
    print(f"{label}: {len(items)} items on {GPU}, max_tokens {max_tokens}")
    if not items:
        return
    convs = []
    for it in items:
        content = [{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{ec.image_b64(p)}"}} for p in it["images"]]
        content.append({"type": "text", "text": it["user_text"] if it.get("user_text") else passages.get(it["eval_id"], "") + ec.prompt_text(it)})
        convs.append([{"role": "user", "content": content}])
    t0 = time.time()
    res = generate.remote(model, convs, max_tokens, think, lora)
    wall = time.time() - t0
    n_ok = 0
    new_rows = {}
    for it, r in zip(items, res["results"]):
        pred = ec.parse_answer(r["text"])
        ok = ec.is_correct(pred, it["answer"])
        n_ok += ok
        new_rows[it["eval_id"]] = {"eval_id": it["eval_id"], "model": label, "pred": pred, "correct": ok, **r,
                                   **({"max_tokens_redo": max_tokens} if redo_truncated else {})}
    if redo_truncated:
        # Replace the truncated rows in place (one row per eval_id; the scorer would otherwise double-count).
        rows = [new_rows.get(r["eval_id"], r) for r in old]
        with open(ec.run_path(label), "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"after redo: {sum(r['correct'] for r in rows)}/{len(rows)} = {sum(r['correct'] for r in rows) / len(rows):.1%}; "
              f"still truncated {sum(r['finish'] == 'length' for r in rows)}")
    else:
        with open(ec.run_path(label), "a", encoding="utf-8") as f:
            for r in new_rows.values():
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    gpu_s = res["load_seconds"] + res["gen_seconds"]
    cost = round(gpu_s / 3600 * GPU_USD_PER_HOUR.get(GPU, 0), 3)
    print(f"accuracy {n_ok}/{len(items)} = {n_ok / len(items):.1%}; load {res['load_seconds']}s, generate {res['gen_seconds']}s, wall {wall:.0f}s, ~${cost} GPU")
    runlog.log("eval_modal", model=label, gpu=GPU, items=len(items), correct=n_ok,
               input_tokens=sum(r["input_tokens"] for r in res["results"]),
               output_tokens=sum(r["output_tokens"] for r in res["results"]),
               load_seconds=res["load_seconds"], gen_seconds=res["gen_seconds"], wall_seconds=round(wall, 1),
               gpu_cost_usd_est=cost)
