"""LoRA fine-tune of Gemma 4 E4B on Modal.

Usage (from the repo root):
  modal run scripts/modal_train.py --max-examples 64          # smoke test, ~5 min
  modal run --detach scripts/modal_train.py --run-name neet-e4b-v1

The local entrypoint uploads data/train/train_v1.jsonl and the figure PNGs it
references to the Modal volume `neet-data`, then trains on one GPU. The remote
function writes everything it produces to the volume `neet-ckpt`:
  /ckpt/<run>/adapter/      LoRA adapter (small; what gets published)
  /ckpt/<run>/merged/       adapter merged into the base weights, for vLLM eval
  /ckpt/<run>/metrics.json  loss curve, timings, config
so a detached run survives the local process. Eval afterwards with:
  modal run scripts/modal_eval.py --model /ckpt/<run>/merged

Training: bf16 base, LoRA r=16 / alpha=32 on the language model's attention
and MLP projections only (vision and audio towers frozen), batch size 1 with
gradient accumulation 16, AdamW lr 1e-4, cosine with warmup, 1 epoch. Loss on
assistant tokens only. Thinking mode off, as in the baseline eval. 1% of
examples are held back for a validation loss.
"""
import json
import os
import sys
import time

import modal

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

GPU = os.environ.get("TRAIN_GPU", "H100")
GPU_USD_PER_HOUR = {"H100": 3.95, "H200": 4.54, "A100-80GB": 2.50}
BASE = "google/gemma-4-E4B-it"

app = modal.App("neet-open-train")
image = (
    modal.Image.from_registry("vllm/vllm-openai:latest")
    .entrypoint([])
    .run_commands("ln -sf \"$(which python3)\" /usr/local/bin/python")
    .run_commands("python -m pip install --no-cache-dir peft accelerate")
    .env({"HF_XET_HIGH_PERFORMANCE": "1"})
)
hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)
data_vol = modal.Volume.from_name("neet-data", create_if_missing=True)
ckpt_vol = modal.Volume.from_name("neet-ckpt", create_if_missing=True)


@app.function(image=image, gpu=GPU, timeout=8 * 3600,
              volumes={"/root/.cache/huggingface": hf_cache, "/data": data_vol, "/ckpt": ckpt_vol})
def train(run_name: str, max_examples: int = 0, lr: float = 1e-4, rank: int = 16, alpha: int = 32,
          grad_accum: int = 16, max_len: int = 4096, epochs: float = 1.0, seed: int = 20251005,
          train_file: str = "/data/train_v1.jsonl", probe_file: str = "", probe_every: int = 250,
          think: bool = False, weight_decay: float = 0.0, probe_max_new: int = 1536, save_every: int = 200) -> dict:
    # v3 (6 Oct): think=True trains on targets that carry a `reasoning` field, rendered by the chat template as the
    # thinking channel, with <|think|> in the prompt, so training matches the shipping setting (thinking on,
    # NCERT passage in the user text). The probe then also generates with thinking (probe_max_new ~4096).
    import math
    import random

    import torch
    from peft import LoraConfig, get_peft_model
    from PIL import Image
    from transformers import AutoModelForMultimodalLM, AutoProcessor

    t_start = time.time()
    torch.manual_seed(seed)
    out_dir = f"/ckpt/{run_name}"
    os.makedirs(out_dir, exist_ok=True)
    rows = [json.loads(l) for l in open(train_file, encoding="utf-8")]
    rng = random.Random(seed)
    rng.shuffle(rows)
    if max_examples:
        rows = rows[:max_examples]
    n_val = max(8, len(rows) // 100)
    val_rows, train_rows = rows[:n_val], rows[n_val:]

    processor = AutoProcessor.from_pretrained(BASE)
    model = AutoModelForMultimodalLM.from_pretrained(BASE, dtype=torch.bfloat16, device_map="cuda")
    hf_cache.commit()
    t_load = time.time() - t_start

    # LoRA on the language model's linear projections only.
    proj = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
    targets = [n for n, m in model.named_modules()
               if isinstance(m, torch.nn.Linear) and "language_model" in n and n.split(".")[-1] in proj]
    print(f"LoRA targets: {len(targets)} modules, e.g. {targets[:2]}", flush=True)
    model = get_peft_model(model, LoraConfig(r=rank, lora_alpha=alpha, lora_dropout=0.05, target_modules=targets,
                                             bias="none", task_type="CAUSAL_LM"))
    for n, p in model.named_parameters():
        if p.requires_grad:
            p.data = p.data.float()  # keep adapter weights in fp32 for stable updates
    # v3 OOM fix: at max_len 6144 the activations plus the 262k-vocab logits exceed 80 GB without checkpointing.
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    # (do not set config.use_cache=False: it also switched the KV cache off in the probe's generate(), ~10x slower)
    n_train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable params: {n_train_params / 1e6:.1f}M", flush=True)

    def encode(row):
        user = [{"type": "image", "image": Image.open(f"/data/{p}").convert("RGB")} for p in row["images"]]
        user.append({"type": "text", "text": row["messages"][0]["content"]})
        assistant = {"role": "assistant", "content": [{"type": "text", "text": row["messages"][1]["content"]}]}
        if think and row.get("reasoning"):
            assistant["reasoning"] = row["reasoning"]  # template renders <|channel>thought\n...\n<channel|> before the content
        msgs = [{"role": "user", "content": user}, assistant]
        full = processor.apply_chat_template(msgs, tokenize=True, return_dict=True, return_tensors="pt", enable_thinking=think)
        prompt = processor.apply_chat_template(msgs[:1], tokenize=True, return_dict=True, return_tensors="pt",
                                               add_generation_prompt=True, enable_thinking=think)
        ids = full["input_ids"][0]
        p_len = prompt["input_ids"].shape[-1]
        if not torch.equal(ids[:p_len], prompt["input_ids"][0]):
            raise ValueError("prompt is not a prefix of the full conversation")
        labels = ids.clone()
        labels[:p_len] = -100
        full["labels"] = labels.unsqueeze(0)
        return {k: (v.to("cuda") if hasattr(v, "to") else v) for k, v in full.items()}

    # quick check on the first examples: template prefix + lengths
    for r in train_rows[:3]:
        b = encode(r)
        print(f"example {r['id']}: {b['input_ids'].shape[-1]} tokens, {(b['labels'] != -100).sum().item()} trained", flush=True)

    steps_total = math.ceil(len(train_rows) * epochs / grad_accum)
    warmup = max(10, steps_total // 30)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=weight_decay, betas=(0.9, 0.999))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warmup) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / max(1, steps_total)))))

    def val_loss():
        model.eval()
        tot, n = 0.0, 0
        with torch.no_grad():
            for r in val_rows:
                b = encode(r)
                if b["input_ids"].shape[-1] > max_len:
                    continue
                tot += model(**b).loss.item()
                n += 1
        model.train()
        torch.cuda.empty_cache()
        return round(tot / max(1, n), 4)

    # Dev probe: greedy generation on held-out dev questions, so a checkpoint
    # that gets worse (or shorter) shows up mid-run rather than at the end.
    import re as _re
    _ANS = _re.compile(r"(?:answer|उत्तर)\s*\**\s*[:：]\s*[^\d\n]{0,30}?([1-4])(?![\d.])(?!\s*[-–]\s*4)", _re.I)
    probe_items = [json.loads(l) for l in open(probe_file, encoding="utf-8")] if probe_file else []
    # Same instruction text as eval_common.INSTRUCTION (kept inline: this file runs remotely on its own).
    instr = {"en": ("Answer this NEET-UG multiple-choice question. Think it through briefly, then give the final answer "
                    "on the last line in exactly this form:\nAnswer: <option number 1-4>"),
             "hi": ("इस NEET-UG बहुविकल्पीय प्रश्न का उत्तर दीजिए। संक्षेप में हिंदी में हल कीजिए, फिर अंतिम पंक्ति में "
                    "ठीक इसी रूप में उत्तर लिखिए:\nAnswer: <विकल्प संख्या 1-4>")}

    def probe(step):
        if not probe_items:
            return None
        model.eval()
        processor.tokenizer.padding_side = "left"
        res = {"step": step, "correct": 0, "n": 0, "chars": [], "by_lang": {"en": [0, 0], "hi": [0, 0]}, "truncated": 0}
        bs = 20 if think else 40
        for i in range(0, len(probe_items), bs):
            batch = probe_items[i:i + bs]
            convs = []
            for it in batch:
                if "user_text" in it:  # dev_probe_v3: already rendered with the retrieved passage
                    text = it["user_text"]
                else:
                    opts = "\n".join(f"({k + 1}) {o}" for k, o in enumerate(it["options"]))
                    text = f"{instr[it['lang']]}\n\n{it['question']}\n\n{opts}"  # = eval_common.prompt_text for text-only items
                convs.append([{"role": "user", "content": [{"type": "text", "text": text}]}])
            enc = processor.apply_chat_template(convs, tokenize=True, return_dict=True, return_tensors="pt", padding=True,
                                                add_generation_prompt=True, enable_thinking=think).to("cuda")
            with torch.no_grad():
                gen = model.generate(**enc, max_new_tokens=probe_max_new, do_sample=False, use_cache=True)
            for it, g in zip(batch, gen[:, enc["input_ids"].shape[1]:]):
                text = processor.tokenizer.decode(g, skip_special_tokens=True)
                m = _ANS.findall(text)
                ok = bool(m) and int(m[-1]) == it.get("gold", it.get("answer"))
                res["correct"] += ok
                res["n"] += 1
                res["chars"].append(len(text))
                res["by_lang"][it["lang"]][0] += ok
                res["by_lang"][it["lang"]][1] += 1
                res["truncated"] += int((g != processor.tokenizer.pad_token_id).sum().item() >= probe_max_new)
        torch.cuda.empty_cache()
        res["acc"] = round(res["correct"] / max(1, res["n"]), 3)
        res["median_chars"] = sorted(res["chars"])[len(res["chars"]) // 2]
        del res["chars"]
        model.train()
        print(f"probe step {step}: acc {res['acc']:.1%} ({res['correct']}/{res['n']}), en {res['by_lang']['en']}, "
              f"hi {res['by_lang']['hi']}, median {res['median_chars']} chars, truncated {res['truncated']}", flush=True)
        return res

    curve = {"train": [], "val": [], "probe": []}
    best_probe, best_step = -1.0, 0
    p0 = probe(0)
    if p0:
        curve["probe"].append(p0)
        best_probe = p0["acc"]  # a checkpoint must beat the untrained model to become "best"
    v0 = val_loss()
    curve["val"].append((0, v0))
    print(f"val loss before: {v0}", flush=True)
    model.train()
    step, micro, run_loss, run_n, skipped, tokens = 0, 0, 0.0, 0, 0, 0
    order = list(range(len(train_rows)))
    n_micro = int(len(train_rows) * epochs)
    t_train = time.time()
    for k in range(n_micro):
        r = train_rows[order[k % len(order)]]
        try:
            b = encode(r)
        except Exception as e:  # noqa: BLE001
            skipped += 1
            print(f"skip {r['id']}: {e}", flush=True)
            continue
        if b["input_ids"].shape[-1] > max_len:
            skipped += 1
            continue
        tokens += b["input_ids"].shape[-1]
        loss = model(**b).loss / grad_accum
        loss.backward()
        run_loss += loss.item() * grad_accum
        run_n += 1
        micro += 1
        if micro % grad_accum == 0:
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if step % 20 == 0:
                avg = run_loss / max(1, run_n)
                curve["train"].append((step, round(avg, 4)))
                el = time.time() - t_train
                print(f"step {step}/{steps_total} loss {avg:.4f} lr {sched.get_last_lr()[0]:.2e} "
                      f"{micro / el:.2f} ex/s, eta {(n_micro - k) / max(1e-6, micro / el) / 60:.0f} min", flush=True)
                run_loss, run_n = 0.0, 0
            if step % save_every == 0:
                model.save_pretrained(f"{out_dir}/adapter-step{step}")  # distinct checkpoints, nothing overwritten
                ckpt_vol.commit()
            if probe_items and step % probe_every == 0:
                pr = probe(step)
                curve["probe"].append(pr)
                if pr["acc"] >= best_probe:  # keep the best-probe checkpoint separately
                    best_probe, best_step = pr["acc"], step
                    model.save_pretrained(f"{out_dir}/adapter-best")
                json.dump(curve, open(f"{out_dir}/curve_partial.json", "w"), indent=1)
                ckpt_vol.commit()
    t_train = time.time() - t_train
    v1 = val_loss()
    curve["val"].append((step, v1))
    if probe_items:
        curve["probe"].append(probe(step))
    print(f"val loss after: {v1}", flush=True)

    model.save_pretrained(f"{out_dir}/adapter")
    merged = model.merge_and_unload()
    for p in merged.parameters():
        p.data = p.data.to(torch.bfloat16)
    merged.save_pretrained(f"{out_dir}/merged", safe_serialization=True)
    processor.save_pretrained(f"{out_dir}/merged")
    metrics = {
        "run": run_name, "base": BASE, "gpu": GPU, "examples_train": len(train_rows), "examples_val": len(val_rows),
        "skipped": skipped, "train_tokens": tokens, "steps": step, "lr": lr, "rank": rank, "alpha": alpha,
        "grad_accum": grad_accum, "epochs": epochs, "max_len": max_len, "trainable_params": n_train_params,
        "think": think, "weight_decay": weight_decay, "train_file": train_file, "probe_file": probe_file,
        "best_probe_acc": best_probe, "best_probe_step": best_step,
        "val_loss_before": v0, "val_loss_after": v1, "curve": curve,
        "load_seconds": round(t_load, 1), "train_seconds": round(t_train, 1), "total_seconds": round(time.time() - t_start, 1),
    }
    metrics["gpu_cost_usd_est"] = round(metrics["total_seconds"] / 3600 * GPU_USD_PER_HOUR.get(GPU, 0), 2)
    json.dump(metrics, open(f"{out_dir}/metrics.json", "w"), indent=1)
    ckpt_vol.commit()
    return metrics


@app.local_entrypoint()
def main(run_name: str = "neet-e4b-v1", max_examples: int = 0, epochs: float = 1.0, lr: float = 1e-4):
    import eval_common as ec
    import runlog

    path = os.path.join(ec.ROOT, "data", "train", "train_v1.jsonl")
    rows = [json.loads(l) for l in open(path, encoding="utf-8")]
    figs = sorted({p for r in rows for p in r["images"]})
    with data_vol.batch_upload(force=True) as up:
        up.put_file(path, "/train_v1.jsonl")
        for p in figs:
            up.put_file(os.path.join(ec.ROOT, p), f"/{p}")
    print(f"uploaded {len(rows)} examples and {len(figs)} figures; training {run_name} on {GPU}")
    m = train.remote(run_name, max_examples=max_examples, epochs=epochs, lr=lr)
    print(json.dumps({k: v for k, v in m.items() if k != "curve"}, indent=1))
    runlog.log("train_modal", **{k: v for k, v in m.items() if k != "curve"})
