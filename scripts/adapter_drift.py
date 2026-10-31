"""How far did a LoRA run move the weights? Compare adapters without a GPU.

Usage:
  python scripts/adapter_drift.py            # v1 (989 steps) vs v2 (stopped, adapter at step 400)
  python scripts/adapter_drift.py neet-e4b-v1:989 neet-e4b-v2:400

For each LoRA module the weight change is dW = (alpha / r) * B @ A. The script
prints the Frobenius norm of dW summed over all modules, the same per step, the
split by projection type and by layer, and the share of the top singular
values of one module (a flat spectrum means the change looks like noise rather
than a few learned directions). Adapters are read from the Modal volume
`neet-ckpt` (/ckpt/<run>/adapter) into the local cache dir on first use.

Finding (6 Oct 2026): v2's training loss never moved (0.23 -> 0.20) yet its
adapter had travelled 65% as far as v1's by step 400, uniformly across layers,
with a flatter spectrum: Adam moves weights at a rate set by the learning rate
whether or not the gradient carries information. On data the model already
reproduces (89% of v2) that movement is drift away from Google's post-training.
"""
import collections
import json
import os
import sys

import numpy as np
from safetensors.numpy import load_file

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "data", "train", "adapters")  # gitignored: 140 MB each


def fetch(run: str) -> str:
    d = os.path.join(CACHE, run)
    os.makedirs(d, exist_ok=True)
    for f in ("adapter_config.json", "adapter_model.safetensors"):
        dst = os.path.join(d, f)
        if os.path.exists(dst) and os.path.getsize(dst) > 0:
            continue
        import modal

        vol = modal.Volume.from_name("neet-ckpt")
        with open(dst, "wb") as out:
            for chunk in vol.read_file(f"{run}/adapter/{f}"):
                out.write(chunk)
        print(f"fetched {run}/{f} ({os.path.getsize(dst) / 1e6:.1f} MB)")
    return d


def analyse(run: str, steps: int) -> dict:
    d = fetch(run)
    cfg = json.load(open(os.path.join(d, "adapter_config.json")))
    scale = cfg["lora_alpha"] / cfg["r"]
    t = load_file(os.path.join(d, "adapter_model.safetensors"))
    keys = sorted(k[: -len(".lora_A.weight")] for k in t if k.endswith(".lora_A.weight"))
    tot, by_proj, by_layer, spectrum = 0.0, collections.defaultdict(float), collections.defaultdict(float), None
    for k in keys:
        dw = scale * (t[k + ".lora_B.weight"].astype(np.float32) @ t[k + ".lora_A.weight"].astype(np.float32))
        n2 = float((dw ** 2).sum())
        tot += n2
        by_proj[k.split(".")[-1]] += n2
        by_layer[int(k.split("layers.")[1].split(".")[0])] += n2
        if "layers.20." in k and k.endswith("q_proj"):
            s = np.linalg.svd(dw, compute_uv=False)
            spectrum = [round(float(x), 3) for x in (s / s.sum())[:6]]
    out = {"run": run, "steps": steps, "modules": len(keys), "dW_fro": round(tot ** 0.5, 2),
           "dW_fro_per_step": round(tot ** 0.5 / steps, 4),
           "by_projection": {p: round(x ** 0.5, 2) for p, x in sorted(by_proj.items())},
           "by_layer": [round(by_layer[l] ** 0.5, 2) for l in sorted(by_layer)],
           "layer20_q_proj_top_singular_shares": spectrum}
    print(json.dumps(out, indent=1))
    return out


def main() -> None:
    args = sys.argv[1:] or ["neet-e4b-v1:989", "neet-e4b-v2:400"]
    res = [analyse(*((a.split(":")[0], int(a.split(":")[1])))) for a in args]
    json.dump(res, open(os.path.join(ROOT, "data", "train", "adapter_drift.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
