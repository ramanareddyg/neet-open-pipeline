"""Launch a training run that does not depend on this machine staying online.

`modal run --detach` still lost the run when the local client crashed on a
DNS failure (5 Oct 2026, step 220/990). This instead calls the *deployed*
function with .spawn(): the call lives entirely on Modal, and its results
(adapter, merged model, metrics.json) land in the `neet-ckpt` volume.

Usage:
  modal deploy scripts/modal_train.py      # once, or after editing modal_train.py
  python scripts/spawn_train.py --run-name neet-e4b-v1
  python scripts/spawn_train.py --status <call-id>

The training data must already be in the `neet-data` volume (the
modal_train.py local entrypoint uploads it; the smoke test did).
"""
import argparse
import json

import modal


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", default="neet-e4b-v1")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--train", default="train_v1", help="training file stem under data/train/")
    ap.add_argument("--probe", action="store_true", help="run the dev probe during training")
    ap.add_argument("--probe-every", type=int, default=250)
    ap.add_argument("--probe-file", default="dev_probe_v1.jsonl", help="under data/eval/ (v3: dev_probe_v3.jsonl, rendered with passages)")
    ap.add_argument("--think", action="store_true", help="v3: targets carry `reasoning`; train and probe with thinking on")
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--max-examples", type=int, default=0, help="pilot: train on the first N examples only")
    ap.add_argument("--max-len", type=int, default=4096)
    ap.add_argument("--probe-max-new", type=int, default=1536)
    ap.add_argument("--save-every", type=int, default=100)
    ap.add_argument("--status", help="function call id to check")
    args = ap.parse_args()
    if args.status:
        fc = modal.FunctionCall.from_id(args.status)
        try:
            m = fc.get(timeout=0)
            print(json.dumps({k: v for k, v in m.items() if k != "curve"}, indent=1))
        except TimeoutError:
            print("still running")
        return
    import os

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    train_local = os.path.join(root, "data", "train", f"{args.train}.jsonl")
    rows = [json.loads(l) for l in open(train_local, encoding="utf-8")]
    figs = sorted({p for r in rows for p in r["images"]})
    vol = modal.Volume.from_name("neet-data")
    with vol.batch_upload(force=True) as up:
        up.put_file(train_local, f"/{args.train}.jsonl")
        up.put_file(os.path.join(root, "data", "eval", args.probe_file), f"/{args.probe_file}")
        for p in figs:
            up.put_file(os.path.join(root, p), f"/{p}")
    print(f"uploaded {len(rows)} examples, {len(figs)} figures and the dev probe")
    fn = modal.Function.from_name("neet-open-train", "train")
    call = fn.spawn(args.run_name, epochs=args.epochs, lr=args.lr, train_file=f"/data/{args.train}.jsonl",
                    probe_file=f"/data/{args.probe_file}" if args.probe else "", probe_every=args.probe_every,
                    think=args.think, weight_decay=args.weight_decay, max_examples=args.max_examples, max_len=args.max_len,
                    probe_max_new=args.probe_max_new, save_every=args.save_every)
    print(f"spawned {args.run_name}: call id {call.object_id}")


if __name__ == "__main__":
    main()
