"""Extract the lines worth keeping from a Modal training log.

Usage:
  python scripts/capture_train_log.py <raw.log> <out.steps.txt> [--last-run]

The raw log (from `modal run` or `modal app logs`) contains progress bars and
library warnings, and `modal app logs` replays earlier runs of the same app.
This keeps setup facts, per-step loss / learning rate / throughput, validation
loss, dev-probe results, memory warnings and errors, de-duplicated (log
replays repeat lines). With --last-run it keeps only the most recent run
(from the last "LoRA targets" line onwards).

The authoritative curve is also written by the run itself to
/ckpt/<run>/curve_partial.json (after each probe) and /ckpt/<run>/metrics.json.
"""
import re
import sys

KEEP = re.compile(r"LoRA targets|trainable params|^example |val loss|^step \d|probe step|OOM|OutOfMemory|Error|Traceback|^skip ")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    src, out = sys.argv[1], sys.argv[2]
    last_run = "--last-run" in sys.argv
    text = open(src, encoding="utf-8", errors="replace").read()
    lines = [l.strip() for l in re.split(r"[\r\n]", text)]
    kept = list(dict.fromkeys(l for l in lines if KEEP.search(l)))
    if last_run:
        starts = [i for i, l in enumerate(kept) if l.startswith("LoRA targets")]
        if starts:
            kept = kept[starts[-1]:]
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(kept) + "\n")
    print("\n".join(kept[-12:]))


if __name__ == "__main__":
    main()
