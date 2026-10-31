"""Score every model run in data/eval/runs/ and print/save a comparison table.

Usage: python scripts/score_eval.py

Writes data/eval/scores_v1.json and docs/eval_baseline.md. Accuracy is
reported overall and split by source (NEET 2025 vs held-out NCERT MCQs),
language, subject and figure/no-figure. Unparseable answers count as wrong
and are reported separately. Cost per 1,000 questions uses the run log's
token counts and list prices (API) or GPU time (Modal).
"""
import glob
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_common as ec  # noqa: E402
import runlog  # noqa: E402

ORDER = ["google/gemma-4-E4B-it", "google/gemma-4-E4B-it#think", "google/gemma-4-E4B-it#passage-retrieved", "google/gemma-4-E4B-it#passage-oracle", "google/gemma-4-E4B-it#think#passage-retrieved", "google/gemma-4-E4B-it#think#passage-retrieved-qwen-qwen3-embedding-0-6b", "/ckpt/neet-e4b-v3/merged#think#passage-retrieved-qwen-qwen3-embedding-0-6b", "Qwen/Qwen3-VL-4B-Instruct#passage-retrieved-qwen-qwen3-embedding-0-6b", "Qwen/Qwen3.5-4B#think#passage-retrieved-qwen-qwen3-embedding-0-6b", "Qwen/Qwen3-VL-8B-Instruct#passage-retrieved-qwen-qwen3-embedding-0-6b", "Qwen/Qwen3-VL-8B-Thinking#think#passage-retrieved-qwen-qwen3-embedding-0-6b", "Qwen/Qwen3.5-9B#think#passage-retrieved-qwen-qwen3-embedding-0-6b", "google/gemma-4-E4B-it-qat-w4a16-ct#think#passage-retrieved", "google/gemma-4-12B-it#think#passage-retrieved", "google/gemma-4-26B-A4B-it#think#passage-retrieved", "google/gemma-4-31B-it#think#passage-retrieved", "/ckpt/neet-e4b-v1/merged", "google/gemma-4-31B-it", "gemini-3.8-flash", "claude-opus-5-5",
         "claude-haiku-4-5-20251001", "gpt-6.1-sol", "gpt-6-luna"]
LABEL = {"google/gemma-4-E4B-it": "Gemma 4 E4B (base)", "google/gemma-4-E4B-it#think": "Gemma 4 E4B (thinking)", "google/gemma-4-E4B-it#passage-retrieved": "E4B + NCERT passage (retrieved)", "google/gemma-4-E4B-it#passage-oracle": "E4B + NCERT passage (oracle)", "google/gemma-4-E4B-it#think#passage-retrieved": "E4B thinking + retrieved passage", "google/gemma-4-E4B-it#think#passage-retrieved-qwen-qwen3-embedding-0-6b": "E4B thinking + offline-retrieved passage (Qwen3-Emb 0.6B)", "/ckpt/neet-e4b-v3/merged#think#passage-retrieved-qwen-qwen3-embedding-0-6b": "E4B + LoRA v3 (31B traces), thinking + passage", "Qwen/Qwen3-VL-4B-Instruct#passage-retrieved-qwen-qwen3-embedding-0-6b": "Qwen3-VL-4B + passage", "Qwen/Qwen3.5-4B#think#passage-retrieved-qwen-qwen3-embedding-0-6b": "Qwen3.5-4B, thinking + passage", "Qwen/Qwen3-VL-8B-Instruct#passage-retrieved-qwen-qwen3-embedding-0-6b": "Qwen3-VL-8B + passage", "Qwen/Qwen3-VL-8B-Thinking#think#passage-retrieved-qwen-qwen3-embedding-0-6b": "Qwen3-VL-8B-Thinking + passage", "Qwen/Qwen3.5-9B#think#passage-retrieved-qwen-qwen3-embedding-0-6b": "Qwen3.5-9B, thinking + passage", "google/gemma-4-E4B-it-qat-w4a16-ct#think#passage-retrieved": "E4B 4-bit QAT, thinking + passage", "google/gemma-4-12B-it#think#passage-retrieved": "Gemma 4 12B, thinking + passage", "google/gemma-4-26B-A4B-it#think#passage-retrieved": "Gemma 4 26B-A4B, thinking + passage", "google/gemma-4-31B-it#think#passage-retrieved": "Gemma 4 31B, thinking + passage", "/ckpt/neet-e4b-v1/merged": "E4B + NEET LoRA v1", "google/gemma-4-31B-it": "Gemma 4 31B",
         "gemini-3.8-flash": "Gemini 3.8 Flash", "claude-opus-5-5": "Claude Opus 5.5",
         "claude-haiku-4-5-20251001": "Claude Haiku 4.5", "gpt-6.1-sol": "GPT-6.1 Sol", "gpt-6-luna": "GPT-6 Luna"}


def pct(a, b):
    return f"{100 * a / b:.1f}" if b else "-"


def main() -> None:
    items = {it["eval_id"]: it for it in ec.load_eval()}
    runs = {}
    for p in glob.glob(os.path.join(ec.RUNS_DIR, "*.jsonl")):
        rows = {}
        for l in open(p, encoding="utf-8"):
            r = json.loads(l)
            # Re-parse from the saved text so every run is scored by the current parser.
            if r["eval_id"] in items:
                r["pred"] = ec.parse_answer(r.get("text"))
                r["correct"] = ec.is_correct(r["pred"], items[r["eval_id"]]["answer"])
            rows[r["eval_id"]] = r  # last write wins
        if rows:
            runs[next(iter(rows.values()))["model"]] = rows
    models = [m for m in ORDER if m in runs] + sorted(m for m in runs if m not in ORDER)

    # cost per model from the run log (sum over eval runs)
    cost = defaultdict(float)
    for l in open(runlog.PATH, encoding="utf-8"):
        r = json.loads(l)
        if r["stage"] == "eval_api":
            cost[r["model"]] += runlog.cost_usd(r["model"], r["input_tokens"], r["output_tokens"]) or 0
        elif r["stage"] == "eval_modal":
            cost[r["model"]] += r.get("gpu_cost_usd_est") or 0

    cuts = [("All", lambda it: True),
            ("NEET 2025 EN", lambda it: it["source"] == "neet2025" and it["lang"] == "en"),
            ("NEET 2025 HI", lambda it: it["source"] == "neet2025" and it["lang"] == "hi"),
            ("NCERT MCQ EN", lambda it: it["source"] == "ncert_mcq" and it["lang"] == "en"),
            ("NCERT MCQ HI", lambda it: it["source"] == "ncert_mcq" and it["lang"] == "hi"),
            ("Physics", lambda it: it["subject"] == "Physics"),
            ("Chemistry", lambda it: it["subject"] == "Chemistry"),
            ("Biology", lambda it: it["subject"] == "Biology"),
            ("With figure", lambda it: it["has_figure"])]
    scores = {}
    for m in models:
        rows = runs[m]
        s = {"n": len(rows), "unparsed": sum(r["pred"] is None for r in rows.values()),
             "cost_usd": round(cost.get(m, 0), 3)}
        for name, f in cuts:
            sel = [r for eid, r in rows.items() if eid in items and f(items[eid])]
            s[name] = (sum(r["correct"] for r in sel), len(sel))
        s["out_tokens_mean"] = round(sum(r.get("output_tokens") or 0 for r in rows.values()) / max(1, len(rows)))
        scores[m] = s

    head = "| Cut | " + " | ".join(LABEL.get(m, m) for m in models) + " |"
    lines = [head, "|---|" + "---:|" * len(models)]
    for name, _ in cuts:
        lines.append(f"| {name} | " + " | ".join(pct(*scores[m][name]) for m in models) + " |")
    lines.append("| Unparsed answers | " + " | ".join(str(scores[m]["unparsed"]) for m in models) + " |")
    lines.append("| Items run | " + " | ".join(str(scores[m]["n"]) for m in models) + " |")
    lines.append("| Mean output tokens | " + " | ".join(str(scores[m]["out_tokens_mean"]) for m in models) + " |")
    lines.append("| Run cost (USD) | " + " | ".join(f"{scores[m]['cost_usd']:.2f}" for m in models) + " |")
    lines.append("| Cost per 1k questions (USD) | " + " | ".join(
        f"{1000 * scores[m]['cost_usd'] / scores[m]['n']:.2f}" if scores[m]["n"] else "-" for m in models) + " |")
    table = "\n".join(lines)
    print(table)
    with open(os.path.join(ec.ROOT, "data", "eval", "scores_v1.json"), "w", encoding="utf-8") as f:
        json.dump(scores, f, indent=1)
    with open(os.path.join(ec.ROOT, "docs", "eval_baseline.md"), "w", encoding="utf-8") as f:
        f.write("# Baseline eval (v1)\n\nAccuracy in %, same prompt, images and parser for every model. "
                "Eval set: NEET 2025 (180 questions, English and Hindi) plus 150 held-out generated NCERT MCQs "
                "(English and Hindi), 660 items, 46 with figures. Generated by `scripts/score_eval.py`.\n\n"
                + table + "\n\nCaveats: NEET 2025 is public and may be in API models' training data. "
                "The NCERT MCQs were written and checked by Gemini 3.8 Flash, which flatters Gemini on that cut. "
                "API cost uses list prices; Modal cost is GPU time at list price, including model load.\n")


if __name__ == "__main__":
    main()
