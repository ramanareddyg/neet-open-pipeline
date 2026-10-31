"""Charts for the Hugging Face blog post (docs/blog/neet-open-blog-draft.md).

Usage: python scripts/blog_charts.py

Writes:
  docs/img/blog-ladder.png    open models (thinking + NCERT page) vs commercial (plain prompt), 660-item eval
  docs/img/blog-v1-loss-vs-accuracy.png   fine-tune v1: validation/train loss falls while accuracy falls
  docs/img/blog-three-finetunes.png       v1, v2, v3 side by side: loss (shared scale) above, accuracy below

Numbers are read from the scored runs (data/eval/runs) and data/train/neet-e4b-v1.metrics.json, not typed in.
Colours: validated two-slot categorical palette (blue = open weights, orange = commercial API);
identity is also carried by text labels, so nothing depends on colour alone.
"""
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_common as ec  # noqa: E402

ROOT = ec.ROOT
OUT = os.path.join(ROOT, "docs", "img")
SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
OPEN, COMMERCIAL = "#2a78d6", "#eb6834"
QWEN_PSG = "#passage-retrieved-qwen-qwen3-embedding-0-6b"

LADDER = [  # (label, run file stem, open?)
    ("Gemma 4 E4B, plain prompt", "google__gemma-4-E4B-it", True),
    ("Claude Haiku 4.5", "claude-haiku-4-5-20251001", False),
    ("Gemma 4 E4B, 4-bit", "google__gemma-4-E4B-it-qat-w4a16-ct#think#passage-retrieved", True),
    ("Gemma 4 E4B, offline", "google__gemma-4-E4B-it#think" + QWEN_PSG, True),
    ("Qwen3.5-4B", "Qwen__Qwen3.5-4B#think" + QWEN_PSG, True),
    ("Gemma 4 26B-A4B", "google__gemma-4-26B-A4B-it#think#passage-retrieved", True),
    ("Qwen3.5-9B", "Qwen__Qwen3.5-9B#think" + QWEN_PSG, True),
    ("GPT-6 Luna", "gpt-6-luna", False),
    ("Gemma 4 31B", "google__gemma-4-31B-it#think#passage-retrieved", True),
    ("Gemini 3.8 Flash", "gemini-3.8-flash", False),
    ("GPT-6.1 Sol", "gpt-6.1-sol", False),
    ("Claude Opus 5.5", "claude-opus-5-5", False),
]


def accuracy(stem: str) -> float:
    ev = {e["eval_id"]: e for e in ec.load_eval()}
    rows = [json.loads(l) for l in open(os.path.join(ec.RUNS_DIR, stem + ".jsonl"), encoding="utf-8") if l.strip()]
    ok = sum(ec.is_correct(ec.parse_answer(r["text"]), ev[r["eval_id"]]["answer"]) for r in rows)
    assert len(rows) == len(ev), (stem, len(rows))
    return 100 * ok / len(rows)


def style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(colors=INK2, labelsize=10, length=0)
    ax.grid(axis="x", color="#e6e5e1", linewidth=0.8)
    ax.set_axisbelow(True)


def ladder():
    data = sorted(((lab, accuracy(stem), op) for lab, stem, op in LADDER), key=lambda x: x[1])
    fig, ax = plt.subplots(figsize=(9, 6.2), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    style(ax)
    ys = range(len(data))
    for y, (lab, acc, op) in zip(ys, data):
        ax.plot([80, acc], [y, y], color="#e6e5e1", linewidth=2, solid_capstyle="round", zorder=1)
        ax.scatter([acc], [y], s=110, color=OPEN if op else COMMERCIAL, edgecolors=SURFACE, linewidths=2, zorder=3)
        ax.text(acc + 0.35, y, f"{acc:.1f}", va="center", ha="left", fontsize=10, color=INK)
    ax.set_yticks(list(ys))
    ax.set_yticklabels([f"{lab}  ·  {'open' if op else 'API'}" for lab, _, op in data], color=INK)
    ax.set_xlim(80, 101.5)
    ax.set_xticks([80, 85, 90, 95, 100])
    ax.set_xlabel("Accuracy on 660 questions (%)", color=INK2, fontsize=10)
    h1 = ax.scatter([], [], s=80, color=OPEN, label="Open weights, thinking + NCERT page")
    h2 = ax.scatter([], [], s=80, color=COMMERCIAL, label="Commercial API, plain prompt")
    leg = ax.legend(handles=[h1, h2], loc="lower right", frameon=False, fontsize=9.5)
    for t in leg.get_texts():
        t.set_color(INK2)
    fig.text(0.01, 0.975, "Open models vs commercial models on NEET 2025", fontsize=14, color=INK, weight="bold", va="top")
    fig.text(0.01, 0.93, "NEET 2025 in English and Hindi plus 150 held-out NCERT questions, 660 items. "
             "E4B plain prompt shown for reference.", fontsize=9.5, color=INK2, va="top")
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    p = os.path.join(OUT, "blog-ladder.png")
    fig.savefig(p, facecolor=SURFACE)
    print("wrote", p, [(l, round(a, 1)) for l, a, _ in data])


def v1_panels():
    m = json.load(open(os.path.join(ROOT, "data", "train", "neet-e4b-v1.metrics.json"), encoding="utf-8"))
    train = m["curve"]["train"]
    val = m["curve"]["val"]
    before, after = accuracy("google__gemma-4-E4B-it"), accuracy("__ckpt__neet-e4b-v1__merged")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4.2), dpi=200, gridspec_kw={"width_ratios": [1.6, 1]})
    fig.patch.set_facecolor(SURFACE)
    for ax in (a1, a2):
        style(ax)
    a1.grid(axis="y", color="#e6e5e1", linewidth=0.8)
    a1.grid(axis="x", visible=False)
    a1.plot([s for s, _ in train], [l for _, l in train], color=OPEN, linewidth=2, solid_capstyle="round")
    a1.scatter([s for s, _ in val], [l for _, l in val], s=70, color=OPEN, edgecolors=SURFACE, linewidths=2, zorder=3)
    a1.annotate("training loss", (train[3][0], train[3][1]), textcoords="offset points", xytext=(10, 14), fontsize=9.5, color=INK2)
    for s, l in val:
        a1.annotate(f"validation {l:.2f}", (s, l), textcoords="offset points", xytext=(8 if s == 0 else -8, 8),
                    ha="left" if s == 0 else "right", fontsize=9.5, color=INK)
    a1.set_ylim(0, 2.2)
    a1.set_xlabel("Training step", color=INK2, fontsize=10)
    a1.set_title("Loss falls", loc="left", color=INK, fontsize=12, weight="bold")
    a2.grid(axis="x", visible=False)
    a2.grid(axis="y", color="#e6e5e1", linewidth=0.8)
    bars = a2.bar(["Before", "After v1"], [before, after], width=0.45, color=OPEN)
    for b, v in zip(bars, (before, after)):
        a2.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:.1f}%", ha="center", fontsize=10, color=INK)
    a2.set_ylim(0, 100)
    a2.set_ylabel("Accuracy, 660 questions (%)", color=INK2, fontsize=10)
    a2.set_title("Accuracy falls too", loc="left", color=INK, fontsize=12, weight="bold")
    fig.text(0.01, 0.99, "Fine-tune v1 on Gemma 4 E4B: the model learned the style of short explanations, not the reasoning",
             fontsize=11, color=INK2, va="top")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    p = os.path.join(OUT, "blog-v1-loss-vs-accuracy.png")
    fig.savefig(p, facecolor=SURFACE)
    print("wrote", p, round(before, 1), round(after, 1))


def three_runs():
    """Small multiples: training loss (top row, shared scale) and accuracy (bottom row) for v1, v2, v3."""
    load = lambda f: json.load(open(os.path.join(ROOT, "data", "train", f), encoding="utf-8"))
    m1, c2, m3 = load("neet-e4b-v1.metrics.json"), load("neet-e4b-v2.curve_partial.json"), load("neet-e4b-v3.metrics.json")
    v3_base = accuracy("google__gemma-4-E4B-it#think" + QWEN_PSG)
    v3_after = accuracy("__ckpt__neet-e4b-v3__merged#think" + QWEN_PSG)
    runs = [
        ("v1: short explanations", m1["curve"]["train"], m1["curve"]["val"],
         [(0, accuracy("google__gemma-4-E4B-it")), (m1["steps"], accuracy("__ckpt__neet-e4b-v1__merged"))], "660-item eval", "stopped reasoning"),
        ("v2: its own correct answers", c2["train"], c2["val"],
         [(p["step"], 100 * p["acc"]) for p in c2["probe"]], "80-item dev probe", "stopped at step ~520"),
        ("v3: 31B reasoning on its failures", m3["curve"]["train"], m3["curve"]["val"],
         [(0, v3_base), (m3["steps"], v3_after)], "660-item eval, thinking + page", "flat"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.4), dpi=200, sharey="row", gridspec_kw={"height_ratios": [1.15, 1]})
    fig.patch.set_facecolor(SURFACE)
    for col, (title, train, val, acc, acc_label, note) in enumerate(runs):
        top, bot = axes[0][col], axes[1][col]
        for ax in (top, bot):
            style(ax)
            ax.grid(axis="x", visible=False)
            ax.grid(axis="y", color="#e6e5e1", linewidth=0.8)
        top.plot([x for x, _ in train], [y for _, y in train], color=OPEN, linewidth=2, solid_capstyle="round")
        top.scatter([x for x, _ in val], [y for _, y in val], s=60, color=OPEN, edgecolors=SURFACE, linewidths=2, zorder=3)
        for x, y in val:
            top.annotate(f"{y:.2f}", (x, y), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=9, color=INK)
        top.set_ylim(0, 2.2)
        top.set_title(title, loc="left", color=INK, fontsize=11.5, weight="bold")
        bot.plot([x for x, _ in acc], [y for _, y in acc], color=COMMERCIAL, linewidth=2, solid_capstyle="round")
        bot.scatter([x for x, _ in acc], [y for _, y in acc], s=60, color=COMMERCIAL, edgecolors=SURFACE, linewidths=2, zorder=3)
        for i, (x, y) in enumerate(acc):
            bot.annotate(f"{y:.1f}", (x, y), textcoords="offset points", xytext=(0, 9), ha="center", fontsize=9, color=INK)
        bot.set_ylim(60, 100)
        bot.set_xlabel("Training step", color=INK2, fontsize=9.5)
        bot.text(0.02, 0.06, f"{acc_label} · {note}", transform=bot.transAxes, fontsize=9, color=INK2)
        pad = max(x for x, _ in train + acc) * 0.06
        for ax in (top, bot):
            ax.set_xlim(-pad, max(x for x, _ in train + acc) + pad)
    axes[0][0].set_ylabel("Loss", color=INK2, fontsize=10)
    axes[1][0].set_ylabel("Accuracy (%)", color=INK2, fontsize=10)
    fig.text(0.01, 0.99, "Three fine-tunes of Gemma 4 E4B: training loss (top, dots = validation loss) and accuracy (bottom)",
             fontsize=12, color=INK, weight="bold", va="top")
    fig.text(0.01, 0.945, "Loss fell in v1 and v3 and stayed flat in v2. Accuracy never rose above the untrained model.",
             fontsize=10, color=INK2, va="top")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    p = os.path.join(OUT, "blog-three-finetunes.png")
    fig.savefig(p, facecolor=SURFACE)
    print("wrote", p, [(r[0], [(x, round(y, 1)) for x, y in r[3]]) for r in runs])


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    ladder()
    v1_panels()
    three_runs()
