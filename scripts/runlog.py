"""Append-only run log for the project story: counts, tokens, time, cost.

Every pipeline stage calls `log(stage, **fields)` once per run. Entries land
in data/runlog.jsonl with a UTC timestamp. Costs are estimates from the
PRICES table (USD per 1M tokens); keep that table honest when models change.
"""
import datetime as _dt
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PATH = os.path.join(ROOT, "data", "runlog.jsonl")

# USD per 1M tokens, (input, output). From ai.google.dev/gemini-api/docs/pricing on 2026-10-04.
# gemini-3.8-flash is a promotional price through 2026-12-31 (doubles from 2027-01-01).
PRICES = {
    "gemini-2.5-flash": (0.30, 2.50),
    "gemini-3.8-flash": (0.75, 3.75),
    "gemini-embedding-001": (0.20, 0.0),  # not listed; gemini-embedding-2 text price used as the estimate
    # OpenRouter list prices on 2026-10-05 (openrouter.ai/api/v1/models). gen_synthetic also logs OpenRouter's own per-call cost.
    "z-ai/glm-5.3": (1.40, 4.40),
    # Eval comparators, list prices on 2026-10-05 (anthropic.com/pricing; OpenAI per press coverage of the Sep 2026 launch).
    "claude-opus-5-5": (4.0, 20.0),
    "claude-sonnet-5-5": (2.0, 10.0),
    "claude-haiku-4-5-20251001": (1.0, 5.0),
    "gpt-6.1-sol": (2.0, 10.0),
    "gpt-6-luna": (0.10, 0.50),
    "z-ai/glm-5.3-flash": (0.15, 0.50),
}


def cost_usd(model: str, input_tokens: int, output_tokens: int) -> float | None:
    p = PRICES.get(model)
    if not p or p[0] is None:
        return None
    return round(input_tokens / 1e6 * p[0] + output_tokens / 1e6 * p[1], 4)


def log(stage: str, **fields) -> dict:
    entry = {"ts": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"), "stage": stage, **fields}
    if "model" in fields and "input_tokens" in fields and "output_tokens" in fields:
        entry["cost_usd"] = cost_usd(fields["model"], fields["input_tokens"] or 0, fields["output_tokens"] or 0)
    os.makedirs(os.path.dirname(PATH), exist_ok=True)
    with open(PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return entry
