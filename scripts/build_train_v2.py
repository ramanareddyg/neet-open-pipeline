"""Build training set v2 from the model's own correct reasoning (rejection sampling).

Usage:
  python scripts/build_train_v2.py --samples data/train/samples/e4b-v2.jsonl            # writes rescue list
  python scripts/build_train_v2.py --samples data/train/samples/e4b-v2.jsonl \
                                   --rescue data/train/samples/31b-rescue.jsonl          # final v2 set

Why: v1 trained on short teacher explanations and the model learned to stop
reasoning (82.1% -> 74.1%, outputs 5x shorter). v2 keeps the model's own long
reasoning and only the attempts that reach the official/verified answer.

Per prompt (same prompts as train_v1, minus the dev probe):
  - an attempt is accepted if its parsed answer equals the gold answer, it
    finished normally (not cut off at the token limit), and it is written in
    the prompt's language (Hindi prompts must be answered in Devanagari);
  - a prompt counts only if at least --min-correct (default 2) attempts reached
    the gold answer: a single hit out of several can be luck or flawed working;
  - up to 2 accepted attempts per real past-paper prompt, 1 per generated
    MCQ prompt, chosen at random among the accepted ones (not the shortest,
    which would reintroduce the brevity bias);
  - prompts with no accepted attempt go to data/train/rescue_ids.json for a
    Gemma 4 31B pass; accepted 31B attempts are added (1 per prompt).
Writes data/train/train_v2.jsonl and data/train/train_v2_stats.json.
"""
import argparse
import json
import os
import random
import re
import statistics as st
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_common as ec  # noqa: E402

ROOT = ec.ROOT
DEV = re.compile(r"[ऀ-ॿ]")


def load(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def accepted(rec):
    out = []
    for s in rec["samples"]:
        t = (s["text"] or "").strip()
        if s.get("finish") != "stop" or ec.parse_answer(t) != rec["gold"]:
            continue
        hindi_share = len(DEV.findall(t)) / max(1, len(t))
        if rec["lang"] == "hi" and hindi_share < 0.2:
            continue
        if rec["lang"] == "en" and hindi_share > 0.05:
            continue
        out.append(t)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", required=True)
    ap.add_argument("--rescue")
    ap.add_argument("--min-correct", type=int, default=2,
                    help="keep a prompt only if at least this many attempts reached the gold answer (consistency filter)")
    args = ap.parse_args()
    rng = random.Random(20251005)
    prompts = {r["id"]: r for r in load(os.path.join(ROOT, "data", "train", "train_v1.jsonl"))}
    recs = load(args.samples)
    stats = Counter()
    by = defaultdict(lambda: [0, 0])
    out, rescue = [], []
    lens_acc = []
    for rec in recs:
        acc = accepted(rec)
        key = f"{'pyq' if rec['source'].startswith('pyq') else 'mcq'}-{rec['lang']}"
        by[key][1] += 1
        stats["samples_total"] += len(rec["samples"])
        stats["samples_accepted"] += len(acc)
        if not acc:
            rescue.append(rec["id"])
            continue
        if len(acc) < args.min_correct:
            # One correct attempt out of several can be luck or flawed working: not trusted as a target.
            stats["dropped_inconsistent"] += 1
            continue
        by[key][0] += 1
        k = 2 if rec["source"].startswith("pyq") else 1
        for j, t in enumerate(rng.sample(acc, min(k, len(acc)))):
            p = prompts[rec["id"]]
            out.append({**{x: p[x] for x in ("source", "lang", "with_passage", "images")}, "id": f"{rec['id']}-s{j}",
                        "teacher": "self:gemma-4-E4B-it", "messages": [p["messages"][0], {"role": "assistant", "content": t}]})
            lens_acc.append(len(t))
    stats["prompts"] = len(recs)
    stats["prompts_with_accepted"] = len(recs) - len(rescue)
    stats["prompt_accept_rate_by"] = {k: f"{a}/{n} ({a / n:.1%})" for k, (a, n) in sorted(by.items())}

    if args.rescue:
        n_res = 0
        for rec in load(args.rescue):
            acc = accepted(rec)
            if not acc:
                stats["rescue_failed"] += 1
                continue
            if len(acc) < args.min_correct:
                stats["rescue_dropped_inconsistent"] += 1
                continue
            p = prompts[rec["id"]]
            t = rng.choice(acc)
            out.append({**{x: p[x] for x in ("source", "lang", "with_passage", "images")}, "id": f"{rec['id']}-r0",
                        "teacher": "gemma-4-31B-it", "messages": [p["messages"][0], {"role": "assistant", "content": t}]})
            lens_acc.append(len(t))
            n_res += 1
        stats["rescued"] = n_res
    else:
        json.dump(rescue, open(os.path.join(ROOT, "data", "train", "rescue_ids.json"), "w"), indent=0)
        stats["rescue_ids_written"] = len(rescue)

    rng.shuffle(out)
    with open(os.path.join(ROOT, "data", "train", "train_v2.jsonl"), "w", encoding="utf-8") as f:
        for row in out:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    v1_len = st.median(len(p["messages"][1]["content"]) for p in prompts.values())
    stats = dict(stats)
    stats.update({"examples": len(out), "by_source": dict(Counter(r["source"] for r in out)),
                  "by_lang": dict(Counter(r["lang"] for r in out)), "by_teacher": dict(Counter(r["teacher"] for r in out)),
                  "with_images": sum(bool(r["images"]) for r in out),
                  "median_target_chars_v2": st.median(lens_acc) if lens_acc else 0, "median_target_chars_v1": v1_len})
    json.dump(stats, open(os.path.join(ROOT, "data", "train", "train_v2_stats.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps(stats, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
