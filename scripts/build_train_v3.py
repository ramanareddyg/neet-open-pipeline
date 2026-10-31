"""Build training set v3: Gemma 4 31B thinking traces on the prompts E4B gets wrong, in the shipping setting.

Usage:
  python scripts/build_train_v3.py --select --student data/train/samples/e4b-v3-greedy.jsonl
      -> data/eval/val_v3.jsonl + val_v3_qids.json   (validation questions held out of training: figures,
                                                        native-Hindi past papers, hard questions; both languages)
      -> data/train/v3_31b_ids.json                   (prompts for the 31B: every remaining prompt E4B got wrong
                                                        or did not finish, plus ballast = --ballast-ratio x that
                                                        many prompts it got right)
      Exits non-zero and writes no id file if a sanity gate fails (coverage, student accuracy 80-97%,
      1,000-6,000 prompts selected).
  python scripts/build_train_v3.py --build --teacher data/train/samples/31b-v3.jsonl [--student ...]
      -> data/train/train_v3.jsonl, train_v3_stats.json (acceptance split by student-wrong vs ballast,
         length distribution and the share that a given max_len would drop)

Lessons applied from v1/v2 (worklog/2026-10-06.md) and the 6 Oct review (docs/finetune_v3_plan.md):
  - targets are long worked reasoning (31B's own thinking trace + its answer), not short explanations;
  - most of the data is prompts the student gets wrong (target mixture ~60/40 wrong/ballast);
  - thinking on + retrieved NCERT passage in every prompt, same as evaluation and the product;
  - a 31B trace is kept only if it finished, reached the gold answer, and the final answer is in the prompt's language;
  - a real validation set exists besides the 80-item probe, and it is excluded from training.
Each row: {id, source, lang, images, with_passage: true, teacher, student_correct, messages: [user, assistant], reasoning}
where `reasoning` is the thought text and the assistant content is the visible answer ending in "Answer: N".
"""
import argparse
import json
import os
import random
import re
import statistics as st
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_common as ec  # noqa: E402

ROOT = ec.ROOT
DEV = re.compile(r"[ऀ-ॿ]")
MAX_IMAGES = 12
IMAGE_TOKENS = 256  # rough per-image budget for length estimates


def load(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def split_thought(text: str):
    """Raw vLLM output with thinking on: '<|channel>thought\\n...<channel|>answer'. Returns (reasoning, content)."""
    if "<channel|>" not in text:
        return None, text.strip()
    thought, content = text.split("<channel|>", 1)
    thought = thought.split("<|channel>thought", 1)[-1].strip("\n")
    return thought.strip(), content.strip()


def student_verdicts(path):
    return {r["id"]: (r["finish"] == "stop" and ec.parse_answer(r["text"]) == r["gold"]) for r in load(path)}


def est_tokens(text: str, n_images: int) -> int:
    dev = len(DEV.findall(text))
    return int((len(text) - dev) / 4 + dev / 2.2) + IMAGE_TOKENS * n_images  # Devanagari tokenises ~2x denser per char


def select(args, prompts, rng):
    v = student_verdicts(args.student)
    expected = [i for i, p in prompts.items() if len(p["images"]) <= MAX_IMAGES]
    missing = [i for i in expected if i not in v]
    if missing:
        sys.exit(f"GATE: student output missing for {len(missing)} prompts (e.g. {missing[:3]}); not selecting")
    acc = sum(v[i] for i in expected) / len(expected)
    if not 0.80 <= acc <= 0.97:
        sys.exit(f"GATE: student accuracy {acc:.1%} outside 80-97%; not selecting")

    # ---- validation set: question ids, both languages held out ----
    by_qid = {}
    for i in expected:
        by_qid.setdefault(prompts[i]["qid"], []).append(i)
    wrong_q = {q for q, ids in by_qid.items() if not all(v[i] for i in ids)}
    fig_q = [q for q, ids in by_qid.items() if prompts[ids[0]]["images"] and prompts[ids[0]]["source"].startswith("pyq")]
    hi_q = [q for q, ids in by_qid.items() if any(prompts[i]["source"] == "pyq_hi_q" for i in ids)]
    hard_pyq = [q for q in wrong_q if prompts[by_qid[q][0]]["source"].startswith("pyq")]
    hard_mcq = [q for q in wrong_q if prompts[by_qid[q][0]]["source"] == "ncert_mcq"]
    val = set()
    for pool, n in ((fig_q, 40), (hi_q, 40), (hard_pyq, 40), (hard_mcq, 20)):
        pool = [q for q in pool if q not in val]
        rng.shuffle(pool)
        val.update(pool[:n])
    val_rows = []
    for q in sorted(val):
        for i in by_qid[q]:
            p = prompts[i]
            val_rows.append({"eval_id": i, "source": p["source"], "qid": q, "lang": p["lang"], "subject": p["subject"],
                             "question": p["question"], "options": p["options"], "answer": p["gold"], "images": p["images"],
                             "has_figure": bool(p["images"]), "style": p.get("style"), "user_text": p["user_text"],
                             "student_correct": v[i], "strata": {"figure": q in fig_q, "native_hi": q in hi_q, "hard": q in wrong_q}})
    with open(os.path.join(ROOT, "data", "eval", "val_v3.jsonl"), "w", encoding="utf-8") as f:
        for r in val_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    json.dump(sorted(val), open(os.path.join(ROOT, "data", "eval", "val_v3_qids.json"), "w"), indent=0)

    # ---- 31B selection on the rest ----
    rest = [i for i in expected if prompts[i]["qid"] not in val]
    wrong = [i for i in rest if not v[i]]
    right = [i for i in rest if v[i]]
    ballast = rng.sample(right, min(len(right), int(round(len(wrong) * args.ballast_ratio))))
    ids = sorted(wrong + ballast)
    if not 1000 <= len(ids) <= 6000:
        sys.exit(f"GATE: {len(ids)} prompts selected, outside 1,000-6,000; not selecting")
    json.dump(ids, open(os.path.join(ROOT, "data", "train", "v3_31b_ids.json"), "w"), indent=0)
    by = Counter((prompts[i]["source"], prompts[i]["lang"]) for i in wrong)
    val_fig = sum(r["has_figure"] for r in val_rows)
    print(f"student: {acc:.1%} right on {len(expected)} prompts (excluded {len(prompts) - len(expected)} with >{MAX_IMAGES} images)")
    print(f"validation: {len(val)} questions / {len(val_rows)} items; with figure {val_fig}, native Hindi {sum(r['strata']['native_hi'] for r in val_rows)}, "
          f"student-wrong items {sum(not r['student_correct'] for r in val_rows)}; student acc on val {sum(r['student_correct'] for r in val_rows) / len(val_rows):.1%}")
    print(f"31B set: {len(wrong)} student-wrong ({len(wrong) / len(ids):.0%}) by (source, lang) {dict(by)} + {len(ballast)} ballast ({len(ballast) / len(ids):.0%}) = {len(ids)}")


def build(args, prompts, rng):
    v = student_verdicts(args.student) if os.path.exists(args.student) else {}
    val_q = set(json.load(open(os.path.join(ROOT, "data", "eval", "val_v3_qids.json")))) if os.path.exists(os.path.join(ROOT, "data", "eval", "val_v3_qids.json")) else set()
    stats, acc_by = Counter(), {True: [0, 0], False: [0, 0]}
    out, lens_r, lens_c, toks = [], [], [], []
    for r in load(args.teacher):
        stats["teacher_rows"] += 1
        sc = v.get(r["id"])
        acc_by[bool(sc)][1] += 1
        if prompts[r["id"]]["qid"] in val_q:
            stats["drop_in_validation"] += 1
            continue
        if r["finish"] != "stop":
            stats["drop_unfinished"] += 1
            continue
        if ec.parse_answer(r["text"]) != r["gold"]:
            stats["drop_wrong"] += 1
            continue
        reasoning, content = split_thought(r["text"])
        if not reasoning or len(content) < 20:
            stats["drop_no_thought"] += 1
            continue
        share = len(DEV.findall(content)) / max(1, len(content))
        if (r["lang"] == "hi" and share < 0.2) or (r["lang"] == "en" and share > 0.05):
            stats["drop_language"] += 1
            continue
        p = prompts[r["id"]]
        acc_by[bool(sc)][0] += 1
        out.append({"id": r["id"], "source": p["source"], "lang": p["lang"], "images": p["images"], "with_passage": True,
                    "teacher": "gemma-4-31B-it:think", "student_correct": sc,
                    "messages": [{"role": "user", "content": p["user_text"]}, {"role": "assistant", "content": content}],
                    "reasoning": reasoning})
        lens_r.append(len(reasoning))
        lens_c.append(len(content))
        toks.append(est_tokens(p["user_text"] + reasoning + content, len(p["images"])))
    rng.shuffle(out)
    with open(os.path.join(ROOT, "data", "train", "train_v3.jsonl"), "w", encoding="utf-8") as f:
        for row in out:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    stats = dict(stats)
    n = len(out)
    stats.update({
        "examples": n, "by_source": dict(Counter(r["source"] for r in out)), "by_lang": dict(Counter(r["lang"] for r in out)),
        "mixture": {"student_wrong": sum(r["student_correct"] is False for r in out), "ballast": sum(r["student_correct"] is True for r in out)},
        "teacher_acceptance": {"on_student_wrong": f"{acc_by[False][0]}/{acc_by[False][1]}", "on_ballast": f"{acc_by[True][0]}/{acc_by[True][1]}"},
        "with_images": sum(bool(r["images"]) for r in out),
        "median_reasoning_chars": st.median(lens_r) if lens_r else 0, "median_answer_chars": st.median(lens_c) if lens_c else 0,
        "est_tokens_p50_p90_max": [int(st.median(toks)), int(sorted(toks)[int(0.9 * n)]), max(toks)] if toks else None,
        "share_dropped_at_max_len": {L: round(sum(t > L for t in toks) / max(1, n), 3) for L in (4096, 6144, 8192, 12288)},
        "share_dropped_at_max_len_student_wrong": {L: round(sum(t > L for t, r in zip(toks, out) if r["student_correct"] is False) /
                                                            max(1, sum(r["student_correct"] is False for r in out)), 3) for L in (4096, 6144, 8192, 12288)},
    })
    json.dump(stats, open(os.path.join(ROOT, "data", "train", "train_v3_stats.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps(stats, ensure_ascii=False, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--select", action="store_true")
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--student", default=os.path.join(ROOT, "data", "train", "samples", "e4b-v3-greedy.jsonl"))
    ap.add_argument("--teacher", default=os.path.join(ROOT, "data", "train", "samples", "31b-v3.jsonl"))
    ap.add_argument("--ballast-ratio", type=float, default=0.67,
                    help="ballast prompts (student already right) as a multiple of the student-wrong prompts; 0.67 -> ~60/40 mixture")
    args = ap.parse_args()
    rng = random.Random(20251006)
    prompts = {r["id"]: r for r in load(os.path.join(ROOT, "data", "train", "v3_prompts_rendered.jsonl"))}
    if args.select:
        select(args, prompts, rng)
    if args.build:
        build(args, prompts, rng)


if __name__ == "__main__":
    main()
