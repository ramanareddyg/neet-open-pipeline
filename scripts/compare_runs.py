"""Compare two eval runs the careful way: paired by question, with flips and a bootstrap interval.

Usage:
  python scripts/compare_runs.py <run_a.jsonl> <run_b.jsonl> [--eval data/eval/eval_v1.jsonl]
  python scripts/compare_runs.py "data/eval/runs/google__gemma-4-E4B-it#think#passage-retrieved-qwen-qwen3-embedding-0-6b.jsonl" \
                                 "data/eval/runs/_ckpt__neet-e4b-v3__merged#think#passage-retrieved-qwen-qwen3-embedding-0-6b.jsonl"

Reports, for the items both runs answered: accuracy of each, right->wrong and wrong->right flips,
the difference with a 95% bootstrap interval resampling *questions* (the English and Hindi items of
one question are kept together, since they are not independent), and the same per cut
(source, language, figure, subject, and val_v3 strata when present). Answers are re-parsed from the
saved text with eval_common.parse_answer, as score_eval.py does.
"""
import argparse
import collections
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_common as ec  # noqa: E402


def load_run(p):
    return {r["eval_id"]: r for r in (json.loads(l) for l in open(p, encoding="utf-8") if l.strip())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--eval", default=os.path.join(ec.ROOT, "data", "eval", "eval_v1.jsonl"))
    ap.add_argument("--boot", type=int, default=5000)
    args = ap.parse_args()
    ev = {e["eval_id"]: e for e in (json.loads(l) for l in open(args.eval, encoding="utf-8") if l.strip())}
    A, B = load_run(args.a), load_run(args.b)
    ids = [i for i in ev if i in A and i in B]
    ok = {i: (ec.is_correct(ec.parse_answer(A[i]["text"]), ev[i]["answer"]), ec.is_correct(ec.parse_answer(B[i]["text"]), ev[i]["answer"])) for i in ids}
    by_q = collections.defaultdict(list)
    for i in ids:
        by_q[ev[i]["qid"]].append(i)
    qids = list(by_q)
    rng = random.Random(0)

    def report(name, sel):
        if not sel:
            return
        a = sum(ok[i][0] for i in sel)
        b = sum(ok[i][1] for i in sel)
        rw = sum(ok[i][0] and not ok[i][1] for i in sel)
        wr = sum(ok[i][1] and not ok[i][0] for i in sel)
        qs = sorted({ev[i]["qid"] for i in sel})
        sel_set = set(sel)
        diffs = []
        for _ in range(args.boot):
            s = rng.choices(qs, k=len(qs))
            d = 0
            n = 0
            for q in s:
                for i in by_q[q]:
                    if i in sel_set:
                        d += ok[i][1] - ok[i][0]
                        n += 1
            diffs.append(d / max(1, n))
        diffs.sort()
        lo, hi = diffs[int(0.025 * len(diffs))], diffs[int(0.975 * len(diffs))]
        print(f"{name:28} n={len(sel):4}  A {100 * a / len(sel):5.1f}  B {100 * b / len(sel):5.1f}  diff {100 * (b - a) / len(sel):+5.1f} "
              f"[{100 * lo:+5.1f}, {100 * hi:+5.1f}]  right->wrong {rw:3}  wrong->right {wr:3}")

    print(f"A = {os.path.basename(args.a)}\nB = {os.path.basename(args.b)}\n(diff = B - A, 95% interval by resampling questions)")
    report("all", ids)
    for key in ("source", "lang", "subject"):
        for v in sorted({ev[i].get(key) for i in ids if ev[i].get(key) is not None}):
            report(f"{key}={v}", [i for i in ids if ev[i].get(key) == v])
    report("has_figure", [i for i in ids if ev[i].get("has_figure")])
    if any("strata" in ev[i] for i in ids):
        for st in ("figure", "native_hi", "hard"):
            report(f"stratum {st}", [i for i in ids if ev[i].get("strata", {}).get(st)])
        report("student was wrong", [i for i in ids if ev[i].get("student_correct") is False])


if __name__ == "__main__":
    main()
