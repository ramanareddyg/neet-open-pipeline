"""Run the eval set through API models (Claude, OpenAI, Gemini).

Usage:
  python scripts/eval_api.py --model claude-opus-5-5 [--limit 5] [--workers 8]
  python scripts/eval_api.py --model gpt-6.1-sol
  python scripts/eval_api.py --model gemini-3.8-flash

Same prompt, images and parser as the Modal runs (eval_common). Default
settings per provider (no extra reasoning flags), max 8k output tokens.
Responses append to data/eval/runs/<model>.jsonl (resumable); a summary with
tokens and estimated cost goes to the run log.
"""
import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_common as ec  # noqa: E402
import runlog  # noqa: E402

load_dotenv(os.path.join(ec.ROOT, ".env"), override=True)
MAX_OUT = 8000
_lock = threading.Lock()


def provider(model: str) -> str:
    if model.startswith("claude"):
        return "anthropic"
    if model.startswith(("gpt", "o1", "o3", "o4")):
        return "openai"
    if model.startswith("gemini"):
        return "gemini"
    raise SystemExit(f"unknown provider for {model}")


def call_anthropic(client, model, item):
    content = [{"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": ec.image_b64(p)}} for p in item["images"]]
    content.append({"type": "text", "text": ec.prompt_text(item)})
    r = client.messages.create(model=model, max_tokens=MAX_OUT, messages=[{"role": "user", "content": content}])
    text = "".join(b.text for b in r.content if getattr(b, "type", "") == "text")
    return text, r.usage.input_tokens, r.usage.output_tokens, r.stop_reason


def call_openai(client, model, item):
    content = [{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{ec.image_b64(p)}"}} for p in item["images"]]
    content.append({"type": "text", "text": ec.prompt_text(item)})
    r = client.chat.completions.create(model=model, max_completion_tokens=MAX_OUT, messages=[{"role": "user", "content": content}])
    return r.choices[0].message.content, r.usage.prompt_tokens, r.usage.completion_tokens, r.choices[0].finish_reason


def call_gemini(client, model, item):
    from google.genai import types
    parts = [types.Part.from_bytes(data=open(os.path.join(ec.ROOT, p), "rb").read(), mime_type="image/png") for p in item["images"]]
    parts.append(ec.prompt_text(item))
    r = client.models.generate_content(model=model, contents=parts, config=types.GenerateContentConfig(max_output_tokens=MAX_OUT))
    u = r.usage_metadata
    fin = str(r.candidates[0].finish_reason) if r.candidates else "NO_CANDIDATE"
    return r.text, u.prompt_token_count or 0, (u.candidates_token_count or 0) + (u.thoughts_token_count or 0), fin


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    prov = provider(args.model)
    if prov == "anthropic":
        import anthropic
        client, fn = anthropic.Anthropic(max_retries=6), call_anthropic
    elif prov == "openai":
        import openai
        client, fn = openai.OpenAI(max_retries=6), call_openai
    else:
        from google import genai
        client, fn = genai.Client(api_key=os.environ["GOOGLE_API_KEY"]), call_gemini

    items = [it for it in ec.load_eval() if it["eval_id"] not in ec.done_ids(args.model)]
    if args.limit:
        items = items[: args.limit]
    print(f"{args.model}: {len(items)} items")
    tot = {"in": 0, "out": 0, "ok": 0, "n": 0, "fail": 0}
    t0 = time.time()

    def work(it):
        last = None
        for attempt in range(4):
            try:
                ts = time.time()
                text, ti, to, fin = fn(client, args.model, it)
                return it, text, ti, to, fin, round(time.time() - ts, 2)
            except Exception as e:  # noqa: BLE001
                last = e
                time.sleep(5 * (attempt + 1))
        raise RuntimeError(f"{it['eval_id']}: {type(last).__name__} {str(last)[:200]}")

    with ThreadPoolExecutor(args.workers) as ex:
        futs = [ex.submit(work, it) for it in items]
        for k, f in enumerate(as_completed(futs), 1):
            try:
                it, text, ti, to, fin, secs = f.result()
            except Exception as e:  # noqa: BLE001
                tot["fail"] += 1
                print("  FAILED", str(e)[:250])
                continue
            pred = ec.parse_answer(text)
            ok = ec.is_correct(pred, it["answer"])
            tot["in"] += ti
            tot["out"] += to
            tot["ok"] += ok
            tot["n"] += 1
            with _lock, open(ec.run_path(args.model), "a", encoding="utf-8") as fh:
                fh.write(json.dumps({"eval_id": it["eval_id"], "model": args.model, "pred": pred, "correct": ok, "text": text,
                                     "input_tokens": ti, "output_tokens": to, "finish": fin, "seconds": secs}, ensure_ascii=False) + "\n")
            if k % 100 == 0:
                print(f"  {k}/{len(items)} acc {tot['ok'] / tot['n']:.1%}", flush=True)
    cost = runlog.cost_usd(args.model, tot["in"], tot["out"])
    acc = tot["ok"] / tot["n"] if tot["n"] else 0
    print(f"done: {tot['ok']}/{tot['n']} = {acc:.1%}, failed {tot['fail']}; tokens in {tot['in']:,} out {tot['out']:,}; ~${cost}")
    runlog.log("eval_api", model=args.model, items=tot["n"], correct=tot["ok"], failed=tot["fail"],
               input_tokens=tot["in"], output_tokens=tot["out"], seconds=round(time.time() - t0, 1))


if __name__ == "__main__":
    main()
