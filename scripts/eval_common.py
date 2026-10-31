"""Shared prompt, answer parsing and scoring for every model in the eval.

Every model, local or API, gets the same text and images, and is scored by
the same parser, so the numbers are comparable.
"""
import base64
import json
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVAL_PATH = os.path.join(ROOT, "data", "eval", "eval_v1.jsonl")
RUNS_DIR = os.path.join(ROOT, "data", "eval", "runs")

INSTRUCTION = {
    "en": ("Answer this NEET-UG multiple-choice question. Think it through briefly, then give the final answer "
           "on the last line in exactly this form:\nAnswer: <option number 1-4>"),
    "hi": ("इस NEET-UG बहुविकल्पीय प्रश्न का उत्तर दीजिए। संक्षेप में हिंदी में हल कीजिए, फिर अंतिम पंक्ति में "
           "ठीक इसी रूप में उत्तर लिखिए:\nAnswer: <विकल्प संख्या 1-4>"),
}


def load_eval() -> list[dict]:
    return [json.loads(l) for l in open(EVAL_PATH, encoding="utf-8") if l.strip()]


def prompt_text(item: dict) -> str:
    opts = "\n".join(f"({i + 1}) {o}" for i, o in enumerate(item["options"]))
    fig = ""
    if item["images"]:
        fig = ("\n[The figure(s) for this question are attached.]\n" if item["lang"] == "en"
               else "\n[इस प्रश्न के चित्र संलग्न हैं।]\n")
    return f"{INSTRUCTION[item['lang']]}\n\n{item['question']}\n{fig}\n{opts}"


def image_b64(path: str) -> str:
    with open(os.path.join(ROOT, path), "rb") as f:
        return base64.b64encode(f.read()).decode()


# Allow a short run of non-digits between the colon and the number: models
# echo the template ("Answer: <विकल्प संख्या 2>", "Answer: <1>", "**Answer:** (3)").
# The lookahead rejects an unfilled template ("<option number 1-4>").
_ANS = re.compile(r"(?:answer|उत्तर)\s*\**\s*[:：]\s*[^\d\n]{0,30}?([1-4])(?![\d.])(?!\s*[-–]\s*4)", re.I)
_FALLBACK = re.compile(r"(?:option|विकल्प)\s*\(?\s*([1-4])\s*\)?", re.I)


def parse_answer(text: str | None) -> int | None:
    """Last 'Answer: N' in the response; else the last 'option N'; else None (counted wrong)."""
    if not text:
        return None
    m = _ANS.findall(text)
    if m:
        return int(m[-1])
    m = _FALLBACK.findall(text[-300:])
    return int(m[-1]) if m else None


def is_correct(pred: int | None, answer) -> bool:
    if pred is None:
        return False
    if isinstance(answer, list):
        return pred in answer
    return pred == answer


def slug(model: str) -> str:
    return model.replace("/", "__")


def run_path(model: str) -> str:
    os.makedirs(RUNS_DIR, exist_ok=True)
    return os.path.join(RUNS_DIR, f"{slug(model)}.jsonl")


def done_ids(model: str) -> set[str]:
    p = run_path(model)
    if not os.path.exists(p):
        return set()
    return {json.loads(l)["eval_id"] for l in open(p, encoding="utf-8") if l.strip()}
