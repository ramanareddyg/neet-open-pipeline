"""Generate fine-tuning data: worked explanations for past-paper questions,
and new NEET-style MCQs from NCERT passages, in English and Hindi.

Default generator: gemini-3.8-flash (GOOGLE_API_KEY in .env; the key is
already in .env). GLM-5.3 via OpenRouter is kept as an alternative:
GEN_MODEL=z-ai/glm-5.3 with OPENROUTER_API_KEY in .env.

Licence note for the dataset card: Google's Gemini API terms restrict using
outputs to develop models that compete with Gemini. GLM-5.3's licence has no
such clause. Each record stores the generating model in `gen.model`.

Two modes:

  explain  For each past-paper question: the model solves it *unaided* first
           (its pick is compared with the official key and logged, a free
           accuracy stat), then writes a worked explanation in English and
           Hindi, grounded in the linked NCERT passage when one exists. If the
           unaided pick is wrong, it is re-asked with the official answer given,
           and the record is marked `answer_given`.

             python scripts/gen_synthetic.py explain --paper 2024_en_T1 [--limit 5]

  mcq      For each NCERT passage: write N new NEET-style MCQs answerable from
           that passage, with four options, the answer, an explanation, and
           Hindi for all of it.

             python scripts/gen_synthetic.py mcq [--chapter kebo101] [--per-passage 4] [--limit 5]

Outputs (resumable; ids already present are skipped unless --force):
  data/synthetic/explain/<paper>.jsonl
  data/synthetic/mcq/<chapter_code>.jsonl

The 2025 paper is the held-out evaluation set: `explain` refuses 2025 papers
unless --allow-holdout, and generated MCQs are later de-duplicated against it.
"""
import argparse
import json
import os
import random
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Literal

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field, ValidationError

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import runlog  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(ROOT, ".env"), override=True)

GEN_MODEL = os.environ.get("GEN_MODEL", "gemini-3.8-flash")
REASONING = os.environ.get("GEN_REASONING", "low")  # low | medium | high
IS_GEMINI = GEN_MODEL.startswith("gemini")
HOLDOUT_YEARS = {2025}


# ---------- schemas ----------

class Explanation(BaseModel):
    chosen_option: int = Field(description="1-4: the option you judge correct")
    explanation_en: str = Field(description="Worked explanation in English, 60-180 words: the concept, the reasoning or calculation step by step, and why the other options are wrong where useful. LaTeX in $...$ for maths.")
    explanation_hi: str = Field(description="The same explanation in natural Hindi (Devanagari), keeping standard scientific terms and symbols as students see them in Hindi-medium NCERT.")
    concept: str = Field(description="The core concept tested, a short phrase")
    ncert_basis: str = Field(description="One sentence on what the NCERT passage provides for this question, in your own words; 'not in passage' if it does not help")


class MCQ(BaseModel):
    question_en: str
    options_en: list[str] = Field(description="Exactly four options, no numbering")
    answer: int = Field(description="1-4")
    explanation_en: str = Field(description="40-120 words, grounded in the passage")
    question_hi: str = Field(description="Hindi (Devanagari) version of the question")
    options_hi: list[str] = Field(description="Hindi versions of the four options, same order")
    explanation_hi: str
    concept: str
    difficulty: Literal["easy", "medium", "hard"]
    style: Literal["direct_recall", "statement_based", "assertion_reason", "match_the_following", "numerical", "diagram_free_application"]


class MCQSet(BaseModel):
    questions: list[MCQ]


# ---------- prompts ----------

SOLVE_PROMPT = """You are an expert NEET-UG tutor. Solve this {subject} question.
{passage_block}
QUESTION:
{question}

OPTIONS:
{options}
{figure_block}{answer_block}
Decide which option is correct, then write the explanation a strong student would want: the concept, the reasoning or calculation, and briefly why the tempting wrong options fail. Then give the same explanation in Hindi.
Return JSON with keys chosen_option, explanation_en, explanation_hi, concept, ncert_basis."""

MCQ_PROMPT = """You write NEET-UG practice questions for Indian students.
Below is a passage from the NCERT {subject} textbook (Class {klass}, chapter "{title}").

PASSAGE:
{passage}

Write {n} new multiple-choice questions in the NEET-UG style that a student can answer from this passage plus standard Class 11-12 knowledge.
Rules:
- Exactly four options each, exactly one correct. Distractors must be plausible to a student who half-knows the topic.
- Vary the style across the set (direct recall, statement-based "how many of the following are correct", assertion-reason, match-the-following, short numerical where the passage supports it). Avoid questions that need a figure.
- Do not copy sentences from the passage; test understanding of it.
- Do not reproduce past NEET questions.
- Give everything in English and in natural Hindi (Devanagari) as used in Hindi-medium NCERT.
Return JSON: {{"questions": [{{question_en, options_en, answer, explanation_en, question_hi, options_hi, explanation_hi, concept, difficulty, style}}, ...]}}."""


# ---------- client ----------

_client = None
_lock = threading.Lock()
_totals = {"calls": 0, "input_tokens": 0, "output_tokens": 0, "reasoning_tokens": 0, "cost_usd": 0.0, "errors": 0, "refusals": 0}


class Refusal(Exception):
    pass


def client():
    global _client
    if _client is None:
        if IS_GEMINI:
            from google import genai
            _client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
        else:
            key = os.environ.get("OPENROUTER_API_KEY")
            if not key:
                sys.exit("OPENROUTER_API_KEY missing: add it to .env (never commit .env)")
            _client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=key)
    return _client


def _call_gemini(prompt, schema, max_tokens, images=None):
    from google.genai import types
    r = client().models.generate_content(
        model=GEN_MODEL, contents=[prompt, *(images or [])],
        config=types.GenerateContentConfig(
            response_mime_type="application/json", response_schema=schema, temperature=0.6,
            max_output_tokens=max_tokens, thinking_config=types.ThinkingConfig(thinking_level=REASONING),
        ),
    )
    fr = str(getattr(r.candidates[0], "finish_reason", "")) if r.candidates else "NO_CANDIDATE"
    if "RECITATION" in fr or "SAFETY" in fr or "PROHIBITED" in fr:
        raise Refusal(fr)
    u = r.usage_metadata
    out, thoughts = u.candidates_token_count or 0, u.thoughts_token_count or 0
    meta = {"input_tokens": u.prompt_token_count, "output_tokens": out + thoughts, "reasoning_tokens": thoughts,
            "cost_usd": runlog.cost_usd(GEN_MODEL, u.prompt_token_count or 0, out + thoughts)}
    return r.text or "", meta


def _call_openrouter(prompt, schema, max_tokens, images=None):  # text-only: figures go in as descriptions
    r = client().chat.completions.create(
        model=GEN_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.6,
        max_tokens=max_tokens,
        response_format={"type": "json_schema", "json_schema": {"name": schema.__name__, "schema": schema.model_json_schema()}},
        extra_body={"reasoning": {"effort": REASONING, "exclude": True}, "usage": {"include": True}},
        extra_headers={"X-Title": "neet-open"},
    )
    u = r.usage
    details = getattr(u, "completion_tokens_details", None)
    meta = {"provider": getattr(r, "provider", None), "input_tokens": u.prompt_tokens, "output_tokens": u.completion_tokens,
            "reasoning_tokens": getattr(details, "reasoning_tokens", None) if details else None, "cost_usd": getattr(u, "cost", None)}
    return r.choices[0].message.content or "", meta


def call(prompt: str, schema: type[BaseModel], max_tokens: int = 8000, retries: int = 3, images=None):
    """One structured call. Returns (parsed, meta). Retries on transport or schema errors."""
    last = None
    for attempt in range(retries):
        try:
            t0 = time.time()
            text, meta = (_call_gemini if IS_GEMINI else _call_openrouter)(prompt, schema, max_tokens, images)
            text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
            parsed = schema.model_validate_json(text)
            meta = {"model": GEN_MODEL, **meta, "seconds": round(time.time() - t0, 1)}
            with _lock:
                _totals["calls"] += 1
                _totals["input_tokens"] += meta["input_tokens"] or 0
                _totals["output_tokens"] += meta["output_tokens"] or 0
                _totals["reasoning_tokens"] += meta["reasoning_tokens"] or 0
                _totals["cost_usd"] += meta["cost_usd"] or 0.0
            return parsed, meta
        except Refusal as e:
            last = f"refused: {e}"
            with _lock:
                _totals["refusals"] += 1
        except (ValidationError, json.JSONDecodeError) as e:
            last = f"schema: {str(e)[:200]}"
        except Exception as e:  # noqa: BLE001 - transport / rate limit
            last = f"{type(e).__name__}: {str(e)[:200]}"
            time.sleep(2 ** attempt * 3)
    with _lock:
        _totals["errors"] += 1
    raise RuntimeError(last)


# ---------- helpers ----------

def load_jsonl(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def done_ids(path: str) -> set[str]:
    return {r["id"] for r in load_jsonl(path)}


def append(path: str, row: dict) -> None:
    with _lock:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def chunks_by_id() -> dict[str, dict]:
    return {c["id"]: c for c in load_jsonl(os.path.join(ROOT, "data", "ncert", "chunks.jsonl"))}


def fmt_options(q: dict) -> str:
    return "\n".join(f"({o['key']}) {o['text_en']}" for o in q["options"])


# ---------- explain ----------

def explain_one(q: dict, link: dict | None, chunks: dict) -> dict:
    passage = chunks.get((link or {}).get("chunk_id") or "")
    if passage:
        passage_block = f"\nRELEVANT NCERT PASSAGE (Class {passage['class']}, \"{passage['title']}\"):\n{passage['text']}\n"
    elif link and link.get("title"):
        passage_block = f"\nRelevant NCERT chapter: Class {link.get('class')}, \"{link['title']}\" (no single passage).\n"
    else:
        passage_block = ""
    figs = [f for f in q.get("figures", []) if os.path.exists(os.path.join(ROOT, f["path"]))]
    images = []
    figure_block = ""
    if figs:
        lines = [f"- figure for the {f['belongs_to'].replace('_', ' ')}: {f['description_en']}" for f in figs]
        if IS_GEMINI:
            from PIL import Image
            images = [Image.open(os.path.join(ROOT, f["path"])) for f in figs]
            figure_block = ("\nFIGURES (attached as images, in this order):\n" + "\n".join(lines) +
                            "\nRead the figures carefully. Base the explanation on what they actually show; never assume a 'standard' setup.\n")
        else:
            figure_block = "\nFIGURE DESCRIPTIONS (images not available):\n" + "\n".join(lines) + "\n"
    base = dict(subject=q["subject"], passage_block=passage_block, question=q["text_en"], options=fmt_options(q), figure_block=figure_block)
    key = q["answer"]
    key_set = set(key) if isinstance(key, list) else ({key} if isinstance(key, int) else set())

    exp, meta = call(SOLVE_PROMPT.format(**base, answer_block=""), Explanation, images=images)
    unaided = exp.chosen_option
    answer_given = False
    metas = [meta]
    if key_set and unaided not in key_set:
        ans = sorted(key_set)[0]
        block = f"\nThe official NTA answer is option ({ans}). Explain why it is correct.\n"
        exp, meta2 = call(SOLVE_PROMPT.format(**base, answer_block=block), Explanation, images=images)
        answer_given = True
        metas.append(meta2)
    return {
        "id": q["id"],
        "year": q["year"],
        "subject": q["subject"],
        "qno": q["qno"],
        "official_answer": key,
        "glm_unaided_answer": unaided,
        "glm_unaided_correct": (unaided in key_set) if key_set else None,
        "answer_given": answer_given,
        "final_answer_matches_key": (exp.chosen_option in key_set) if key_set else None,
        "has_figure": q.get("has_figure", False),
        "figures_seen": "image" if images else ("description" if figure_block else None),
        "ncert_chunk_id": passage["id"] if passage else None,
        "ncert_level": (link or {}).get("level"),
        **exp.model_dump(),
        "gen": {"model": GEN_MODEL, "reasoning": REASONING, "calls": metas},
    }


def run_explain(args) -> None:
    paper = args.paper
    year = int(paper.split("_")[0])
    if year in HOLDOUT_YEARS and not args.allow_holdout:
        sys.exit(f"{year} is the held-out eval year; pass --allow-holdout to generate reference explanations anyway")
    qs = load_jsonl(os.path.join(ROOT, "data", "questions", f"{paper}.jsonl"))
    if not qs:
        sys.exit(f"no questions for {paper}")
    links = {l["id"]: l for l in load_jsonl(os.path.join(ROOT, "data", "links", f"{paper}.jsonl"))}
    if not links:
        print(f"note: no NCERT links for {paper}; explanations will be ungrounded (run link_ncert.py first)")
    chunks = chunks_by_id()
    out_dir = os.path.join(ROOT, "data", "synthetic", "explain")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"{paper}.jsonl")
    skip = set() if args.force else done_ids(out)
    todo = [q for q in qs if q["id"] not in skip and len(q["options"]) == 4 and q["text_en"].strip()]
    if args.limit:
        todo = todo[: args.limit]
    print(f"{paper}: {len(todo)} to explain with {GEN_MODEL} (reasoning={REASONING}), {len(skip)} already done")
    t0 = time.time()
    stats = {"ok": 0, "unaided_correct": 0, "answer_given": 0, "failed": 0}
    with ThreadPoolExecutor(args.workers) as ex:
        futs = {ex.submit(explain_one, q, links.get(q["id"]), chunks): q for q in todo}
        for f in as_completed(futs):
            q = futs[f]
            try:
                row = f.result()
            except Exception as e:  # noqa: BLE001
                stats["failed"] += 1
                print(f"  {q['id']}: FAILED {e}")
                continue
            append(out, row)
            stats["ok"] += 1
            stats["unaided_correct"] += bool(row["glm_unaided_correct"])
            stats["answer_given"] += row["answer_given"]
            if stats["ok"] % 20 == 0:
                print(f"  {stats['ok']}/{len(todo)}  unaided correct {stats['unaided_correct']}  ${_totals['cost_usd']:.3f}")
    print(f"done: {stats}  tokens in {_totals['input_tokens']:,} out {_totals['output_tokens']:,}  ${_totals['cost_usd']:.3f}")
    runlog.log("gen_explain", paper=paper, model=GEN_MODEL, reasoning=REASONING, items=stats["ok"], failed=stats["failed"],
               unaided_correct=stats["unaided_correct"], answer_given=stats["answer_given"],
               input_tokens=_totals["input_tokens"], output_tokens=_totals["output_tokens"],
               reasoning_tokens=_totals["reasoning_tokens"], openrouter_cost_usd=round(_totals["cost_usd"], 4),
               seconds=round(time.time() - t0, 1))


# ---------- mcq ----------

def mcq_one(c: dict, n: int) -> list[dict]:
    prompt = MCQ_PROMPT.format(subject=c["subject"], klass=c["class"], title=c["title"], passage=c["text"], n=n)
    res, meta = call(prompt, MCQSet, max_tokens=12000)
    rows = []
    for i, m in enumerate(res.questions):
        ok = len(m.options_en) == 4 and len(m.options_hi) == 4 and 1 <= m.answer <= 4 and m.question_en.strip()
        d = m.model_dump()
        # Models put the answer first far too often. Shuffle option order (same
        # permutation for Hindi), except assertion-reason, whose four options are
        # a fixed convention that students expect in that order.
        if ok and m.style != "assertion_reason":
            perm = list(range(4))
            random.Random(f"{c['id']}-q{i}").shuffle(perm)
            d["options_en"] = [m.options_en[k] for k in perm]
            d["options_hi"] = [m.options_hi[k] for k in perm]
            d["answer"] = perm.index(m.answer - 1) + 1
            d["answer_as_generated"] = m.answer
        rows.append({
            "id": f"{c['id']}-q{i}",
            "source_chunk": c["id"],
            "chapter_code": c["chapter_code"],
            "subject": c["subject"],
            "class": c["class"],
            "title": c["title"],
            "valid_shape": bool(ok),
            **d,
            "gen": {"model": GEN_MODEL, "reasoning": REASONING, **(meta if i == 0 else {"shared_call": True})},
        })
    return rows


def run_mcq(args) -> None:
    chunks = list(chunks_by_id().values())
    if args.chapter:
        chunks = [c for c in chunks if c["chapter_code"] in args.chapter.split(",")]
    chunks = [c for c in chunks if len(c["text"].split()) >= 80]  # skip headers, captions, exercises stubs
    out_dir = os.path.join(ROOT, "data", "synthetic", "mcq")
    os.makedirs(out_dir, exist_ok=True)
    done_chunks = set()
    if not args.force:
        for p in os.listdir(out_dir):
            done_chunks |= {r["source_chunk"] for r in load_jsonl(os.path.join(out_dir, p))}
    todo = [c for c in chunks if c["id"] not in done_chunks]
    if args.limit:
        todo = todo[: args.limit]
    print(f"mcq: {len(todo)} passages x {args.per_passage} with {GEN_MODEL} (reasoning={REASONING}), {len(done_chunks)} passages already done")
    t0 = time.time()
    n_q = n_bad = n_fail = 0
    with ThreadPoolExecutor(args.workers) as ex:
        futs = {ex.submit(mcq_one, c, args.per_passage): c for c in todo}
        for k, f in enumerate(as_completed(futs), 1):
            c = futs[f]
            try:
                rows = f.result()
            except Exception as e:  # noqa: BLE001
                n_fail += 1
                print(f"  {c['id']}: FAILED {e}")
                continue
            for row in rows:
                append(os.path.join(out_dir, f"{c['chapter_code']}.jsonl"), row)
            n_q += len(rows)
            n_bad += sum(not r["valid_shape"] for r in rows)
            if k % 20 == 0:
                print(f"  {k}/{len(todo)} passages, {n_q} questions  ${_totals['cost_usd']:.3f}")
    print(f"done: {n_q} questions ({n_bad} bad shape), {n_fail} passages failed  tokens in {_totals['input_tokens']:,} out {_totals['output_tokens']:,}  ${_totals['cost_usd']:.3f}")
    runlog.log("gen_mcq", model=GEN_MODEL, reasoning=REASONING, passages=len(todo) - n_fail, failed=n_fail, questions=n_q, bad_shape=n_bad,
               input_tokens=_totals["input_tokens"], output_tokens=_totals["output_tokens"],
               reasoning_tokens=_totals["reasoning_tokens"], openrouter_cost_usd=round(_totals["cost_usd"], 4),
               seconds=round(time.time() - t0, 1))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)
    e = sub.add_parser("explain")
    e.add_argument("--paper", required=True, help="e.g. 2024_en_T1 or 2024_merged")
    e.add_argument("--allow-holdout", action="store_true")
    m = sub.add_parser("mcq")
    m.add_argument("--chapter", help="comma-separated chapter codes, e.g. kebo101,leph101")
    m.add_argument("--per-passage", type=int, default=4)
    for p in (e, m):
        p.add_argument("--limit", type=int)
        p.add_argument("--workers", type=int, default=8)
        p.add_argument("--force", action="store_true")
    args = ap.parse_args()
    run_explain(args) if args.mode == "explain" else run_mcq(args)


if __name__ == "__main__":
    main()
