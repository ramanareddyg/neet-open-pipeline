"""Extract NEET questions from rendered page images with Gemini (vision).

Usage:
  python scripts/extract_pages.py --year 2025 --lang en [--pages 3-5] [--force]

For every page PNG under data/pages/<year>/<lang>/<code>/ this writes
data/extracted/<year>/<lang>/<code>/p<NNN>.json holding the questions that
start on that page, in the model's structured output. Pages already
extracted are skipped unless --force. Assembly into per-paper question
files happens in assemble_questions.py; this stage only transcribes.

Design notes
- One page per call. NEET pages carry 4-6 questions in two columns; a page
  is small enough for reliable transcription and large enough to amortise
  the prompt.
- Bilingual booklets print English in the left column and Hindi in the
  right. The model returns both for each question.
- Questions can run over a page break. The model flags `incomplete` on the
  first page and `continuation` on the next; assembly stitches them.
- Figures are not transcribed; the model returns a bounding box
  (0-1000 normalised, [ymin, xmin, ymax, xmax]) so assembly can crop them.
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Literal, Optional

import yaml
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import runlog  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# override=True: the repo's .env must win over any stale key in the process environment
load_dotenv(os.path.join(ROOT, ".env"), override=True)

MODEL = os.environ.get("GEMINI_MODEL", "gemini-flash-latest")


class Option(BaseModel):
    key: Literal["1", "2", "3", "4"]
    text_en: str = Field(description="Option text in English, LaTeX for math")
    text_hi: Optional[str] = Field(default=None, description="Option text in Hindi if printed")


class Figure(BaseModel):
    box_2d: list[int] = Field(description="[ymin, xmin, ymax, xmax] on a 0-1000 scale of the full page")
    belongs_to: Literal["question", "option_1", "option_2", "option_3", "option_4"]
    description_en: str = Field(description="One-sentence description of what the figure shows")


class Question(BaseModel):
    qno: int = Field(description="Printed question number")
    section_header: Optional[str] = Field(
        default=None,
        description="Section header printed just before this question on this page, e.g. 'PHYSICS SECTION A', else null",
    )
    text_en: str = Field(description="Full English question stem. Use LaTeX ($...$) for math and chemistry. Keep tables as Markdown.")
    text_hi: Optional[str] = Field(default=None, description="Full Hindi stem if printed, else null")
    options: list[Option] = Field(description="Exactly four options unless the question is cut off by the page end")
    figures: list[Figure] = Field(default_factory=list)
    incomplete: bool = Field(default=False, description="True if the question continues on the next page")
    continuation: bool = Field(default=False, description="True if this entry continues a question that started on the previous page")
    notes: Optional[str] = Field(default=None, description="Anything the transcription could not capture faithfully")


class Page(BaseModel):
    booklet_code: Optional[str] = Field(default=None, description="Booklet code from header/footer, e.g. 'T1' or '45'")
    printed_page_number: Optional[int] = None
    languages: list[Literal["en", "hi"]]
    questions: list[Question]


PROMPT = """You are transcribing one page of an official NEET-UG question paper into structured data.

Rules:
- Transcribe every question that starts on this page, in printed order. Also include a question that started on the previous page and finishes here, with continuation=true and only the parts printed on this page.
- Copy text exactly. Do not fix, solve, or shorten anything. Keep units, symbols and numbering as printed.
- Write mathematics and chemical formulae in LaTeX inside $...$. Write tables as Markdown tables. For match-the-following / list-based questions, keep both lists in text_en as Markdown.
- If the page is bilingual (English left, Hindi right), fill text_hi and each option's text_hi with the Hindi as printed. Otherwise leave them null.
- Do not describe figures in the stem. For every diagram, graph, structure drawing or circuit, add a Figure with a tight bounding box on a 0-1000 scale of the whole page image and say which question or option it belongs to. If an option itself is a drawing (e.g. a chemical structure), set its text_en to "[figure]" and add a Figure with belongs_to=option_N.
- Record a section header (e.g. "PHYSICS SECTION B") only on the question immediately following it.
- Read the booklet code from the page header/footer if printed (e.g. "45_English", "T1_English", "S6_Hindi+English").
- If a question is cut off by the page end, set incomplete=true and include whatever is printed.
"""


def parse_pages(spec: str, n: int) -> list[int]:
    if not spec:
        return list(range(1, n + 1))
    out = []
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return [p for p in out if 1 <= p <= n]


class Recitation(Exception):
    """The model refused with finish_reason RECITATION (verbatim-text filter)."""


# Models tried in turn on a single-question crop that the primary model refuses.
# The recitation filter fires per model and per snippet, so rotation recovers most stubs.
# Bump when the crop rung changes so --redo-boxes revisits pages done under the old logic.
REDO_VERSION = 2
BILINGUAL = False  # set from --lang in main(); bilingual pages print the two languages side by side
FALLBACK_MODELS = [m for m in os.environ.get("GEMINI_FALLBACK_MODELS", "gemini-3.1-flash-lite,gemini-2.5-pro").split(",") if m]


def call_model(client: genai.Client, img: bytes, model: str | None = None) -> tuple[Page, dict]:
    resp = client.models.generate_content(
        model=model or MODEL,
        contents=[types.Part.from_bytes(data=img, mime_type="image/png"), PROMPT],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=Page,
            temperature=0,
            max_output_tokens=16384,
        ),
    )
    fin = str(resp.candidates[0].finish_reason) if resp.candidates else "no candidates"
    # The filter can stop generation midway and still return partial text; treat
    # any RECITATION finish as a refusal so the fallback ladder runs.
    if "RECITATION" in fin:
        raise Recitation(fin)
    if resp.text is None:
        raise RuntimeError(f"empty response: {fin}")
    usage = resp.usage_metadata
    try:
        page = Page.model_validate_json(resp.text)
    except Exception as e:  # noqa: BLE001
        # Keep the raw output for diagnosis; validation failures are rare and worth seeing.
        dbg = os.path.join(ROOT, "data", "extract_debug")
        os.makedirs(dbg, exist_ok=True)
        with open(os.path.join(dbg, f"invalid_{int(time.time()*1000)}.json"), "w", encoding="utf-8") as f:
            f.write(resp.text)
        raise RuntimeError(f"schema validation: {str(e)[:200]}") from e
    return page, {
        "input_tokens": getattr(usage, "prompt_token_count", None),
        "output_tokens": (getattr(usage, "candidates_token_count", None) or 0) + (getattr(usage, "thoughts_token_count", None) or 0),  # thinking is billed as output
    }


def split_image(img: bytes, parts: int) -> list[bytes]:
    """Split a page image into `parts` horizontal bands with a small overlap.

    Horizontal bands keep the English and Hindi columns together on bilingual
    pages; the overlap means a question cut by the band edge is seen whole in
    one of the two bands (the model flags the fragment as incomplete/continuation).
    """
    from io import BytesIO

    from PIL import Image

    im = Image.open(BytesIO(img))
    out = []
    for rect in band_rects(im.size, parts):
        buf = BytesIO()
        im.crop(rect).save(buf, "PNG")
        out.append(buf.getvalue())
    return out


def band_rects(size: tuple[int, int], parts: int) -> list[tuple[int, int, int, int]]:
    """Pixel rectangles (left, top, right, bottom) of the bands split_image cuts."""
    w, h = size
    band = h // parts
    overlap = h // 25
    return [(0, max(0, i * band - overlap), w, min(h, (i + 1) * band + overlap)) for i in range(parts)]


def image_size(img: bytes) -> tuple[int, int]:
    from io import BytesIO

    from PIL import Image

    return Image.open(BytesIO(img)).size


def remap_figures(page: "Page", rect: tuple[int, int, int, int], size: tuple[int, int]) -> "Page":
    """Convert figure boxes from a sub-image's 0-1000 frame to the full page's.

    The model reports box_2d relative to the image it was shown. Band and crop
    rungs show it part of the page, so without this the boxes point at the
    wrong region when assembly cuts figures from the full page PNG.
    """
    left, top, right, bottom = rect
    w, h = size
    for q in page.questions:
        for fg in q.figures:
            if len(fg.box_2d) != 4:
                continue
            y0, x0, y1, x1 = fg.box_2d
            fg.box_2d = [
                round((top + y0 / 1000 * (bottom - top)) / h * 1000),
                round((left + x0 / 1000 * (right - left)) / w * 1000),
                round((top + y1 / 1000 * (bottom - top)) / h * 1000),
                round((left + x1 / 1000 * (right - left)) / w * 1000),
            ]
    return page


def call_bands(client, img: bytes, parts: int) -> list[tuple["Page", dict]]:
    """Transcribe each band; figure boxes come back in full-page coordinates."""
    size = image_size(img)
    out = []
    for band, rect in zip(split_image(img, parts), band_rects(size, parts)):
        page, usage = call_model(client, band)
        out.append((remap_figures(page, rect, size), usage))
    return out


def merge_pages(pages: list[Page]) -> Page:
    """Merge band results: later bands override an incomplete copy of the same qno."""
    by_qno: dict[int, Question] = {}
    for p in pages:
        for q in p.questions:
            prev = by_qno.get(q.qno)
            if prev is None or (prev.incomplete and not q.incomplete) or (q.continuation is False and prev.continuation):
                by_qno[q.qno] = q
    first = pages[0]
    return Page(
        booklet_code=next((p.booklet_code for p in pages if p.booklet_code), None),
        printed_page_number=next((p.printed_page_number for p in pages if p.printed_page_number), None),
        languages=sorted({l for p in pages for l in p.languages}) or first.languages,
        questions=[by_qno[k] for k in sorted(by_qno)],
    )


class Box(BaseModel):
    qno: int
    box_2d: list[int] = Field(description="[ymin, xmin, ymax, xmax] on a 0-1000 scale of the image")


class Layout(BaseModel):
    boxes: list[Box]


LAYOUT_PROMPT = (
    "Locate every numbered question on this exam page. For each, return its printed number and a bounding box "
    "[ymin, xmin, ymax, xmax] on a 0-1000 scale covering the question text, any figure, and all of its options "
    "(in a bilingual page include both language columns). Do not transcribe any text."
)


def layout_boxes(client: genai.Client, img: bytes) -> list[Box]:
    resp = client.models.generate_content(
        model=MODEL,
        contents=[types.Part.from_bytes(data=img, mime_type="image/png"), LAYOUT_PROMPT],
        config=types.GenerateContentConfig(response_mime_type="application/json", response_schema=Layout, temperature=0),
    )
    if not resp.text:
        return []
    # Drop malformed boxes (wrong arity or out of range) rather than crash on them.
    return [b for b in Layout.model_validate_json(resp.text).boxes
            if len(b.box_2d) == 4 and all(0 <= v <= 1000 for v in b.box_2d) and b.box_2d[2] > b.box_2d[0]]


def crop_rect(size: tuple[int, int], box: list[int], pad_frac: float = 0.012) -> tuple[int, int, int, int]:
    """Pixel rectangle that crop() cuts for a 0-1000 box (with padding)."""
    w, h = size
    y0, x0, y1, x1 = box
    px, py = int(w * pad_frac), int(h * pad_frac)
    return (max(0, x0 * w // 1000 - px), max(0, y0 * h // 1000 - py),
            min(w, x1 * w // 1000 + px), min(h, y1 * h // 1000 + py))


def crop(img: bytes, box: list[int], pad_frac: float = 0.012) -> bytes:
    from io import BytesIO

    from PIL import Image

    im = Image.open(BytesIO(img))
    buf = BytesIO()
    im.crop(crop_rect(im.size, box, pad_frac)).save(buf, "PNG")
    return buf.getvalue()


def sum_usage(results: list[tuple[Page, dict]]) -> dict:
    return {
        "input_tokens": sum(r[1]["input_tokens"] or 0 for r in results),
        "output_tokens": sum(r[1]["output_tokens"] or 0 for r in results),
    }


def extract_page(client: genai.Client, png_path: str) -> dict:
    """Transcribe one page, descending a fallback ladder when the recitation filter fires.

    1. whole page  2. 2 bands  3. 3 bands  4. per-question crops from a layout pass.
    At rung 4 a crop that still refuses is recorded as a stub (`recitation_failed`)
    so the run continues and the gap is visible in assembly.
    """
    with open(png_path, "rb") as f:
        img = f.read()
    last_err = None
    for attempt in range(4):
        try:
            page, usage = call_model(client, img)
            return {"model": MODEL, "usage": usage, **page.model_dump()}
        except Recitation:
            break
        except RuntimeError as e:
            # Truncated/invalid JSON on a whole page behaves like a refusal in
            # practice (the output is cut mid-stream); go down the ladder too.
            last_err = e
            if "schema validation" in str(e) and attempt >= 1:
                break
            time.sleep(2 * (attempt + 1))
        except Exception as e:  # noqa: BLE001
            last_err = e
            time.sleep(2 * (attempt + 1))
    else:
        raise RuntimeError(f"failed after retries: {last_err}")

    for parts in (2, 3):
        try:
            results = call_bands(client, img, parts)
            page = merge_pages([r[0] for r in results])
            return {"model": MODEL, "usage": sum_usage(results), "split_into": parts, **page.model_dump()}
        except (Recitation, RuntimeError):
            continue  # refused or truncated on a band: try finer rungs

    boxes = layout_boxes(client, img)
    if not boxes:
        raise RuntimeError("recitation on every band and layout pass returned no boxes")
    results, failed, by_model = transcribe_boxes(client, img, boxes)
    page = merge_pages([r[0] for r in results])
    out = {"model": MODEL, "usage": sum_usage(results), "split_into": f"boxes:{len(boxes)}", **page.model_dump()}
    if by_model:
        out["fallback_models"] = by_model
    out["boxes"] = [b.model_dump() for b in boxes]
    if failed:
        out["recitation_failed"] = failed
    return out


def transcribe_boxes(client: genai.Client, img: bytes, boxes: list[Box]):
    """Transcribe each question crop, rotating models when one refuses."""
    results: list[tuple[Page, dict]] = []
    failed: list[int] = []
    by_model: dict[str, list[int]] = {}
    # Reading order: left column top-to-bottom, then right column.
    boxes = sorted(boxes, key=lambda b: (b.box_2d[1] >= 500, b.box_2d[0]))
    # Layout boxes tend to stop at the stem and clip the options, which sit
    # below in the same column. Extend each crop to the full column width and
    # down to the next box in that column (or the page bottom).
    expanded: dict[int, list[int]] = {}
    for idx, b in enumerate(boxes):
        y0, x0, y1, x1 = b.box_2d
        right_col = x0 >= 500
        col_x0, col_x1 = (500, 1000) if right_col else (0, 500)
        if x1 - x0 > 600:  # a box spanning both columns: keep its own width
            col_x0, col_x1 = x0, x1
        if BILINGUAL:
            # English left, Hindi right: the two columns are one question. Full width.
            col_x0, col_x1 = 0, 1000
            right_col = False
        nxt = [nb.box_2d[0] for nb in boxes[idx + 1:] if (nb.box_2d[1] >= 500) == right_col and nb.box_2d[0] > y0]
        bottom = max(y1, (min(nxt) - 5) if nxt else 985)
        expanded[idx] = [y0, min(x0, col_x0 + 10), min(bottom, 990), max(x1, col_x1 - 10)]
    for idx, b in enumerate(boxes):
        c = crop(img, expanded[idx])
        done = False
        for model in [MODEL] + FALLBACK_MODELS:
            try:
                page, usage = call_model(client, c, model)
                remap_figures(page, crop_rect(image_size(img), expanded[idx]), image_size(img))
                # A crop often clips the printed number; the layout pass saw the whole page.
                if len(page.questions) == 1 and page.questions[0].qno != b.qno:
                    page.questions[0].notes = f"qno_from_layout (crop read {page.questions[0].qno})" + (f"; {page.questions[0].notes}" if page.questions[0].notes else "")
                    page.questions[0].qno = b.qno
                # A numbered box starts a question; it is never a continuation of the previous page.
                for q in page.questions:
                    if q.qno == b.qno:
                        q.continuation = False
                        if len(q.options) == 4:
                            q.incomplete = False
                for q in page.questions:
                    if model != MODEL:
                        q.notes = f"transcribed_by:{model}" + (f"; {q.notes}" if q.notes else "")
                results.append((page, usage))
                if model != MODEL:
                    by_model.setdefault(model, []).append(b.qno)
                done = True
                break
            except (Recitation, RuntimeError):
                continue
            except Exception:  # noqa: BLE001
                continue
        if not done:
            failed.append(b.qno)
            results.append((Page(languages=[], questions=[Question(
                qno=b.qno, text_en="", options=[], notes="recitation_failed: transcribe from crop", incomplete=False)]), {"input_tokens": 0, "output_tokens": 0}))
    return results, failed, by_model


def fix_qnos(client: genai.Client, out_dir: str, manifest: dict, qnos: list[int]) -> None:
    """Re-transcribe specific questions by number, patching the page JSON they belong to.

    For each qno: locate the page (the page holding it, else the page whose
    neighbours bracket it), run a fresh layout pass, take the box with that
    number, and transcribe the column-extended crop with model rotation. If the
    layout pass does not list the number, fall back to the 2-band transcription
    of the page and take the question from there.
    """
    import glob

    pages = []
    for path in sorted(glob.glob(os.path.join(out_dir, "p*.json"))):
        with open(path, encoding="utf-8") as f:
            pages.append((path, json.load(f)))

    def page_for(qno: int):
        for path, d in pages:
            if any(q["qno"] == qno for q in d["questions"]):
                return path, d
        # bracket by neighbours
        for idx, (path, d) in enumerate(pages):
            nums = [q["qno"] for q in d["questions"] if q["qno"] > 0]
            if not nums:
                continue
            nxt = [q["qno"] for _, dd in pages[idx + 1:idx + 2] for q in dd["questions"] if q["qno"] > 0]
            if min(nums) <= qno <= (min(nxt) if nxt else max(nums) + 1):
                return path, d
        return None, None

    for qno in qnos:
        path, d = page_for(qno)
        if d is None:
            print(f"q{qno}: no page found", flush=True)
            continue
        p = d["page"]
        with open(os.path.join(ROOT, manifest["pages"][p - 1]), "rb") as f:
            img = f.read()
        boxes = layout_boxes(client, img)
        target = [b for b in boxes if b.qno == qno]
        got = None
        via = {}
        if target:
            results, failed, via = transcribe_boxes(client, img, boxes)  # column extension needs all boxes
            for page, _ in results:
                for q in page.questions:
                    # Only a complete transcription counts; partial ones fall through to the next rung.
                    if q.qno == qno and len(q.options) == 4 and (q.text_en or "").strip() and not (q.notes or "").startswith("recitation_failed"):
                        got = q
        if got is None:
            # Fall back: bands of the page, pick the question out.
            for parts in (2, 3):
                try:
                    merged = merge_pages([r[0] for r in call_bands(client, img, parts)])
                    cand = [q for q in merged.questions if q.qno == qno and len(q.options) == 4 and (q.text_en or "").strip()]
                    if cand:
                        got = cand[0]
                        via = {"bands": parts}
                        break
                except Recitation:
                    continue
        if got is None:
            # Last rung: whole page for context (questions can run across columns),
            # but ask for this one question only; small outputs tend to pass the filter.
            for model in [MODEL] + FALLBACK_MODELS:
                try:
                    resp = client.models.generate_content(
                        model=model,
                        contents=[types.Part.from_bytes(data=img, mime_type="image/png"),
                                  PROMPT + f"\n\nTranscribe ONLY question number {qno} from this page (it may continue from one column into the next). Return it as the single entry in `questions`."],
                        config=types.GenerateContentConfig(response_mime_type="application/json", response_schema=Page, temperature=0, max_output_tokens=8192),
                    )
                    fin = str(resp.candidates[0].finish_reason) if resp.candidates else ""
                    if "RECITATION" in fin or not resp.text:
                        continue
                    cand = [q for q in Page.model_validate_json(resp.text).questions if q.qno == qno and len(q.options) == 4 and (q.text_en or "").strip()]
                    if cand:
                        got = cand[0]
                        got.continuation = False
                        got.notes = f"transcribed_by:{model}; single_question_full_page" + (f"; {got.notes}" if got.notes else "")
                        via = {"full_page_single": model}
                        break
                except Exception:  # noqa: BLE001
                    continue
        if got is None:
            print(f"q{qno}: p{p:03d} could not be recovered", flush=True)
            continue
        got.continuation = False  # explicitly targeted by number: it is this question, not a tail
        if len(got.options) == 4:
            got.incomplete = False
        got.notes = f"fixed_by_qno" + (f"; {got.notes}" if got.notes else "")
        d["questions"] = [q for q in d["questions"] if q["qno"] != qno] + [got.model_dump()]
        d["questions"].sort(key=lambda q: q["qno"])
        with open(path, "w", encoding="utf-8") as f:
            json.dump(d, f, ensure_ascii=False, indent=1)
        print(f"q{qno}: p{p:03d} fixed, options={len(got.options)} via {via}", flush=True)


def retry_stubs(client: genai.Client, out_dir: str, manifest: dict, redo_all_boxes: bool = False) -> None:
    """Re-transcribe recitation stubs on extracted pages.

    With redo_all_boxes, every box-split page is re-transcribed crop by crop
    from its stored layout boxes (used after fixes to the crop rung).
    """
    import glob

    for path in sorted(glob.glob(os.path.join(out_dir, "p*.json"))):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        is_box_page = str(data.get("split_into", "")).startswith("boxes")
        stubs = data.get("recitation_failed") or []
        already = data.get("redo_version") == REDO_VERSION
        if not stubs and not (redo_all_boxes and is_box_page and not already):
            continue
        p = data["page"]
        with open(os.path.join(ROOT, manifest["pages"][p - 1]), "rb") as f:
            img = f.read()
        if redo_all_boxes and is_box_page:
            boxes = layout_boxes(client, img)  # stored boxes may carry misread numbers; redo the layout pass
        else:
            boxes = [Box(**b) for b in data.get("boxes", []) if b["qno"] in stubs] or [b for b in layout_boxes(client, img) if b.qno in stubs]
        results, failed, by_model = transcribe_boxes(client, img, boxes)
        page = merge_pages([r[0] for r in results])
        if redo_all_boxes and is_box_page:
            data["questions"] = [q.model_dump() for q in page.questions]
        else:
            fixed = {q.qno: q for q in page.questions if not (q.notes or "").startswith("recitation_failed")}
            data["questions"] = [fixed[q["qno"]].model_dump() if q["qno"] in fixed else q for q in data["questions"]]
        data["boxes"] = [b.model_dump() for b in boxes] if redo_all_boxes else data.get("boxes", [])
        data["recitation_failed"] = failed
        if not failed:
            data.pop("recitation_failed", None)
        data["fallback_models"] = by_model
        if redo_all_boxes:
            data["redo_version"] = REDO_VERSION
        u = sum_usage(results)
        data["usage"] = {k: (data["usage"].get(k) or 0) + u[k] for k in u}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=1)
        print(f"p{p:03d} boxes {[b.qno for b in boxes]} -> got {[q.qno for q in page.questions]} still_failed {failed} via {by_model}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--lang", choices=["en", "hi"], required=True)
    ap.add_argument("--pages", default="", help="e.g. 3-5,9")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--workers", type=int, default=4, help="parallel page requests")
    ap.add_argument("--retry-stubs", action="store_true", help="only re-transcribe recitation stubs on extracted pages")
    ap.add_argument("--redo-boxes", action="store_true", help="re-transcribe every box-split page crop by crop")
    ap.add_argument("--fix-qnos", default="", help="comma-separated question numbers to re-transcribe individually")
    args = ap.parse_args()

    with open(os.path.join(ROOT, "data", "papers.yaml"), encoding="utf-8") as f:
        papers = yaml.safe_load(f)["papers"]
    entry = next((p for p in papers if p["year"] == args.year and p["lang"] == args.lang), None)
    if not entry:
        sys.exit(f"no paper for {args.year} {args.lang} in data/papers.yaml")
    code = str(entry["code"])
    global BILINGUAL
    BILINGUAL = args.lang == "hi"
    pages_dir = os.path.join(ROOT, "data", "pages", str(args.year), args.lang, code)
    out_dir = os.path.join(ROOT, "data", "extracted", str(args.year), args.lang, code)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(pages_dir, "manifest.json"), encoding="utf-8") as f:
        manifest = json.load(f)

    client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
    if args.fix_qnos:
        fix_qnos(client, out_dir, manifest, [int(x) for x in args.fix_qnos.split(",") if x.strip()])
        return
    if args.retry_stubs or args.redo_boxes:
        retry_stubs(client, out_dir, manifest, redo_all_boxes=args.redo_boxes)
        return
    pages = parse_pages(args.pages, manifest["page_count"])
    tot_in = tot_out = 0
    n_pages = n_q = n_figs = 0
    t_start = time.time()
    todo = [p for p in pages if args.force or not os.path.exists(os.path.join(out_dir, f"p{p:03d}.json"))]

    def work(p: int) -> tuple[int, dict, float]:
        t0 = time.time()
        png = os.path.join(ROOT, manifest["pages"][p - 1])
        data = extract_page(client, png)
        data["page"] = p
        data["source_png"] = manifest["pages"][p - 1]
        with open(os.path.join(out_dir, f"p{p:03d}.json"), "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=1)
        return p, data, time.time() - t0

    def safe_work(p: int):
        try:
            return work(p)
        except Exception as e:  # noqa: BLE001
            print(f"p{p:03d} FAILED: {str(e)[:160]}", flush=True)
            return None

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = [r for r in pool.map(safe_work, todo) if r is not None]
    n_failed = len(todo) - len(results)
    for p, data, dt in results:
        u = data["usage"]
        tot_in += u["input_tokens"] or 0
        tot_out += u["output_tokens"] or 0
        qs = data["questions"]
        qnos = [q["qno"] for q in qs]
        flags = "".join("i" if q["incomplete"] else "c" if q["continuation"] else "." for q in qs)
        nf = sum(len(q["figures"]) for q in qs)
        n_pages += 1
        n_q += len(qs)
        n_figs += nf
        split = f" split={data['split_into']}" if data.get("split_into") else ""
        rf = f" RECITATION_FAILED={data['recitation_failed']}" if data.get("recitation_failed") else ""
        print(f"p{p:03d} code={data['booklet_code']} q={qnos} {flags} figs={nf}{split}{rf} {dt:.1f}s", flush=True)
    print(f"tokens in={tot_in} out={tot_out} pages_failed={n_failed}")
    if n_pages or n_failed:
        runlog.log(
            "extract_pages", year=args.year, lang=args.lang, code=code, model=MODEL,
            pages=n_pages, pages_failed=n_failed, questions_seen=n_q, figures=n_figs,
            input_tokens=tot_in, output_tokens=tot_out, seconds=round(time.time() - t_start, 1),
        )


if __name__ == "__main__":
    main()
