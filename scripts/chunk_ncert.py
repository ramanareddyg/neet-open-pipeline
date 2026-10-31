"""Chunk NCERT chapter PDFs into page-anchored passages.

Usage: python scripts/chunk_ncert.py

Reads  data/raw/ncert/<Subject>/Class <n>/Part <n>/<code>.pdf
Writes data/ncert/chapters.json   (one entry per chapter: code, subject, class, part, chapter no, title, pages)
       data/ncert/chunks.jsonl     (passages of ~120-350 words, each with chapter code, printed page, chunk index)

NCERT file codes: <k|l><e><subj><part><nn>.pdf
  k = Class 11, l = Class 12; e = English medium; subj in {ph, ch, bo};
  part = 1|2; nn = chapter number, or ps/an/ap/gl/in/pre for front/back matter (skipped).
Chunks keep the printed page number from the PDF text where it can be read,
so a link can cite "Class 12 Physics, Ch. 4, p. 109".
"""
import glob
import json
import os
import re

import fitz  # PyMuPDF: an order of magnitude faster than pypdf on these scans
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "data", "raw", "ncert")
OUT = os.path.join(ROOT, "data", "ncert")

CODE_RE = re.compile(r"^([kl])e(ph|ch|bo)([12])(\d\d)$")
SUBJ = {"ph": "Physics", "ch": "Chemistry", "bo": "Biology"}
MIN_WORDS, TARGET_WORDS, MAX_WORDS = 120, 250, 350


def clean(text: str) -> str:
    text = text.replace("­", "")
    text = re.sub(r"-\n(?=[a-z])", "", text)          # de-hyphenate line breaks
    text = re.sub(r"[ \t]*\n[ \t]*", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def printed_page(text: str) -> int | None:
    # NCERT pages start or end with the page number; take a bare integer on the first/last short line.
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    for l in lines[:2] + lines[-2:]:
        if re.fullmatch(r"\d{1,3}", l):
            return int(l)
    return None


def guess_title(pages: list[str], chapter_no: int) -> str:
    # The opening-page title is often drawn as a graphic, but NCERT repeats the
    # chapter title as an all-caps running header on later pages (sometimes
    # with the page number attached). Take the most frequent such header.
    counts: dict[str, int] = {}
    for text in pages[1:12]:
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        for l in lines[:3] + lines[-3:]:
            cand = re.sub(r"^\d+\s*|\s*\d+$", "", l).strip()
            if len(cand) >= 5 and cand.upper() == cand and re.search(r"[A-Z]{3}", cand):
                if cand in {"EXERCISES", "SUMMARY", "POINTS TO PONDER", "PHYSICS", "CHEMISTRY", "BIOLOGY"}:
                    continue
                counts[cand] = counts.get(cand, 0) + 1
    if counts:
        best = max(counts, key=counts.get)
        return re.sub(r"\s+", " ", best).title()
    return f"Chapter {chapter_no}"


def split_paragraphs(text: str) -> list[str]:
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks, buf, n = [], [], 0
    for p in paras:
        w = len(p.split())
        if n + w > MAX_WORDS and buf:
            chunks.append("\n\n".join(buf))
            buf, n = [], 0
        buf.append(p)
        n += w
        if n >= TARGET_WORDS:
            chunks.append("\n\n".join(buf))
            buf, n = [], 0
    if buf:
        if chunks and n < MIN_WORDS:
            chunks[-1] += "\n\n" + "\n\n".join(buf)
        else:
            chunks.append("\n\n".join(buf))
    return chunks


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "titles.yaml"), encoding="utf-8") as f:
        titles = yaml.safe_load(f)
    chapters, chunks = [], []
    for f in sorted(glob.glob(os.path.join(SRC, "**", "*.pdf"), recursive=True)):
        stem = os.path.splitext(os.path.basename(f))[0]
        m = CODE_RE.match(stem)
        if not m:
            continue  # prelims, answers, appendices, glossary, index
        cls = 11 if m.group(1) == "k" else 12
        subject, part, chap = SUBJ[m.group(2)], int(m.group(3)), int(m.group(4))
        with fitz.open(f) as doc:
            pages = [clean(p.get_text("text") or "") for p in doc]
        meta = titles.get(stem)
        if meta:
            chap, title = meta["chapter"], meta["title"]
        else:
            title = guess_title(pages, chap)
            print(f"  WARNING: {stem} not in titles.yaml; using heuristic '{title}'")
        chapters.append({
            "code": stem, "subject": subject, "class": cls, "part": part,
            "chapter": chap, "title": title, "pdf_pages": len(pages),
            "source": os.path.relpath(f, ROOT).replace(os.sep, "/"),
        })
        for pi, text in enumerate(pages, start=1):
            if len(text.split()) < 30:
                continue
            pp = printed_page(text)
            for ci, body in enumerate(split_paragraphs(text)):
                chunks.append({
                    "id": f"{stem}-p{pi:03d}-{ci}",
                    "chapter_code": stem, "subject": subject, "class": cls,
                    "chapter": chap, "title": title,
                    "pdf_page": pi, "printed_page": pp,
                    "text": body, "words": len(body.split()),
                })
        print(f"{stem}: Class {cls} {subject} ch{chap:02d} '{title}' {len(pages)}p", flush=True)
    with open(os.path.join(OUT, "chapters.json"), "w", encoding="utf-8") as fo:
        json.dump(chapters, fo, ensure_ascii=False, indent=1)
    with open(os.path.join(OUT, "chunks.jsonl"), "w", encoding="utf-8") as fo:
        for c in chunks:
            fo.write(json.dumps(c, ensure_ascii=False) + "\n")
    words = sum(c["words"] for c in chunks)
    print(f"\n{len(chapters)} chapters, {len(chunks)} chunks, {words:,} words")


if __name__ == "__main__":
    main()
