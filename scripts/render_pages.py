"""Render selected NEET paper PDFs to page PNGs for vision extraction.

Usage: python scripts/render_pages.py [--dpi 130]

Reads data/papers.yaml for the canonical paper per (year, language), writes
data/pages/<year>/<lang>/<code>/p<NNN>.png and a manifest JSON per paper.
Outputs are gitignored; the manifest records source file, booklet code and
page count so extraction is reproducible.
"""
import argparse
import hashlib
import json
import os

import fitz  # PyMuPDF
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAPERS = os.path.join(ROOT, "data", "papers.yaml")
OUT = os.path.join(ROOT, "data", "pages")


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def render(entry: dict, dpi: int) -> None:
    src = os.path.join(ROOT, entry["file"])
    dest = os.path.join(OUT, str(entry["year"]), entry["lang"], entry["code"])
    os.makedirs(dest, exist_ok=True)
    doc = fitz.open(src)
    pages = []
    for i, page in enumerate(doc, start=1):
        out = os.path.join(dest, f"p{i:03d}.png")
        if not os.path.exists(out):
            page.get_pixmap(dpi=dpi).save(out)
        pages.append(os.path.relpath(out, ROOT).replace(os.sep, "/"))
    manifest = {
        "year": entry["year"],
        "lang": entry["lang"],
        "code": entry["code"],
        "source_file": entry["file"],
        "source_sha256": sha256(src),
        "dpi": dpi,
        "page_count": len(pages),
        "pages": pages,
    }
    with open(os.path.join(dest, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"{entry['year']} {entry['lang']:5} {entry['code']:>3}: {len(pages)} pages -> {os.path.relpath(dest, ROOT)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dpi", type=int, default=130)
    args = ap.parse_args()
    with open(PAPERS, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for entry in cfg["papers"]:
        render(entry, args.dpi)


if __name__ == "__main__":
    main()
