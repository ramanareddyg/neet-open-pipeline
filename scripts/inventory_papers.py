"""Inventory the raw NEET paper PDFs: pages, text yield, Hindi presence, booklet code."""
import glob
import os
import re
import sys

import pypdf

BASE = os.path.join("data", "raw", sys.argv[1] if len(sys.argv) > 1 else "papers")
CODE_RE = re.compile(r"(?:Test Booklet Code|Booklet Code|Code)\s*[:\-]?\s*([A-Z]\d|\d{2})")

for f in sorted(glob.glob(os.path.join(BASE, "**", "*.pdf"), recursive=True)):
    name = os.path.relpath(f, BASE).replace(os.sep, "/")
    kb = os.path.getsize(f) // 1024
    try:
        r = pypdf.PdfReader(f)
        n = len(r.pages)
        if n == 0:
            print(f"{name[:72]:72} EMPTY {kb}KB")
            continue
        t = "".join((r.pages[i].extract_text() or "") for i in range(min(3, n)))
        words = len(t.split())
        hindi = bool(re.search(r"[ऀ-ॿ]", t))
        m = CODE_RE.search(t)
        code = m.group(1) if m else "-"
        print(f"{name[:72]:72} p={n:>3} w3={words:>5} hi={int(hindi)} code={code:>3} {kb}KB")
    except Exception as e:  # noqa: BLE001
        print(f"{name[:72]:72} ERR {str(e)[:50]}")
