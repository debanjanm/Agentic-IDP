"""Turn any document (PDF or image) into per-page (PNG, text) pairs.

PyMuPDF opens PDF and PNG/JPG/TIFF through the same API, so there is one code
path for both. The text layer comes free for digital PDFs; when it's empty
(a scan, a photo, or any image - images never have a PDF text layer at all)
app/ocr.py fills it in instead, so every page carries real text into the
pipeline's hybrid text+image prompts, not just digital PDFs.
"""

import base64
from typing import NamedTuple

import fitz

from . import ocr

SUPPORTED_EXTS = {".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}


class Page(NamedTuple):
    number: int  # 1-based
    png_b64: str
    text: str
    source: str = "text-layer"  # or "ocr" - shown in the UI's "Read pages" stage


def load_pages(path, max_pages=10, dpi=150, max_edge=1400) -> list[Page]:
    """Render every page, clamped so the longest edge <= max_edge px.

    A document over max_pages is refused, not truncated: extracting from the
    first 10 pages of 14 and calling it done loses data without a trace.
    dpi and max_edge are the accuracy/latency knobs: higher means the model
    reads small print better and takes proportionally longer.
    """
    pages = []
    with fitz.open(path) as doc:
        if doc.page_count > max_pages:
            raise ValueError(f"document has {doc.page_count} pages; the limit is {max_pages}. "
                             "Split it into smaller files.")
        for i, page in enumerate(doc):
            png_bytes = _render(page, dpi, max_edge)
            text, source = page.get_text().strip(), "text-layer"
            if not text:
                text, source = ocr.text_from_png(png_bytes), "ocr"
            pages.append(Page(i + 1, base64.b64encode(png_bytes).decode(), text, source))
    if not pages:
        raise ValueError(f"no pages found in {path}")
    return pages


def render_png(path, page_number, dpi=150, max_edge=1400) -> bytes:
    """Single page as raw PNG bytes, for the UI preview endpoint."""
    with fitz.open(path) as doc:
        if not 1 <= page_number <= doc.page_count:
            raise ValueError(f"page {page_number} out of range (1..{doc.page_count})")
        return _render(doc[page_number - 1], dpi, max_edge)


def _render(page, dpi, max_edge) -> bytes:
    # A page rect is in points (1/72"). Drop the dpi until the render fits
    # max_edge, rather than rendering big and downscaling in Pillow.
    longest_pt = max(page.rect.width, page.rect.height)
    if longest_pt > 0:
        dpi = min(dpi, max_edge * 72 / longest_pt)
    return page.get_pixmap(dpi=round(dpi)).tobytes("png")
