"""PaddleOCR: fills in text for pages with no usable text layer (scanned or
photographed documents, and images, which never have a PyMuPDF text layer).

Loading the models takes a few seconds - lazy singleton, built once per
process on first use, not per page.
"""

import os
import tempfile
import threading

OCR_LANG = os.environ.get("IDP_OCR_LANG", "en")

_engine = None
_lock = threading.Lock()


def _get_engine():
    global _engine
    with _lock:
        if _engine is None:
            from paddleocr import PaddleOCR
            # ponytail: orientation/unwarping detection off - our documents
            # come from doc.load_pages already right-side-up. Turn on if
            # phone-photo uploads start arriving sideways.
            _engine = PaddleOCR(
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
                lang=OCR_LANG,
            )
        return _engine


def text_from_png(png_bytes) -> str:
    """OCR one page image, return recognized lines joined in reading order.

    Empty string (never an exception) when nothing is recognized - callers
    already treat empty text as "no text available for this page".
    """
    with tempfile.NamedTemporaryFile(suffix=".png") as f:
        f.write(png_bytes)
        f.flush()
        results = _get_engine().predict(f.name)
    lines = []
    for r in results:
        lines.extend(r.get("rec_texts") or [])
    return "\n".join(lines)
