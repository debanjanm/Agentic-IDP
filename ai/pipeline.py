"""classify -> extract -> verify -> bounded retry.

The verify pass is the agentic part: a second look that can overrule the first
answer. With a 3B model that is the main defense against confident wrong reads.
"""

import json
import re
import time

from . import doc, ground, llm, registry, validate

MAX_IMAGES_PER_CALL = 4  # a 3B VLM on 16GB loses the thread past ~4 pages
CLASSIFY_MIN_CONFIDENCE = 0.6
TEXT_CLASSIFY_MIN_CHARS = 200  # below this, trust the image over sparse/noisy text
FIELD_MIN_CONFIDENCE = 0.6
# A field with a warning sign is capped here, below FIELD_MIN_CONFIDENCE, so it always
# goes to review whatever confidence the verifier claimed for it.
SIGNAL_CAP = 0.4
VERIFY_RETRIES = 1
NULL_WORDS = ("", "null", "none", "n/a", "not present")  # what a typed value means "empty"

CLASSIFY_SCHEMA = {
    "type": "object",
    "properties": {
        "doc_type": {"type": "string"},
        "confidence": {"type": "number"},
        "reasoning": {"type": "string"},
        "language": {"type": ["string", "null"], "description": "ISO 639-1 code, e.g. en, de, fr"},
    },
    "required": ["doc_type", "confidence", "reasoning", "language"],
}

def _verify_schema(names) -> dict:
    """One required property per field being checked.

    An array of {name, ...} lets the model quietly return three entries for ten
    fields, and the missing ones then score as unknown. Making each field a
    required property means constrained decoding cannot omit any of them.
    """
    verdict = {
        "type": "object",
        "properties": {
            "ok": {"type": "boolean"},
            "confidence": {"type": "number"},
            "correction": {"type": ["string", "null"]},
        },
        "required": ["ok", "confidence", "correction"],
    }
    return {
        "type": "object",
        "properties": {n: dict(verdict) for n in names},
        "required": list(names),
    }


def run(path, doc_type=None, schema=None, max_pages=10, on_step=None) -> dict:
    """Full pipeline on a file path. Loads pages, then delegates to run_pages."""
    note = on_step or (lambda *_: None)
    note("reading")
    pages = doc.load_pages(path, max_pages=max_pages)
    return run_pages(pages, doc_type=doc_type, schema=schema, on_step=on_step)


def run_pages(pages, doc_type=None, schema=None, cls=None, on_step=None) -> dict:
    """Full pipeline on already-loaded pages (also the API response body).

    `cls` lets a caller that already classified these pages - segment.split,
    grouping a multi-document upload - skip a second classify() call.
    """
    trace = []
    note = on_step or (lambda *_: None)

    note("classifying")
    if cls is None:
        cls = ({"doc_type": doc_type, "confidence": 1.0, "reasoning": "supplied by caller", "language": None}
               if doc_type else classify(pages))
    trace.append({"step": "classify", **cls})

    if schema is None:
        try:
            schema = registry.load(cls["doc_type"])
        except KeyError:
            # No registered schema for this type - extract loose key/value
            # pairs instead of aborting. Every document produces *something*.
            schema = registry.load(registry.GENERIC)
            trace.append({"step": "fallback", "reason": f"no schema for {cls['doc_type']!r}, using generic"})

    note("extracting")
    t0 = time.time()
    extracted = extract(pages, schema)
    trace.append({"step": "extract", "seconds": round(time.time() - t0, 1), "fields": dict(extracted)})

    note("verifying")
    fields, confidence, vtrace = verify(pages, schema, extracted)
    trace.extend(vtrace)
    signals = _signals(pages, extracted, fields, confidence)
    for name in signals:
        confidence[name] = min(confidence[name], SIGNAL_CAP)
    if signals:
        trace.append({"step": "signals", "fields": signals})

    note("validating")
    checks = validate.check(schema, fields)
    trace.append({"step": "validate", **checks})

    low = [k for k, v in confidence.items() if v < FIELD_MIN_CONFIDENCE]
    return {
        "doc_type": cls["doc_type"],
        "confidence": cls["confidence"],
        "language": cls.get("language"),
        "fields": fields,
        "field_confidence": confidence,
        "needs_review": bool(low) or bool(checks["failed"]) or cls["confidence"] < CLASSIFY_MIN_CONFIDENCE,
        "low_confidence_fields": low,
        "confidence_signals": signals,
        "validation": checks,
        "pages": len(pages),
        "text_source": "+".join(sorted({p.source for p in pages})),
        "schema": schema,  # the caller (worker.py) needs this for entity resolution
        "trace": trace,
    }


def _signals(pages, extracted, fields, confidence) -> dict:
    """{field: [why it is doubtful]} from evidence other than the model's own claim.

    verifier_overruled: the extract pass and the verify pass disagreed - two looks
    at the same page gave different answers. not_on_page: the final value appears
    nowhere in the page text. A verifier's self-reported confidence collapses toward
    "sure" on small models, so these are what the confidence is capped by.
    """
    text = "\n".join(p.text for p in pages)
    out = {}
    for name in confidence:
        why = []
        if extracted.get(name) != fields.get(name):
            why.append("verifier_overruled")
        if ground.grounded(fields.get(name), text) is False:
            why.append("not_on_page")
        if why:
            out[name] = why
    return out


def classify(pages) -> dict:
    """Identify the document type from its first page.

    A page with plenty of real text (native PDF layer, or OCR - doc.py fills
    both in) is classified from text alone with the small text-only model:
    picking a label doesn't need pixels, so this skips a slow vision call.
    Sparse/empty text (a page the OCR pass couldn't read well) falls back to
    the vision model reading the image directly.
    """
    labels = registry.labels()
    page = pages[0]  # the first page is what tells you what a document is
    schema = dict(CLASSIFY_SCHEMA)
    schema["properties"] = dict(schema["properties"])
    schema["properties"]["doc_type"] = {"type": "string", "enum": labels}

    prompt = (
        "Identify what kind of document this is.\n"
        f"Choose exactly one: {', '.join(labels)}.\n"
        "Use 'unknown' if it matches none of them.\n"
        "confidence is 0.0-1.0. language is the document's main language, ISO 639-1.\n"
        + _text_block(page.text, _is_ocr([page]))
    )
    use_vision = len(page.text) < TEXT_CLASSIFY_MIN_CHARS
    model = llm.CLASSIFY_MODEL if use_vision else llm.TEXT_CLASSIFY_MODEL
    out = llm.chat(
        prompt,
        schema=schema,
        images=[page.png_b64] if use_vision else None,
        model=model,
        system="You classify documents. Answer only from what is visible/given.",
        num_ctx=_ctx(1 if use_vision else 0, page.text),
    )
    out["confidence"] = _clamp(out.get("confidence"))
    out["via"], out["model"] = ("vision" if use_vision else "text"), model  # shown in the UI
    return out


def extract(pages, schema) -> dict:
    """Extract against `schema`. Short documents go in one call, no merging."""
    chunks = [pages[i:i + MAX_IMAGES_PER_CALL] for i in range(0, len(pages), MAX_IMAGES_PER_CALL)]
    results = [_extract_chunk(c, schema, len(pages)) for c in chunks]
    merged = results[0]
    for r in results[1:]:
        merged = _merge(merged, r)
    return merged


def _extract_chunk(chunk, schema, total_pages) -> dict:
    text = "\n\n".join(f"--- page {p.number} ---\n{p.text}" for p in chunk if p.text)
    prompt = (
        f"Extract the requested fields from this document "
        f"(pages {chunk[0].number}-{chunk[-1].number} of {total_pages}).\n\n"
        f"{_field_docs(schema)}\n"
        "Rules:\n"
        "- Copy values exactly as they appear. Do not reformat except dates, which use YYYY-MM-DD.\n"
        "- Amounts are plain numbers: no currency symbols, no thousands separators.\n"
        "- If a field is not present in the document, return null. Never guess or invent a value.\n"
        + _text_block(text, _is_ocr(chunk))
    )
    return llm.chat(
        prompt,
        schema=schema,
        images=[p.png_b64 for p in chunk],
        model=llm.EXTRACT_MODEL,
        system="You extract structured data from documents. Accuracy matters more than completeness.",
        num_ctx=_ctx(len(chunk), text),
    )


def verify(pages, schema, fields):
    """Re-check scalar fields against the pages; apply corrections, re-check once.

    Arrays are skipped: row-by-row verification of line items costs far more
    than it buys at this model size.
    ponytail: scalars only, extend to arrays if line-item errors show up.
    """
    props = schema.get("properties", {})
    scalars = {k: v for k, v in fields.items() if not isinstance(v, (list, dict))}
    # Only verified fields get a score. Arrays are deliberately absent rather
    # than given a made-up 1.0 - nobody checked them, so don't claim otherwise.
    confidence = {}
    trace = []
    if not scalars:
        return fields, confidence, trace

    for attempt in range(VERIFY_RETRIES + 1):
        report = _verify_all(pages, scalars, schema)
        trace.append({"step": "verify", "attempt": attempt + 1, "report": report})

        fixes = {}
        for name, item in report.items():
            if name not in scalars or not isinstance(item, dict):
                continue
            confidence[name] = _clamp(item.get("confidence"))
            if item.get("ok"):
                continue
            fixed = _coerce(item["correction"], props.get(name, {})) \
                if item.get("correction") is not None else None
            # An adversarial verifier will happily say "wrong!" and then hand back
            # the value it was given. Only a change is a correction; restating the
            # current value is agreement, whatever the model called it.
            if fixed != scalars[name]:
                fixes[name] = fixed
            else:
                confidence[name] = max(confidence[name], _clamp(item.get("confidence")))

        if not fixes or attempt == VERIFY_RETRIES:
            if fixes:  # out of retries - take the corrections, flag them as shaky
                scalars.update(fixes)
                trace.append({"step": "verify", "note": "applied final corrections unchecked",
                              "fields": list(fixes)})
            break
        scalars.update(fixes)
        trace.append({"step": "correct", "applied": fixes})

    fields = {**fields, **scalars}
    for k in scalars:  # a scalar the verifier ignored is a coin flip
        confidence.setdefault(k, 0.5)
    return fields, confidence, trace


def _verify_all(pages, scalars, schema) -> dict:
    """Verify against every page, MAX_IMAGES_PER_CALL at a time.

    A verifier that sees pages 1-4 of 10 would "correct" a field that lives on
    page 7 to null. So a field passes if any chunk confirms it; otherwise the
    most confident non-null correction wins. Short documents are one call.
    """
    reports = [_verify_once(pages[i:i + MAX_IMAGES_PER_CALL], scalars, schema)
               for i in range(0, len(pages), MAX_IMAGES_PER_CALL)]
    if len(reports) == 1:
        return reports[0]

    def rank(item):  # confirmed beats corrected beats null-correction, then confidence
        return (bool(item.get("ok")), item.get("correction") is not None, _clamp(item.get("confidence")))

    out = {}
    for name in scalars:
        items = [r[name] for r in reports if isinstance(r.get(name), dict)]
        if items:
            out[name] = max(items, key=rank)
    return out


def _verify_once(chunk, scalars, schema) -> dict:
    text = "\n\n".join(f"--- page {p.number} ---\n{p.text}" for p in chunk if p.text)
    prompt = (
        "A junior clerk extracted the values below from the document shown. Some of "
        "them are wrong - that is why you are checking. Your job is to find the "
        "mistakes, not to agree.\n\n"
        f"{json.dumps(scalars, indent=2, default=str)}\n\n"
        "Here is what each field is supposed to mean:\n"
        f"{_field_docs(schema, only=list(scalars))}\n"
        "For each field, first find that information in the document yourself, then "
        "compare it to the clerk's value.\n"
        "Watch for these specific mistakes:\n"
        "- two parties swapped (the seller put where the buyer belongs, or the reverse)\n"
        "- a date left in the document's own wording instead of YYYY-MM-DD\n"
        "- a value that is not in the document at all, which must be null\n"
        "- a value left null when the document does state it\n\n"
        f"Judge all {len(scalars)} fields: {', '.join(scalars)}.\n"
        "Set ok=false and give the right value as a string in correction when it is "
        "wrong; use correction null to mean the field should be null. Set confidence "
        "to how sure you are, and below 0.5 when the document is unclear.\n"
        + _text_block(text, _is_ocr(chunk))
    )
    out = llm.chat(
        prompt,
        schema=_verify_schema(list(scalars)),
        images=[p.png_b64 for p in chunk],
        model=llm.EXTRACT_MODEL,
        system="You audit extracted data against source documents. Be strict.",
        num_ctx=_ctx(len(chunk), text + json.dumps(scalars, default=str)),
    )
    return out if isinstance(out, dict) else {}


def _merge(a, b) -> dict:
    """First non-null wins for scalars; arrays concatenate."""
    out = dict(a)
    for k, v in b.items():
        if isinstance(v, list):
            out[k] = list(a.get(k) or []) + v
        elif out.get(k) in (None, "", []):
            out[k] = v
    return out


def _coerce(value, field_schema):
    """Turn a correction string into the type the schema asks for."""
    types = field_schema.get("type", "string")
    types = types if isinstance(types, list) else [types]
    s = str(value).strip()
    if s.lower() in NULL_WORDS:
        return None
    if "number" in types or "integer" in types:
        n = _number(s)
        if n is None:
            return None
        return int(n) if "integer" in types and n.is_integer() else n
    if "boolean" in types:
        return s.lower() in ("true", "yes", "y", "1")
    return s


def _number(s):
    """Parse "1,240.55", "1.240,55", "1 240,55", "(120.00)", "12%", "EUR -42".

    The last separator is the decimal point, unless it repeats ("1.234.567")
    or is a lone comma before exactly three digits ("1,240"), which is thousands.
    A lone dot is always a decimal point: "1.240" is 1.24, since the extraction
    prompt asks for plain numbers and US style is the common case.
    """
    s = re.sub(r"[\s\u00a0\u202f']", "", s)
    m = re.search(r"\d[\d.,]*\d|\d", s)
    if not m:
        return None
    t = m.group()
    neg = s[:m.start()].endswith("-") or (s.startswith("(") and s.endswith(")"))
    last = max(t.rfind(","), t.rfind("."))
    if last >= 0:
        sep, head, frac = t[last], t[:last], t[last + 1:]
        other = "." if sep == "," else ","
        if sep in head or (sep == "," and len(frac) == 3 and other not in head):
            t = t.replace(",", "").replace(".", "")
        else:
            t = head.replace(other, "") + "." + frac
    n = float(t)
    return -n if neg else n


def _field_docs(schema, only=None) -> str:
    """The schema's own field descriptions, as prompt text.

    Without this the model only sees bare names like `vendor_name` and guesses
    at what they mean - which is how buyer and seller end up swapped.
    """
    lines = []
    for name, spec in schema.get("properties", {}).items():
        if only is not None and name not in only:
            continue
        raw_type = spec.get("type", "string")
        types = "|".join(raw_type) if isinstance(raw_type, list) else raw_type
        desc = spec.get("description") or ""
        if raw_type == "array":
            inner = spec.get("items", {}).get("properties", {})
            desc = desc or f"one entry per row, each with: {', '.join(inner)}"
        lines.append(f"- {name} ({types}){': ' + desc if desc else ''}")
    return "Fields:\n" + "\n".join(lines) + "\n"


def _text_block(text, ocr=False, limit=4000) -> str:
    if not text:
        return "\nThis page has no text layer - read it from the image."
    if ocr:  # OCR misreads 0/O, 1/l, and drops characters - don't call it exact
        return (
            "\nOCR text of the page is below. It may contain recognition errors "
            "(look-alike characters, dropped or merged characters), so check "
            f"it against the image.\n<text>\n{text[:limit]}\n</text>"
        )
    return (
        "\nThe document's embedded text layer is below. It is exact - prefer it over "
        f"reading characters off the image, but use the image for layout.\n<text>\n{text[:limit]}\n</text>"
    )


def _is_ocr(pages) -> bool:
    return any(p.source == "ocr" for p in pages)


def _ctx(n_images, text) -> int:
    # ponytail: rough token budget - ~1600 tokens per rendered page, ~3 chars
    # per token of text. Tune if you see truncated output.
    return min(32768, 3072 + 1600 * n_images + len(text or "") // 3)


def _clamp(x) -> float:
    try:
        return max(0.0, min(1.0, float(x)))
    except (TypeError, ValueError):
        return 0.0


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        sys.exit("usage: python -m app.pipeline <file> [doc_type]")
    ok, detail = llm.available()
    if not ok:
        sys.exit(detail)
    result = run(sys.argv[1], doc_type=sys.argv[2] if len(sys.argv) > 2 else None,
                 on_step=lambda s: print(f"[{s}]", file=sys.stderr, flush=True))
    print(json.dumps(result, indent=2, default=str))
