"""One end-to-end check plus the pure-logic bits.

    .venv/backend/bin/python tests/test_pipeline.py   # or: pytest tests/

The offline tests always run. The end-to-end test needs Ollama up with the
vision model pulled, and skips with a clear message when it isn't.
"""

import json
import os
import sys
import tempfile
from pathlib import Path

# app.db reads IDP_DATA_DIR once, at its own first import, and caches a single
# connection - this must run before ANYTHING imports app.db, including
# transitively (app.resolve imports it). Set it before the `from app import`
# line below, or these tests point at the real project data/idp.db instead of
# a throwaway directory.
os.environ.setdefault("IDP_DATA_DIR", tempfile.mkdtemp(prefix="idp_test_"))

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "backend"), str(ROOT / "scripts")]  # ai, app, eval.py

import fitz  # noqa: E402

from ai import doc, llm, pipeline, registry, segment, validate
from app import resolve, review

INVOICE_NUMBER = "INV-2024-0093"
TOTAL = 1240.55
RECEIPT_MERCHANT = "CORNER CAFE"


def make_invoice_pdf(path):
    d = fitz.open()
    page = d.new_page()
    lines = [
        "NORTHWIND SUPPLIES LTD",
        "12 Harbour Road, Bristol, UK",
        "",
        "INVOICE",
        f"Invoice number: {INVOICE_NUMBER}",
        "Invoice date: 14 March 2024",
        "Due date: 13 April 2024",
        "",
        "Bill to: Acme Robotics GmbH",
        "",
        "Description            Qty   Unit price     Amount",
        "Widget, 40mm             10       82.00      820.00",
        "Gasket set                4       55.00      220.00",
        "",
        "Subtotal                                    1040.00",
        "VAT 20%                                      200.55",
        f"TOTAL DUE                              EUR {TOTAL:,.2f}",
    ]
    y = 70
    for line in lines:
        page.insert_text((60, y), line, fontsize=11, fontname="cour")
        y += 18
    d.save(path)
    d.close()


def make_receipt_pdf(path):
    d = fitz.open()
    page = d.new_page()
    lines = [
        RECEIPT_MERCHANT,
        "4 Market Street, Leeds, UK",
        "",
        "RECEIPT",
        "Date: 2024-03-20  Time: 14:05",
        "",
        "Flat white               3.20",
        "Croissant                2.80",
        "",
        "Subtotal                 6.00",
        "Tax                      0.00",
        "TOTAL                    6.00",
        "Paid by card",
    ]
    y = 70
    for line in lines:
        page.insert_text((60, y), line, fontsize=11, fontname="cour")
        y += 18
    d.save(path)
    d.close()


def make_two_document_pdf(path):
    """One PDF file containing an invoice followed by an unrelated receipt -
    the segmentation boundary case: two documents, one upload."""
    with tempfile.TemporaryDirectory() as tmp:
        inv, rcpt = Path(tmp) / "a.pdf", Path(tmp) / "b.pdf"
        make_invoice_pdf(inv)
        make_receipt_pdf(rcpt)
        merged = fitz.open()
        for p in (inv, rcpt):
            with fitz.open(p) as d:
                merged.insert_pdf(d)
        merged.save(path)
        merged.close()


# --- offline: no model needed -------------------------------------------------

def test_coerce_handles_messy_corrections():
    num = {"type": ["number", "null"]}
    assert pipeline._coerce("EUR 1,240.55", num) == TOTAL
    assert pipeline._coerce("  -42 ", num) == -42
    assert pipeline._coerce("n/a", num) is None
    assert pipeline._coerce("no digits here", num) is None
    assert pipeline._coerce("7", {"type": ["integer", "null"]}) == 7
    assert pipeline._coerce("INV-9", {"type": ["string", "null"]}) == "INV-9"
    # European and accounting formats: the old regex read "1 240,55" as 124055
    assert pipeline._coerce("1.240,55", num) == TOTAL
    assert pipeline._coerce("1 240,55", num) == TOTAL
    assert pipeline._coerce("1,240", num) == 1240          # lone comma + 3 digits = thousands
    assert pipeline._coerce("12,5", num) == 12.5
    assert pipeline._coerce("1,234,567", num) == 1234567
    assert pipeline._coerce("(120.00)", num) == -120       # accounting negative
    assert pipeline._coerce("12%", num) == 12


def test_verify_applies_real_corrections_and_ignores_restated_values():
    """The correction path, without needing a model.

    A verifier told to hunt for mistakes will often say "wrong!" and then hand
    back the value it was given. Only an actual change counts as a correction.
    """
    schema = registry.load("invoice")
    pages = [doc.Page(1, "", "")]
    report = {
        # genuinely wrong -> must be replaced, and coerced to a number
        "total_amount": {"ok": False, "confidence": 0.9, "correction": "EUR 1,240.55"},
        # "wrong" but restates the same value -> must be left alone, no retry
        "invoice_number": {"ok": False, "confidence": 0.8, "correction": INVOICE_NUMBER},
        # hallucinated value the document does not contain -> null it out
        "vendor_tax_id": {"ok": False, "confidence": 0.7, "correction": None},
    }
    calls = []
    original = pipeline._verify_once
    pipeline._verify_once = lambda *a: (calls.append(1), report)[1]
    try:
        fields, conf, trace = pipeline.verify(
            pages, schema,
            {"total_amount": 999.0, "invoice_number": INVOICE_NUMBER, "vendor_tax_id": "GB-FAKE"},
        )
    finally:
        pipeline._verify_once = original

    assert fields["total_amount"] == TOTAL, fields["total_amount"]
    assert fields["invoice_number"] == INVOICE_NUMBER
    assert fields["vendor_tax_id"] is None
    assert conf["invoice_number"] == 0.8
    assert calls == [1, 1], f"expected exactly one retry after real fixes, got {len(calls)} calls"
    assert any(t["step"] == "correct" for t in trace)


def test_classify_routes_text_vs_vision_by_text_length():
    """Plenty of text -> text-only qwen3:4b, no image sent. Sparse text ->
    vision model, image sent. Stubbed llm.chat, no model needed."""
    calls = []

    def fake_chat(prompt, **kw):
        calls.append(kw)
        return {"doc_type": "invoice", "confidence": 0.9, "reasoning": "x", "language": "en"}

    original = pipeline.llm.chat
    pipeline.llm.chat = fake_chat
    try:
        long_page = doc.Page(1, "b64", "x" * 300)
        text_out = pipeline.classify([long_page])
        short_page = doc.Page(1, "b64", "short")
        vision_out = pipeline.classify([short_page])
    finally:
        pipeline.llm.chat = original

    # the UI's "Identify document" stage reports which path and model ran
    assert (text_out["via"], text_out["model"]) == ("text", pipeline.llm.TEXT_CLASSIFY_MODEL)
    assert (vision_out["via"], vision_out["model"]) == ("vision", pipeline.llm.CLASSIFY_MODEL)

    assert calls[0]["model"] == pipeline.llm.TEXT_CLASSIFY_MODEL
    assert calls[0]["images"] is None, "text path must not send an image"
    assert calls[1]["model"] == pipeline.llm.CLASSIFY_MODEL
    assert calls[1]["images"] == ["b64"], "vision path must send the page image"


def test_step_timer_adds_up_repeated_steps():
    """The worker marks 'classifying', then the pipeline marks it again; the
    UI's per-stage seconds must be the sum, and the last step runs to 'now'."""
    from app import worker

    steps = worker._Steps("unused")  # seconds() never touches the db
    steps.marks = [("reading", 0.0), ("classifying", 2.0), ("classifying", 5.0), ("extracting", 6.0)]
    real = worker.time.monotonic
    worker.time.monotonic = lambda: 10.0
    try:
        assert steps.seconds() == {"reading": 2.0, "classifying": 4.0, "extracting": 4.0}
    finally:
        worker.time.monotonic = real


def test_segment_groups_consecutive_same_type_pages():
    """Boundary logic only, no model: stub classify() to return a fixed
    doc_type per page index and check the grouping."""
    fake_pages = [doc.Page(i + 1, "", "") for i in range(5)]
    fake_types = ["invoice", "invoice", "receipt", "receipt", "receipt"]

    original = pipeline.classify
    pipeline.classify = lambda pages: {"doc_type": fake_types[pages[0].number - 1],
                                       "confidence": 0.9, "reasoning": "stub", "language": "en"}
    try:
        segments = segment.split(fake_pages)
    finally:
        pipeline.classify = original

    assert [s.cls["doc_type"] for s in segments] == ["invoice", "receipt"]
    assert [len(s.pages) for s in segments] == [2, 3]
    assert [p.number for p in segments[0].pages] == [1, 2]
    assert [p.number for p in segments[1].pages] == [3, 4, 5]


def test_resolve_matches_name_variants_but_not_different_vendors():
    a = resolve.EntityGroup("Vendor", {"primary_name": "NORTHWIND SUPPLIES LTD"}, set())
    candidates = [{"id": "e1", "canonical_attrs": {"primary_name": "Northwind Supplies Ltd."}}]
    assert resolve.match(a, candidates) == "e1", "punctuation/case/legal-suffix variants should match"

    b = resolve.EntityGroup("Vendor", {"primary_name": "Southwind Traders"}, set())
    assert resolve.match(b, candidates) is None, "an unrelated name must not match"


def test_resolve_strong_identifier_overrides_name_mismatch():
    group = resolve.EntityGroup(
        "Vendor", {"primary_name": "Northwind Supplies (UK) Ltd", "tax_id": "GB123456"}, {"tax_id"}
    )
    candidates = [{"id": "e1", "canonical_attrs": {"primary_name": "Totally Different Name", "tax_id": "GB123456"}}]
    assert resolve.match(group, candidates) == "e1", "a shared tax id should force-match despite the name"


def test_resolve_entity_groups_reads_x_entity_hints():
    schema = registry.load("invoice")
    fields = {"vendor_name": "Acme Corp", "vendor_tax_id": "X1", "customer_name": "Bob", "currency": "EUR"}
    groups = {g.type: g for g in resolve.entity_groups(schema, fields)}
    assert set(groups) == {"Vendor", "Customer"}
    assert groups["Vendor"].attrs == {"primary_name": "Acme Corp", "tax_id": "X1"}
    assert groups["Vendor"].strong_keys == {"tax_id"}
    assert groups["Customer"].attrs == {"primary_name": "Bob"}


def test_resolve_creates_one_entity_across_two_jobs_with_name_variants():
    """Integration: two invoices from the same vendor, spelled slightly
    differently, must collapse into one golden record with two mentions."""
    from app import db

    schema = registry.load("invoice")
    fields_a = {"vendor_name": "NORTHWIND SUPPLIES LTD", "vendor_tax_id": None, "customer_name": "Acme"}
    fields_b = {"vendor_name": "Northwind Supplies Ltd.", "vendor_tax_id": None, "customer_name": "Acme"}

    db.create("rj1", "a.pdf", "/tmp/a.pdf")
    db.create("rj2", "b.pdf", "/tmp/b.pdf")
    out_a = resolve.resolve("rj1", schema, fields_a)
    out_b = resolve.resolve("rj2", schema, fields_b)

    assert out_a["Vendor"] == out_b["Vendor"], "spelling variants of the same vendor must resolve to one entity"
    mentions = db.entity_mentions(out_a["Vendor"])
    assert len(mentions) == 2 and {m["job_id"] for m in mentions} == {"rj1", "rj2"}


def _seed_done_job(job_id, vendor, needs_review=1, extra=None):
    """A finished, entity-resolved invoice job, as the worker would leave it."""
    from app import db

    schema = registry.load("invoice")
    fields = {"vendor_name": vendor, "vendor_tax_id": None, "customer_name": None,
              "total_amount": 100.0, "line_items": []}
    fields.update(extra or {})
    db.create(job_id, f"{job_id}.pdf", f"/tmp/{job_id}.pdf")
    entities = resolve.resolve(job_id, schema, fields)
    db.update(job_id, status="done", doc_type="invoice", confidence=0.9, needs_review=needs_review,
              finished_at=db.now(), trace=[],
              result={"fields": fields, "field_confidence": {"vendor_name": 0.4, "total_amount": 0.9},
                      "low_confidence_fields": ["vendor_name"], "entities": entities,
                      "validation": validate.check(schema, fields)})
    return entities["Vendor"]


def test_review_correction_moves_entity_and_rebuilds_the_old_one():
    from app import db

    shared = _seed_done_job("rv1", "Zephyr Freight Ltd")
    assert _seed_done_job("rv2", "Zephyr Freight") == shared  # spelling variant joins the same record
    assert db.get_entity(shared)["canonical_attrs"]["primary_name"] == "Zephyr Freight"  # newest wins

    review.apply_review("rv2", {"vendor_name": "Quartz Logistics", "total_amount": "1,250.50"})

    job = db.get("rv2")
    assert job["result"]["fields"]["total_amount"] == 1250.5
    moved = job["result"]["entities"]["Vendor"]
    assert moved != shared and db.get_entity(moved)["canonical_attrs"]["primary_name"] == "Quartz Logistics"
    # rv2's old spelling must not linger on the record it left
    assert db.get_entity(shared)["canonical_attrs"]["primary_name"] == "Zephyr Freight Ltd"
    assert db.mention_count(shared) == 1
    assert job["reviewed_at"] and job["trace"][-1]["corrected"]["vendor_name"]["from"] == "Zephyr Freight"
    assert job["result"]["field_confidence"]["vendor_name"] == 0.4, "the model's confidence must stay as measured"

    for bad in ({"total_amount": "abc"}, {"line_items": "x"}, {"nope": "x"}):
        try:
            review.apply_review("rv2", bad)
        except review.ReviewError:
            continue
        raise AssertionError(f"accepted a bad correction: {bad}")


def test_validate_checks_amounts_columns_and_dates():
    rules = {"x-checks": ["subtotal + tax_amount = total_amount", "sum(line_items.amount) = subtotal",
                          "invoice_date <= due_date"]}
    ok = {"subtotal": 1040.0, "tax_amount": 200.55, "total_amount": 1240.55,
          "line_items": [{"amount": 820.0}, {"amount": 220.0}], "invoice_date": "2024-03-14", "due_date": "2024-04-13"}
    assert validate.check(rules, ok) == {"ran": 3, "skipped": 0, "failed": []}

    out = validate.check(rules, {**ok, "total_amount": 1250.55, "due_date": "2024-01-01"})
    assert [f["rule"] for f in out["failed"]] == ["subtotal + tax_amount = total_amount", "invoice_date <= due_date"]
    assert out["failed"][0]["fields"] == ["subtotal", "tax_amount", "total_amount"]
    assert [f["rule"] for f in validate.check(rules, {**ok, "subtotal": 900.0})["failed"]][:2] == [
        "subtotal + tax_amount = total_amount", "sum(line_items.amount) = subtotal"]

    # cent rounding is tolerated; a missing value skips the rule instead of failing it
    assert validate.check(rules, {**ok, "total_amount": 1240.56})["failed"] == []
    assert validate.check(rules, {**ok, "tax_amount": None})["skipped"] == 1
    # ... including a table column with a hole in it, and dates the model left unnormalised
    assert validate.check(rules, {**ok, "line_items": [{"amount": 820.0}, {"amount": None}]})["skipped"] == 1
    assert validate.check(rules, {**ok, "due_date": "13 April 2024"})["skipped"] == 1

    # subtraction, strict <, and rules that must never raise
    bank = {"x-checks": ["opening_balance + total_deposits - total_withdrawals = closing_balance",
                         "date_of_birth < issue_date", "a * b = c", "", 5]}
    out = validate.check(bank, {"opening_balance": 100.0, "total_deposits": 50.0, "total_withdrawals": 30.0,
                                "closing_balance": 120.0, "date_of_birth": "2000-01-01", "issue_date": "2000-01-01"})
    assert (out["ran"], out["skipped"]) == (2, 3)
    assert [f["rule"] for f in out["failed"]] == ["date_of_birth < issue_date"]  # equal dates aren't strictly before


def test_seed_schema_rules_parse_and_name_real_fields():
    """A typo in a schema's rule would skip silently and disable the check."""
    for doc_type in registry.doc_types():
        schema = registry.load(doc_type)
        props = schema["properties"]
        for rule in schema.get("x-checks", []):
            m = validate._RULE.fullmatch(rule)
            assert m, f"{doc_type}: rule doesn't parse: {rule!r}"
            for side in (m[1], m[3]):
                for _sign, arr, key, name in validate._PART.findall(side):
                    if arr:
                        assert key in props[arr]["items"]["properties"], f"{doc_type}: {rule!r} - no {arr}.{key}"
                    else:
                        assert name in props, f"{doc_type}: {rule!r} - no field {name}"


def test_failed_check_forces_review_even_when_the_model_is_confident():
    """extract/verify stubbed: every field at 0.95, but the total doesn't add up."""
    fake_cls = {"doc_type": "invoice", "confidence": 0.95, "reasoning": "x", "language": "en"}

    def run(total):
        fields = {"subtotal": 1040.0, "tax_amount": 200.55, "total_amount": total,
                  "line_items": [{"amount": 1040.0}], "invoice_date": "2024-03-14", "due_date": "2024-04-13"}
        real_extract, real_verify = pipeline.extract, pipeline.verify
        pipeline.extract = lambda pages, schema: dict(fields)
        pipeline.verify = lambda pages, schema, f: (f, {k: 0.95 for k in f if k != "line_items"}, [])
        try:
            return pipeline.run_pages([doc.Page(1, "", "text")], cls=fake_cls)
        finally:
            pipeline.extract, pipeline.verify = real_extract, real_verify

    good, bad = run(1240.55), run(1250.55)
    assert good["needs_review"] is False and good["validation"]["failed"] == []
    assert bad["needs_review"] is True and bad["low_confidence_fields"] == []
    assert [f["rule"] for f in bad["validation"]["failed"]] == ["subtotal + tax_amount = total_amount"]
    assert any(t["step"] == "validate" for t in bad["trace"])


def test_review_fixing_the_numbers_clears_the_failed_check():
    from app import db

    _seed_done_job("rv7", "Helix Couriers", extra={"subtotal": 1040.0, "tax_amount": 200.55, "total_amount": 1250.55})
    assert [f["rule"] for f in db.get("rv7")["result"]["validation"]["failed"]] == ["subtotal + tax_amount = total_amount"]
    assert db.review_queue() and "subtotal + tax_amount = total_amount" in \
        next(j for j in db.review_queue() if j["id"] == "rv7")["failed_checks"]

    review.apply_review("rv7", {"total_amount": "1240.55"})
    assert db.get("rv7")["result"]["validation"]["failed"] == []
    assert "rv7" not in [j["id"] for j in db.review_queue()]


def test_unmerge_splits_a_job_off_and_guards_single_mention_entities():
    from app import db

    shared = _seed_done_job("rv3", "Nimbus Cargo")
    assert _seed_done_job("rv4", "Nimbus Cargo Inc") == shared

    new = review.unmerge("rv4", "Vendor")
    assert new != shared
    assert db.get("rv4")["result"]["entities"]["Vendor"] == new
    assert db.mention_count(shared) == 1 and db.mention_count(new) == 1
    assert db.get_entity(new)["canonical_attrs"]["primary_name"] == "Nimbus Cargo Inc"
    assert db.get_entity(shared)["canonical_attrs"]["primary_name"] == "Nimbus Cargo"

    for job_id in ("rv3", "rv4"):  # each now stands alone - nothing left to split from
        try:
            review.unmerge(job_id, "Vendor")
        except review.ReviewError:
            continue
        raise AssertionError(f"unmerged a single-mention entity for {job_id}")


def test_stats_count_new_vs_matched_entities_and_human_corrections():
    def vendor(stats):
        return next((e for e in stats["entities"] if e["type"] == "Vendor"), {"entities": 0, "mentions": 0, "matched": 0})

    before = review.stats()
    _seed_done_job("rv5", "Orion Haulage", needs_review=1)
    _seed_done_job("rv6", "Orion Haulage Ltd", needs_review=0)  # joins rv5's record
    review.apply_review("rv5", {"total_amount": "99"})
    after = review.stats()

    assert after["jobs"] - before["jobs"] == 2
    v0, v1 = vendor(before), vendor(after)
    assert (v1["entities"] - v0["entities"], v1["mentions"] - v0["mentions"], v1["matched"] - v0["matched"]) == (1, 2, 1)
    assert after["reviewed"]["jobs"] - before["reviewed"]["jobs"] == 1
    assert after["reviewed"]["with_corrections"] - before["reviewed"]["with_corrections"] == 1
    assert after["reviewed"]["fields"]["total_amount"] - before["reviewed"]["fields"].get("total_amount", 0) == 1


def test_worker_retries_then_marks_failed():
    """Retry/dead-letter transition, no model needed."""
    from app import db, worker

    db.create("j1", "a.pdf", "/tmp/a.pdf")
    job = db.get("j1")
    worker._fail_or_retry("j1", job["attempts"], RuntimeError("boom"))
    job = db.get("j1")
    assert job["status"] == "queued" and job["attempts"] == 1, job
    assert "boom" in job["error"]

    worker._fail_or_retry("j1", job["attempts"], RuntimeError("boom again"))
    job = db.get("j1")
    assert job["status"] == "failed" and job["attempts"] == 2, job
    assert job["finished_at"] is not None


def test_batch_retry_skips_already_done_children():
    """A batch parent retried after a partial failure must not recreate or
    rerun a child that already finished - that used to raise sqlite's
    duplicate-id error and redo finished VLM work."""
    from app import db, worker

    db.create("p1", "batch.pdf", "/tmp/batch.pdf")
    db.create("p1-1", "batch.pdf (part 1/2)", "/tmp/batch.pdf", parent_job_id="p1")
    db.update("p1-1", status="done", doc_type="invoice", needs_review=0)

    calls = []
    original = worker._run_pipeline
    worker._run_pipeline = lambda job_id, pages, **kw: calls.append(job_id)
    try:
        fake_cls = {"doc_type": "x", "confidence": 1.0, "reasoning": "x", "language": None}
        segments = [
            segment.Segment([doc.Page(1, "", "")], fake_cls),
            segment.Segment([doc.Page(2, "", "")], fake_cls),
        ]
        worker._run_batch("p1", "/tmp/batch.pdf", "batch.pdf", segments)
    finally:
        worker._run_pipeline = original

    assert calls == ["p1-2"], f"should only run the missing child, got {calls}"
    assert db.get("p1")["status"] == "done"


def test_merge_prefers_first_non_null_and_concatenates_arrays():
    a = {"total": None, "id": "A1", "rows": [1]}
    b = {"total": 5, "id": "B2", "rows": [2]}
    assert pipeline._merge(a, b) == {"total": 5, "id": "A1", "rows": [1, 2]}


def test_registry_rejects_schemas_ollama_cannot_constrain():
    for bad in ['{"nope', {"type": "string"}, {"type": "object", "properties": {"a": {"$ref": "#/x"}}}]:
        try:
            registry.validate_schema(bad)
        except ValueError:
            continue
        raise AssertionError(f"accepted a bad schema: {bad}")
    assert registry.validate_schema(registry.load("invoice"))["type"] == "object"


def test_loads_pdf_and_image_through_one_path():
    with tempfile.TemporaryDirectory() as tmp:
        pdf = Path(tmp) / "inv.pdf"
        make_invoice_pdf(pdf)
        pages = doc.load_pages(pdf)
        assert len(pages) == 1
        assert INVOICE_NUMBER in pages[0].text  # digital PDF keeps its text layer

        png = Path(tmp) / "inv.png"
        png.write_bytes(doc.render_png(pdf, 1))
        # a raw image has no PDF text layer, but ai/ocr.py (PaddleOCR) fills
        # one in - this is what lets classify() skip vision for scans too
        assert INVOICE_NUMBER in doc.load_pages(png)[0].text


def test_verify_sees_every_page_not_just_the_first_four():
    """A field on page 7 must not be nulled because the verifier only saw 1-4."""
    seen = []

    def fake(chunk, scalars, schema):
        seen.append([p.number for p in chunk])
        on_this_chunk = 7 in [p.number for p in chunk]
        return {"invoice_number": {"ok": on_this_chunk, "confidence": 0.9 if on_this_chunk else 0.3,
                                   "correction": None}}

    original = pipeline._verify_once
    pipeline._verify_once = fake
    try:
        pages = [doc.Page(i, "", "") for i in range(1, 11)]
        fields, conf, _ = pipeline.verify(pages, registry.load("invoice"), {"invoice_number": "INV-7"})
    finally:
        pipeline._verify_once = original

    assert seen == [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10]], seen
    assert fields["invoice_number"] == "INV-7" and conf["invoice_number"] == 0.9


def test_load_pages_refuses_instead_of_truncating():
    import fitz

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "long.pdf"
        d = fitz.open()
        for _ in range(4):
            d.new_page().insert_text((72, 72), "text")
        d.save(path)
        d.close()
        assert len(doc.load_pages(path, max_pages=4)) == 4
        try:
            doc.load_pages(path, max_pages=3)
        except ValueError as e:
            assert "4 pages" in str(e)
        else:
            raise AssertionError("silently truncated a 4-page document at 3")


def test_segment_splits_same_type_documents_at_page_one_of_n():
    pages = [doc.Page(1, "", "Page 1 of 2"), doc.Page(2, "", "Page 2 of 2"),
             doc.Page(3, "", "Page 1 of 1"), doc.Page(4, "", "no numbering")]
    original = pipeline.classify
    pipeline.classify = lambda p: {"doc_type": "invoice", "confidence": 0.9, "reasoning": "", "language": "en"}
    try:
        segments = segment.split(pages)
    finally:
        pipeline.classify = original
    assert [[p.number for p in s.pages] for s in segments] == [[1, 2], [3, 4]]


def test_only_text_layer_text_is_called_exact():
    assert "exact" in pipeline._text_block("INV-1")
    ocr_text = pipeline._text_block("INV-1", ocr=True)
    assert "exact" not in ocr_text and "errors" in ocr_text
    assert pipeline._is_ocr([doc.Page(1, "", "t", "ocr")]) and not pipeline._is_ocr([doc.Page(1, "", "t")])


def test_registry_rejects_malformed_x_entity_hints():
    def schema(hint):
        return {"type": "object", "properties": {"v": {"type": "string", "x-entity": hint}}}

    for bad in ["Vendor", {"key": "primary_name"}, {"type": 3}, {"type": "Vendor", "strong": "yes"}]:
        try:
            registry.validate_schema(schema(bad))
        except ValueError as e:
            assert "x-entity" in str(e)
        else:
            raise AssertionError(f"accepted x-entity {bad!r}")
    registry.validate_schema(schema({"type": "Vendor", "key": "tax_id", "strong": True}))
    for t in registry.doc_types():  # the seed schemas pass their own validator
        registry.validate_schema(registry.load(t))


def test_eval_loads_sroie_labels_and_scores_the_verifier():
    import eval as ev  # the eval.py script; no model needed for this

    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / "r1.jpg").write_bytes(b"x")
        (Path(tmp) / "r1.txt").write_text('{"company": "ACME SDN BHD", "date": "25/12/2018", "address": "x", "total": "9.00"}')
        (Path(tmp) / "r2.jpg").write_bytes(b"x")  # no label -> skipped
        [(path, truth)] = ev.load_dir(tmp)
    assert path.name == "r1.jpg"
    assert truth["merchant_name"] == "ACME SDN BHD" and truth["total_amount"] == "9.00"

    schema = registry.load("receipt")
    assert ev.same("transaction_date", schema, "25/12/2018", "2018-12-25")
    assert ev.same("total_amount", schema, "1,240.55", 1240.55)
    assert ev.same("merchant_name", schema, "ACME SDN. BHD.", "acme sdn bhd")
    assert not ev.same("total_amount", schema, "9.00", None)
    assert ev.same("items", schema, 2, [{}, {}]) and not ev.same("items", schema, 3, [{}, {}])

    def result(extracted, final, flagged=False, failed=()):
        return {"schema": schema, "fields": final, "needs_review": flagged,
                "validation": {"failed": list(failed)},
                "trace": [{"step": "extract", "fields": extracted}], "doc_type": "receipt"}

    truth = {"merchant_name": "ACME", "total_amount": "9.00"}
    fixed = ev.score_case(truth, result({"merchant_name": "ACME", "total_amount": 99.0},
                                        {"merchant_name": "ACME", "total_amount": 9.0}), "receipt")
    broke = ev.score_case(truth, result({"merchant_name": "ACME", "total_amount": 9.0},
                                        {"merchant_name": "ACME", "total_amount": 99.0}, True, ["rule"]), "receipt")
    s = ev.summarize([fixed, broke, {"error": "boom"}])
    assert (s["verifier_fixed"], s["verifier_broke"], s["errors"]) == (1, 1, 1)
    assert s["fields"]["total_amount"] == (2, 1, 1)
    assert (s["wrong_docs"], s["wrong_flagged"], s["wrong_caught_by_rules"], s["right_docs"]) == (1, 1, 1, 1)
    assert s["type_ok"] == "2/2"


# --- end to end: needs Ollama -------------------------------------------------

def test_end_to_end_invoice():
    ok, detail = llm.available()
    if not ok:
        print(f"SKIP end-to-end: {detail}")
        return
    with tempfile.TemporaryDirectory() as tmp:
        pdf = Path(tmp) / "inv.pdf"
        make_invoice_pdf(pdf)
        result = pipeline.run(str(pdf), on_step=lambda s: print(f"  [{s}]", flush=True))

    print(json.dumps(result["fields"], indent=2, default=str))
    assert result["doc_type"] == "invoice", f"classified as {result['doc_type']}"
    f = result["fields"]
    assert f["invoice_number"] == INVOICE_NUMBER, f"got {f['invoice_number']!r}"
    assert f["total_amount"] == TOTAL, f"got {f['total_amount']!r}"
    # buyer and seller are the classic small-VLM swap; the schema descriptions
    # fed into the prompt are what keep these apart
    assert "NORTHWIND" in (f["vendor_name"] or "").upper(), f"vendor: {f['vendor_name']!r}"
    assert "Acme" in (f["customer_name"] or ""), f"customer: {f['customer_name']!r}"
    assert f["invoice_date"] == "2024-03-14", f"date not normalised: {f['invoice_date']!r}"
    assert len(f["line_items"]) == 2, f"line items: {f['line_items']}"
    # the verify schema makes every scalar a required property, so none may be missing
    scalars = [k for k, v in f.items() if not isinstance(v, (list, dict))]
    missing = [k for k in scalars if k not in result["field_confidence"]]
    assert not missing, f"verify skipped {missing}"
    assert any(t["step"] == "verify" for t in result["trace"]), "verify never ran"


def test_end_to_end_segmentation_splits_two_documents():
    ok, detail = llm.available()
    if not ok:
        print(f"SKIP segmentation e2e: {detail}")
        return
    with tempfile.TemporaryDirectory() as tmp:
        merged = Path(tmp) / "batch.pdf"
        make_two_document_pdf(merged)
        pages = doc.load_pages(merged)
        assert len(pages) == 2, "test fixture should be a 2-page upload"

        segments = segment.split(pages)
        print("segments:", [(s.cls["doc_type"], len(s.pages)) for s in segments])
        assert len(segments) == 2, f"expected 2 documents, got {len(segments)}"
        assert segments[0].cls["doc_type"] == "invoice", segments[0].cls
        assert segments[1].cls["doc_type"] == "receipt", segments[1].cls

        inv_result = pipeline.run_pages(segments[0].pages, cls=segments[0].cls)
        rcpt_result = pipeline.run_pages(segments[1].pages, cls=segments[1].cls)

    assert inv_result["fields"]["invoice_number"] == INVOICE_NUMBER
    assert rcpt_result["fields"]["merchant_name"] == RECEIPT_MERCHANT, rcpt_result["fields"]


def test_generic_fallback_extracts_when_type_is_unregistered():
    ok, detail = llm.available()
    if not ok:
        print(f"SKIP generic-fallback e2e: {detail}")
        return
    with tempfile.TemporaryDirectory() as tmp:
        pdf = Path(tmp) / "inv.pdf"
        make_invoice_pdf(pdf)
        # force an unregistered type - the point is the pipeline must not abort
        result = pipeline.run(str(pdf), doc_type="purchase_order_xyz")

    assert result["fields"], "generic fallback produced no fields at all"
    assert any(t["step"] == "fallback" for t in result["trace"]), "fallback step missing from trace"
    assert INVOICE_NUMBER in json.dumps(result["fields"]), \
        f"generic extraction lost the invoice number: {result['fields']}"


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_"):
            continue
        try:
            fn()
            print(f"PASS {name}")
        except AssertionError as e:
            failures += 1
            print(f"FAIL {name}: {e}")
    sys.exit(1 if failures else 0)
