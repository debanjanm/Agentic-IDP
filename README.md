# Agentic IDP

Give it a document — PDF or image — and it classifies the document, then extracts
entities against a JSON Schema. Everything runs locally on Ollama; no document
ever leaves the machine.

## How it works

```
file ──► doc.py ──► classify ──► extract ──► verify ──► validate ──► link ──► JSON
         PDF/image   which type?  schema-     second      do the       match to
         → PNG +     (enum-       constrained  look at    numbers and  earlier
         text layer  constrained)  decoding    each field dates agree?  entities
```

Two ideas carry the design:

**The JSON Schema is the decoding constraint.** Ollama's `format` parameter takes
a raw JSON Schema and restricts generation to it, so schema-valid JSON comes back
without an output parser, regex repair, or retry-on-malformed loop. Your schema is
passed through untouched.

**Image and text layer together.** Digital PDFs carry exact text, so the model
copies long invoice numbers instead of guessing at pixels. Scans have no text
layer and fall back to vision automatically — same code path, no switch.

The verify pass is a second model call that re-reads the page with the extracted
JSON in hand and can overrule it. At 3B parameters that is the main defense
against a confident wrong read.

A model's own "I'm 90% sure" is a weak signal, so field confidence is also capped
by evidence that does not come from the model: the verifier overruling the first
extraction, or the value appearing nowhere in the page text (see
[Confidence](#confidence)).

## Layout

```
backend/app/   API, worker, database, entity resolution, review (package `app`)
ai/            document pipeline: OCR, classify, extract, verify, validate, schemas (package `ai`)
frontend/      Streamlit UI (ui.py, .streamlit/ theme)
scripts/       backend.sh, frontend.sh, eval.py
docs/          architecture, solution flow, changelog
tests/         test_pipeline.py
assets/        sample documents + labels (assets/samples)
artifacts/     generated output such as eval reports (git-ignored)
data/          runtime database and uploads (git-ignored)
```

Diagrams and the full flow are in [docs/](docs/architecture.md); history in [docs/CHANGELOG.md](docs/CHANGELOG.md).

## Setup

Needs Python 3.12 and [Ollama](https://ollama.com). Nothing else to install by
hand: `./scripts/backend.sh` and `./scripts/frontend.sh` (below) each create their own virtualenv
under `.venv/` on first run and install what they need. Prefer your own
environment? The dependency lists are at the top of each script.

`paddleocr` (~130MB of models, downloaded on first use, cached under
`~/.paddlex/official_models`) OCRs any page with no PDF text layer - scans,
photos, plain images - so those pages carry real text into the pipeline
instead of going in blind. Digital PDFs never touch it: PyMuPDF's own text
layer is used first, and OCR only runs when that's empty.

Pull the models (~5.7 GB total):

```bash
ollama pull qwen2.5vl:3b
ollama pull qwen3:4b
```

`qwen3:4b` is text-only and does classification for any page with enough
text (native or OCR'd) - about half the speed of a vision call, since it
skips sending the image. `qwen2.5vl:3b` remains the vision model: extraction,
verification, and classification for the sparse-text pages OCR couldn't
help with.

## Run

Two scripts, one terminal each:

```bash
./scripts/backend.sh
```

```bash
./scripts/frontend.sh
```

`./scripts/backend.sh` runs the API (`http://127.0.0.1:8000`) and the worker together
and stops both on Ctrl-C. The API only accepts uploads and reports status;
`backend/app/worker.py` does the actual work as its own process, so a job survives
retries and an API restart. `./scripts/frontend.sh` runs the UI at
`http://localhost:8501` and needs none of the backend's heavy dependencies
(PaddlePaddle, PyMuPDF), so its environment is a third the size.

The first run takes a few minutes (PaddlePaddle is large); later runs start in
a couple of seconds. Both scripts pick Python 3.12 (override with
`PYTHON=/path/to/python`) and reinstall only when the script itself changes.
`PORT` and `HOST` change where they listen; `IDP_API=http://host:port` points
the UI at a backend elsewhere.

Or skip the web layer entirely:

```bash
PYTHONPATH=. .venv/backend/bin/python -m ai.pipeline assets/samples/invoice.pdf
```

## API

| Endpoint | |
|---|---|
| `POST /jobs` | multipart `file`, optional `doc_type`, optional `schema` → `202 {job_id}` |
| `POST /jobs/batch` | multipart `files` (repeated), optional `doc_type`/`schema` applied to all → `202 {job_ids}` |
| `GET /jobs/{id}` | status, fields, per-field confidence, full agent trace, `children` if this upload was split into sub-documents |
| `GET /jobs` | recent jobs |
| `GET /review` | jobs the pipeline flagged for a human that nobody has reviewed yet |
| `POST /jobs/{id}/review` | JSON `{"corrections": {field: value}}` - apply corrections and mark reviewed (empty = "checked, fine") |
| `POST /jobs/{id}/entities/{type}/unmerge` | split this job off a golden record it was wrongly matched to |
| `GET /stats` | review rate, confidence, entity match rate, what reviewers corrected |
| `GET /schemas`, `GET /schemas/{type}` | the schema registry |
| `GET /jobs/{id}/pages/{n}` | rendered page PNG |
| `GET /health` | Ollama reachable, models present |

The UI keeps the current job in the URL (`?job=<id>`), so a result survives a
reload and the link can be shared.

```bash
curl -F file=@invoice.pdf http://localhost:8000/jobs
curl -F file=@doc.pdf -F doc_type=receipt -F schema="$(cat my_schema.json)" http://localhost:8000/jobs
```

## Schemas

`ai/schemas/*.json` — one JSON Schema per file, filename is the document type.
Ships with invoice, receipt, bank_statement and id_card. Drop in a new file and it
is immediately both a classifier label and an extraction target; no code change.

A job may also POST its own schema. Use plain types and nullable unions
(`{"type": ["string", "null"]}`) — `$ref`, `oneOf`, `allOf` and `anyOf` are
rejected at submit time because Ollama's constrained decoding mishandles them.
Make every field `required` and nullable so the model can say "absent" instead of
inventing a value.

### Validation rules

A schema can carry `x-checks`: plain-text rules the extracted values must
satisfy. They run after the double-check, on the corrected values.

```json
"x-checks": [
  "subtotal + tax_amount = total_amount",
  "sum(line_items.amount) = subtotal",
  "invoice_date <= due_date"
]
```

Three kinds: amounts that add up (`+`, `-`, `=`, within 2 cents), a table column
that adds up (`sum(rows.column)`), and dates in order (`<=` or `<`, ISO dates).
A failed rule sends the job to the review queue **even when the model was
confident** - a wrong total read with full confidence is exactly what this
catches - and the Validate stage on the job page says which rule failed and
the two numbers it compared. A rule with a missing value (a null tax amount, a
date the model left as "14 March 2024") is skipped, not failed, and a rule that
doesn't parse is skipped too, so a typo in your own schema can't fail a job.
The seed schemas' rules are tested for typos. Reviewing and fixing the numbers
re-runs the rules.

### Entity resolution

A property can carry an `"x-entity": {"type": "Vendor", "key": "primary_name"}`
hint — properties sharing a `type` become one entity group (e.g. `vendor_name` +
`vendor_tax_id` are both part of the `Vendor` group). After verification,
`backend/app/resolve.py` matches each group against existing golden records (fuzzy name
match via stdlib `difflib`, or an exact match on any field marked `"strong":
true`, which overrides name similarity entirely — a shared tax ID or account
number means "same entity" even if the printed name differs) and either attaches
a mention to the match or creates a new golden record. Every job's result gets
an `entities: {type: entity_id}` map; the `entities`/`entity_mentions` tables
hold the golden records and their provenance. A schema with no `x-entity` hints
(the generic fallback, `id_card`) simply resolves nothing.

## Configuration

| Variable | Default |
|---|---|
| `IDP_EXTRACT_MODEL` | `qwen2.5vl:3b` |
| `IDP_CLASSIFY_MODEL` | `qwen2.5vl:3b` (vision fallback, sparse-text pages) |
| `IDP_TEXT_CLASSIFY_MODEL` | `qwen3:4b` (text-only, pages with ≥200 chars) |
| `IDP_OCR_LANG` | `en` (PaddleOCR language) |
| `OLLAMA_HOST` | `http://localhost:11434` |
| `IDP_DATA_DIR` | `./data` |
| `IDP_API` (UI only) | `http://localhost:8000` |

## Tests

```bash
.venv/backend/bin/python tests/test_pipeline.py
```

Offline tests always run. The end-to-end test builds a synthetic invoice PDF,
runs the real pipeline, and asserts the invoice number and total come back exact.
It skips with a clear message when Ollama isn't up.

### Accuracy on public data

`scripts/eval.py` scores the pipeline against labelled documents: exact-match per field
before and after the verify pass, how often a wrong document was flagged for
review, and how often a correct one was flagged anyway.

```bash
.venv/backend/bin/python scripts/eval.py path/to/sroie_entities -n 20   # images + X.txt labels
.venv/backend/bin/pip install datasets pillow                   # once, for CORD
.venv/backend/bin/python scripts/eval.py --cord 20 --known-type
.venv/backend/bin/python scripts/eval.py assets/samples --doc-type invoice --known-type   # 1 sample, ~1 min
```

SROIE maps to `merchant_name`, `transaction_date`, `total_amount`; CORD to
subtotal, tax, total and item count. Both are receipts; there is no public
bank-statement set, so those schemas are only measured on your own documents
(a folder of files each with an `X.json` label in schema field names).
A page takes ~1-3 minutes, so keep `-n` small. Reports land in `artifacts/`.

## Confidence

The verifier reports a confidence per field, but small models say "sure" about
nearly everything. Two warning signs that do not depend on that number cap a
field at 0.4, below the 0.6 review threshold, so it always goes to the review queue:

- `verifier_overruled` - the extract pass and the verify pass disagreed.
- `not_on_page` - the final value is in none of the page text. Numbers are looked
  for in every common notation (`1,240.55`, `1.240,55`, `1 240,55`), dates in
  written forms (`14 March 2024`), other values by their words. Not checked when
  the page has no text, the value is null, or a string is under 4 characters.

The job result lists them as `confidence_signals`, and the review queue says why
a field was flagged. `scripts/eval.py` prints how often a field is wrong when each
sign fires versus when none does, which is how to judge whether a sign is worth
its review load.

## Review and quality

The UI has three views: **Process**, **Review queue** and **Quality**. A job
lands in the review queue when the pipeline flags it (a field or the document
type came back below 60% confidence). Reviewing writes back to what the
pipeline wrote:

- **Corrections** change the job's field values. The model's own confidence is
  left as measured (the dashboard reports it), and the correction - what it
  was, what it became - goes in the job's trace.
- A correction to an entity field (a vendor name, a tax id) moves that job's
  mention to whichever golden record now matches, and the record it left is
  rebuilt from its remaining mentions, so the old spelling doesn't linger.
- **Unmerge** splits a job off a golden record it was wrongly matched to.
  Unmerging two jobs from the same record makes two separate records (there is
  no merge yet - see the backlog).

The **Quality** view reports what's knowable without ground truth: how many
jobs were flagged, average confidence per document type, how often documents
joined an existing entity vs started a new one, and - the only real error
measure - how many reviewed jobs needed a correction and which fields.

## Known limits

- A 3B VLM is mediocre on dense or low-quality scans. First lever:
  `IDP_EXTRACT_MODEL=qwen2.5vl:7b` (still fits 16 GB, roughly 2× slower).
- Measured on an M2/16GB, a one-page invoice takes **~3 minutes** end to end —
  three model calls (classify ≈45s, extract ≈60s, verify ≈85s), each re-reading
  the page image. Jobs are serialized on purpose: one model, one document at a
  time. Drop `VERIFY_RETRIES` to 0 or skip verification if you need it faster.
- `dpi` and `max_edge` in `ai/doc.py` are the accuracy/latency knobs and want
  tuning against your actual documents.
- Documents over 10 pages are refused with an error (`max_pages` in `ai/doc.py`),
  not truncated. Longer ones need splitting first.
- A multi-document upload is split where the page type changes or a page reads
  "Page 1 of N". Two same-type documents with no page numbering stay merged.
- Text from OCR is labelled "may contain errors" in the prompts; only a PDF's own
  text layer is called exact.
- Numbers in corrections parse `1,240.55`, `1.240,55`, `1 240,55`, `(120.00)` and
  `12%`. A lone dot is always a decimal point (`1.240` is 1.24).
- `not_on_page` needs page text, so on a page OCR could not read it stays silent.
  A value the model correctly derives (not printed as such) can trip it.
- Verification covers scalar fields; line items are extracted but not re-checked.
- No bounding boxes or field-level provenance yet — the natural next feature.
- A failed job retries once (`backend/app/worker.py`'s `MAX_ATTEMPTS`), then its status
  becomes `failed` with the error message attached — it does not retry forever.
- The worker process itself crashing (not just the API) can strand a job at
  `status="running"` with nothing to resume it — there's no lease/heartbeat.
  Restart the worker and requeue that job by hand (`UPDATE jobs SET status='queued' WHERE id=...`)
  if it happens; add a staleness sweep if it happens often.

## Backlog

Deliberately not built. Each item says when it becomes worth building.

- **Entity linking / knowledge graph** (was Phase 4): `/entities` API, an
  Entities tab, entity-to-entity `edges`, manual merge. Dropped as not needed
  yet - "which documents mention X" is already one query on `entity_mentions`.
  First piece worth building if wrong matches pile up: **merge** (the review
  view can only unmerge, and unmerging two jobs makes two separate records).
- **Graduated review**: today it's flagged or not. A failed arithmetic check
  should force a full review; one weak field should flag only that field.
- **Image quality checks**: blur, skew, rotation before OCR. Needed once phone
  photos arrive.
- **File handling**: magic-byte type detection, a clear error for encrypted PDFs
  (today they retry pointlessly), Office/email/ZIP inputs.
- **Provenance**: which part of the page each value came from (bounding boxes).
- **UI**: a batch upload's child documents aren't shown; line items can't be
  edited in review.
- **Cross-type identity**: the same company is a `Customer` on an invoice and a
  `Person` as a bank account holder; the two never link.

## Not included, on purpose

Layout model · vector DB / RAG · Docker · Celery/Redis · auth · React. Add a
real queue when inference moves off-box.
