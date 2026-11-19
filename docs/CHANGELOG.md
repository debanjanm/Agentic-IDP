# Changelog

Dates are commit dates. Newest first.

## Unreleased

### Added
- Evidence-based confidence signals. A field is capped at 0.4 (below the 0.6
  review threshold) when the verifier overruled the first extraction
  (`verifier_overruled`) or its value is in none of the page text (`not_on_page`,
  new `ai/ground.py`). Results carry `confidence_signals`; the review queue says
  why a field was flagged.
- `scripts/eval.py` prints how often a field is wrong when each signal fires
  versus when none does.

### Changed
- Fields can now be flagged for review even when the verifier was confident.
  Expect a higher review rate than before.

## 2026-11-19

### Changed
- Restructured the repository into `backend/`, `ai/`, `frontend/`, `scripts/`,
  `tests/`, `assets/`, `artifacts/` and `docs/`.
- `ai/` holds the document pipeline (OCR, classify, extract, verify, validate,
  schemas). `backend/app/` holds the API, worker, database, entity resolution
  and review. `ai/` never imports from `backend/`.
- `backend.sh`, `frontend.sh` and `eval.py` moved to `scripts/`. Run them as
  `./scripts/backend.sh` and `./scripts/frontend.sh`.
- The default data directory resolves from the repo root, so it no longer
  depends on where a process starts.
- `eval.py` writes its report to `artifacts/eval-<time>.json` by default.

### Added
- `assets/samples/`: a sample invoice and receipt with label files.
- `docs/`: architecture, solution flow and this changelog.

## 2026-11-12

### Added
- `scripts/eval.py`: scores the pipeline against labelled documents (SROIE-style
  folders, CORD v2, or your own). Reports exact match per field before and after
  verification, how many fields the verifier fixed or broke, and how often wrong
  and correct documents were flagged for review.
- The first full implementation, committed in one piece:
  - classify, extract, verify with bounded retry, validate, entity resolution
    and human review
  - PaddleOCR for pages with no text layer; `qwen3:4b` classifies pages with
    enough text, `qwen2.5vl:3b` does vision work
  - multi-document uploads split into child jobs
  - FastAPI API and a separate SQLite-polling worker with retries
  - Streamlit UI with a live stage tracker, a review queue and a quality
    dashboard, in a muted neobrutalist theme
  - schema extensions `x-checks` (validation rules) and `x-entity` (entity hints)
  - seed schemas for invoice, receipt, bank statement and ID card, plus a
    generic fallback
  - `backend.sh` and `frontend.sh`, each with its own virtualenv

### Fixed
- The verifier only saw the first 4 pages of a longer document and could null a
  field that lived later. It now checks every page, and a field passes if any
  chunk confirms it.
- Documents over 10 pages were silently cut to 10. They are now refused with a
  clear error.
- Two same-type documents back to back were merged. A page reading "Page 1 of N"
  now starts a new document.
- OCR text was described to the model as "exact". Only a PDF's own text layer is
  now labelled that way; OCR text is labelled "may contain recognition errors".
- A malformed `x-entity` hint crashed entity resolution. It is now rejected at
  upload with a clear message.
- Number parsing misread European and accounting formats: `1 240,55` became
  124055. It now handles `1.240,55`, `1 240,55`, `(120.00)` and `12%`.

## 2026-11-10
- Added the MIT licence.

## 2026-09-18
- Initial commit.

## Not yet built

Tracked in the README backlog: atomic job claiming, a lease for stranded jobs,
graduated review, bounding-box provenance, image quality checks, and an
independent second extraction pass as a further confidence signal.
