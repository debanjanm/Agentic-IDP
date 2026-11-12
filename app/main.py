"""FastAPI surface. Only handles HTTP concerns - validating and saving an
upload, then enqueuing it. app/worker.py (a separate process) does the actual
work, so a job survives an API restart."""

import json
import uuid
from pathlib import Path

from fastapi import Body, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import Response

from . import db, doc, llm, registry, review

MAX_UPLOAD_BYTES = 25 * 1024 * 1024
CHUNK = 1 << 20

app = FastAPI(title="Agentic IDP", version="0.1.0")


@app.get("/health")
def health():
    ok, detail = llm.available()
    return {"ok": ok, "detail": detail, "classify_model": llm.CLASSIFY_MODEL,
            "extract_model": llm.EXTRACT_MODEL}


@app.get("/schemas")
def list_schemas():
    return {"doc_types": registry.doc_types(), "labels": registry.labels()}


@app.get("/schemas/{doc_type}")
def get_schema(doc_type: str):
    try:
        return registry.load(doc_type)
    except KeyError as e:
        raise HTTPException(404, str(e)) from e


@app.post("/jobs", status_code=202)
async def create_job(file: UploadFile = File(...), doc_type: str = Form(None), schema: str = Form(None)):
    parsed = _validate_request(doc_type, schema)
    job_id = await _ingest(file, doc_type or None, parsed)
    return {"job_id": job_id, "status": "queued"}


@app.post("/jobs/batch", status_code=202)
async def create_batch(files: list[UploadFile] = File(...), doc_type: str = Form(None), schema: str = Form(None)):
    """Multiple files, one call - the on-ramp for "give it a folder of
    documents." doc_type/schema, if given, apply to every file the same way."""
    parsed = _validate_request(doc_type, schema)
    job_ids = [await _ingest(f, doc_type or None, parsed) for f in files]
    return {"job_ids": job_ids, "status": "queued", "count": len(job_ids)}


@app.get("/jobs")
def list_jobs(limit: int = 25):
    return {"jobs": db.recent(min(max(limit, 1), 100))}


@app.get("/jobs/{job_id}")
def get_job(job_id: str):
    job = db.get(job_id)
    if not job:
        raise HTTPException(404, "no such job")
    job.pop("path", None)
    kids = db.children(job_id)
    if kids:
        job["children"] = [{k: v for k, v in c.items() if k != "path"} for c in kids]
    ents = job["result"].get("entities") if isinstance(job["result"], dict) else None
    if ents:
        job["entity_info"] = {
            t: {"id": i, "documents": db.mention_count(i),
                "name": ((db.get_entity(i) or {}).get("canonical_attrs") or {}).get("primary_name")}
            for t, i in ents.items()
        }
    return job


@app.get("/review")
def review_queue():
    """Jobs the pipeline flagged for a human that nobody has reviewed yet."""
    return {"jobs": db.review_queue()}


@app.post("/jobs/{job_id}/review")
def review_job(job_id: str, body: dict = Body(default={})):
    """Body: {"corrections": {field: new value}}. Empty means "checked, fine"."""
    try:
        review.apply_review(job_id, body.get("corrections") or {})
    except review.ReviewError as e:
        raise HTTPException(400, str(e)) from e
    return get_job(job_id)


@app.post("/jobs/{job_id}/entities/{entity_type}/unmerge")
def unmerge_entity(job_id: str, entity_type: str):
    """Split this job off a golden record it was wrongly matched to."""
    try:
        return {"entity_id": review.unmerge(job_id, entity_type)}
    except review.ReviewError as e:
        raise HTTPException(400, str(e)) from e


@app.get("/stats")
def stats():
    return review.stats()


@app.get("/jobs/{job_id}/pages/{page_number}")
def get_page_image(job_id: str, page_number: int):
    job = db.get(job_id)
    if not job:
        raise HTTPException(404, "no such job")
    try:
        # page_offset translates a child job's own page numbering back to the
        # shared original file's page numbering; 0 for a job that isn't a
        # segment of a larger upload.
        png = doc.render_png(job["path"], job["page_offset"] + page_number)
    except ValueError as e:
        raise HTTPException(404, str(e)) from e
    return Response(png, media_type="image/png")


def _validate_request(doc_type, schema) -> dict | None:
    """Shared doc_type/schema validation for /jobs and /jobs/batch - checked
    before anything is written to disk or enqueued."""
    parsed = None
    if schema:
        try:
            parsed = registry.validate_schema(schema)
        except ValueError as e:
            raise HTTPException(400, str(e)) from e
    if doc_type and doc_type != registry.UNKNOWN and parsed is None:
        try:
            registry.load(doc_type)
        except KeyError as e:
            raise HTTPException(400, str(e)) from e
    return parsed


async def _ingest(file: UploadFile, doc_type, parsed_schema) -> str:
    """Validate one file's type/size, save it, and enqueue it. Returns job_id."""
    ext = Path(file.filename or "").suffix.lower()
    if ext not in doc.SUPPORTED_EXTS:
        raise HTTPException(400, f"unsupported file type {ext!r} ({file.filename}); "
                                 f"allowed: {', '.join(sorted(doc.SUPPORTED_EXTS))}")

    job_id = uuid.uuid4().hex[:12]
    path = db.UPLOAD_DIR / f"{job_id}{ext}"
    db.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

    size = 0
    with open(path, "wb") as out:
        while chunk := await file.read(CHUNK):
            size += len(chunk)
            if size > MAX_UPLOAD_BYTES:
                out.close()
                path.unlink(missing_ok=True)
                raise HTTPException(413, f"{file.filename} exceeds {MAX_UPLOAD_BYTES // 1024 // 1024} MB")
            out.write(chunk)
    if size == 0:
        path.unlink(missing_ok=True)
        raise HTTPException(400, f"{file.filename} is empty")

    db.create(job_id, file.filename, path,
              requested_doc_type=doc_type,
              requested_schema=json.dumps(parsed_schema) if parsed_schema else None)
    return job_id
