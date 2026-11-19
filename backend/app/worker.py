"""Background worker: polls the jobs table for queued work and runs the
pipeline. Runs as its own process (`python -m app.worker`), separate from the
API - a job that's already queued keeps running even if uvicorn restarts, and
a job that fails transiently gets a couple of automatic retries before it's
marked failed for good.

ponytail: one worker, one poll loop, no lease/heartbeat mechanism - a crash of
the worker process itself (not just uvicorn) can still strand a job at
status='running' forever. Add a lease column and a sweep for stale 'running'
rows if that actually happens; it's not worth the complexity pre-emptively.
"""

import json
import time

from ai import doc, pipeline, segment

from . import db, resolve

POLL_SECONDS = 2
MAX_ATTEMPTS = 2  # total tries before a job is marked failed


def main():
    print(f"worker: watching {db.DB_PATH}, polling every {POLL_SECONDS}s")
    while True:
        job = db.next_queued()
        if job is None:
            time.sleep(POLL_SECONDS)
            continue
        _run_job(job)


class _Steps:
    """Publishes the running step for the UI and times each one.

    Steps are the UI's stage names (reading, classifying, extracting,
    verifying, resolving). A step marked twice in a row (the worker marks
    classifying, then the pipeline marks it again) just adds up.
    """

    def __init__(self, job_id):
        self.job_id, self.marks = job_id, []

    def mark(self, step):
        self.marks.append((step, time.monotonic()))
        db.update(self.job_id, step=step)

    def seconds(self) -> dict:
        ends = [t for _, t in self.marks[1:]] + [time.monotonic()]
        out = {}
        for (step, t0), t1 in zip(self.marks, ends):
            out[step] = round(out.get(step, 0) + t1 - t0, 1)
        return out


def _run_job(job):
    """A caller-supplied doc_type/schema means "treat the whole upload as one
    document of this type" - segmentation only runs on auto-detect, matching
    the behavior main.py's create_job has always had."""
    job_id = job["id"]
    steps = _Steps(job_id)
    try:
        db.update(job_id, status="running")
        steps.mark("reading")
        pages = doc.load_pages(job["path"])
        doc_type = job.get("requested_doc_type")
        schema = json.loads(job["requested_schema"]) if job.get("requested_schema") else None
        if not (doc_type or schema):
            steps.mark("classifying")  # segment.split classifies every page
        segments = None if (doc_type or schema) else segment.split(pages)

        if segments is None or len(segments) == 1:
            cls = segments[0].cls if segments else None
            _run_pipeline(job_id, pages, doc_type=doc_type, schema=schema, cls=cls, steps=steps)
        else:
            _run_batch(job_id, job["path"], job["filename"], segments)
    except Exception as e:
        _fail_or_retry(job_id, job["attempts"], e)


def _fail_or_retry(job_id, attempts, exc):
    attempts += 1
    msg = f"{type(exc).__name__}: {exc}"
    if attempts < MAX_ATTEMPTS:
        db.update(job_id, status="queued", step="retry pending", attempts=attempts, error=msg)
    else:
        # step is left as-is on purpose: it names the stage that failed
        db.update(job_id, status="failed", attempts=attempts, error=msg, finished_at=db.now())


def _run_batch(parent_id, path, filename, segments):
    """One child job per detected sub-document, then roll the parent up.

    Safe to call again if the parent itself gets retried (some non-child step
    raised): a child already marked done is left alone rather than recreated
    or rerun, so a retry can't hit a duplicate-id error or redo finished work.
    """
    total_pages = sum(len(s.pages) for s in segments)
    db.update(parent_id, status="running", step="processing segments",
              doc_type="batch", pages=total_pages)

    child_ids = [f"{parent_id}-{i}" for i in range(1, len(segments) + 1)]
    for i, (child_id, seg) in enumerate(zip(child_ids, segments), 1):
        existing = db.get(child_id)
        if existing and existing["status"] == "done":
            continue
        if not existing:
            db.create(child_id, f"{filename} (part {i}/{len(segments)})", path,
                      parent_job_id=parent_id, page_offset=seg.pages[0].number - 1)
        try:
            _run_pipeline(child_id, seg.pages, cls=seg.cls)
        except Exception as e:
            # A child's failure doesn't retry the whole batch - it's recorded
            # on the child, and the parent rollup below reflects it.
            db.update(child_id, status="failed",
                      error=f"{type(e).__name__}: {e}", finished_at=db.now())

    rows = [db.get(c) for c in child_ids]
    db.update(
        parent_id,
        status="failed" if any(r["status"] == "failed" for r in rows) else "done",
        step="done",
        needs_review=int(any(r["needs_review"] for r in rows)),
        result={"segments": len(segments), "child_ids": child_ids,
                "doc_types": [r["doc_type"] for r in rows]},
        trace=[{"step": "segment", "count": len(segments),
                "doc_types": [r["doc_type"] for r in rows]}],
        finished_at=db.now(),
    )


def _run_pipeline(job_id, pages, doc_type=None, schema=None, cls=None, steps=None):
    """Run classify/extract/verify on one job's pages, resolve any entities
    the schema tags, and write the result.

    Raises on failure - retry/dead-letter bookkeeping happens one level up.
    Entity resolution lives here rather than in pipeline.py on purpose:
    pipeline.py stays a pure function of (pages, schema) -> fields, with no
    job_id or database of its own - the CLI and every test call it that way.
    """
    steps = steps or _Steps(job_id)
    db.update(job_id, status="running")
    result = pipeline.run_pages(pages, doc_type=doc_type, schema=schema, cls=cls, on_step=steps.mark)
    steps.mark("resolving")
    entities = resolve.resolve(job_id, result["schema"], result["fields"])
    db.update(
        job_id, status="done", step="done",
        doc_type=result["doc_type"], confidence=result["confidence"],
        needs_review=int(result.get("needs_review", False)),
        pages=result.get("pages"),
        result={**{k: v for k, v in result.items() if k not in ("trace", "schema")},
                "entities": entities, "timings": steps.seconds()},
        trace=result["trace"], finished_at=db.now(),
    )


if __name__ == "__main__":
    main()
