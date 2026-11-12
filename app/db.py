"""Job storage: stdlib sqlite3, no ORM."""

import json
import os
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

DATA_DIR = Path(os.environ.get("IDP_DATA_DIR", Path(__file__).parent.parent / "data"))
UPLOAD_DIR = DATA_DIR / "uploads"
DB_PATH = DATA_DIR / "idp.db"

# ponytail: one global lock serializes pipeline runs - a 3B VLM on 16GB unified
# memory can only do one document at a time anyway. Swap for a queue + worker
# pool when inference moves off this box.
# Kept separate from _write_lock on purpose: the worker holds RUN_LOCK for the
# whole pipeline and writes progress rows while holding it, and threading.Lock
# is not reentrant.
RUN_LOCK = threading.Lock()

_conn = None
_conn_lock = threading.Lock()
_write_lock = threading.Lock()

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs(
  id TEXT PRIMARY KEY,
  parent_job_id TEXT,
  filename TEXT NOT NULL,
  path TEXT NOT NULL,
  page_offset INTEGER DEFAULT 0,
  status TEXT NOT NULL,
  step TEXT,
  attempts INTEGER DEFAULT 0,
  requested_doc_type TEXT,
  requested_schema TEXT,
  doc_type TEXT,
  confidence REAL,
  needs_review INTEGER DEFAULT 0,
  pages INTEGER,
  result TEXT,
  trace TEXT,
  error TEXT,
  created_at TEXT NOT NULL,
  finished_at TEXT,
  reviewed_at TEXT
);

-- Golden records (app/resolve.py). Brand-new tables, unlike jobs, so
-- CREATE TABLE IF NOT EXISTS alone is enough - no _migrate() needed.
CREATE TABLE IF NOT EXISTS entities(
  id TEXT PRIMARY KEY,
  type TEXT NOT NULL,
  canonical_attrs TEXT NOT NULL,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS entity_mentions(
  id TEXT PRIMARY KEY,
  entity_id TEXT NOT NULL,
  job_id TEXT NOT NULL,
  attrs TEXT NOT NULL,
  created_at TEXT NOT NULL
);
"""

# Kept separate from SCHEMA and run only after _migrate(): on a jobs table
# that predates parent_job_id, indexing that column before ALTER TABLE has
# added it fails outright, even though CREATE TABLE IF NOT EXISTS itself
# would have silently no-op'd.
_INDEXES = """
CREATE INDEX IF NOT EXISTS idx_jobs_parent ON jobs(parent_job_id);
CREATE INDEX IF NOT EXISTS idx_entities_type ON entities(type);
CREATE INDEX IF NOT EXISTS idx_mentions_entity ON entity_mentions(entity_id);
"""

# CREATE TABLE IF NOT EXISTS only helps a brand-new db file - it does nothing
# to a jobs table that already exists from before these columns were added.
# ponytail: ALTER TABLE ADD COLUMN, not a migrations framework - fine for a
# table that only ever grows optional nullable/defaulted columns.
_NEW_COLUMNS = {
    "parent_job_id": "TEXT",
    "page_offset": "INTEGER DEFAULT 0",
    "attempts": "INTEGER DEFAULT 0",
    "requested_doc_type": "TEXT",
    "requested_schema": "TEXT",
    "reviewed_at": "TEXT",
}


def _migrate(conn):
    existing = {row["name"] for row in conn.execute("PRAGMA table_info(jobs)")}
    for col, decl in _NEW_COLUMNS.items():
        if col not in existing:
            conn.execute(f"ALTER TABLE jobs ADD COLUMN {col} {decl}")


def connect():
    global _conn
    with _conn_lock:
        if _conn is None:
            UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
            _conn = sqlite3.connect(DB_PATH, check_same_thread=False)
            _conn.row_factory = sqlite3.Row
            _conn.executescript(SCHEMA)
            _migrate(_conn)
            _conn.executescript(_INDEXES)
            _conn.commit()
        return _conn


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def create(job_id, filename, path, parent_job_id=None, page_offset=0,
           requested_doc_type=None, requested_schema=None):
    """requested_doc_type/requested_schema persist the caller's override so a
    separate worker process can pick this row up later and honor it - Phase 0
    passed these as plain function arguments, which only works when the same
    process both enqueues and runs the job."""
    with _write_lock:
        connect().execute(
            "INSERT INTO jobs(id, parent_job_id, filename, path, page_offset, "
            "requested_doc_type, requested_schema, status, step, created_at) "
            "VALUES(?,?,?,?,?,?,?,'queued','queued',?)",
            (job_id, parent_job_id, filename, str(path), page_offset,
             requested_doc_type, requested_schema, now()),
        )
        connect().commit()


def update(job_id, **cols):
    for k in ("result", "trace"):
        if k in cols and not isinstance(cols[k], (str, type(None))):
            cols[k] = json.dumps(cols[k], default=str)
    sets = ", ".join(f"{k}=?" for k in cols)
    with _write_lock:
        connect().execute(f"UPDATE jobs SET {sets} WHERE id=?", (*cols.values(), job_id))
        connect().commit()


def get(job_id):
    row = connect().execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
    return _row(row) if row else None


def recent(limit=25):
    rows = connect().execute(
        "SELECT id, parent_job_id, filename, status, step, doc_type, confidence, needs_review, "
        "pages, error, created_at, finished_at FROM jobs ORDER BY created_at DESC LIMIT ?",
        (limit,),
    ).fetchall()
    return [dict(r) for r in rows]


def next_queued():
    """Oldest queued job, or None. The worker's poll loop calls this - a
    single worker process, so no claim/lease mechanism is needed to avoid two
    workers grabbing the same row."""
    row = connect().execute(
        "SELECT * FROM jobs WHERE status='queued' ORDER BY created_at LIMIT 1"
    ).fetchone()
    return _row(row) if row else None


def children(parent_job_id):
    """Sub-document jobs a batch upload split into (app/segment.py). Empty for a normal job."""
    rows = connect().execute(
        "SELECT * FROM jobs WHERE parent_job_id=? ORDER BY created_at", (parent_job_id,)
    ).fetchall()
    return [_row(r) for r in rows]


def review_queue():
    """Finished jobs the pipeline flagged for a human that nobody has reviewed.
    Batch parents are skipped - their child jobs carry the fields."""
    rows = connect().execute(
        "SELECT id, filename, doc_type, confidence, result, finished_at FROM jobs "
        "WHERE status='done' AND needs_review=1 AND reviewed_at IS NULL AND doc_type != 'batch' "
        "ORDER BY finished_at"
    ).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        result = json.loads(d.pop("result") or "{}")
        d["low_confidence_fields"] = result.get("low_confidence_fields") or []
        d["failed_checks"] = [f["rule"] for f in (result.get("validation") or {}).get("failed", [])]
        out.append(d)
    return out


def done_jobs():
    """Every finished non-batch job with parsed result and trace - what the
    quality dashboard (app/review.py stats) aggregates over."""
    rows = connect().execute(
        "SELECT id, doc_type, confidence, needs_review, reviewed_at, result, trace FROM jobs "
        "WHERE status='done' AND doc_type != 'batch'"
    ).fetchall()
    return [_row(r) for r in rows]


# --- entities: golden records (app/resolve.py) --------------------------------

def entities_of_type(etype):
    """Every golden record of one type - app/resolve.py blocks/matches against
    this list in Python rather than in SQL; a single user's document set is
    small enough that this needs no index beyond the plain type lookup."""
    rows = connect().execute("SELECT id, canonical_attrs FROM entities WHERE type=?", (etype,)).fetchall()
    return [{"id": r["id"], "canonical_attrs": json.loads(r["canonical_attrs"])} for r in rows]


def create_entity(etype, attrs) -> str:
    entity_id = uuid.uuid4().hex[:12]
    with _write_lock:
        connect().execute(
            "INSERT INTO entities(id, type, canonical_attrs, created_at, updated_at) VALUES(?,?,?,?,?)",
            (entity_id, etype, json.dumps(attrs, default=str), now(), now()),
        )
        connect().commit()
    return entity_id


def update_entity_attrs(entity_id, attrs):
    """Most-recent-non-null-wins per attribute - the same trust-the-latest-
    pass rule pipeline.verify already uses, not a survivorship rule engine."""
    row = connect().execute("SELECT canonical_attrs FROM entities WHERE id=?", (entity_id,)).fetchone()
    current = json.loads(row["canonical_attrs"]) if row else {}
    merged = {**current, **{k: v for k, v in attrs.items() if v not in (None, "")}}
    with _write_lock:
        connect().execute(
            "UPDATE entities SET canonical_attrs=?, updated_at=? WHERE id=?",
            (json.dumps(merged, default=str), now(), entity_id),
        )
        connect().commit()


def record_mention(entity_id, job_id, attrs):
    """Provenance: which job contributed which raw values to this entity."""
    with _write_lock:
        connect().execute(
            "INSERT INTO entity_mentions(id, entity_id, job_id, attrs, created_at) VALUES(?,?,?,?,?)",
            (uuid.uuid4().hex[:12], entity_id, job_id, json.dumps(attrs, default=str), now()),
        )
        connect().commit()


def delete_mentions(job_id, entity_ids):
    """Drop this job's mentions of the given entities (a review is redoing them)."""
    if not entity_ids:
        return
    marks = ",".join("?" * len(entity_ids))
    with _write_lock:
        connect().execute(
            f"DELETE FROM entity_mentions WHERE job_id=? AND entity_id IN ({marks})", (job_id, *entity_ids)
        )
        connect().commit()


def mention_for(job_id, entity_id):
    row = connect().execute(
        "SELECT * FROM entity_mentions WHERE job_id=? AND entity_id=?", (job_id, entity_id)
    ).fetchone()
    if not row:
        return None
    d = dict(row)
    d["attrs"] = json.loads(d["attrs"])
    return d


def mention_count(entity_id) -> int:
    return connect().execute(
        "SELECT COUNT(*) FROM entity_mentions WHERE entity_id=?", (entity_id,)
    ).fetchone()[0]


def move_mention(mention_id, entity_id):
    with _write_lock:
        connect().execute("UPDATE entity_mentions SET entity_id=? WHERE id=?", (entity_id, mention_id))
        connect().commit()


def rebuild_entity(entity_id):
    """Recompute a golden record from the mentions it has left.

    canonical_attrs is the fold of its mentions, oldest first, newest non-empty
    value winning - the same rule update_entity_attrs applies one mention at a
    time - so removing a mention can't leave its values behind. An entity with
    no mentions left is deleted."""
    rows = connect().execute(
        "SELECT attrs FROM entity_mentions WHERE entity_id=? ORDER BY created_at, rowid", (entity_id,)
    ).fetchall()
    with _write_lock:
        if not rows:
            connect().execute("DELETE FROM entities WHERE id=?", (entity_id,))
        else:
            merged = {}
            for r in rows:
                merged.update({k: v for k, v in json.loads(r["attrs"]).items() if v not in (None, "")})
            connect().execute(
                "UPDATE entities SET canonical_attrs=?, updated_at=? WHERE id=?",
                (json.dumps(merged, default=str), now(), entity_id),
            )
        connect().commit()


def entity_counts():
    """Per entity type: golden records, and the mentions feeding them."""
    rows = connect().execute(
        "SELECT e.type AS type, COUNT(DISTINCT e.id) AS entities, COUNT(m.id) AS mentions "
        "FROM entities e LEFT JOIN entity_mentions m ON m.entity_id = e.id GROUP BY e.type"
    ).fetchall()
    return [dict(r) for r in rows]


def get_entity(entity_id):
    row = connect().execute("SELECT * FROM entities WHERE id=?", (entity_id,)).fetchone()
    if not row:
        return None
    d = dict(row)
    d["canonical_attrs"] = json.loads(d["canonical_attrs"])
    return d


def entity_mentions(entity_id):
    rows = connect().execute(
        "SELECT * FROM entity_mentions WHERE entity_id=? ORDER BY created_at", (entity_id,)
    ).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        d["attrs"] = json.loads(d["attrs"])
        out.append(d)
    return out


def _row(row) -> dict:
    d = dict(row)
    for k in ("result", "trace"):
        if d.get(k):
            try:
                d[k] = json.loads(d[k])
            except json.JSONDecodeError:
                pass
    d["needs_review"] = bool(d.get("needs_review"))
    return d
