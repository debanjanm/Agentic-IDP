"""Human in the loop: apply a reviewer's corrections, undo a wrong entity
match, and summarize quality.

A review writes back to the same places the pipeline wrote - the job's result
and its entity mentions - so the corrected values are the ones the rest of the
system sees. The model's own confidence numbers are never overwritten: they are
the signal the quality dashboard measures, and a correction is recorded in the
job's trace instead (what it was, what it became).
"""

import json
from collections import Counter
from statistics import mean

from . import db, pipeline, registry, resolve, validate


class ReviewError(ValueError):
    pass


def schema_for(job) -> dict:
    """The schema this job was extracted against (worker.py doesn't store it)."""
    if job.get("requested_schema"):
        return json.loads(job["requested_schema"])
    try:
        return registry.load(job["doc_type"])
    except KeyError:
        return registry.load(registry.GENERIC)


def apply_review(job_id, corrections) -> None:
    """Apply {field: new value} corrections and mark the job reviewed.

    Empty corrections mean "checked, all fine". Only scalar fields are
    editable. A correction to an entity-tagged field (a vendor name, a tax id)
    moves this job's mention to whichever golden record now matches it.
    """
    job = db.get(job_id)
    result = (job or {}).get("result")
    if not isinstance(result, dict) or "fields" not in result:
        raise ReviewError("this job has no extracted fields to review")

    schema = schema_for(job)
    props = schema.get("properties", {})
    fields = result["fields"]
    changes = {}
    for name, raw in corrections.items():
        if name not in fields or isinstance(fields[name], (list, dict)):
            raise ReviewError(f"{name!r} is not an editable field")
        new = pipeline._coerce(raw, props.get(name, {}))
        if new is None and str(raw).strip().lower() not in pipeline.NULL_WORDS:
            raise ReviewError(f"{name}: {raw!r} is not a valid value")
        if new != fields[name]:
            changes[name] = {"from": fields[name], "to": new}
    for name, ch in changes.items():
        fields[name] = ch["to"]

    touched = {props[n]["x-entity"]["type"] for n in changes if "x-entity" in props.get(n, {})}
    entities = dict(result.get("entities") or {})
    if touched:
        old_ids = [entities.pop(t) for t in touched if t in entities]
        db.delete_mentions(job_id, old_ids)
        for eid in old_ids:
            db.rebuild_entity(eid)  # forget this job's old values, or delete if it was the only mention
        entities.update(resolve.resolve(job_id, schema, fields, only=touched))
    result["entities"] = entities
    result["validation"] = validate.check(schema, fields)

    db.update(job_id, result=result, trace=(job["trace"] or []) + [{"step": "review", "corrected": changes}],
              reviewed_at=db.now())


def unmerge(job_id, etype) -> str:
    """This job was matched to an existing golden record it doesn't belong to:
    split it off into a new record of its own. Returns the new entity id.

    Two jobs unmerged from the same record become two separate new records -
    there is no merge yet (see the README backlog)."""
    job = db.get(job_id)
    result = job["result"] if job and isinstance(job["result"], dict) else {}
    old = (result.get("entities") or {}).get(etype)
    if not old:
        raise ReviewError(f"this job has no linked {etype!r} entity")
    mention = db.mention_for(job_id, old)
    if not mention or db.mention_count(old) < 2:
        raise ReviewError("no other documents share this entity, so there is nothing to split from")

    new = db.create_entity(etype, mention["attrs"])
    db.move_mention(mention["id"], new)
    db.rebuild_entity(old)
    result["entities"][etype] = new
    db.update(job_id, result=result,
              trace=(job["trace"] or []) + [{"step": "unmerge", "type": etype, "from": old, "to": new}])
    return new


def stats() -> dict:
    """Quality signals that are actually knowable without ground truth.

    Review rate and confidence describe what the model was unsure about; entity
    match rate shows how often a document joined an existing record vs started a
    new one; the reviewed section is the only real error measure - what humans
    had to correct."""
    # a "done" row without a parsed result has nothing to measure
    jobs = [j for j in db.done_jobs() if isinstance(j["result"], dict)]

    def summarize(subset):
        confs = [j["confidence"] for j in subset if j["confidence"] is not None]
        fconfs = [c for j in subset for c in (j["result"].get("field_confidence") or {}).values()]
        return {
            "jobs": len(subset),
            "review_rate": sum(j["needs_review"] for j in subset) / len(subset) if subset else None,
            "avg_classifier_confidence": mean(confs) if confs else None,
            "avg_field_confidence": mean(fconfs) if fconfs else None,
        }

    by_type = {}
    for j in jobs:
        by_type.setdefault(j["doc_type"], []).append(j)

    entities = [
        {**c, "matched": c["mentions"] - c["entities"],
         "match_rate": (c["mentions"] - c["entities"]) / c["mentions"] if c["mentions"] else None}
        for c in db.entity_counts()
    ]

    reviewed = [j for j in jobs if j["reviewed_at"]]
    corrected = Counter()
    with_corrections = 0
    for j in reviewed:
        fixes = {f for t in (j["trace"] or []) if t["step"] == "review" for f in t["corrected"]}
        with_corrections += bool(fixes)
        corrected.update(fixes)

    return {
        **summarize(jobs),
        "by_type": [{"doc_type": t, **summarize(js)} for t, js in sorted(by_type.items())],
        "entities": entities,
        "reviewed": {"jobs": len(reviewed), "with_corrections": with_corrections,
                     "fields": dict(corrected.most_common())},
    }
