"""Entity resolution: fold repeated extractions of the same real-world thing
(a vendor, a customer, a bank...) into one golden record, using the x-entity
hints a schema's own properties carry.

Matching is plain string logic - no embeddings, no ML - because a single
user's document set has at most a few thousand entities per type, and a
difflib.SequenceMatcher scan over a same-first-word block is instant at that
scale. Upgrade to rapidfuzz or an embedding match only if this measurably
misses; nothing here is pip-installed that isn't already stdlib.
"""

import difflib
import re
from typing import NamedTuple

from . import db

NAME_MATCH_THRESHOLD = 0.85

# Words that carry no identity, stripped before comparing two names so
# "Northwind Supplies Ltd" and "Northwind Supplies Ltd." (or "... GmbH") match
# on what actually distinguishes one company from another.
_LEGAL_SUFFIXES = {"ltd", "limited", "inc", "incorporated", "llc", "gmbh",
                   "corp", "corporation", "co", "plc", "sa", "ag", "bv"}


class EntityGroup(NamedTuple):
    type: str
    attrs: dict       # {"primary_name": "...", "tax_id": "...", ...}
    strong_keys: set  # subset of attrs' keys that are exact-match identifiers


def entity_groups(schema, fields) -> list[EntityGroup]:
    """Read x-entity hints off the schema and group the extracted fields that
    carry them. A group needs a non-empty primary_name to be resolvable - a
    schema with no x-entity hints at all (the generic fallback) yields none."""
    groups = {}
    for name, spec in schema.get("properties", {}).items():
        hint = spec.get("x-entity")
        value = fields.get(name)
        if not hint or value in (None, ""):
            continue
        g = groups.setdefault(hint["type"], {"attrs": {}, "strong_keys": set()})
        key = hint.get("key", "primary_name")
        g["attrs"][key] = value
        if hint.get("strong"):
            g["strong_keys"].add(key)
    return [EntityGroup(t, g["attrs"], g["strong_keys"])
            for t, g in groups.items() if g["attrs"].get("primary_name")]


def normalize(name) -> str:
    words = re.sub(r"[^\w\s]", " ", str(name or "").casefold()).split()
    return " ".join(w for w in words if w not in _LEGAL_SUFFIXES)


def match(group: EntityGroup, candidates: list[dict]):
    """candidates: [{"id": ..., "canonical_attrs": {...}}, ...], already
    filtered to group.type. Returns a matching id, or None to make a new one.

    Pure and DB-free on purpose: the match logic is what needs testing
    without a database, not the storage around it.
    """
    # A strong identifier (tax id, account number) matches regardless of name
    # similarity - that's the whole point of it - so it scans every candidate
    # of this type, never the name-block below.
    for key in group.strong_keys:
        want = normalize(group.attrs.get(key))
        if not want:
            continue
        for cand in candidates:
            if normalize(cand["canonical_attrs"].get(key)) == want:
                return cand["id"]

    norm = normalize(group.attrs.get("primary_name"))
    block = norm.split()[:1]
    blocked = [c for c in candidates
               if normalize(c["canonical_attrs"].get("primary_name")).split()[:1] == block]

    best_id, best_score = None, 0.0
    for cand in blocked:
        score = difflib.SequenceMatcher(
            None, norm, normalize(cand["canonical_attrs"].get("primary_name"))
        ).ratio()
        if score > best_score:
            best_id, best_score = cand["id"], score
    return best_id if best_score >= NAME_MATCH_THRESHOLD else None


def resolve(job_id, schema, fields, only=None) -> dict:
    """Match or create a golden record per entity group in `fields`, record
    the mention, and return {entity_type: entity_id}. `only` limits it to those
    entity types - a review re-resolves just the groups its corrections touched,
    so it can't undo an earlier manual unmerge of another group."""
    out = {}
    for group in entity_groups(schema, fields):
        if only is not None and group.type not in only:
            continue
        candidates = db.entities_of_type(group.type)
        entity_id = match(group, candidates)
        if entity_id is None:
            entity_id = db.create_entity(group.type, group.attrs)
        else:
            db.update_entity_attrs(entity_id, group.attrs)
        db.record_mention(entity_id, job_id, group.attrs)
        out[group.type] = entity_id
    return out
