"""The schema registry: doc_type -> JSON Schema, loaded from app/schemas/*.json.

Does double duty - the filenames define the classifier's label set, and the
schema itself constrains extraction.
"""

import json
from pathlib import Path

import jsonschema

SCHEMA_DIR = Path(__file__).parent / "schemas"
UNKNOWN = "unknown"
GENERIC = "_generic"  # fallback extraction target, never a classifier label

# Ollama's constrained decoding does not handle these reliably; a schema using
# them fails deep inside the model with an opaque error, so reject it up front.
UNSUPPORTED = ("$ref", "oneOf", "allOf", "anyOf", "not", "if", "patternProperties")


def doc_types() -> list[str]:
    """Real, classifiable document types - filenames starting with '_' (like
    the generic fallback) are registry entries but not classification targets."""
    return sorted(p.stem for p in SCHEMA_DIR.glob("*.json") if not p.stem.startswith("_"))


def labels() -> list[str]:
    """Classifier label set: every registered type, plus an escape hatch."""
    return doc_types() + [UNKNOWN]


def load(doc_type: str) -> dict:
    path = SCHEMA_DIR / f"{doc_type}.json"
    if not path.is_file() or path.parent != SCHEMA_DIR:
        raise KeyError(f"unknown doc_type {doc_type!r}; have {doc_types()}")
    return json.loads(path.read_text())


def validate_schema(schema) -> dict:
    """Check a user-supplied schema before it reaches the model.

    Trust boundary: this runs on request input, so it stays strict and gives a
    usable error message instead of letting the failure surface mid-job.
    """
    if isinstance(schema, str):
        try:
            schema = json.loads(schema)
        except json.JSONDecodeError as e:
            raise ValueError(f"schema is not valid JSON: {e}") from e
    if not isinstance(schema, dict):
        raise ValueError("schema must be a JSON object")
    try:
        jsonschema.Draft202012Validator.check_schema(schema)
    except jsonschema.SchemaError as e:
        raise ValueError(f"invalid JSON Schema: {e.message}") from e
    if schema.get("type") != "object" or not schema.get("properties"):
        raise ValueError("schema must be an object schema with a non-empty 'properties'")
    found = _find_unsupported(schema)
    if found:
        raise ValueError(
            f"schema uses {', '.join(sorted(found))}, which Ollama's constrained decoding "
            "does not support. Use plain types and nullable unions like "
            '{"type": ["string", "null"]} instead.'
        )
    for name, spec in schema["properties"].items():
        _check_entity_hint(name, spec.get("x-entity") if isinstance(spec, dict) else None)
    return schema


def _check_entity_hint(name, hint):
    if hint is None:
        return
    ok = (isinstance(hint, dict) and isinstance(hint.get("type"), str) and hint["type"]
          and isinstance(hint.get("key", ""), str) and isinstance(hint.get("strong", False), bool))
    if not ok:
        raise ValueError(f"property {name!r}: x-entity must look like "
                         '{"type": "Vendor", "key": "primary_name", "strong": false}')


def _find_unsupported(node, found=None) -> set:
    found = set() if found is None else found
    if isinstance(node, dict):
        found |= {k for k in node if k in UNSUPPORTED}
        for v in node.values():
            _find_unsupported(v, found)
    elif isinstance(node, list):
        for v in node:
            _find_unsupported(v, found)
    return found
