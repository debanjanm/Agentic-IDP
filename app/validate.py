"""Do the extracted values agree with each other?

Rules live in the schema as `x-checks`, one plain-text rule each, so a new
document type brings its own with no Python change:

    "subtotal + tax_amount = total_amount"     amounts add up (within a cent or two)
    "sum(line_items.amount) = subtotal"        a table column adds up
    "invoice_date <= due_date"                 dates in order (also <)

A rule with a missing operand is skipped, not failed: a null total is not
evidence of a wrong one. A rule that doesn't parse is skipped too - a typo in a
user-supplied schema must never fail the job. (The seed schemas are tested so
their typos can't silently disable a check.)
"""

import re
from datetime import date

# ponytail: absolute, enough to absorb cent rounding. Go relative if totals reach the millions.
TOLERANCE = 0.02

_TERM = r"(?:sum\(\w+\.\w+\)|\w+)"
_SIDE = rf"\s*[+-]?\s*{_TERM}(?:\s*[+-]\s*{_TERM})*\s*"
_RULE = re.compile(rf"({_SIDE})(<=|<|=)({_SIDE})")
_PART = re.compile(r"([+-]?)\s*(?:sum\((\w+)\.(\w+)\)|(\w+))")


def check(schema, fields) -> dict:
    """Run the schema's x-checks. Never raises.

    {"ran": n, "skipped": n, "failed": [{"rule", "detail", "fields"}]} - `fields`
    are the scalar fields the rule mentions, so a reviewer knows what to look at."""
    ran, skipped, failed = 0, 0, []
    for rule in schema.get("x-checks") or []:
        outcome = _evaluate(rule, fields)
        if outcome is None:
            skipped += 1
            continue
        ran += 1
        ok, detail = outcome
        if not ok:
            names = dict.fromkeys(re.findall(r"[A-Za-z_]\w*", rule))
            failed.append({"rule": rule, "detail": detail, "fields": [
                n for n in names if n in fields and not isinstance(fields[n], (list, dict))]})
    return {"ran": ran, "skipped": skipped, "failed": failed}


def _evaluate(rule, fields):
    """(passed, detail), or None when the rule can't be checked."""
    m = _RULE.fullmatch(rule.strip()) if isinstance(rule, str) else None
    if not m:
        return None
    left, op, right = m.groups()
    if op == "=":
        a, b = _amount(left, fields), _amount(right, fields)
        return None if a is None or b is None else (abs(a - b) <= TOLERANCE, f"{a:.2f} vs {b:.2f}")
    a, b = _day(fields, left.strip()), _day(fields, right.strip())
    return None if a is None or b is None else ((a <= b) if op == "<=" else (a < b), f"{a} vs {b}")


def _number(v):
    return v if isinstance(v, (int, float)) and not isinstance(v, bool) else None  # True is not 1


def _amount(side, fields):
    """Value of 'a + sum(rows.col) - b', or None if any operand is missing."""
    total = 0.0
    for sign, arr, key, name in _PART.findall(side):
        if arr:
            rows = fields.get(arr)
            vals = [_number(r.get(key)) if isinstance(r, dict) else None for r in rows] \
                if isinstance(rows, list) and rows else [None]
            v = None if None in vals else sum(vals)
        else:
            v = _number(fields.get(name))
        if v is None:
            return None
        total += -v if sign == "-" else v
    return total


def _day(fields, name):
    try:
        return date.fromisoformat(fields.get(name))
    except (TypeError, ValueError):
        return None
