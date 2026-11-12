"""Score the pipeline against labelled documents.

    .venv/backend/bin/python eval.py DIR -n 20        # your own labelled folder / SROIE
    .venv/backend/bin/python eval.py --cord 20        # CORD v2 receipts from HuggingFace

DIR holds documents (pdf/png/jpg/tif) each next to a label file of the same name,
`X.json` or `X.txt`, containing a JSON object. Keys are our schema's field names;
SROIE's own keys (company, date, total) are renamed on load, so an unpacked SROIE
"entities" folder works as-is. Fields missing from a label are not scored.

--cord needs `datasets` and Pillow in the backend env, and downloads the dataset on
first use: .venv/backend/bin/pip install datasets pillow
CORD labels no merchant or date, so it scores subtotal, tax, total and item count.

Per field it reports exact-match before the verify pass and after it, so the
verifier's net effect is visible (fixed vs broke). It also reports how often a
wrong document was flagged for review, and how often a correct one was flagged.
A one-page job is ~3 minutes on an M2/16GB: keep -n small.
"""

import argparse
import json
import re
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

from app import llm, pipeline, registry
from app.doc import SUPPORTED_EXTS

SROIE_KEYS = {"company": "merchant_name", "date": "transaction_date", "total": "total_amount"}
DATE_FORMATS = ("%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y", "%d.%m.%Y", "%d/%m/%y", "%d-%m-%y",
                "%d %b %Y", "%d-%b-%Y", "%d %B %Y", "%b %d %Y", "%b %d, %Y")  # day first: SROIE is Malaysian
TOLERANCE = 0.005


# --- loading ------------------------------------------------------------------

def load_dir(folder) -> list:
    cases = []
    for doc in sorted(Path(folder).iterdir()):
        label = next((doc.with_suffix(e) for e in (".json", ".txt") if doc.with_suffix(e).is_file()), None)
        if doc.suffix.lower() in SUPPORTED_EXTS and label:
            truth = json.loads(label.read_text(errors="ignore"))
            cases.append((doc, {SROIE_KEYS.get(k, k): v for k, v in truth.items()}))
    return cases


def load_cord(n, tmp) -> list:
    from datasets import load_dataset  # not a project dependency, see docstring

    cases = []
    for i, row in enumerate(load_dataset("naver-clova-ix/cord-v2", split=f"test[:{n}]")):
        gt = json.loads(row["ground_truth"])["gt_parse"]
        menu = gt.get("menu") or []
        sub, total = gt.get("sub_total"), gt.get("total")
        sub, total = sub if isinstance(sub, dict) else {}, total if isinstance(total, dict) else {}
        truth = {"subtotal": sub.get("subtotal_price"), "tax_amount": sub.get("tax_price"),
                 "total_amount": total.get("total_price"), "items": 1 if isinstance(menu, dict) else len(menu)}
        path = Path(tmp) / f"cord{i:03}.png"
        row["image"].convert("RGB").save(path)
        cases.append((path, {k: v for k, v in truth.items() if v not in (None, "")}))
    return cases


# --- scoring ------------------------------------------------------------------

def _norm(s) -> str:
    return re.sub(r"\W+", " ", str(s).casefold()).strip()


def _iso(s) -> str:
    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(str(s).strip(), fmt).date().isoformat()
        except ValueError:
            pass
    return _norm(s)


def same(field, schema, truth, pred) -> bool:
    if field == "items":  # compared by row count only
        return isinstance(pred, list) and len(pred) == int(truth)
    if pred is None:
        return False
    types = schema.get("properties", {}).get(field, {}).get("type", "string")
    if "number" in types or "integer" in types:
        t, p = pipeline._number(str(truth)), pipeline._number(str(pred))
        return t is not None and p is not None and abs(t - p) <= TOLERANCE
    if field.endswith("_date"):
        return _iso(truth) == _iso(pred)
    return _norm(truth) == _norm(pred)


def score_case(truth, result, expected_type=None) -> dict:
    """One document: per field (right before verify, right after verify)."""
    schema = result["schema"]
    extracted = next((t["fields"] for t in result["trace"] if t["step"] == "extract"), {})
    rows = {f: (same(f, schema, t, extracted.get(f)), same(f, schema, t, result["fields"].get(f)))
            for f, t in truth.items()}
    return {"rows": rows, "flagged": result["needs_review"],
            "rules_failed": bool(result["validation"]["failed"]),
            "type_ok": None if expected_type is None else result["doc_type"] == expected_type}


def summarize(cases) -> dict:
    ok = [c for c in cases if "error" not in c]
    fields = {}
    for c in ok:
        for f, (before, after) in c["rows"].items():
            n, b, a = fields.get(f, (0, 0, 0))
            fields[f] = (n + 1, b + before, a + after)
    wrong = [c for c in ok if not all(a for _, a in c["rows"].values())]
    right = [c for c in ok if c not in wrong]
    typed = [c for c in ok if c["type_ok"] is not None]
    return {
        "documents": len(cases), "errors": len(cases) - len(ok),
        "fields": fields,
        "verifier_fixed": sum(1 for c in ok for b, a in c["rows"].values() if a and not b),
        "verifier_broke": sum(1 for c in ok for b, a in c["rows"].values() if b and not a),
        "wrong_docs": len(wrong), "wrong_flagged": sum(c["flagged"] for c in wrong),
        "wrong_caught_by_rules": sum(c["rules_failed"] for c in wrong),
        "right_docs": len(right), "right_flagged": sum(c["flagged"] for c in right),
        "type_ok": f"{sum(c['type_ok'] for c in typed)}/{len(typed)}" if typed else "not tested",
    }


def show(s) -> None:
    print(f"\n{'field':<18}{'n':>4}{'before verify':>16}{'after verify':>14}")
    for f, (n, b, a) in sorted(s["fields"].items()):
        print(f"{f:<18}{n:>4}{b / n:>15.0%} {a / n:>13.0%}")
    print(f"\ndocuments {s['documents']} (errors {s['errors']}), document type correct: {s['type_ok']}")
    print(f"verifier: fixed {s['verifier_fixed']} fields, broke {s['verifier_broke']}")
    print(f"documents with a wrong field: {s['wrong_docs']} - flagged for review {s['wrong_flagged']}, "
          f"caught by a failed rule {s['wrong_caught_by_rules']}")
    print(f"documents all correct: {s['right_docs']} - flagged anyway {s['right_flagged']} (review load)")


# --- run ----------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="score the pipeline against labelled documents")
    ap.add_argument("folder", nargs="?", help="documents + label files (see eval.py docstring)")
    ap.add_argument("--cord", type=int, metavar="N", help="use the first N CORD v2 test receipts")
    ap.add_argument("-n", type=int, default=20, help="documents to run from FOLDER (default 20)")
    ap.add_argument("--doc-type", default="receipt", help="what the documents are (default receipt)")
    ap.add_argument("--known-type", action="store_true", help="skip classification, saves ~45s each")
    ap.add_argument("--out", help="also write the per-document results to this JSON file")
    args = ap.parse_args()
    if bool(args.folder) == bool(args.cord):
        ap.error("give either FOLDER or --cord N")
    registry.load(args.doc_type)  # fail now on a typo, not after the first 3 minutes

    ok, why = llm.available()
    if not ok:
        sys.exit(why)
    with tempfile.TemporaryDirectory() as tmp:
        cases = load_cord(args.cord, tmp) if args.cord else load_dir(args.folder)[:args.n]
        if not cases:
            sys.exit("no labelled documents found")
        results = []
        for i, (path, truth) in enumerate(cases, 1):
            t0 = time.time()
            try:
                res = pipeline.run(path, doc_type=args.doc_type if args.known_type else None)
                row = score_case(truth, res, None if args.known_type else args.doc_type)
            except Exception as e:  # one bad document must not lose the run so far
                row = {"error": f"{type(e).__name__}: {e}"}
            results.append({"name": path.name, **row})
            hit = "ERROR" if "error" in row else f"{sum(a for _, a in row['rows'].values())}/{len(row['rows'])} right"
            print(f"[{i}/{len(cases)}] {path.name}: {hit} ({time.time() - t0:.0f}s)", flush=True)
    show(summarize(results))
    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
