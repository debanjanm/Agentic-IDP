"""Streamlit client. Talks to the API over HTTP and holds no pipeline logic."""

import html
import json
import os
import time

import pandas as pd
import requests
import streamlit as st

API = os.environ.get("IDP_API", "http://localhost:8000")
POLL_SECONDS = 2
POLL_TIMEOUT = 900

st.set_page_config(page_title="Agentic IDP", page_icon="📄", layout="wide")

# Neobrutalism: thick black borders, hard offset shadows (no blur), flat
# saturated colors. Colors come from .streamlit/config.toml; a theme file can't
# express borders or shadows, so those live here. Selectors are data-testid
# hooks, which Streamlit keeps stable - the generated .st-emotion-cache-* class
# names are not, so none are used.
NEO_CSS = """<style>
:root { --ink: #171717; --yellow: #F4EBC8; --magenta: #EFB3CF; --teal: #A9D6CB; --cyan: #B3DDEB;
        --green: #A6E3B8; --red: #F09A93;
        --border: 3px solid var(--ink); --shadow: 4px 4px 0 var(--ink); }
::selection { background: var(--magenta); color: var(--ink) }

.stApp { background: var(--yellow) }
[data-testid="stHeader"] { background: transparent }

/* the page is one big bordered panel floating on the yellow, like a browser window */
[data-testid="stAppViewBlockContainer"] {
    background: #fff; border: 4px solid var(--ink); border-radius: 18px;
    box-shadow: 7px 7px 0 var(--ink); margin: 1.5rem auto 3rem; padding: 2rem 2.5rem 2.5rem;
    max-width: 1300px; }

[data-testid="stSidebar"] { background: var(--teal); border-right: 4px solid var(--ink) }
[data-testid="stSidebar"] [data-testid="stAlert"] { background: #fff }

h1, h2, h3 { font-family: "Arial Black", "Helvetica Neue", Helvetica, Arial, sans-serif;
             font-weight: 900; letter-spacing: -.02em; color: var(--ink) }
h1 { display: inline-block; background: var(--magenta); border: var(--border);
     box-shadow: var(--shadow); border-radius: 12px; padding: .15em .55em !important; margin-bottom: .5rem }
label, [data-testid="stWidgetLabel"] p { font-weight: 700; color: var(--ink) }
[data-testid="stCaptionContainer"] { color: var(--ink); opacity: .75 }

/* buttons: pills that press down onto their own shadow */
[data-testid="baseButton-secondary"], [data-testid="baseButton-primary"] {
    border: var(--border); border-radius: 999px; box-shadow: 3px 3px 0 var(--ink);
    color: var(--ink); font-weight: 800; transition: transform .08s, box-shadow .08s }
[data-testid="baseButton-secondary"] { background: #fff }
[data-testid="baseButton-primary"] { background: var(--magenta) }
[data-testid="baseButton-secondary"]:hover, [data-testid="baseButton-primary"]:hover {
    background: var(--cyan); color: var(--ink); border-color: var(--ink);
    transform: translate(-2px, -2px); box-shadow: 5px 5px 0 var(--ink) }
[data-testid="baseButton-secondary"]:active, [data-testid="baseButton-primary"]:active {
    transform: translate(3px, 3px); box-shadow: 0 0 0 var(--ink) }

/* view switcher: radio buttons drawn as pills, the selected one filled */
[data-testid="stRadio"] [role="radiogroup"] { gap: .6rem }
label[data-baseweb="radio"] { border: var(--border); border-radius: 999px; padding: .2rem 1rem; margin: 0;
    background: #fff; box-shadow: 3px 3px 0 var(--ink); font-weight: 800 }
label[data-baseweb="radio"] > div:first-child { display: none }
label[data-baseweb="radio"]:has(input:checked) { background: var(--magenta) }

/* inputs */
[data-baseweb="input"], [data-baseweb="textarea"], [data-baseweb="select"] > div {
    border: var(--border) !important; border-radius: 10px !important; background: #fff !important;
    box-shadow: 2px 2px 0 var(--ink) }
[data-testid="stCheckbox"] span[role="checkbox"] { border: var(--border); border-radius: 4px }

[data-testid="stFileUploaderDropzone"] {
    background: var(--teal); border: var(--border); border-radius: 16px; box-shadow: var(--shadow) }
[data-testid="stFileUploaderDropzone"] * { color: var(--ink) }

[data-testid="stMetric"] {
    background: #fff; border: var(--border); border-radius: 12px; box-shadow: var(--shadow); padding: .8rem 1rem }
[data-testid="stMetricValue"] { font-weight: 900 }

[data-testid="stAlert"] > div { border: var(--border); border-radius: 12px; box-shadow: var(--shadow); color: var(--ink) }

/* expanders and st.status share this markup */
[data-testid="stExpander"] details {
    background: #fff; border: var(--border); border-radius: 12px; box-shadow: var(--shadow) }
[data-testid="stExpander"] summary { font-weight: 800 }
[data-testid="stExpander"] summary:hover { background: var(--yellow) }

[data-testid="stProgress"] [data-baseweb="progress-bar"] > div {
    background: #fff; border: var(--border); border-radius: 999px; height: 1.1rem; overflow: hidden }
[data-testid="stProgress"] [data-baseweb="progress-bar"] > div > div { background: var(--magenta) }

[data-testid="stDataFrame"] { border: var(--border); border-radius: 10px; box-shadow: var(--shadow); overflow: hidden }
[data-testid="stImage"] img { border: var(--border); box-shadow: var(--shadow); border-radius: 6px }
hr { border: 0; border-top: var(--border) }
</style>"""


st.markdown(NEO_CSS, unsafe_allow_html=True)


def api(path, **kw):
    r = requests.request(kw.pop("method", "GET"), f"{API}{path}", timeout=30, **kw)
    r.raise_for_status()
    return r.json()


# The pipeline's stages, in order. The keys are the exact `step` values
# app/worker.py publishes while a job runs, so the tracker needs no mapping.
STAGES = [
    ("reading", "Read pages", "Render each page. Uses the PDF's own text, or OCR when a page has none."),
    ("classifying", "Identify document", "Decide what kind of document this is."),
    ("extracting", "Extract fields", "Pull out the values the schema asks for."),
    ("verifying", "Double-check", "Re-read every value against the page and fix mistakes."),
    ("validating", "Validate", "Check the numbers and dates agree with each other (totals add up, dates in order)."),
    ("resolving", "Link entities", "Match vendors and customers to ones seen before, or add new ones."),
]
STAGE_KEYS = [k for k, _, _ in STAGES]
STAGE_NAMES = {k: n for k, n, _ in STAGES}

STAGE_CSS = """<style>
@keyframes idp-pulse { 0%, 100% { opacity: .45 } 50% { opacity: 1 } }
.idp-row { display: flex; gap: .9rem; align-items: center; padding: .55rem 0;
           border-bottom: 2px solid var(--ink); color: var(--ink) }
.idp-row:last-child { border-bottom: 0 }
.idp-row.pending { opacity: .5 }
.idp-dot { width: 1.7rem; height: 1.7rem; display: grid; place-items: center; font-weight: 900;
           border: 3px solid var(--ink); border-radius: 6px; box-shadow: 2px 2px 0 var(--ink); background: #fff }
.idp-row.done .idp-dot { background: var(--green) }
.idp-row.failed .idp-dot { background: var(--red) }
.idp-row.running .idp-dot { background: var(--cyan); animation: idp-pulse 1s infinite }
.idp-row.running .idp-name { animation: idp-pulse 1s infinite }
.idp-name { font-weight: 800; min-width: 12rem }
.idp-note { flex: 1 }
.idp-time { font-weight: 800; font-variant-numeric: tabular-nums }
</style>"""


def stage_rows(step, notes=None, timings=None, failed=False):
    """HTML checklist of the stages. `step` is the job's current step; "done"
    marks every stage complete, `failed` marks the current one as the one that
    died. Everything interpolated is escaped - notes come from model output."""
    notes, timings = notes or {}, timings or {}
    cur = STAGE_KEYS.index(step) if step in STAGE_KEYS else len(STAGES) if step == "done" else -1
    rows = []
    for i, (key, name, blurb) in enumerate(STAGES):
        state = "done" if i < cur else ("failed" if failed else "running") if i == cur else "pending"
        dot = {"done": "✓", "running": "●", "failed": "✗", "pending": "○"}[state]
        secs = f"{timings[key]:.0f}s" if key in timings else ""
        rows.append(
            f'<div class="idp-row {state}"><span class="idp-dot">{dot}</span>'
            f'<span class="idp-name">{i + 1} · {name}</span>'
            f'<span class="idp-note">{html.escape(notes.get(key) or blurb)}</span>'
            f'<span class="idp-time">{secs}</span></div>'
        )
    return STAGE_CSS + "".join(rows)


def stage_notes(job):
    """One-line outcome per stage, read back from what the pipeline stored."""
    r, trace, notes = job["result"], job.get("trace") or [], {}
    fields = r.get("fields", {})
    sources = {"text-layer": "PDF text layer", "ocr": "OCR"}

    if r.get("text_source"):
        used = " + ".join(sources.get(x, x) for x in r["text_source"].split("+"))
        notes["reading"] = f"{job['pages']} page(s) · text from {used}"

    c = next((t for t in trace if t["step"] == "classify"), None)
    if c and c.get("via"):
        lang = f" · {c['language']}" if c.get("language") else ""
        notes["classifying"] = f"{c['doc_type']} · {c['confidence']:.0%} sure · {c['model']} ({c['via']}){lang}"
    elif c:
        notes["classifying"] = f"{c['doc_type']} · type chosen by you"

    filled = sum(v not in (None, "", []) for v in fields.values())
    rows = ", ".join(f"{len(v)} × {k}" for k, v in fields.items() if isinstance(v, list) and v)
    generic = "generic fallback · " if any(t["step"] == "fallback" for t in trace) else ""
    notes["extracting"] = f"{generic}{filled} of {len(fields)} fields filled" + (f" · {rows}" if rows else "")

    fixed = set()
    for t in trace:
        if t["step"] == "correct":
            fixed |= set(t["applied"])
        elif t["step"] == "verify" and "note" in t:
            fixed |= set(t["fields"])
    low = r.get("low_confidence_fields") or []
    notes["verifying"] = (
        f"{len(r.get('field_confidence') or {})} values checked · "
        + (f"corrected {', '.join(sorted(fixed))}" if fixed else "no corrections")
        + (f" · unsure: {', '.join(low)}" if low else "")
    )

    v = r.get("validation")
    if v:
        failed = "; ".join(f"{f['rule']} ({f['detail']})" for f in v["failed"])
        notes["validating"] = (f"{v['ran']} check(s) run"
                               + (f" · {v['skipped']} skipped (a value was missing)" if v["skipped"] else "")
                               + (f" · FAILED: {failed}" if failed else " · all passed"))

    ents = r.get("entities") or {}
    notes["resolving"] = (", ".join(f"{t} #{i[:6]}" for t, i in ents.items())
                          if ents else "no entity fields in this schema")
    return notes


def follow(job_id):
    """Poll a queued/running job, animating the stage checklist, until it
    finishes (or POLL_TIMEOUT). Returns the latest job."""
    started = time.time()
    with st.status(f"Job {job_id}", expanded=True) as status:
        tracker, bar = st.empty(), st.progress(0.0)
        while True:
            job = api(f"/jobs/{job_id}")
            step = job["step"] or job["status"]
            if job["status"] not in ("queued", "running") or time.time() - started > POLL_TIMEOUT:
                break
            i = STAGE_KEYS.index(step) if step in STAGE_KEYS else 0
            name = STAGE_NAMES.get(step, step)
            tracker.markdown(stage_rows(step), unsafe_allow_html=True)
            bar.progress(i / len(STAGES), text=f"Stage {i + 1} of {len(STAGES)} · {name}" if step in STAGE_KEYS else name)
            status.update(label=f"Job {job_id} · {name} · {int(time.time() - started)}s")
            time.sleep(POLL_SECONDS)
        status.update(label=f"Job {job_id} · {job['status']}",
                      state="complete" if job["status"] == "done" else "error", expanded=False)
    return job


def pct(x):
    return "-" if x is None else f"{x:.0%}"


def corrected_fields(job):
    """Fields a reviewer changed, read back from the job's trace."""
    return sorted({f for t in job.get("trace") or [] if t["step"] == "review" for f in t["corrected"]})


def review_view():
    """Jobs the pipeline flagged for a human. Edits and entity unmerges write back to the API."""
    queue = api("/review")["jobs"]
    if not queue:
        st.success("Nothing waiting for review.")
        return
    def why(j):
        parts = ([f"unsure: {', '.join(j['low_confidence_fields'])}"] if j["low_confidence_fields"] else []) \
            + [f"failed: {rule}" for rule in j["failed_checks"]]
        return " · ".join(parts) or "document type"

    labels = {j["id"]: f"{j['filename']} · {j['doc_type']} · {why(j)}" for j in queue}
    job_id = st.selectbox(f"{len(queue)} waiting for review", list(labels), format_func=labels.get)
    job = api(f"/jobs/{job_id}")
    fields, conf = job["result"]["fields"], job["result"].get("field_confidence", {})
    for f in (job["result"].get("validation") or {}).get("failed", []):
        st.warning(f"Check failed: {f['rule']} ({f['detail']})")

    left, right = st.columns([1, 1.4])
    with left:
        page = st.number_input("Page", 1, max(job["pages"] or 1, 1), 1, key=f"rv-page-{job_id}")
        st.image(f"{API}/jobs/{job_id}/pages/{page}", use_column_width=True)
    with right:
        st.caption("Edit any value that's wrong, then save. Line items aren't editable yet.")
        before = pd.DataFrame([{"field": k, "value": "" if v is None else str(v), "confidence": conf.get(k)}
                               for k, v in fields.items() if not isinstance(v, (list, dict))])
        after = st.data_editor(
            before, key=f"rv-edit-{job_id}", hide_index=True, use_container_width=True,
            disabled=["field", "confidence"],
            column_config={"confidence": st.column_config.ProgressColumn(
                "confidence", min_value=0.0, max_value=1.0, format="%.2f")},
        )
        changed = after[after["value"] != before["value"]]
        corrections = dict(zip(changed["field"], changed["value"]))
        label = f"Save {len(corrections)} correction(s) & mark reviewed" if corrections else "Looks right - mark reviewed"
        if st.button(label, type="primary", key=f"rv-save-{job_id}"):
            try:
                api(f"/jobs/{job_id}/review", method="POST", json={"corrections": corrections})
            except requests.HTTPError as e:
                st.error(e.response.json().get("detail", str(e)))
            else:
                st.rerun()

        for etype, info in (job.get("entity_info") or {}).items():
            st.markdown(f"**{etype}:** {info['name']} · linked to {info['documents']} document(s)")
            if info["documents"] > 1 and st.button(f"Not the same {etype} - split this document off",
                                                   key=f"rv-unmerge-{job_id}-{etype}"):
                try:
                    api(f"/jobs/{job_id}/entities/{etype}/unmerge", method="POST")
                except requests.HTTPError as e:
                    st.error(e.response.json().get("detail", str(e)))
                else:
                    st.rerun()


def quality_view():
    """What the model was unsure about, how entity matching is behaving, and what humans had to fix."""
    s = api("/stats")
    if not s["jobs"]:
        st.info("No finished jobs yet.")
        return
    a, b, c, d = st.columns(4)
    a.metric("Jobs finished", s["jobs"])
    b.metric("Sent to review", pct(s["review_rate"]))
    c.metric("Avg classifier confidence", pct(s["avg_classifier_confidence"]))
    d.metric("Avg field confidence", pct(s["avg_field_confidence"]))

    st.subheader("By document type")
    by_type = pd.DataFrame(s["by_type"])
    for col in ("review_rate", "avg_classifier_confidence", "avg_field_confidence"):
        by_type[col] = by_type[col].map(pct)
    st.dataframe(by_type.rename(columns={
        "doc_type": "type", "review_rate": "sent to review",
        "avg_classifier_confidence": "classifier conf.", "avg_field_confidence": "field conf."}),
        hide_index=True, use_container_width=True)

    st.subheader("Entity matching")
    if s["entities"]:
        ents = pd.DataFrame(s["entities"])
        ents["match_rate"] = ents["match_rate"].map(pct)
        st.dataframe(ents.rename(columns={
            "entities": "records", "mentions": "documents", "matched": "joined existing",
            "match_rate": "match rate"}), hide_index=True, use_container_width=True)
        st.caption("A type that should repeat (vendors) with a low match rate suggests the name threshold is "
                   "too strict; a type that should stay distinct with a high one, too loose.")
    else:
        st.caption("No entities yet.")

    st.subheader("Reviewed by humans")
    r = s["reviewed"]
    if not r["jobs"]:
        st.caption("Nothing reviewed yet - this is the only real error measure, so review a few.")
        return
    x, y = st.columns(2)
    x.metric("Jobs reviewed", r["jobs"])
    y.metric("Needed a correction", pct(r["with_corrections"] / r["jobs"]))
    if r["fields"]:
        st.dataframe(pd.DataFrame(list(r["fields"].items()), columns=["most corrected field", "times"]),
                     hide_index=True, use_container_width=True)


with st.sidebar:
    st.header("Setup")
    API = st.text_input("Backend", API)
    try:
        health = api("/health")
        doc_types = api("/schemas")["doc_types"]
    except Exception as e:
        st.error(f"Backend unreachable: {e}")
        st.caption("Start it with: `uvicorn app.main:app --port 8000`")
        st.stop()

    if health["ok"]:
        st.success(f"Ollama ready · {health['extract_model']}")
    else:
        st.error(health["detail"])

    st.divider()
    choice = st.selectbox("Document type", ["auto-detect"] + doc_types,
                          help="Auto-detect runs the classifier first.")
    doc_type = None if choice == "auto-detect" else choice

    edit = st.checkbox("Edit the extraction schema")
    schema_text = None
    if edit:
        base = doc_type or doc_types[0]
        schema_text = st.text_area("JSON Schema", json.dumps(api(f"/schemas/{base}"), indent=2),
                                   height=300)

st.title("📄 Agentic IDP")
st.caption("Classify a document, then extract entities against a JSON Schema. "
           "Everything runs locally on Ollama.")

view = st.radio("View", ["Process", "Review queue", "Quality"], horizontal=True, label_visibility="collapsed")
if view == "Review queue":
    review_view()
    st.stop()
if view == "Quality":
    quality_view()
    st.stop()

upload = st.file_uploader("Document", type=["pdf", "png", "jpg", "jpeg", "tif", "tiff"])

if upload and st.button("Process", type="primary"):
    data = {}
    if doc_type:
        data["doc_type"] = doc_type
    if schema_text:
        data["schema"] = schema_text
    try:
        job_id = api("/jobs", method="POST", files={"file": (upload.name, upload.getvalue())},
                     data=data)["job_id"]
    except requests.HTTPError as e:
        st.error(e.response.json().get("detail", str(e)))
        st.stop()

    st.session_state["job_id"] = job_id
    st.query_params["job"] = job_id  # so a reload keeps the result, and it is linkable

job_id = st.session_state.get("job_id") or st.query_params.get("job")
if job_id:
    try:
        job = api(f"/jobs/{job_id}")
    except requests.HTTPError:
        st.warning(f"Job {job_id} not found.")
        st.stop()

    if job["status"] in ("queued", "running"):
        job = follow(job_id)  # also what a page reload mid-run lands on

    if job["status"] in ("error", "failed"):
        st.error(job["error"])
        with st.expander("Pipeline stages", expanded=True):
            st.markdown(stage_rows(job["step"], failed=True), unsafe_allow_html=True)
    elif job["status"] != "done":
        st.info("Still running - reload this page to keep watching.")
    else:
        result = job["result"]
        fields = result.get("fields", {})
        conf = result.get("field_confidence", {})

        a, b, c = st.columns(3)
        a.metric("Document type", job["doc_type"] or "-")
        b.metric("Classifier confidence", f"{(job['confidence'] or 0):.0%}")
        c.metric("Pages", job["pages"] or "-")
        sig = result.get("confidence_signals") or {}
        why = [f"{f} ({', '.join(w.replace('_', ' ') for w in sig[f])})" if f in sig else f
               for f in result.get("low_confidence_fields") or []] + [
            f"check failed: {f['rule']}" for f in (result.get("validation") or {}).get("failed", [])]
        if job.get("reviewed_at"):
            fixed = corrected_fields(job)
            st.success("Reviewed" + (f" · corrected: {', '.join(fixed)}" if fixed else " · no corrections needed"))
        elif job["needs_review"]:
            st.warning("Needs review — " + (", ".join(why) if why else "low classifier confidence")
                       + " · open the Review queue")

        if "fields" in result:  # a batch parent has no stages of its own
            with st.expander("Pipeline stages", expanded=True):
                st.markdown(stage_rows("done", stage_notes(job), result.get("timings")),
                            unsafe_allow_html=True)

        left, right = st.columns([1, 1.4])
        with left:
            page = st.number_input("Page", 1, max(job["pages"] or 1, 1), 1)
            st.image(f"{API}/jobs/{job_id}/pages/{page}", use_column_width=True)
        with right:
            scalars = {k: v for k, v in fields.items() if not isinstance(v, (list, dict))}
            st.dataframe(
                pd.DataFrame(
                    [{"field": k, "value": v, "confidence": conf.get(k)} for k, v in scalars.items()]
                ),
                use_container_width=True, hide_index=True,
                column_config={"confidence": st.column_config.ProgressColumn(
                    "confidence", min_value=0.0, max_value=1.0, format="%.2f")},
            )
            for k, v in fields.items():
                if isinstance(v, list) and v:
                    st.caption(k)
                    st.dataframe(pd.DataFrame(v), use_container_width=True, hide_index=True)

            st.download_button("Download JSON", json.dumps(result, indent=2),
                               f"{job_id}.json", "application/json")

        with st.expander("Raw output"):
            st.json(result)
        with st.expander("Agent trace"):
            st.json(job["trace"])

st.divider()
with st.expander("Recent jobs"):
    jobs = api("/jobs")["jobs"]
    if jobs:
        st.dataframe(pd.DataFrame(jobs), use_container_width=True, hide_index=True)
    else:
        st.caption("Nothing processed yet.")
