"""Ollama /api/chat wrapper.

This is the only module that knows which inference provider we use. Swapping to
a hosted API means rewriting this file and nothing else.

The key mechanism: Ollama's `format` takes a raw JSON Schema and constrains
decoding to it, so a schema-valid dict comes back without any output parsing.
"""

import json
import os

import httpx

HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
CLASSIFY_MODEL = os.environ.get("IDP_CLASSIFY_MODEL", "qwen2.5vl:3b")
EXTRACT_MODEL = os.environ.get("IDP_EXTRACT_MODEL", "qwen2.5vl:3b")
# Text-only, no vision - used when a page already has real text (native PDF
# layer or OCR), so classification skips a slow vision call entirely.
TEXT_CLASSIFY_MODEL = os.environ.get("IDP_TEXT_CLASSIFY_MODEL", "qwen3:4b")
TIMEOUT = float(os.environ.get("IDP_TIMEOUT", "600"))


class LLMError(RuntimeError):
    pass


def chat(prompt, schema=None, images=None, model=None, system=None, num_ctx=8192):
    """One turn. Returns a dict when `schema` is given, else the raw string.

    images: list of base64 PNG strings.
    """
    msg = {"role": "user", "content": prompt}
    if images:
        msg["images"] = list(images)
    messages = ([{"role": "system", "content": system}] if system else []) + [msg]

    body = {
        "model": model or EXTRACT_MODEL,
        "messages": messages,
        "stream": False,
        "think": False,
        "options": {"temperature": 0, "num_ctx": num_ctx},
    }
    if schema:
        body["format"] = schema

    content = _post(body)
    if not schema:
        return content
    try:
        return json.loads(content)
    except json.JSONDecodeError as e:
        # Constrained decoding makes this near-impossible, but a truncated
        # response (context overflow) can still get here.
        raise LLMError(f"model returned non-JSON: {content[:300]}") from e


def _post(body):
    last = None
    for _ in range(2):  # one retry: cold model loads can blow the first timeout
        try:
            r = httpx.post(f"{HOST}/api/chat", json=body, timeout=TIMEOUT)
            r.raise_for_status()
            return r.json()["message"]["content"]
        except (httpx.TimeoutException, httpx.ConnectError) as e:
            last = e
    raise LLMError(f"Ollama unreachable or too slow at {HOST}: {last}")


def available() -> tuple[bool, str]:
    """(reachable, detail) - used by the API health check and the test skip."""
    try:
        tags = httpx.get(f"{HOST}/api/tags", timeout=5).json()
        names = {m["name"] for m in tags.get("models", [])}
        missing = {CLASSIFY_MODEL, EXTRACT_MODEL, TEXT_CLASSIFY_MODEL} - names
        if missing:
            return False, f"missing model(s): {', '.join(sorted(missing))} - run: ollama pull {sorted(missing)[0]}"
        return True, "ok"
    except Exception as e:
        return False, f"Ollama not reachable at {HOST}: {e}"
