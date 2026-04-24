from __future__ import annotations

import json
import re
import time
from typing import Any


class RetryableError(Exception):
    pass


RETRYABLE_FRAGMENTS = (
    "timed out",
    "timeout",
    "temporarily unavailable",
    "temporary failure",
    "engine is currently overloaded",
    "engine_overloaded_error",
    "overloaded",
    "rate limit",
    "too many requests",
    "connection error",
    "connection reset",
    "connection aborted",
    "server error",
)


def is_retryable(exc: Exception) -> bool:
    status_code = getattr(exc, "status_code", None)
    if isinstance(status_code, int):
        if status_code in {408, 409, 429} or status_code >= 500:
            return True
    lowered = str(exc).lower()
    return any(fragment in lowered for fragment in RETRYABLE_FRAGMENTS)


def retry_delay_seconds(attempt: int, base: float = 1.0, limit: float = 8.0) -> float:
    delay = base * (2 ** max(attempt - 1, 0))
    return min(delay, limit)


def retry_with_backoff(
    fn,
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 8.0,
):
    last_exc: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            return fn()
        except Exception as exc:
            last_exc = exc
            if attempt == max_retries or not is_retryable(exc):
                raise
            time.sleep(retry_delay_seconds(attempt, base_delay, max_delay))
    assert last_exc is not None
    raise last_exc


def dedupe_strings(values: list[str]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for value in values:
        cleaned = value.strip()
        if not cleaned:
            continue
        key = " ".join(re.sub(r"[^\w\u4e00-\u9fff]+", " ", cleaned.lower()).split())
        if key in seen:
            continue
        seen.add(key)
        deduped.append(cleaned)
    return deduped


def extract_json_block(text: str) -> dict[str, Any]:
    fenced_match = re.search(r"```json\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    if fenced_match:
        return json.loads(fenced_match.group(1))
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("The model response did not contain a JSON object.")
    return json.loads(text[start : end + 1])


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _collapse_single_char_tokens(tokens: list[str]) -> list[str]:
    collapsed: list[str] = []
    run: list[str] = []

    def flush_run() -> None:
        nonlocal run
        if not run:
            return
        if len(run) >= 2:
            collapsed.append("".join(run))
        else:
            collapsed.extend(run)
        run = []

    for token in tokens:
        if re.fullmatch(r"[0-9A-Za-z]", token):
            run.append(token)
            continue
        flush_run()
        collapsed.append(token)
    flush_run()
    return collapsed


def _strip_inline_math_markup(text: str) -> str:
    def replace_math(match: re.Match[str]) -> str:
        math_text = match.group(1)
        math_text = re.sub(r"\\[A-Za-z]+", " ", math_text)
        math_text = re.sub(r"[_^{}]", " ", math_text)
        math_text = re.sub(r"[^0-9a-zA-Z\u4e00-\u9fff\s]+", " ", math_text)
        math_tokens = normalize_whitespace(math_text).split()
        math_tokens = _collapse_single_char_tokens(math_tokens)
        return " ".join(math_tokens)

    return re.sub(r"\$(.+?)\$", replace_math, text)


def clean_title_text(text: str) -> str:
    cleaned = _strip_inline_math_markup(text)
    cleaned = re.sub(r"\(\s*([ivxlcdm]+)\s*\)", r"\1", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\\[A-Za-z]+", " ", cleaned)
    cleaned = re.sub(r"[_^{}]", " ", cleaned)
    return normalize_whitespace(cleaned)


def normalize_for_search(text: str) -> str:
    lowered = text.lower()
    lowered = re.sub(r"[^0-9a-zA-Z\u4e00-\u9fff\s\-]", " ", lowered)
    return normalize_whitespace(lowered)


def normalize_title(text: str) -> str:
    lowered = clean_title_text(text).lower()
    lowered = re.sub(r"[^0-9a-zA-Z\u4e00-\u9fff\s]+", " ", lowered)
    return normalize_whitespace(lowered)
