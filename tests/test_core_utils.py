from agentic_rag.core.utils import (
    dedupe_strings,
    extract_json_block,
    is_retryable,
    normalize_for_search,
    normalize_whitespace,
    retry_with_backoff,
)


def test_normalize_whitespace():
    assert normalize_whitespace("  hello   world  ") == "hello world"
    assert normalize_whitespace("a\n\nb\t\tc") == "a b c"


def test_normalize_for_search():
    assert normalize_for_search("Hello, World!") == "hello world"


def test_dedupe_strings():
    # dedupe is case-insensitive by normalized key
    assert dedupe_strings(["a", "a", "b", "A "]) == ["a", "b"]
    assert dedupe_strings(["x", "y", "z"]) == ["x", "y", "z"]


def test_extract_json_block_fenced():
    text = 'Some text\n```json\n{"key": "value"}\n```\nMore text'
    result = extract_json_block(text)
    assert result == {"key": "value"}


def test_extract_json_block_inline():
    text = 'Some text {"key": "value"} more text'
    result = extract_json_block(text)
    assert result == {"key": "value"}


def test_is_retryable_timeout():
    assert is_retryable(RuntimeError("Request timed out"))
    assert is_retryable(RuntimeError("connection error"))
    assert not is_retryable(RuntimeError("invalid format"))


def test_retry_with_backoff_retries_retryable_error():
    attempts = {"count": 0}

    def flaky():
        attempts["count"] += 1
        if attempts["count"] < 2:
            raise RuntimeError("connection error")
        return "ok"

    assert retry_with_backoff(flaky, max_retries=2, base_delay=0.0, max_delay=0.0) == "ok"
    assert attempts["count"] == 2
