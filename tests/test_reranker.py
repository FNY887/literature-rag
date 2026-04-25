import json
import http.client
from types import SimpleNamespace

import pytest

from agentic_rag.core.config import Settings
from agentic_rag.query.reranker import DashScopeRerankClient


class FakeUrlopen:
    def __init__(self, payload: dict):
        self.payload = payload

    def __call__(self, request, timeout):
        del request, timeout
        return self

    def __enter__(self):
        return SimpleNamespace(read=lambda: json.dumps(self.payload).encode("utf-8"))

    def __exit__(self, exc_type, exc, traceback):
        return False


class FlakyUrlopen:
    def __init__(self):
        self.calls = 0

    def __call__(self, request, timeout):
        del request, timeout
        self.calls += 1
        if self.calls == 1:
            raise http.client.RemoteDisconnected("Remote end closed connection without response")
        return self

    def __enter__(self):
        return SimpleNamespace(
            read=lambda: json.dumps({
                "results": [
                    {"index": 0, "relevance_score": 0.9},
                ]
            }).encode("utf-8")
        )

    def __exit__(self, exc_type, exc, traceback):
        return False


def test_dashscope_reranker_accepts_top_level_results(monkeypatch):
    monkeypatch.setattr(
        "agentic_rag.query.reranker.urllib.request.urlopen",
        FakeUrlopen({
            "results": [
                {"index": 1, "relevance_score": 0.9},
                {"index": 0, "score": 0.5},
            ]
        }),
    )
    settings = Settings(dashscope_api_key="key")
    client = DashScopeRerankClient(settings)

    results = client.rerank(query="q", documents=["a", "b"])

    assert results == [(1, 0.9), (0, 0.5)]


def test_dashscope_reranker_retries_remote_disconnected(monkeypatch):
    fake_urlopen = FlakyUrlopen()
    monkeypatch.setattr("agentic_rag.query.reranker.urllib.request.urlopen", fake_urlopen)
    monkeypatch.setattr("agentic_rag.query.reranker.time.sleep", lambda delay: None)
    settings = Settings(dashscope_api_key="key", rerank_max_retries=2)
    client = DashScopeRerankClient(settings)

    results = client.rerank(query="q", documents=["a"])

    assert fake_urlopen.calls == 2
    assert results == [(0, 0.9)]


def test_dashscope_reranker_surfaces_nested_api_error(monkeypatch):
    monkeypatch.setattr(
        "agentic_rag.query.reranker.urllib.request.urlopen",
        FakeUrlopen({
            "error": {
                "code": "InvalidParameter",
                "message": "Range of input length should be [1, 8192]",
            }
        }),
    )
    settings = Settings(dashscope_api_key="key")
    client = DashScopeRerankClient(settings)

    with pytest.raises(RuntimeError, match="Range of input length"):
        client.rerank(query="q", documents=["a"])
