from __future__ import annotations

import http.client
import json
import socket
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

from agentic_rag.core.config import Settings
from agentic_rag.core.utils import is_retryable


def _rerank_retryable(exc: Exception) -> bool:
    if isinstance(exc, (http.client.RemoteDisconnected, http.client.HTTPException)):
        return True
    if isinstance(exc, (ConnectionError, ConnectionResetError, ConnectionAbortedError)):
        return True
    if isinstance(exc, urllib.error.HTTPError):
        if exc.code in {408, 409, 429}:
            return True
        if exc.code >= 500:
            return True
    if isinstance(exc, urllib.error.URLError):
        return True
    if isinstance(exc, TimeoutError | socket.timeout):
        return True
    return is_retryable(exc)


def _response_excerpt(raw_body: str, *, limit: int = 600) -> str:
    compact = " ".join(raw_body.split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3].rstrip() + "..."


def _raise_api_error_if_present(body: dict[str, Any], raw_body: str) -> None:
    error = body.get("error")
    if isinstance(error, dict):
        code = str(error.get("code", "")).strip()
        message = str(error.get("message", "")).strip()
        if code or message:
            raise RuntimeError(f"{code}: {message}".strip(": "))
        raise RuntimeError(f"Rerank API error: {_response_excerpt(raw_body)}")


def _extract_results(body: dict[str, Any], raw_body: str) -> list[Any]:
    output = body.get("output")
    if isinstance(output, dict):
        raw_results = output.get("results")
        if isinstance(raw_results, list):
            return raw_results

    raw_results = body.get("results")
    if isinstance(raw_results, list):
        return raw_results

    raw_results = body.get("data")
    if isinstance(raw_results, list):
        return raw_results

    code = str(body.get("code", "")).strip()
    message = str(body.get("message", "")).strip()
    if code or message:
        raise RuntimeError(f"{code}: {message}".strip(": "))
    raise ValueError(f"Missing output/results in rerank response: {_response_excerpt(raw_body)}")


@dataclass(slots=True)
class DashScopeRerankClient:
    settings: Settings

    def rerank(
        self,
        *,
        query: str,
        documents: list[str],
        top_n: int | None = None,
        instruct: str | None = None,
    ) -> list[tuple[int, float]]:
        if not documents:
            return []
        if not self.settings.dashscope_api_key:
            raise ValueError("DASHSCOPE_API_KEY is required for rerank generation.")

        payload: dict[str, object] = {
            "model": self.settings.rerank_model,
            "documents": documents,
            "query": query,
        }
        if top_n is not None:
            payload["top_n"] = top_n
        if instruct:
            payload["instruct"] = instruct

        request = urllib.request.Request(
            url=f"{self.settings.dashscope_rerank_base_url.rstrip('/')}/reranks",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.settings.dashscope_api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        last_exc: Exception | None = None
        for attempt in range(1, max(1, self.settings.rerank_max_retries) + 1):
            try:
                with urllib.request.urlopen(
                    request,
                    timeout=max(1.0, self.settings.rerank_timeout_seconds),
                ) as response:
                    raw_body = response.read().decode("utf-8")
                body = json.loads(raw_body)
                if not isinstance(body, dict):
                    raise ValueError("Unexpected rerank response payload.")
                _raise_api_error_if_present(body, raw_body)
                raw_results = _extract_results(body, raw_body)
                parsed: list[tuple[int, float]] = []
                for item in raw_results:
                    if not isinstance(item, dict):
                        continue
                    index = item.get("index")
                    score = item.get("relevance_score", item.get("score"))
                    if not isinstance(index, int):
                        continue
                    try:
                        parsed.append((index, float(score)))
                    except (TypeError, ValueError):
                        continue
                if not parsed:
                    raise ValueError("No valid rerank results were returned.")
                return parsed
            except Exception as exc:
                last_exc = exc
                if attempt == max(1, self.settings.rerank_max_retries) or not _rerank_retryable(exc):
                    break
                from agentic_rag.core.utils import retry_delay_seconds

                time.sleep(
                    retry_delay_seconds(
                        attempt,
                        self.settings.rerank_retry_base_delay_seconds,
                        self.settings.rerank_retry_max_delay_seconds,
                    )
                )

        assert last_exc is not None
        raise last_exc
