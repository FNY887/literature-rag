from __future__ import annotations

import asyncio
import inspect
import re
from dataclasses import dataclass, field
from typing import Any, Protocol

from openai import AsyncOpenAI

from .config import Settings
from .utils import extract_json_block, is_retryable, retry_delay_seconds


class LLMResponseError(RuntimeError):
    def __init__(self, message: str, *, raw_response: str | None = None):
        super().__init__(message)
        self.raw_response = raw_response


def _status_code_from_exception(exc: Exception) -> int | None:
    status_code = getattr(exc, "status_code", None)
    if isinstance(status_code, int):
        return status_code
    match = re.search(r"error code:\s*(\d{3})", str(exc), flags=re.IGNORECASE)
    if match:
        return int(match.group(1))
    return None


def _is_retryable_exception(exc: Exception) -> bool:
    if isinstance(exc, LLMResponseError):
        return False
    status_code = _status_code_from_exception(exc)
    if status_code in {408, 409, 429}:
        return True
    if status_code is not None and status_code >= 500:
        return True
    lowered = str(exc).lower()
    retryable_fragments = (
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
    return any(fragment in lowered for fragment in retryable_fragments)


class LLMClient(Protocol):
    async def complete_json(self, system_prompt: str, user_prompt: str) -> dict[str, Any]: ...


@dataclass(slots=True)
class OpenAICompatibleChatClient:
    settings: Settings
    _client: AsyncOpenAI = field(init=False, repr=False)
    max_retries: int = field(init=False, default=1)
    retry_base_delay_seconds: float = field(init=False, default=1.0)
    retry_max_delay_seconds: float = field(init=False, default=8.0)
    _json_response_format_supported: bool = field(init=False, default=True, repr=False)
    _closed: bool = field(init=False, default=False, repr=False)

    def __post_init__(self) -> None:
        if not self.settings.chat_api_key:
            raise ValueError("CHAT_API_KEY is required for ask/chat agent reasoning.")
        self._client = AsyncOpenAI(
            api_key=self.settings.chat_api_key,
            base_url=self.settings.chat_base_url,
            timeout=self.settings.chat_timeout_seconds,
        )
        self.max_retries = max(1, self.settings.chat_max_retries)
        self.retry_base_delay_seconds = max(0.0, self.settings.chat_retry_base_delay_seconds)
        self.retry_max_delay_seconds = max(
            self.retry_base_delay_seconds,
            self.settings.chat_retry_max_delay_seconds,
        )

    async def _create_completion(self, system_prompt: str, user_prompt: str, *, use_response_format: bool):
        request: dict[str, Any] = {
            "model": self.settings.chat_model,
            "temperature": 1,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        if use_response_format:
            request["response_format"] = {"type": "json_object"}
        return await self._client.chat.completions.create(**request)

    async def complete_json(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        for attempt in range(1, self.max_retries + 1):
            try:
                try:
                    response = await self._create_completion(
                        system_prompt,
                        user_prompt,
                        use_response_format=self._json_response_format_supported,
                    )
                except Exception as exc:
                    if self._json_response_format_supported and _is_unsupported_response_format_exception(exc):
                        self._json_response_format_supported = False
                        response = await self._create_completion(
                            system_prompt,
                            user_prompt,
                            use_response_format=False,
                        )
                    else:
                        raise
                content = response.choices[0].message.content or "{}"
                try:
                    return extract_json_block(content)
                except Exception as exc:
                    raise LLMResponseError(
                        "The model response could not be parsed as JSON.",
                        raw_response=content,
                    ) from exc
            except Exception as exc:
                if attempt == self.max_retries or not _is_retryable_exception(exc):
                    raise
                await asyncio.sleep(
                    retry_delay_seconds(
                        attempt,
                        self.retry_base_delay_seconds,
                        self.retry_max_delay_seconds,
                    )
                )
        raise RuntimeError("Unreachable retry state in OpenAICompatibleChatClient.complete_json")

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        close_method = getattr(self._client, "close", None)
        if callable(close_method):
            result = close_method()
            if inspect.isawaitable(result):
                await result
            return
        aclose_method = getattr(self._client, "aclose", None)
        if callable(aclose_method):
            result = aclose_method()
            if inspect.isawaitable(result):
                await result


def _is_unsupported_response_format_exception(exc: Exception) -> bool:
    if _status_code_from_exception(exc) != 400:
        return False
    lowered = str(exc).lower()
    return "response_format" in lowered or "json_object" in lowered
