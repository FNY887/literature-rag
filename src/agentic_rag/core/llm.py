from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, field
from typing import Any

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


class LLMClient:
    async def complete_json(self, system_prompt: str, user_prompt: str) -> dict[str, Any]: ...


@dataclass(slots=True)
class DeepSeekChatClient:
    settings: Settings
    _client: AsyncOpenAI = field(init=False, repr=False)
    max_retries: int = field(init=False, default=1)
    retry_base_delay_seconds: float = field(init=False, default=1.0)
    retry_max_delay_seconds: float = field(init=False, default=8.0)

    def __post_init__(self) -> None:
        if not self.settings.deepseek_api_key:
            raise ValueError("DEEPSEEK_API_KEY is required for DeepSeek-backed agent reasoning.")
        self._client = AsyncOpenAI(
            api_key=self.settings.deepseek_api_key,
            base_url=self.settings.deepseek_base_url,
            timeout=self.settings.deepseek_timeout_seconds,
        )
        self.max_retries = max(1, self.settings.deepseek_max_retries)
        self.retry_base_delay_seconds = max(0.0, self.settings.deepseek_retry_base_delay_seconds)
        self.retry_max_delay_seconds = max(
            self.retry_base_delay_seconds,
            self.settings.deepseek_retry_max_delay_seconds,
        )

    async def complete_json(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        for attempt in range(1, self.max_retries + 1):
            try:
                response = await self._client.chat.completions.create(
                    model=self.settings.deepseek_model,
                    temperature=1,
                    response_format={"type": "json_object"},
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                )
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
        raise RuntimeError("Unreachable retry state in DeepSeekChatClient.complete_json")
