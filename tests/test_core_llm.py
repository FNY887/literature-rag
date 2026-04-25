import asyncio
from types import SimpleNamespace

from agentic_rag.core.config import Settings
from agentic_rag.core.llm import OpenAICompatibleChatClient


class FakeCompletions:
    def __init__(self, *, reject_response_format: bool = False):
        self.reject_response_format = reject_response_format
        self.requests: list[dict] = []

    async def create(self, **kwargs):
        self.requests.append(kwargs)
        if self.reject_response_format and "response_format" in kwargs:
            exc = RuntimeError("unsupported response_format json_object")
            exc.status_code = 400
            raise exc
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content='{"ok": true, "provider": "fake"}')
                )
            ]
        )


class FakeAsyncOpenAI:
    completions = FakeCompletions()
    init_kwargs: dict = {}
    close_calls = 0

    def __init__(self, **kwargs):
        type(self).init_kwargs = kwargs
        self.chat = SimpleNamespace(completions=type(self).completions)

    async def close(self):
        type(self).close_calls += 1


class FakeAsyncOpenAIWithAclose:
    completions = FakeCompletions()
    init_kwargs: dict = {}
    aclose_calls = 0

    def __init__(self, **kwargs):
        type(self).init_kwargs = kwargs
        self.chat = SimpleNamespace(completions=type(self).completions)

    async def aclose(self):
        type(self).aclose_calls += 1


def test_openai_compatible_chat_client_uses_chat_settings(monkeypatch):
    FakeAsyncOpenAI.completions = FakeCompletions()
    monkeypatch.setattr("agentic_rag.core.llm.AsyncOpenAI", FakeAsyncOpenAI)
    settings = Settings(
        chat_api_key="chat-key",
        chat_base_url="https://example.test/v1",
        chat_model="provider-model",
        chat_timeout_seconds=123,
    )

    client = OpenAICompatibleChatClient(settings)
    result = asyncio.run(client.complete_json("system", "user"))

    assert result == {"ok": True, "provider": "fake"}
    assert FakeAsyncOpenAI.init_kwargs == {
        "api_key": "chat-key",
        "base_url": "https://example.test/v1",
        "timeout": 123,
    }
    request = FakeAsyncOpenAI.completions.requests[0]
    assert request["model"] == "provider-model"
    assert request["response_format"] == {"type": "json_object"}
    assert request["messages"] == [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "user"},
    ]


def test_openai_compatible_chat_client_falls_back_without_response_format(monkeypatch):
    FakeAsyncOpenAI.completions = FakeCompletions(reject_response_format=True)
    monkeypatch.setattr("agentic_rag.core.llm.AsyncOpenAI", FakeAsyncOpenAI)
    settings = Settings(chat_api_key="chat-key")

    client = OpenAICompatibleChatClient(settings)
    result = asyncio.run(client.complete_json("system", "user"))
    second_result = asyncio.run(client.complete_json("system", "user"))

    assert result["ok"] is True
    assert second_result["ok"] is True
    requests = FakeAsyncOpenAI.completions.requests
    assert len(requests) == 3
    assert "response_format" in requests[0]
    assert "response_format" not in requests[1]
    assert "response_format" not in requests[2]


def test_openai_compatible_chat_client_close_uses_close(monkeypatch):
    FakeAsyncOpenAI.completions = FakeCompletions()
    FakeAsyncOpenAI.close_calls = 0
    monkeypatch.setattr("agentic_rag.core.llm.AsyncOpenAI", FakeAsyncOpenAI)
    settings = Settings(chat_api_key="chat-key")

    client = OpenAICompatibleChatClient(settings)
    asyncio.run(client.close())

    assert FakeAsyncOpenAI.close_calls == 1


def test_openai_compatible_chat_client_close_falls_back_to_aclose(monkeypatch):
    FakeAsyncOpenAIWithAclose.completions = FakeCompletions()
    FakeAsyncOpenAIWithAclose.aclose_calls = 0
    monkeypatch.setattr("agentic_rag.core.llm.AsyncOpenAI", FakeAsyncOpenAIWithAclose)
    settings = Settings(chat_api_key="chat-key")

    client = OpenAICompatibleChatClient(settings)
    asyncio.run(client.close())

    assert FakeAsyncOpenAIWithAclose.aclose_calls == 1
