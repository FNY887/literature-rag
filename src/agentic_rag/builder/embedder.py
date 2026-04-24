from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np
from openai import OpenAI

from agentic_rag.core.config import Settings
from agentic_rag.core.models import ChunkRecord
from agentic_rag.core.utils import retry_with_backoff


def chunk_embedding_input(chunk: ChunkRecord) -> str:
    parts = [f"Title: {chunk.title}"]
    if chunk.section_hint:
        parts.append(f"Section: {chunk.section_hint}")
    if chunk.keywords_hint:
        parts.append(f"Keywords: {chunk.keywords_hint}")
    parts.append(f"Text: {chunk.text}")
    return "\n".join(parts)


@dataclass(slots=True)
class DashScopeEmbeddingClient:
    settings: Settings
    max_retries: int = 3
    _client: OpenAI = field(init=False, repr=False)
    max_batch_size: int = 10
    query_cache_size: int = 128
    _query_cache: ClassVar[OrderedDict[tuple[str, str, int, str], np.ndarray]] = OrderedDict()

    def __post_init__(self) -> None:
        if not self.settings.dashscope_api_key:
            raise ValueError("DASHSCOPE_API_KEY is required for embedding generation.")
        self._client = OpenAI(
            api_key=self.settings.dashscope_api_key,
            base_url=self.settings.dashscope_base_url,
        )

    def embed_texts(self, texts: list[str]) -> list[np.ndarray]:
        if not texts:
            return []
        vectors: list[np.ndarray] = []
        batch_size = min(self.max_batch_size, max(1, self.settings.embedding_batch_size))
        for batch_start in range(0, len(texts), batch_size):
            batch = texts[batch_start : batch_start + batch_size]

            def _call():
                response = self._client.embeddings.create(
                    model=self.settings.embedding_model,
                    input=batch,
                    dimensions=self.settings.embedding_dimensions,
                )
                return [np.asarray(item.embedding, dtype=np.float32) for item in response.data]

            batch_vectors = retry_with_backoff(
                _call,
                max_retries=self.max_retries,
                base_delay=1.0,
                max_delay=8.0,
            )
            vectors.extend(batch_vectors)
        return vectors

    def embed_query(self, query: str) -> np.ndarray:
        cache_key = (
            self.settings.dashscope_base_url,
            self.settings.embedding_model,
            self.settings.embedding_dimensions,
            " ".join(query.split()),
        )
        cached = self._query_cache.get(cache_key)
        if cached is not None:
            self._query_cache.move_to_end(cache_key)
            return cached.copy()

        vector = self.embed_texts([query])[0]
        self._query_cache[cache_key] = vector.copy()
        while len(self._query_cache) > self.query_cache_size:
            self._query_cache.popitem(last=False)
        return vector
