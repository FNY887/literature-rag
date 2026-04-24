from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, AsyncIterator, Protocol

import numpy as np

from agentic_rag.builder.embedder import DashScopeEmbeddingClient
from agentic_rag.builder.store import SQLiteIndexStore
from agentic_rag.core.config import Settings, get_settings
from agentic_rag.core.llm import DeepSeekChatClient
from agentic_rag.core.models import AgentAnswer, AnswerStreamEvent, SearchHit
from agentic_rag.query.agent import AgentStageError, AgenticAnswerer, DashScopeRequiredError, DeepSeekRequiredError
from agentic_rag.query.retriever import HybridRetriever
from agentic_rag.query.reranker import DashScopeRerankClient


class Reranker(Protocol):
    def rerank(
        self,
        *,
        query: str,
        documents: list[str],
        top_n: int | None = None,
        instruct: str | None = None,
    ) -> list[tuple[int, float]]: ...


class Embedder(Protocol):
    def embed_texts(self, texts: list[str]) -> list[np.ndarray]: ...
    def embed_query(self, query: str) -> np.ndarray: ...


@dataclass(frozen=True, slots=True)
class WarmIndexStats:
    chunks_loaded: int
    vector_dimensions: int


def _resolve_settings(
    settings: Settings | None,
    *,
    index_path: str | Path | None = None,
) -> Settings:
    resolved = settings.resolved() if settings is not None else get_settings()
    if index_path is not None:
        resolved.index_path = Path(index_path) if Path(index_path).is_absolute() else Path.cwd() / index_path
        resolved.index_dir = resolved.index_path.parent
    return resolved


def _validate_default_retrieval_stack(
    *,
    question: str,
    settings: Settings,
    embedder: Embedder | None,
    reranker: Reranker | None,
) -> None:
    if settings.dashscope_api_key:
        return
    missing_requirements: list[str] = []
    if embedder is None:
        missing_requirements.append("query embeddings for hybrid retrieval")
    if reranker is None:
        missing_requirements.append("qwen3-rerank document reranking")
    if missing_requirements:
        raise DashScopeRequiredError(
            question=question,
            requirement=" and ".join(missing_requirements),
        )


def _make_search_fn(
    retriever: HybridRetriever,
    settings: Settings,
    embedder: Embedder,
) -> Any:
    query_vector_cache: dict[str, np.ndarray] = {}

    def search_fn(*, query: str, mode: str, top_k_vector: int, top_k_keyword: int, keyword_fts_query: str | None = None) -> list[SearchHit]:
        if mode == "keyword":
            return retriever.keyword_candidates(query, limit=None, fts_query=keyword_fts_query)
        query_key = " ".join(query.split())
        query_vector = query_vector_cache.get(query_key)
        if query_vector is None:
            query_vector = embedder.embed_query(query)
            query_vector_cache[query_key] = query_vector
        if mode == "vector":
            return retriever.vector_candidates(query_vector, limit=None)
        return retriever.hybrid_candidates(
            query=query,
            query_vector=query_vector,
            top_k_vector=None,
            top_k_keyword=None,
            limit=None,
            keyword_fts_query=keyword_fts_query,
        )

    return search_fn


def warm_index(
    *,
    index_path: str | Path | None = None,
    settings: Settings | None = None,
) -> WarmIndexStats:
    resolved = _resolve_settings(settings, index_path=index_path)
    if not resolved.index_path.exists():
        raise RuntimeError(
            f"向量库不存在：{resolved.index_path}。请先运行 literature-rag build <存放Markdown文献的目录>。"
        )
    store = SQLiteIndexStore(resolved.index_path)
    try:
        rows, matrix = store.fetch_all_vectors()
    except Exception as exc:
        raise RuntimeError(
            f"向量库无法加载：{resolved.index_path}。请先运行 literature-rag build <存放Markdown文献的目录> 重建。"
        ) from exc
    vector_dimensions = int(matrix.shape[1]) if matrix.ndim == 2 and matrix.shape[0] else 0
    return WarmIndexStats(chunks_loaded=len(rows), vector_dimensions=vector_dimensions)


def search(
    *,
    query: str,
    mode: str = "hybrid",
    top_k: int = 5,
    top_k_vector: int | None = None,
    top_k_keyword: int | None = None,
    keyword_fts_query: str | None = None,
    index_path: str | Path | None = None,
    settings: Settings | None = None,
    embedder: Embedder | None = None,
) -> list[SearchHit]:
    resolved = _resolve_settings(settings, index_path=index_path)
    store = SQLiteIndexStore(resolved.index_path)
    retriever = HybridRetriever(
        store=store,
        settings=resolved,
    )

    top_k_vector = top_k_vector or resolved.top_k_vector
    top_k_keyword = top_k_keyword or resolved.top_k_keyword

    if mode == "keyword":
        return retriever.keyword_search(query, limit=top_k, fts_query=keyword_fts_query)

    embedding_client = embedder or DashScopeEmbeddingClient(resolved)
    query_vector = embedding_client.embed_query(query)
    if mode == "vector":
        return retriever.vector_search(query_vector, limit=top_k)
    if mode == "hybrid":
        return retriever.hybrid_search(
            query=query,
            query_vector=query_vector,
            top_k_vector=top_k_vector,
            top_k_keyword=top_k_keyword,
            limit=top_k,
            keyword_fts_query=keyword_fts_query,
        )
    raise ValueError(f"Unsupported search mode: {mode}")


async def answer(
    *,
    question: str,
    index_path: str | Path | None = None,
    settings: Settings | None = None,
    embedder: Embedder | None = None,
    llm: DeepSeekChatClient | None = None,
    reranker: Reranker | None = None,
) -> AgentAnswer:
    resolved = _resolve_settings(settings, index_path=index_path)
    store = SQLiteIndexStore(resolved.index_path)
    retriever = HybridRetriever(
        store=store,
        settings=resolved,
    )
    _validate_default_retrieval_stack(
        question=question,
        settings=resolved,
        embedder=embedder,
        reranker=reranker,
    )
    embedding_client = embedder or DashScopeEmbeddingClient(resolved)
    llm_client = llm
    if llm_client is None and resolved.deepseek_api_key:
        llm_client = DeepSeekChatClient(resolved)
    if llm_client is None:
        raise DeepSeekRequiredError(question=question)
    rerank_client = reranker
    if rerank_client is None and resolved.dashscope_api_key:
        rerank_client = DashScopeRerankClient(resolved)
    answerer = AgenticAnswerer(
        retriever=retriever,
        settings=resolved,
        llm_client=llm_client,
        rerank_client=rerank_client,
    )
    search_fn = _make_search_fn(retriever, resolved, embedding_client)
    return await answerer.answer(question=question, search_fn=search_fn)


def answer_stream(
    *,
    question: str,
    index_path: str | Path | None = None,
    settings: Settings | None = None,
    embedder: Embedder | None = None,
    llm: DeepSeekChatClient | None = None,
    reranker: Reranker | None = None,
) -> AsyncIterator[AnswerStreamEvent]:
    resolved = _resolve_settings(settings, index_path=index_path)
    store = SQLiteIndexStore(resolved.index_path)
    retriever = HybridRetriever(
        store=store,
        settings=resolved,
    )
    _validate_default_retrieval_stack(
        question=question,
        settings=resolved,
        embedder=embedder,
        reranker=reranker,
    )
    embedding_client = embedder or DashScopeEmbeddingClient(resolved)
    llm_client = llm
    if llm_client is None and resolved.deepseek_api_key:
        llm_client = DeepSeekChatClient(resolved)
    if llm_client is None:
        raise DeepSeekRequiredError(question=question)
    rerank_client = reranker
    if rerank_client is None and resolved.dashscope_api_key:
        rerank_client = DashScopeRerankClient(resolved)
    answerer = AgenticAnswerer(
        retriever=retriever,
        settings=resolved,
        llm_client=llm_client,
        rerank_client=rerank_client,
    )
    search_fn = _make_search_fn(retriever, resolved, embedding_client)
    return answerer.answer_stream(question=question, search_fn=search_fn)
