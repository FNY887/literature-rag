import asyncio
from pathlib import Path

import numpy as np
import pytest

from agentic_rag.builder import build_index
from agentic_rag.core.config import get_settings
from agentic_rag.query import answer, answer_stream, search, warm_index
from agentic_rag.query.agent import DashScopeRequiredError, DeepSeekRequiredError


class FakeEmbedder:
    def embed_texts(self, texts: list[str]) -> list[np.ndarray]:
        return [self._embed(text) for text in texts]

    def embed_query(self, query: str) -> np.ndarray:
        return self._embed(query)

    def _embed(self, text: str) -> np.ndarray:
        lowered = text.lower()
        if "alkaline" in lowered or "ph" in lowered:
            return np.asarray([1.0, 0.0], dtype=np.float32)
        if "mineral" in lowered:
            return np.asarray([0.8, 0.2], dtype=np.float32)
        return np.asarray([0.0, 1.0], dtype=np.float32)


class DirectEvidenceLLMClient:
    async def complete_json(self, system_prompt: str, user_prompt: str):
        if "target_claim" in system_prompt and "query_bundles" in system_prompt:
            return {
                "original_question": "test",
                "target_claim": "Find direct evidence.",
                "evidence_type": "direct_support",
                "question_language": "en",
                "intent_type": "evidence_request",
                "analysis": "Test analysis.",
                "entity_terms_en": ["calcium phosphate"],
                "condition_terms_en": ["alkaline"],
                "relation_terms_en": ["occurs under"],
                "numeric_constraints": [],
                "query_bundles": [
                    {
                        "bundle_name": "exact_condition",
                        "query": "alkaline calcium phosphate",
                        "keyword_phrases": ["alkaline", "calcium phosphate"],
                        "required_terms": ["alkaline", "calcium phosphate"],
                    }
                ],
                "must_match_groups": [["calcium phosphate"], ["alkaline"]],
                "diagnostic_notes": [],
                "mode": "hybrid",
                "top_k_vector": 10,
                "top_k_keyword": 10,
            }
        if "Document candidate:" in user_prompt:
            if '"title": "Direct Paper"' not in user_prompt:
                return {
                    "doc_id": "background",
                    "label": "background_only",
                    "supporting_chunk_ids": [],
                    "matched_constraints": ["calcium phosphate"],
                    "missing_constraints": ["alkaline"],
                    "reason": "Related but not direct.",
                    "answer_line": "",
                }
            return {
                "doc_id": "doc1",
                "label": "direct_support",
                "supporting_chunk_ids": ["doc1:0001"],
                "matched_constraints": ["calcium phosphate", "alkaline"],
                "missing_constraints": [],
                "reason": "Direct support.",
                "answer_line": (
                    "The provided chunk states that calcium phosphate mineralization proceeded rapidly "
                    "under alkaline conditions. This directly supports the question because it connects "
                    "the target material and the required condition in the same passage."
                ),
            }
        raise AssertionError("Unexpected LLM call")


class NoDirectSupportLLMClient:
    async def complete_json(self, system_prompt: str, user_prompt: str):
        if "target_claim" in system_prompt and "query_bundles" in system_prompt:
            return {
                "original_question": "test",
                "target_claim": "Find direct evidence.",
                "evidence_type": "direct_support",
                "question_language": "zh",
                "intent_type": "evidence_request",
                "analysis": "Test analysis.",
                "entity_terms_en": ["calcium phosphate"],
                "condition_terms_en": ["alkaline"],
                "relation_terms_en": ["occurs under"],
                "numeric_constraints": [],
                "query_bundles": [
                    {
                        "bundle_name": "exact_condition",
                        "query": "alkaline calcium phosphate",
                        "keyword_phrases": ["alkaline", "calcium phosphate"],
                        "required_terms": ["alkaline", "calcium phosphate"],
                    }
                ],
                "must_match_groups": [["calcium phosphate"], ["alkaline"]],
                "diagnostic_notes": [],
                "mode": "hybrid",
                "top_k_vector": 10,
                "top_k_keyword": 10,
            }
        if "Document candidate:" in user_prompt:
            return {
                "doc_id": "doc1",
                "label": "background_only",
                "supporting_chunk_ids": ["doc1:0001"],
                "matched_constraints": ["calcium phosphate"],
                "missing_constraints": ["alkaline"],
                "reason": "Related but not direct.",
                "answer_line": "",
            }
        raise AssertionError("Unexpected LLM call")


class PassthroughReranker:
    def rerank(self, *, query: str, documents: list[str], top_n: int | None = None, instruct: str | None = None):
        limit = len(documents) if top_n is None else min(top_n, len(documents))
        return [(index, float(limit - index)) for index in range(limit)]


def _write_sample_corpus(source_dir: Path) -> None:
    source_dir.mkdir(exist_ok=True)
    (source_dir / "direct.md").write_text(
        "# Direct Paper\n\n## Abstract\n\n"
        "In alkaline conditions, calcium phosphate mineralization proceeded rapidly.\n",
        encoding="utf-8",
    )
    (source_dir / "background.md").write_text(
        "# Background Paper\n\n## Abstract\n\n"
        "Bone mineralization depends on pH and biological regulation.\n",
        encoding="utf-8",
    )
    (source_dir / "offtarget.md").write_text(
        "# Off Target\n\n## Abstract\n\n"
        "Nanoconfinement changes water transport.\n",
        encoding="utf-8",
    )


async def _collect_answer_events(*, question: str, index_path: Path, settings, embedder, llm, reranker):
    events = []
    async for event in answer_stream(
        question=question,
        index_path=index_path,
        settings=settings,
        embedder=embedder,
        llm=llm,
        reranker=reranker,
    ):
        events.append(event)
    return events


def test_search_hybrid(tmp_path: Path):
    source_dir = tmp_path / "literature"
    _write_sample_corpus(source_dir)
    index_path = tmp_path / "index.sqlite3"
    embedder = FakeEmbedder()
    settings = get_settings()
    settings.dashscope_api_key = None

    build_index(source_dir=source_dir, index_path=index_path, settings=settings, embedder=embedder)

    hits = search(query="alkaline calcium phosphate", mode="hybrid", top_k=5, index_path=index_path, embedder=embedder)
    assert hits
    assert hits[0].title == "Direct Paper"


def test_search_uses_real_paper_title_instead_of_masthead(tmp_path: Path):
    source_dir = tmp_path / "literature"
    source_dir.mkdir()
    (source_dir / "masthead.md").write_text(
        "# Journal of Materials Chemistry B\n\n"
        "# Bacterial S-layer protein inspired multifunctional peptide\n\n"
        "## Abstract\n\n"
        "This peptide promotes alkaline calcium phosphate mineralization inside collagen fibrils.\n",
        encoding="utf-8",
    )
    index_path = tmp_path / "index.sqlite3"
    embedder = FakeEmbedder()
    settings = get_settings()
    settings.dashscope_api_key = None

    build_index(source_dir=source_dir, index_path=index_path, settings=settings, embedder=embedder)

    hits = search(query="alkaline calcium phosphate peptide", mode="hybrid", top_k=5, index_path=index_path, embedder=embedder)

    assert hits
    assert hits[0].title == "Bacterial S-layer protein inspired multifunctional peptide"
    assert hits[0].citation.startswith("Bacterial S-layer protein inspired multifunctional peptide (p. 0, chunk ")


def test_search_does_not_require_sidecar(tmp_path: Path):
    source_dir = tmp_path / "literature"
    _write_sample_corpus(source_dir)
    index_path = tmp_path / "index.sqlite3"
    embedder = FakeEmbedder()
    settings = get_settings()
    settings.dashscope_api_key = None

    build_index(source_dir=source_dir, index_path=index_path, settings=settings, embedder=embedder)
    assert not Path(f"{index_path}.sidecar").exists()

    hits = search(query="alkaline calcium phosphate", mode="hybrid", top_k=5, index_path=index_path, embedder=embedder)
    assert hits


def test_warm_index_loads_sqlite_vectors(tmp_path: Path):
    source_dir = tmp_path / "literature"
    _write_sample_corpus(source_dir)
    index_path = tmp_path / "index.sqlite3"
    embedder = FakeEmbedder()
    settings = get_settings()
    settings.dashscope_api_key = None

    build_index(source_dir=source_dir, index_path=index_path, settings=settings, embedder=embedder)

    stats = warm_index(index_path=index_path, settings=settings)

    assert stats.chunks_loaded > 0
    assert stats.vector_dimensions == 2


def test_warm_index_reports_missing_database(tmp_path: Path):
    with pytest.raises(RuntimeError, match="向量库不存在"):
        warm_index(index_path=tmp_path / "missing.sqlite3")


def test_answer_requires_deepseek_when_no_chat_client(tmp_path: Path, monkeypatch):
    source_dir = tmp_path / "literature"
    _write_sample_corpus(source_dir)
    index_path = tmp_path / "index.sqlite3"
    embedder = FakeEmbedder()
    settings = get_settings()
    settings.dashscope_api_key = None
    settings.deepseek_api_key = None
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)

    build_index(source_dir=source_dir, index_path=index_path, settings=settings, embedder=embedder)

    with pytest.raises(DeepSeekRequiredError):
        asyncio.run(
            answer(
                question="What evidence?",
                index_path=index_path,
                settings=settings,
                embedder=embedder,
                reranker=PassthroughReranker(),
            )
        )


def test_answer_keeps_only_direct_support(tmp_path: Path):
    source_dir = tmp_path / "literature"
    _write_sample_corpus(source_dir)
    index_path = tmp_path / "index.sqlite3"
    embedder = FakeEmbedder()
    settings = get_settings()
    settings.dashscope_api_key = None

    build_index(source_dir=source_dir, index_path=index_path, settings=settings, embedder=embedder)

    result = asyncio.run(
        answer(
            question="test",
            index_path=index_path,
            settings=settings,
            embedder=embedder,
            llm=DirectEvidenceLLMClient(),
            reranker=PassthroughReranker(),
        )
    )
    assert result.answer
    assert result.stopped_early is False
    assert result.stage_timings.analyze_seconds >= 0.0
    assert result.stage_timings.retrieve_seconds >= 0.0
    assert result.stage_timings.rerank_seconds >= 0.0
    assert result.stage_timings.judge_seconds >= 0.0
    assert result.stage_timings.total_seconds >= 0.0
    assert result.performance_counters.judge_documents_total >= 1
    assert result.performance_counters.judge_concurrency_initial == 5
    assert result.performance_counters.judge_concurrency_final >= 1
    assert result.answer == (
        "1. The provided chunk states that calcium phosphate mineralization proceeded rapidly "
        "under alkaline conditions. This directly supports the question because it connects "
        "the target material and the required condition in the same passage."
    )
    assert len(result.citations) == 1
    assert result.citations[0].startswith("[1] Direct Paper (p. 0, chunk ")
    assert result.citations[0].endswith(":0001)")


def test_answer_stream_emits_events(tmp_path: Path):
    source_dir = tmp_path / "literature"
    _write_sample_corpus(source_dir)
    index_path = tmp_path / "index.sqlite3"
    embedder = FakeEmbedder()
    settings = get_settings()
    settings.dashscope_api_key = None

    build_index(source_dir=source_dir, index_path=index_path, settings=settings, embedder=embedder)

    events = asyncio.run(
        _collect_answer_events(
            question="test",
            index_path=index_path,
            settings=settings,
            embedder=embedder,
            llm=DirectEvidenceLLMClient(),
            reranker=PassthroughReranker(),
        )
    )
    assert events[0].event == "scan_started"
    assert events[-1].event == "scan_completed"
    assert events[-1].answer is not None
    assert events[-1].stage_timings is not None
    assert events[-1].performance_counters is not None
    assert events[-1].stage_timings.total_seconds >= 0.0
    assert events[-1].performance_counters.judge_concurrency_initial == 5
    assert "synthesize_seconds" not in events[-1].stage_timings.model_dump()


def test_answer_without_direct_support_returns_fixed_message(tmp_path: Path):
    source_dir = tmp_path / "literature"
    _write_sample_corpus(source_dir)
    index_path = tmp_path / "index.sqlite3"
    embedder = FakeEmbedder()
    settings = get_settings()
    settings.dashscope_api_key = None

    build_index(source_dir=source_dir, index_path=index_path, settings=settings, embedder=embedder)

    result = asyncio.run(
        answer(
            question="test",
            index_path=index_path,
            settings=settings,
            embedder=embedder,
            llm=NoDirectSupportLLMClient(),
            reranker=PassthroughReranker(),
        )
    )

    assert result.answer == "现有检索到的文献片段中，没有找到直接支持该问题的文献内容。"
    assert result.citations == []
    assert "synthesize_seconds" not in result.stage_timings.model_dump()
