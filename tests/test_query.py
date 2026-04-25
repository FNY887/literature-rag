import asyncio
from pathlib import Path

import numpy as np
import pytest

from agentic_rag.builder import build_index
from agentic_rag.core.config import get_settings
from agentic_rag.query import answer, answer_stream, research, search, warm_index
from agentic_rag.query.agent import ChatModelRequiredError, DashScopeRequiredError


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


VALID_RESEARCH_REPORT = (
    "这是一段基于检索片段生成的自然研究报告，围绕用户问题综合已有证据展开说明。"
    "报告只使用提供的 chunk 内容，不引入外部知识，并通过引用标注关键依据[1]。"
    "它不会强制拆成固定条目，而是根据证据之间的关系组织为连续论述，说明材料、方法、应用和限制之间的联系。"
    "如果某些证据来自同一篇论文的不同片段，程序会在最终参考文献中自动合并这些引用。"
    "报告可以自然地交代已有研究显示了什么、这些发现之间如何相互支撑，以及当前材料还不能充分回答的问题。"
    "这种方式让模型根据证据组织内容，而不是被迫输出固定的提纲或分类。"
    "当问题需要概括时，模型可以在同一段落中完成归纳、比较和限定。"
    "如果 chunks 内容较多，报告可以继续展开，不需要为了固定上限压缩掉重要证据。"
    "如果 chunks 内容较少，报告也应保持谨慎，只在证据允许的范围内补充解释。"
    "这种无上限但有下限的规则，让输出既不会过短，也不会人为限制模型根据材料充分发挥。"
    "它更适合深度研究模式，因为模型可以根据材料密度决定展开程度，而程序只负责保证回答不是一句话式摘要。"
    "引用仍然来自给定 chunk，后续参考文献会按论文合并，方便检查证据来源。"
    "这种写法保留了深度研究所需的综合性，同时避免把模型限制成僵硬的提纲输出。"
)


class ResearchLLMClient:
    def __init__(self, *, reports: list[str] | None = None, repair_report: str = VALID_RESEARCH_REPORT):
        self.reports = reports or [VALID_RESEARCH_REPORT]
        self.repair_report = repair_report
        self.research_prompts: list[str] = []
        self.repair_prompts: list[str] = []
        self.research_system_prompts: list[str] = []
        self.repair_system_prompts: list[str] = []

    async def complete_json(self, system_prompt: str, user_prompt: str):
        if "retrieval planner" in system_prompt.lower():
            return {
                "original_question": "test",
                "target_claim": "Find direct evidence.",
                "evidence_type": "direct_support",
                "question_language": "zh",
                "intent_type": "research_report",
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
        if "research report writer" in system_prompt.lower():
            self.research_system_prompts.append(system_prompt)
            self.research_prompts.append(user_prompt)
            return {"report": self.reports.pop(0), "used_ref_ids": [1]}
        if "revise a chunk-grounded research report" in system_prompt.lower():
            self.repair_system_prompts.append(system_prompt)
            self.repair_prompts.append(user_prompt)
            return {"report": self.repair_report, "used_ref_ids": [1]}
        raise AssertionError("Unexpected LLM call")


class PassthroughReranker:
    def rerank(self, *, query: str, documents: list[str], top_n: int | None = None, instruct: str | None = None):
        limit = len(documents) if top_n is None else min(top_n, len(documents))
        return [(index, float(limit - index)) for index in range(limit)]


class RecordingPassthroughReranker(PassthroughReranker):
    def __init__(self):
        self.calls: list[list[str]] = []

    def rerank(self, *, query: str, documents: list[str], top_n: int | None = None, instruct: str | None = None):
        self.calls.append(documents)
        return super().rerank(query=query, documents=documents, top_n=top_n, instruct=instruct)


class TrackingOwnedLLMClient:
    instances: list["TrackingOwnedLLMClient"] = []

    def __init__(self, settings):
        del settings
        self.closed = 0
        type(self).instances.append(self)

    async def complete_json(self, system_prompt: str, user_prompt: str):
        if "target_claim" in system_prompt and "query_bundles" in system_prompt:
            return {
                "original_question": "test",
                "target_claim": "Find direct evidence.",
                "evidence_type": "direct_support",
                "question_language": "zh",
                "intent_type": "research_report",
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
        if "research report writer" in system_prompt.lower():
            return {"report": VALID_RESEARCH_REPORT, "used_ref_ids": [1]}
        if "revise a chunk-grounded research report" in system_prompt.lower():
            return {"report": VALID_RESEARCH_REPORT, "used_ref_ids": [1]}
        raise AssertionError("Unexpected LLM call")

    async def close(self):
        self.closed += 1


class ExternallyManagedLLMClient(TrackingOwnedLLMClient):
    async def close(self):
        self.closed += 1


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


def test_answer_requires_chat_model_when_no_chat_client(tmp_path: Path, monkeypatch):
    source_dir = tmp_path / "literature"
    _write_sample_corpus(source_dir)
    index_path = tmp_path / "index.sqlite3"
    embedder = FakeEmbedder()
    settings = get_settings()
    settings.dashscope_api_key = None
    settings.chat_api_key = None
    monkeypatch.delenv("API_KEY", raising=False)

    build_index(source_dir=source_dir, index_path=index_path, settings=settings, embedder=embedder)

    with pytest.raises(ChatModelRequiredError):
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


def test_research_generates_report_and_citations(tmp_path: Path):
    source_dir = tmp_path / "literature"
    _write_sample_corpus(source_dir)
    index_path = tmp_path / "index.sqlite3"
    embedder = FakeEmbedder()
    settings = get_settings()
    settings.dashscope_api_key = None
    settings.chat_api_key = None
    llm = ResearchLLMClient()
    reranker = RecordingPassthroughReranker()

    build_index(source_dir=source_dir, index_path=index_path, settings=settings, embedder=embedder)

    result = asyncio.run(
        research(
            question="请写一份研究报告",
            index_path=index_path,
            settings=settings,
            embedder=embedder,
            llm=llm,
            reranker=reranker,
        )
    )

    assert result.report == VALID_RESEARCH_REPORT
    assert result.citations
    assert result.citations[0].startswith("[1] ")
    assert result.chunks_recalled > 0
    assert result.chunks_reranked > 0
    assert 0 < result.chunks_in_context <= settings.research_final_chunk_limit
    assert reranker.calls
    assert "Evidence chunks:" in llm.research_prompts[0]
    assert "Structured query plan" not in llm.research_prompts[0]
    assert "请写一份研究报告" in llm.research_prompts[0]
    assert "taxonomy" not in llm.research_system_prompts[0].lower()
    assert "major direction" not in llm.research_system_prompts[0].lower()


def test_research_expands_short_report(tmp_path: Path):
    source_dir = tmp_path / "literature"
    _write_sample_corpus(source_dir)
    index_path = tmp_path / "index.sqlite3"
    embedder = FakeEmbedder()
    settings = get_settings()
    settings.dashscope_api_key = None
    settings.chat_api_key = None
    llm = ResearchLLMClient(reports=["太短[1]"], repair_report=VALID_RESEARCH_REPORT)

    build_index(source_dir=source_dir, index_path=index_path, settings=settings, embedder=embedder)

    result = asyncio.run(
        research(
            question="请写一份研究报告",
            index_path=index_path,
            settings=settings,
            embedder=embedder,
            llm=llm,
            reranker=PassthroughReranker(),
        )
    )

    assert result.report == VALID_RESEARCH_REPORT
    assert llm.repair_prompts


def test_answer_closes_owned_llm_client(tmp_path: Path, monkeypatch):
    source_dir = tmp_path / "literature"
    _write_sample_corpus(source_dir)
    index_path = tmp_path / "index.sqlite3"
    embedder = FakeEmbedder()
    settings = get_settings()
    settings.dashscope_api_key = None
    settings.chat_api_key = "chat-key"
    TrackingOwnedLLMClient.instances = []
    monkeypatch.setattr("agentic_rag.query.OpenAICompatibleChatClient", TrackingOwnedLLMClient)

    build_index(source_dir=source_dir, index_path=index_path, settings=settings, embedder=embedder)

    result = asyncio.run(
        answer(
            question="test",
            index_path=index_path,
            settings=settings,
            embedder=embedder,
            reranker=PassthroughReranker(),
        )
    )

    assert result.answer
    assert len(TrackingOwnedLLMClient.instances) == 1
    assert TrackingOwnedLLMClient.instances[0].closed == 1


def test_research_closes_owned_llm_client(tmp_path: Path, monkeypatch):
    source_dir = tmp_path / "literature"
    _write_sample_corpus(source_dir)
    index_path = tmp_path / "index.sqlite3"
    embedder = FakeEmbedder()
    settings = get_settings()
    settings.dashscope_api_key = None
    settings.chat_api_key = "chat-key"
    TrackingOwnedLLMClient.instances = []
    monkeypatch.setattr("agentic_rag.query.OpenAICompatibleChatClient", TrackingOwnedLLMClient)

    build_index(source_dir=source_dir, index_path=index_path, settings=settings, embedder=embedder)

    result = asyncio.run(
        research(
            question="请写一份研究报告",
            index_path=index_path,
            settings=settings,
            embedder=embedder,
            reranker=PassthroughReranker(),
        )
    )

    assert result.report == VALID_RESEARCH_REPORT
    assert len(TrackingOwnedLLMClient.instances) == 1
    assert TrackingOwnedLLMClient.instances[0].closed == 1


def test_answer_stream_closes_owned_llm_client(tmp_path: Path, monkeypatch):
    source_dir = tmp_path / "literature"
    _write_sample_corpus(source_dir)
    index_path = tmp_path / "index.sqlite3"
    embedder = FakeEmbedder()
    settings = get_settings()
    settings.dashscope_api_key = None
    settings.chat_api_key = "chat-key"
    TrackingOwnedLLMClient.instances = []
    monkeypatch.setattr("agentic_rag.query.OpenAICompatibleChatClient", TrackingOwnedLLMClient)

    build_index(source_dir=source_dir, index_path=index_path, settings=settings, embedder=embedder)

    events = asyncio.run(
        _collect_answer_events(
            question="test",
            index_path=index_path,
            settings=settings,
            embedder=embedder,
            llm=None,
            reranker=PassthroughReranker(),
        )
    )

    assert events[-1].event == "scan_completed"
    assert len(TrackingOwnedLLMClient.instances) == 1
    assert TrackingOwnedLLMClient.instances[0].closed == 1


def test_research_does_not_close_external_llm_client(tmp_path: Path):
    source_dir = tmp_path / "literature"
    _write_sample_corpus(source_dir)
    index_path = tmp_path / "index.sqlite3"
    embedder = FakeEmbedder()
    settings = get_settings()
    settings.dashscope_api_key = None
    settings.chat_api_key = None
    llm = ExternallyManagedLLMClient(settings)

    build_index(source_dir=source_dir, index_path=index_path, settings=settings, embedder=embedder)

    result = asyncio.run(
        research(
            question="请写一份研究报告",
            index_path=index_path,
            settings=settings,
            embedder=embedder,
            llm=llm,
            reranker=PassthroughReranker(),
        )
    )

    assert result.report == VALID_RESEARCH_REPORT
    assert llm.closed == 0
