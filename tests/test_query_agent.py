import asyncio
import json

import pytest

from agentic_rag.core.config import Settings
from agentic_rag.core.models import AgentAnswer, DocumentCandidate, DocumentAssessment, QueryAnalysis, QueryBundle, SearchHit
from agentic_rag.query.agent import (
    AgentStageError,
    AgenticAnswerer,
    _estimate_rerank_tokens,
    _merge_research_report_citations,
)


class FakeLLMClient:
    def __init__(self, payload):
        self.payload = payload

    async def complete_json(self, system_prompt: str, user_prompt: str):
        del system_prompt, user_prompt
        return self.payload


class FailingLLMClient:
    async def complete_json(self, system_prompt: str, user_prompt: str):
        del system_prompt, user_prompt
        raise RuntimeError("simulated llm failure")


class SingleDocumentJudgeLLMClient:
    def __init__(self):
        self.prompts: list[str] = []
        self.system_prompts: list[str] = []

    async def complete_json(self, system_prompt: str, user_prompt: str):
        assert "strict document-level evidence judge" in system_prompt.lower()
        self.system_prompts.append(system_prompt)
        self.prompts.append(user_prompt)
        assert "Document candidate:" in user_prompt
        return {
            "doc_id": "direct-doc",
            "label": "direct_support",
            "supporting_chunk_ids": ["doc:0002"],
            "matched_constraints": ["calcium phosphate mineralization", "alkaline conditions"],
            "missing_constraints": [],
            "reason": "The provided paper directly supports the claim.",
            "answer_line": (
                "The provided chunks describe alkaline-condition evidence for calcium phosphate mineralization. "
                "They directly support the question by linking the observed mineralization text to the required condition."
            ),
        }


class RetryingJudgeLLMClient:
    def __init__(self):
        self.prompts: list[str] = []
        self.calls = 0

    async def complete_json(self, system_prompt: str, user_prompt: str):
        assert "strict document-level evidence judge" in system_prompt.lower()
        self.prompts.append(user_prompt)
        self.calls += 1
        if self.calls == 1:
            return {
                "doc_id": "direct-doc",
                "label": "direct_support",
                "supporting_chunk_ids": ["doc:0002"],
                "matched_constraints": ["calcium phosphate mineralization", "alkaline conditions"],
                "missing_constraints": [],
                "reason": "The provided paper directly supports the claim.",
                "answer_line": "",
            }
        return {
            "doc_id": "direct-doc",
            "label": "direct_support",
            "supporting_chunk_ids": ["doc:0002", "doc:0003"],
            "matched_constraints": ["calcium phosphate mineralization", "alkaline conditions"],
            "missing_constraints": [],
            "reason": "The provided paper directly supports the claim.",
            "answer_line": (
                "The chunks identify calcium phosphate mineralization under the requested alkaline condition. "
                "They directly support the claim because the cited passages satisfy both the material and condition constraints."
            ),
        }


class RecordingReranker:
    def __init__(self, *, fail_on_call: int | None = None):
        self.calls: list[list[str]] = []
        self.fail_on_call = fail_on_call

    def rerank(self, *, query: str, documents: list[str], top_n: int | None = None, instruct: str | None = None):
        del query, top_n, instruct
        self.calls.append(documents)
        if self.fail_on_call is not None and len(self.calls) == self.fail_on_call:
            raise RuntimeError("simulated rerank failure")
        results: list[tuple[int, float]] = []
        for index, document in enumerate(documents):
            score = 0.0
            for doc_number in range(1, 20):
                if f"Title: doc{doc_number}" in document:
                    score = float(doc_number)
                    break
            results.append((index, score))
        return results


class SequentialResearchLLMClient:
    def __init__(self, payloads: list[dict]):
        self.payloads = payloads
        self.system_prompts: list[str] = []
        self.user_prompts: list[str] = []

    async def complete_json(self, system_prompt: str, user_prompt: str):
        self.system_prompts.append(system_prompt)
        self.user_prompts.append(user_prompt)
        return self.payloads.pop(0)


def _valid_research_report(ref_id: int = 1) -> str:
    return (
        "这是一段满足长度要求的深度研究报告，完全依据给定片段组织内容，围绕用户问题解释证据之间的联系。"
        f"报告使用引用标注关键依据[{ref_id}]，但不把回答强制拆成固定分点，而是让模型根据问题自然展开。"
        "它可以综合材料性质、实验观察、应用场景和证据限制，说明当前文献能够支持哪些判断、哪些地方仍然需要更多资料。"
        "如果检索片段显示多个研究路径，报告会在自然叙述中串联这些路径，而不是机械列出模板化标题。"
        "如果证据集中在少数片段，报告也会保持谨慎，只说明这些片段能够直接支撑的结论。"
        "这种输出方式更接近研究备忘录：先回答核心问题，再补充证据强弱、可能应用和仍需验证的环节。"
        "同时它仍然保留引用编号，便于用户回到参考文献和具体 chunk 检查依据。"
        "当检索到的 chunks 内容很多时，报告可以继续展开更多细节，不需要为了固定上限而删掉重要证据。"
        "当 chunks 内容较少时，报告也应避免空泛扩写，只在证据允许的范围内补充背景、联系和限制。"
        "这种规则使深度研究更接近真实阅读后的综合判断：既保留关键事实，也允许模型说明不同证据之间的层次。"
        "同时，引用仍然限定在给定片段中，避免模型把未检索到的外部知识混入结论。"
        "这段文本刻意保持在五百字符以上，用于验证正常输出不会触发二次修复。"
    )


def _make_research_hit(index: int, *, text: str = "Evidence text") -> SearchHit:
    return SearchHit(
        doc_id=f"doc{index}",
        chunk_id=f"doc{index}:0001",
        score_final=float(index),
        score_constraint=0.5,
        retrieval_source="hybrid",
        text=text,
        citation=f"Paper {index} (p. 0, chunk doc{index}:0001)",
        title=f"doc{index}",
        source_path=f"doc{index}.md",
        page_start=0,
        page_end=0,
        section_hint="Abstract",
    )


def test_analyze_success():
    answerer = AgenticAnswerer(
        retriever=None,
        settings=Settings(),
        llm_client=FakeLLMClient({
            "target_claim": "Test claim",
            "query_bundles": [{"bundle_name": "b1", "query": "test query"}],
        }),
    )
    result = asyncio.run(answerer.analyze("Test question"))
    assert result.target_claim == "Test claim"


def test_analyze_missing_target_claim_raises():
    answerer = AgenticAnswerer(
        retriever=None,
        settings=Settings(),
        llm_client=FakeLLMClient({}),
    )
    with pytest.raises(AgentStageError):
        asyncio.run(answerer.analyze("Test question"))


def test_analyze_llm_failure_raises():
    answerer = AgenticAnswerer(
        retriever=None,
        settings=Settings(),
        llm_client=FailingLLMClient(),
    )
    with pytest.raises(AgentStageError):
        asyncio.run(answerer.analyze("Test question"))


def test_judge_document_success():
    answerer = AgenticAnswerer(
        retriever=None,
        settings=Settings(),
        llm_client=SingleDocumentJudgeLLMClient(),
    )
    candidate = DocumentCandidate(
        doc_id="direct-doc",
        title="Direct Paper",
        source_path="direct.md",
        aggregate_score=0.9,
        top_chunks=[
            SearchHit(
                chunk_id="doc:0001",
                score_final=0.8,
                retrieval_source="hybrid",
                text="Some text",
                citation="citation1",
                title="Direct Paper",
                source_path="direct.md",
                page_start=1,
                page_end=1,
            ),
            SearchHit(
                chunk_id="doc:0002",
                score_final=0.9,
                retrieval_source="hybrid",
                text="Alkaline conditions text",
                citation="citation2",
                title="Direct Paper",
                source_path="direct.md",
                page_start=1,
                page_end=1,
            ),
            SearchHit(
                chunk_id="doc:0003",
                score_final=0.7,
                retrieval_source="hybrid",
                text="More support",
                citation="citation3",
                title="Direct Paper",
                source_path="direct.md",
                page_start=2,
                page_end=2,
            ),
        ],
    )
    assessment = asyncio.run(
        answerer.judge_document(
            question="test",
            query_analysis=QueryAnalysis(
                target_claim="Find evidence",
                query_bundles=[QueryBundle(bundle_name="b1", query="test")],
            ),
            document_candidate=candidate,
        )
    )
    assert assessment.label == "direct_support"
    assert "doc:0002" in assessment.supporting_chunk_ids
    assert "alkaline-condition evidence" in assessment.answer_line
    system_prompt = answerer.llm_client.system_prompts[0]
    user_prompt = answerer.llm_client.prompts[0]
    assert "one concise answer sentence" not in system_prompt
    assert "2-3 sentence chunk-grounded explanation" in system_prompt
    assert "how the provided chunks directly support" in system_prompt
    assert "Do not invent methods, experimental design, or evidence sources" in system_prompt
    assert "2-3 sentence chunk-grounded explanation" in user_prompt

def test_judge_document_retries_same_payload_with_three_chunks():
    answerer = AgenticAnswerer(
        retriever=None,
        settings=Settings(),
        llm_client=RetryingJudgeLLMClient(),
    )
    candidate = DocumentCandidate(
        doc_id="direct-doc",
        title="Direct Paper",
        source_path="direct.md",
        aggregate_score=0.9,
        top_chunks=[
            SearchHit(
                chunk_id="doc:0001",
                score_final=0.8,
                retrieval_source="hybrid",
                text="Chunk 1",
                citation="citation1",
                title="Direct Paper",
                source_path="direct.md",
                page_start=1,
                page_end=1,
            ),
            SearchHit(
                chunk_id="doc:0002",
                score_final=0.9,
                retrieval_source="hybrid",
                text="Chunk 2",
                citation="citation2",
                title="Direct Paper",
                source_path="direct.md",
                page_start=1,
                page_end=1,
            ),
            SearchHit(
                chunk_id="doc:0003",
                score_final=0.85,
                retrieval_source="hybrid",
                text="Chunk 3",
                citation="citation3",
                title="Direct Paper",
                source_path="direct.md",
                page_start=2,
                page_end=2,
            ),
        ],
    )

    assessment = asyncio.run(
        answerer.judge_document(
            question="test",
            query_analysis=QueryAnalysis(
                target_claim="Find evidence",
                query_bundles=[QueryBundle(bundle_name="b1", query="test")],
            ),
            document_candidate=candidate,
        )
    )
    payloads = []
    for prompt in answerer.llm_client.prompts:
        candidate_payload = prompt.split("Document candidate:\n", 1)[1].split("\n\nReturn JSON", 1)[0]
        payloads.append(json.loads(candidate_payload))

    assert "calcium phosphate mineralization" in assessment.answer_line
    assert len(payloads) == 2
    assert payloads[0] == payloads[1]
    assert len(payloads[0]["chunks"]) == 3


def test_judge_document_payload_uses_full_chunk_text_without_truncation():
    llm_client = SingleDocumentJudgeLLMClient()
    answerer = AgenticAnswerer(
        retriever=None,
        settings=Settings(),
        llm_client=llm_client,
    )
    long_text = "prefix " + ("x" * 700) + " suffix"
    candidate = DocumentCandidate(
        doc_id="direct-doc",
        title="Direct Paper",
        source_path="direct.md",
        aggregate_score=0.9,
        top_chunks=[
            SearchHit(
                chunk_id="doc:0001",
                score_final=0.95,
                retrieval_source="hybrid",
                text=long_text,
                citation="citation1",
                title="Direct Paper",
                source_path="direct.md",
                page_start=1,
                page_end=1,
            ),
            SearchHit(
                chunk_id="doc:0002",
                score_final=0.9,
                retrieval_source="hybrid",
                text="second chunk",
                citation="citation2",
                title="Direct Paper",
                source_path="direct.md",
                page_start=1,
                page_end=1,
            ),
            SearchHit(
                chunk_id="doc:0003",
                score_final=0.85,
                retrieval_source="hybrid",
                text="third chunk",
                citation="citation3",
                title="Direct Paper",
                source_path="direct.md",
                page_start=2,
                page_end=2,
            ),
            SearchHit(
                chunk_id="doc:0004",
                score_final=0.8,
                retrieval_source="hybrid",
                text="fourth chunk should not be sent",
                citation="citation4",
                title="Direct Paper",
                source_path="direct.md",
                page_start=3,
                page_end=3,
            ),
        ],
    )

    asyncio.run(
        answerer.judge_document(
            question="test",
            query_analysis=QueryAnalysis(
                target_claim="Find evidence",
                query_bundles=[QueryBundle(bundle_name="b1", query="test")],
            ),
            document_candidate=candidate,
        )
    )

    candidate_payload = llm_client.prompts[0].split("Document candidate:\n", 1)[1].split("\n\nReturn JSON", 1)[0]
    payload = json.loads(candidate_payload)

    assert len(payload["chunks"]) == 3
    assert payload["chunks"][0]["text"] == long_text
    assert len(payload["chunks"][0]["text"]) > 520
    assert "fourth chunk should not be sent" not in candidate_payload


def test_document_rerank_input_uses_top_three_chunks_without_default_truncation():
    reranker = RecordingReranker()
    answerer = AgenticAnswerer(
        retriever=None,
        settings=Settings(rerank_request_token_budget=100000),
        llm_client=object(),
        rerank_client=reranker,
    )
    long_tail = "x" * 600
    candidates = [
        _candidate_with_chunks(
            "doc1",
            chunk_texts=[
                f"first chunk {long_tail}",
                "second chunk evidence",
                "third chunk evidence",
                "fourth chunk should not be sent",
            ],
        ),
        _candidate_with_chunks("doc2", chunk_texts=["other first", "other second", "other third"]),
    ]

    answerer._rerank_document_candidates(
        question="test",
        query_analysis=QueryAnalysis(target_claim="Find evidence"),
        document_candidates=candidates,
    )

    assert len(reranker.calls) == 1
    first_document = reranker.calls[0][0]
    assert "first chunk" in first_document
    assert "second chunk evidence" in first_document
    assert "third chunk evidence" in first_document
    assert "fourth chunk should not be sent" not in first_document
    assert long_tail in first_document
    assert len(first_document) > 420


def test_document_rerank_input_keeps_hierarchical_section_hint_text():
    answerer = AgenticAnswerer(
        retriever=None,
        settings=Settings(rerank_request_token_budget=100000),
        llm_client=object(),
        rerank_client=RecordingReranker(),
    )
    candidate = _candidate_with_chunks("doc1", chunk_texts=["first chunk", "second chunk", "third chunk"])
    candidate.top_chunks[0].section_hint = (
        "2. Experimental section > 2.1. Preparation of samples > 2.1.1. Synthesis of composites beads"
    )

    document_input = answerer._document_rerank_input(candidate)

    assert (
        "Section: 2. Experimental section > 2.1. Preparation of samples > 2.1.1. Synthesis of composites beads"
        in document_input
    )


def test_document_rerank_batches_by_token_budget():
    reranker = RecordingReranker()
    settings = Settings(
        rerank_document_chunk_limit=3,
        rerank_document_text_limit=0,
        rerank_request_token_budget=250,
        rerank_instruct="",
    )
    answerer = AgenticAnswerer(
        retriever=None,
        settings=settings,
        llm_client=object(),
        rerank_client=reranker,
    )
    candidates = [
        _candidate_with_chunks(f"doc{index}", chunk_texts=[f"{index} " + "x" * 40] * 3)
        for index in range(1, 7)
    ]

    answerer._rerank_document_candidates(
        question="q",
        query_analysis=QueryAnalysis(target_claim="q"),
        document_candidates=candidates,
    )

    assert len(reranker.calls) > 1
    for batch in reranker.calls:
        estimated = (
            _estimate_rerank_tokens("q")
            + _estimate_rerank_tokens(settings.rerank_instruct)
            + 64
            + sum(_estimate_rerank_tokens(document) + 16 for document in batch)
        )
        assert estimated <= settings.rerank_request_token_budget


def test_document_rerank_maps_batch_local_indices_to_original_candidates():
    reranker = RecordingReranker()
    answerer = AgenticAnswerer(
        retriever=None,
        settings=Settings(rerank_request_token_budget=250, rerank_instruct=""),
        llm_client=object(),
        rerank_client=reranker,
    )
    candidates = [
        _candidate_with_chunks(f"doc{index}", chunk_texts=[f"{index} " + "x" * 40] * 3)
        for index in range(1, 7)
    ]

    ranked = answerer._rerank_document_candidates(
        question="q",
        query_analysis=QueryAnalysis(target_claim="q"),
        document_candidates=candidates,
    )

    assert len(reranker.calls) > 1
    assert [candidate.doc_id for candidate in ranked[:3]] == ["doc6", "doc5", "doc4"]
    assert ranked[0].rerank_score == 6.0


def test_document_rerank_failure_raises_instead_of_falling_back():
    reranker = RecordingReranker(fail_on_call=2)
    answerer = AgenticAnswerer(
        retriever=None,
        settings=Settings(rerank_request_token_budget=250, rerank_instruct=""),
        llm_client=object(),
        rerank_client=reranker,
    )
    candidates = [
        _candidate_with_chunks(f"doc{index}", chunk_texts=[f"{index} " + "x" * 40] * 3)
        for index in range(1, 7)
    ]

    with pytest.raises(RuntimeError, match="simulated rerank failure"):
        answerer._rerank_document_candidates(
            question="q",
            query_analysis=QueryAnalysis(target_claim="q"),
            document_candidates=candidates,
        )

    assert len(reranker.calls) == 2


class ScriptedAsyncAnswerer(AgenticAnswerer):
    def __init__(self, *, candidates, query_analysis, outcomes, settings: Settings):
        super().__init__(retriever=None, settings=settings, llm_client=object())
        self._candidates = candidates
        self._query_analysis = query_analysis
        self._outcomes = outcomes
        self._attempts: dict[str, int] = {}

    async def analyze(self, question: str) -> QueryAnalysis:
        del question
        return self._query_analysis

    def _retrieve_document_candidates(
        self,
        *,
        question: str,
        query_analysis: QueryAnalysis,
        search_fn,
    ) -> tuple[list[DocumentCandidate], list[str]]:
        del question, query_analysis, search_fn
        return self._candidates, ["test-query"]

    async def judge_document(
        self,
        *,
        question: str,
        query_analysis: QueryAnalysis,
        document_candidate: DocumentCandidate,
    ) -> DocumentAssessment:
        del question, query_analysis
        outcome = self._outcomes[document_candidate.doc_id]
        if isinstance(outcome, list):
            attempt_index = self._attempts.get(document_candidate.doc_id, 0)
            delay, payload = outcome[min(attempt_index, len(outcome) - 1)]
            self._attempts[document_candidate.doc_id] = attempt_index + 1
        else:
            delay, payload = outcome
        await asyncio.sleep(delay)
        if isinstance(payload, Exception):
            raise payload
        return payload

    async def _build_answer_from_document_assessments(
        self,
        *,
        question: str,
        query_analysis: QueryAnalysis,
        document_candidates: list[DocumentCandidate],
        direct_assessments: dict[str, DocumentAssessment],
        background_assessments: dict[str, DocumentAssessment],
        used_queries: list[str],
    ) -> AgentAnswer:
        del question, query_analysis, document_candidates, direct_assessments, background_assessments
        return AgentAnswer(
            answer="final",
            citations=[],
            used_queries=used_queries,
            rounds=1,
            confidence="high",
        )


class ScriptedRecallAnswerer(AgenticAnswerer):
    def __init__(self, *, candidates, query_analysis, settings: Settings, rerank_client):
        super().__init__(
            retriever=None,
            settings=settings,
            llm_client=object(),
            rerank_client=rerank_client,
        )
        self._candidates = candidates
        self._query_analysis = query_analysis

    async def analyze(self, question: str) -> QueryAnalysis:
        del question
        return self._query_analysis

    def _recall_document_candidates(
        self,
        *,
        question: str,
        query_analysis: QueryAnalysis,
        search_fn,
    ) -> tuple[list[DocumentCandidate], list[str]]:
        del question, query_analysis, search_fn
        return self._candidates, ["test-query"]

    async def judge_document(
        self,
        *,
        question: str,
        query_analysis: QueryAnalysis,
        document_candidate: DocumentCandidate,
    ) -> DocumentAssessment:
        del question, query_analysis, document_candidate
        raise AssertionError("judge_document should not run when rerank fails")


class ScriptedResearchAnswerer(AgenticAnswerer):
    def __init__(self, *, hits, query_analysis, settings: Settings, rerank_client):
        super().__init__(
            retriever=None,
            settings=settings,
            llm_client=object(),
            rerank_client=rerank_client,
        )
        self._hits = hits
        self._query_analysis = query_analysis

    async def analyze(self, question: str) -> QueryAnalysis:
        del question
        return self._query_analysis

    def _recall_research_hits(
        self,
        *,
        question: str,
        query_analysis: QueryAnalysis,
        search_fn,
    ) -> tuple[list[SearchHit], list[str]]:
        del question, query_analysis, search_fn
        return self._hits, ["test-query"]


class ScriptedJudgeAnswerer(AgenticAnswerer):
    def __init__(self, *, candidates, query_analysis, outcomes, settings: Settings):
        super().__init__(retriever=None, settings=settings, llm_client=object())
        self._candidates = candidates
        self._query_analysis = query_analysis
        self._outcomes = outcomes

    async def analyze(self, question: str) -> QueryAnalysis:
        del question
        return self._query_analysis

    def _retrieve_document_candidates(
        self,
        *,
        question: str,
        query_analysis: QueryAnalysis,
        search_fn,
    ) -> tuple[list[DocumentCandidate], list[str]]:
        del question, query_analysis, search_fn
        return self._candidates, ["test-query"]

    async def judge_document(
        self,
        *,
        question: str,
        query_analysis: QueryAnalysis,
        document_candidate: DocumentCandidate,
    ) -> DocumentAssessment:
        del question, query_analysis
        delay, payload = self._outcomes[document_candidate.doc_id]
        await asyncio.sleep(delay)
        if isinstance(payload, Exception):
            raise payload
        return payload


async def _collect_answer_stream_events(answerer: AgenticAnswerer):
    events = []
    async for event in answerer.answer_stream("test", search_fn=lambda **kwargs: []):
        events.append(event)
    return events


def test_answer_stream_rerank_failure_raises_stage_error():
    query_analysis = QueryAnalysis(
        target_claim="Find evidence",
        query_bundles=[QueryBundle(bundle_name="b1", query="test")],
    )
    answerer = ScriptedRecallAnswerer(
        candidates=[
            _candidate_with_chunks("doc1", chunk_texts=["one"] * 3),
            _candidate_with_chunks("doc2", chunk_texts=["two"] * 3),
        ],
        query_analysis=query_analysis,
        settings=Settings(rerank_request_token_budget=100000, rerank_instruct=""),
        rerank_client=RecordingReranker(fail_on_call=1),
    )

    with pytest.raises(AgentStageError) as exc_info:
        asyncio.run(_collect_answer_stream_events(answerer))

    assert exc_info.value.stage == "rerank"
    assert "simulated rerank failure" in exc_info.value.message


def _candidate(doc_id: str) -> DocumentCandidate:
    return DocumentCandidate(
        doc_id=doc_id,
        title=doc_id,
        source_path=f"{doc_id}.md",
        aggregate_score=1.0,
        top_chunks=[
            SearchHit(
                chunk_id=f"{doc_id}:0001",
                score_final=1.0,
                retrieval_source="hybrid",
                text="text",
                citation=f"{doc_id} citation",
                title=doc_id,
                source_path=f"{doc_id}.md",
                page_start=1,
                page_end=1,
            )
        ],
    )


def _candidate_with_chunks(doc_id: str, *, chunk_texts: list[str]) -> DocumentCandidate:
    return DocumentCandidate(
        doc_id=doc_id,
        title=doc_id,
        source_path=f"{doc_id}.md",
        aggregate_score=1.0,
        top_chunks=[
            SearchHit(
                chunk_id=f"{doc_id}:{index:04d}",
                score_final=1.0 / index,
                retrieval_source="hybrid",
                text=text,
                citation=f"{doc_id} citation {index}",
                title=doc_id,
                source_path=f"{doc_id}.md",
                page_start=index,
                page_end=index,
                section_hint=f"section {index}",
            )
            for index, text in enumerate(chunk_texts, start=1)
        ],
    )


def _assessment(doc_id: str, label: str, *, answer_line: str = "") -> DocumentAssessment:
    return DocumentAssessment(
        doc_id=doc_id,
        label=label,
        supporting_chunk_ids=[f"{doc_id}:0001"] if label != "off_target" else [],
        reason=label,
        answer_line=answer_line,
    )


def test_answer_stream_commits_document_results_in_rank_order():
    query_analysis = QueryAnalysis(
        target_claim="Find evidence",
        query_bundles=[QueryBundle(bundle_name="b1", query="test")],
    )
    answerer = ScriptedAsyncAnswerer(
        candidates=[_candidate("doc1"), _candidate("doc2"), _candidate("doc3")],
        query_analysis=query_analysis,
        outcomes={
            "doc1": (0.05, _assessment("doc1", "direct_support")),
            "doc2": (0.0, _assessment("doc2", "background_only")),
            "doc3": (0.01, _assessment("doc3", "off_target")),
        },
        settings=Settings(document_judge_initial_concurrency=5),
    )

    events = asyncio.run(_collect_answer_stream_events(answerer))
    judged_doc_ids = [event.document.doc_id for event in events if event.event == "document_judged"]

    assert judged_doc_ids == ["doc1", "doc2", "doc3"]
    assert events[-1].event == "scan_completed"
    assert events[-1].stage_timings is not None
    assert events[-1].performance_counters is not None


def test_answer_stream_uses_default_judge_concurrency_of_5():
    query_analysis = QueryAnalysis(
        target_claim="Find evidence",
        query_bundles=[QueryBundle(bundle_name="b1", query="test")],
    )
    answerer = ScriptedAsyncAnswerer(
        candidates=[_candidate("doc1")],
        query_analysis=query_analysis,
        outcomes={"doc1": (0.0, _assessment("doc1", "direct_support"))},
        settings=Settings(),
    )

    events = asyncio.run(_collect_answer_stream_events(answerer))

    assert events[0].event == "scan_started"
    assert events[0].active_judge_concurrency == 5
    assert events[-1].performance_counters is not None
    assert events[-1].performance_counters.judge_concurrency_initial == 5


def test_answer_stream_retries_provider_pressure_until_success():
    query_analysis = QueryAnalysis(
        target_claim="Find evidence",
        query_bundles=[QueryBundle(bundle_name="b1", query="test")],
    )
    answerer = ScriptedAsyncAnswerer(
        candidates=[_candidate(f"doc{i}") for i in range(1, 7)],
        query_analysis=query_analysis,
        outcomes={
            "doc1": [
                (
                    0.0,
                    AgentStageError(
                        stage="judge_document",
                        message="429 Too Many Requests",
                        question="test",
                    ),
                ),
                (0.0, _assessment("doc1", "direct_support")),
            ],
            "doc2": (0.01, _assessment("doc2", "background_only")),
            "doc3": (0.01, _assessment("doc3", "background_only")),
            "doc4": (0.01, _assessment("doc4", "background_only")),
            "doc5": (0.01, _assessment("doc5", "background_only")),
            "doc6": (0.01, _assessment("doc6", "background_only")),
        },
        settings=Settings(),
    )

    events = asyncio.run(_collect_answer_stream_events(answerer))
    judged_events = [event for event in events if event.event == "document_judged"]

    assert judged_events[0].document.doc_id == "doc1"
    assert judged_events[0].assessment is not None
    assert judged_events[0].assessment.label == "direct_support"
    assert judged_events[0].error is None
    assert judged_events[0].active_judge_concurrency == 4
    assert all(event.error is None for event in judged_events)
    assert events[-1].failed_documents == 0
    assert events[-1].performance_counters is not None
    assert events[-1].performance_counters.provider_pressure_events == 1
    assert events[-1].performance_counters.judge_concurrency_initial == 5
    assert events[-1].performance_counters.judge_concurrency_final == 4


def test_answer_stream_provider_pressure_reduces_concurrency_to_one_without_recovering():
    pressure_error = AgentStageError(
        stage="judge_document",
        message="429 Too Many Requests",
        question="test",
    )
    query_analysis = QueryAnalysis(
        target_claim="Find evidence",
        query_bundles=[QueryBundle(bundle_name="b1", query="test")],
    )
    answerer = ScriptedAsyncAnswerer(
        candidates=[_candidate("doc1")],
        query_analysis=query_analysis,
        outcomes={
            "doc1": [
                (0.0, pressure_error),
                (0.0, pressure_error),
                (0.0, pressure_error),
                (0.0, pressure_error),
                (0.0, _assessment("doc1", "background_only")),
            ],
        },
        settings=Settings(),
    )

    events = asyncio.run(_collect_answer_stream_events(answerer))
    judged_events = [event for event in events if event.event == "document_judged"]

    assert len(judged_events) == 1
    assert judged_events[0].error is None
    assert judged_events[0].active_judge_concurrency == 1
    assert events[-1].failed_documents == 0
    assert events[-1].performance_counters is not None
    assert events[-1].performance_counters.provider_pressure_events == 4
    assert events[-1].performance_counters.judge_concurrency_initial == 5
    assert events[-1].performance_counters.judge_concurrency_final == 1


def test_answer_stream_preserves_rank_order_when_first_document_is_retried():
    query_analysis = QueryAnalysis(
        target_claim="Find evidence",
        query_bundles=[QueryBundle(bundle_name="b1", query="test")],
    )
    answerer = ScriptedAsyncAnswerer(
        candidates=[_candidate("doc1"), _candidate("doc2"), _candidate("doc3")],
        query_analysis=query_analysis,
        outcomes={
            "doc1": [
                (
                    0.0,
                    AgentStageError(
                        stage="judge_document",
                        message="request timed out",
                        question="test",
                    ),
                ),
                (0.03, _assessment("doc1", "background_only")),
            ],
            "doc2": (0.0, _assessment("doc2", "direct_support")),
            "doc3": (0.0, _assessment("doc3", "off_target")),
        },
        settings=Settings(),
    )

    events = asyncio.run(_collect_answer_stream_events(answerer))
    judged_doc_ids = [event.document.doc_id for event in events if event.event == "document_judged"]

    assert judged_doc_ids == ["doc1", "doc2", "doc3"]
    assert events[-1].failed_documents == 0
    assert events[-1].performance_counters is not None
    assert events[-1].performance_counters.provider_pressure_events == 1


def test_answer_stream_non_provider_error_is_not_retried_forever():
    query_analysis = QueryAnalysis(
        target_claim="Find evidence",
        query_bundles=[QueryBundle(bundle_name="b1", query="test")],
    )
    answerer = ScriptedAsyncAnswerer(
        candidates=[_candidate("doc1"), _candidate("doc2")],
        query_analysis=query_analysis,
        outcomes={
            "doc1": (
                0.0,
                AgentStageError(
                    stage="judge_document",
                    message="schema validation failed",
                    question="test",
                ),
            ),
            "doc2": (0.0, _assessment("doc2", "background_only")),
        },
        settings=Settings(),
    )

    events = asyncio.run(_collect_answer_stream_events(answerer))
    judged_events = [event for event in events if event.event == "document_judged"]

    assert judged_events[0].document.doc_id == "doc1"
    assert judged_events[0].error == "schema validation failed"
    assert events[-1].failed_documents == 1
    assert events[-1].performance_counters is not None
    assert events[-1].performance_counters.provider_pressure_events == 0


def test_answer_stream_outputs_all_direct_support_documents():
    query_analysis = QueryAnalysis(
        target_claim="Find evidence",
        query_bundles=[QueryBundle(bundle_name="b1", query="test")],
    )
    candidates = [_candidate(f"doc{i}") for i in range(1, 17)]
    outcomes = {
        candidate.doc_id: (
            0.0,
            _assessment(
                candidate.doc_id,
                "direct_support",
                answer_line=(
                    f"{candidate.doc_id} provides chunk-grounded support for the claim. "
                    f"The cited chunks explain why {candidate.doc_id} directly answers the question."
                ),
            ),
        )
        for candidate in candidates
    }
    answerer = ScriptedJudgeAnswerer(
        candidates=candidates,
        query_analysis=query_analysis,
        outcomes=outcomes,
        settings=Settings(),
    )

    result = asyncio.run(answerer.answer("test", search_fn=lambda **kwargs: []))

    assert result.answer.count("\n\n") == 15
    assert result.answer.startswith("1. doc1 provides chunk-grounded support for the claim.")
    assert "16. doc16 provides chunk-grounded support for the claim." in result.answer
    assert len(result.citations) == 16
    assert result.citations[0] == "[1] doc1 citation"
    assert result.citations[-1] == "[16] doc16 citation"


def test_answer_without_direct_support_returns_fixed_message():
    query_analysis = QueryAnalysis(
        target_claim="Find evidence",
        query_bundles=[QueryBundle(bundle_name="b1", query="test")],
    )
    answerer = ScriptedJudgeAnswerer(
        candidates=[_candidate("doc1"), _candidate("doc2")],
        query_analysis=query_analysis,
        outcomes={
            "doc1": (0.0, _assessment("doc1", "background_only")),
            "doc2": (0.0, _assessment("doc2", "off_target")),
        },
        settings=Settings(),
    )

    result = asyncio.run(answerer.answer("test", search_fn=lambda **kwargs: []))

    assert result.answer == "现有检索到的文献片段中，没有找到直接支持该问题的文献内容。"
    assert result.citations == []


def test_research_chunk_rerank_processes_default_top_500():
    settings = Settings()
    settings.research_rerank_chunk_limit = 500
    reranker = RecordingReranker()
    answerer = AgenticAnswerer(
        retriever=None,
        settings=settings,
        llm_client=FakeLLMClient({}),
        rerank_client=reranker,
    )
    hits = [_make_research_hit(index) for index in range(600)]
    query_analysis = QueryAnalysis(
        target_claim="Find evidence",
        query_bundles=[QueryBundle(bundle_name="b1", query="test")],
    )

    reranked, chunks_reranked = answerer._rerank_research_hits(
        question="test",
        query_analysis=query_analysis,
        hits=hits,
    )

    assert chunks_reranked == 500
    assert sum(len(call) for call in reranker.calls) == 500
    assert len(reranked) == 600


def test_research_rerank_batches_by_shared_token_budget():
    reranker = RecordingReranker()
    settings = Settings(rerank_request_token_budget=250, rerank_instruct="")
    answerer = AgenticAnswerer(
        retriever=None,
        settings=settings,
        llm_client=FakeLLMClient({}),
        rerank_client=reranker,
    )
    hits = [
        _make_research_hit(index, text=f"{index} " + "x" * 160)
        for index in range(1, 7)
    ]
    query_analysis = QueryAnalysis(
        target_claim="q",
        query_bundles=[QueryBundle(bundle_name="b1", query="test")],
    )

    answerer._rerank_research_hits(
        question="q",
        query_analysis=query_analysis,
        hits=hits,
    )

    assert len(reranker.calls) > 1
    for batch in reranker.calls:
        estimated = (
            _estimate_rerank_tokens("q")
            + _estimate_rerank_tokens(settings.rerank_instruct)
            + 64
            + sum(_estimate_rerank_tokens(document) + 16 for document in batch)
        )
        assert estimated <= settings.rerank_request_token_budget


def test_research_rerank_failure_diagnostics_include_batch_info():
    query_analysis = QueryAnalysis(
        target_claim="q",
        query_bundles=[QueryBundle(bundle_name="b1", query="test")],
    )
    settings = Settings(rerank_request_token_budget=250, rerank_instruct="")
    hits = [
        _make_research_hit(index, text=f"{index} " + "x" * 160)
        for index in range(1, 7)
    ]
    answerer = ScriptedResearchAnswerer(
        hits=hits,
        query_analysis=query_analysis,
        settings=settings,
        rerank_client=RecordingReranker(fail_on_call=1),
    )

    with pytest.raises(AgentStageError) as exc_info:
        asyncio.run(answerer.research("test", search_fn=lambda **kwargs: []))

    payload = exc_info.value.diagnostic_payload()
    assert payload["stage"] == "research_rerank"
    assert "simulated rerank failure" in payload["message"]
    assert payload["rerank_documents"] == 6
    assert payload["rerank_batches"] > 1
    assert payload["rerank_request_token_budget"] == 250


def test_research_context_limits_chunks_without_truncating_text():
    settings = Settings()
    settings.research_final_chunk_limit = 100
    settings.research_context_token_budget = 3000
    answerer = AgenticAnswerer(
        retriever=None,
        settings=settings,
        llm_client=FakeLLMClient({}),
    )
    short_text = "short evidence text"
    long_text = "long evidence text " * 2000
    hits = [
        _make_research_hit(1, text=short_text),
        _make_research_hit(2, text=long_text),
        _make_research_hit(3, text=long_text),
    ]

    selected = answerer._select_research_context_hits(hits)

    assert selected == [hits[0]]
    assert selected[0].text == short_text


def test_research_context_uses_at_most_100_chunks():
    settings = Settings()
    settings.research_final_chunk_limit = 100
    settings.research_context_token_budget = 120000
    answerer = AgenticAnswerer(
        retriever=None,
        settings=settings,
        llm_client=FakeLLMClient({}),
    )
    hits = [_make_research_hit(index, text="compact evidence") for index in range(120)]

    selected = answerer._select_research_context_hits(hits)

    assert len(selected) == 100


def test_research_report_prompt_is_natural_and_omits_query_plan():
    report = _valid_research_report()
    llm = SequentialResearchLLMClient([{"report": report, "used_ref_ids": [1]}])
    answerer = AgenticAnswerer(
        retriever=None,
        settings=Settings(),
        llm_client=llm,
    )
    context_hits = [_make_research_hit(1, text="完整证据片段内容")]

    result, used_ref_ids = asyncio.run(
        answerer._generate_research_report(
            question="请自然说明羟基磷灰石的医学应用",
            query_analysis=QueryAnalysis(target_claim="Find evidence"),
            context_hits=context_hits,
        )
    )

    assert result == report
    assert used_ref_ids == [1]
    assert "Structured query plan" not in llm.user_prompts[0]
    assert "请自然说明羟基磷灰石的医学应用" in llm.user_prompts[0]
    assert "完整证据片段内容" in llm.user_prompts[0]
    assert "There is no maximum length limit" in llm.user_prompts[0]
    assert "taxonomy" not in llm.system_prompts[0].lower()
    assert "major direction" not in llm.system_prompts[0].lower()


def test_research_report_expands_short_report_once():
    repaired = _valid_research_report()
    llm = SequentialResearchLLMClient([
        {"report": "太短[1]", "used_ref_ids": [1]},
        {"report": repaired, "used_ref_ids": [1]},
    ])
    answerer = AgenticAnswerer(
        retriever=None,
        settings=Settings(),
        llm_client=llm,
    )

    result, used_ref_ids = asyncio.run(
        answerer._generate_research_report(
            question="test",
            query_analysis=QueryAnalysis(target_claim="Find evidence"),
            context_hits=[_make_research_hit(1, text="evidence")],
        )
    )

    assert result == repaired
    assert used_ref_ids == [1]
    assert len(llm.user_prompts) == 2
    assert "shorter than the required" in llm.user_prompts[1]
    assert "Evidence chunks:" in llm.user_prompts[1]


def test_research_report_accepts_long_report_without_repair():
    long_report = "长报告内容" * 160 + "[1]"
    llm = SequentialResearchLLMClient([
        {"report": long_report, "used_ref_ids": [1]},
    ])
    answerer = AgenticAnswerer(
        retriever=None,
        settings=Settings(),
        llm_client=llm,
    )

    result, used_ref_ids = asyncio.run(
        answerer._generate_research_report(
            question="test",
            query_analysis=QueryAnalysis(target_claim="Find evidence"),
            context_hits=[_make_research_hit(1, text="evidence")],
        )
    )

    assert result == long_report
    assert used_ref_ids == [1]
    assert len(llm.user_prompts) == 1


def test_research_report_raises_when_repair_still_out_of_range():
    llm = SequentialResearchLLMClient([
        {"report": "太短[1]", "used_ref_ids": [1]},
        {"report": "仍然太短[1]", "used_ref_ids": [1]},
    ])
    answerer = AgenticAnswerer(
        retriever=None,
        settings=Settings(),
        llm_client=llm,
    )

    with pytest.raises(AgentStageError) as exc_info:
        asyncio.run(
            answerer._generate_research_report(
                question="test",
                query_analysis=QueryAnalysis(target_claim="Find evidence"),
                context_hits=[_make_research_hit(1, text="evidence")],
            )
        )

    assert exc_info.value.stage == "research_generate"
    assert "shorter than required minimum" in exc_info.value.message


def test_research_citations_merge_multiple_chunks_from_same_paper():
    first = _make_research_hit(1, text="first evidence")
    first.doc_id = "review-doc"
    first.title = "Hydroxyapatite Review"
    first.chunk_id = "review-doc:0001"
    first.citation = "Hydroxyapatite Review (p. 1, chunk review-doc:0001)"
    second = _make_research_hit(2, text="second evidence")
    second.doc_id = "review-doc"
    second.title = "Hydroxyapatite Review"
    second.chunk_id = "review-doc:0002"
    second.citation = "Hydroxyapatite Review (p. 2, chunk review-doc:0002)"
    third = _make_research_hit(3, text="other evidence")
    third.doc_id = "coating-doc"
    third.title = "Hydroxyapatite Coatings"
    third.chunk_id = "coating-doc:0001"
    third.citation = "Hydroxyapatite Coatings (p. 3, chunk coating-doc:0001)"

    report, citations = _merge_research_report_citations(
        "骨修复方向由多段证据支持[1][2]，涂层方向由另一篇支持[3]。",
        [1, 2, 3],
        [first, second, third],
    )

    assert report == "骨修复方向由多段证据支持[1]，涂层方向由另一篇支持[2]。"
    assert len(citations) == 2
    assert citations[0].startswith("[1] Hydroxyapatite Review")
    assert "doc:0001" in citations[0]
    assert "doc:0002" in citations[0]
    assert citations[1].startswith("[2] Hydroxyapatite Coatings")


def test_research_citations_rewrite_comma_group_same_paper():
    hits = [_make_research_hit(index) for index in range(1, 83)]
    for ref_id in (8, 82):
        hit = hits[ref_id - 1]
        hit.doc_id = "md-doc"
        hit.title = "MD Hydroxyapatite Paper"
        hit.chunk_id = f"md-doc:{ref_id:04d}"
        hit.citation = f"MD Hydroxyapatite Paper (p. 0, chunk md-doc:{ref_id:04d})"

    report, citations = _merge_research_report_citations(
        "MD 模拟适合研究界面水和离子吸附[8, 82]。",
        [8, 82],
        hits,
    )

    assert report == "MD 模拟适合研究界面水和离子吸附[1]。"
    assert len(citations) == 1
    assert citations[0].startswith("[1] MD Hydroxyapatite Paper")
    assert "doc:0008" in citations[0]
    assert "doc:0082" in citations[0]


def test_research_citations_rewrite_comma_group_multiple_papers():
    hits = [_make_research_hit(index) for index in range(1, 31)]
    for ref_id in (9, 18):
        hit = hits[ref_id - 1]
        hit.doc_id = "method-a"
        hit.title = "Method A"
        hit.chunk_id = f"method-a:{ref_id:04d}"
        hit.citation = f"Method A (p. 0, chunk method-a:{ref_id:04d})"
    hit = hits[29]
    hit.doc_id = "method-b"
    hit.title = "Method B"
    hit.chunk_id = "method-b:0030"
    hit.citation = "Method B (p. 0, chunk method-b:0030)"

    report, citations = _merge_research_report_citations(
        "机器学习势函数可连接 DFT 精度和 MD 尺度[9, 18, 30]。",
        [9, 18, 30],
        hits,
    )

    assert report == "机器学习势函数可连接 DFT 精度和 MD 尺度[1][2]。"
    assert len(citations) == 2
    assert citations[0].startswith("[1] Method A")
    assert citations[1].startswith("[2] Method B")


def test_research_citations_drop_invalid_ref_ids():
    hit = _make_research_hit(1)

    report, citations = _merge_research_report_citations(
        "这个结论没有有效引用[999]，但这个引用有效[1]。",
        [1, 999],
        [hit],
    )

    assert report == "这个结论没有有效引用，但这个引用有效[1]。"
    assert len(citations) == 1
