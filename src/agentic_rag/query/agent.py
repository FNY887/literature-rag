from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass
from time import perf_counter
from typing import Any, AsyncIterator

from agentic_rag.core.config import Settings
from agentic_rag.core.llm import LLMClient
from agentic_rag.core.models import (
    AgentAnswer,
    AnswerStreamEvent,
    DocumentAssessment,
    DocumentCandidate,
    PerformanceCounters,
    QueryAnalysis,
    QueryBundle,
    ResearchAnswer,
    ResearchStageTimings,
    SearchHit,
    StageTimings,
)
from agentic_rag.core.utils import dedupe_strings
from agentic_rag.builder.store import _shorten_chunk_id
from agentic_rag.query.retriever import build_fts_query_from_phrases, rerank_hits_for_query_plan


ANALYZE_SYSTEM_PROMPT = """You are a retrieval planner for scientific literature QA.
The corpus is in English, but the user may ask in Chinese.
Your task is to translate the user's scientific request into a strict, constraint-aware retrieval plan.

Rules:
1. Preserve the user's exact claim and hard constraints. Do not generalize away conditions such as pH, alkaline/basic media, confinement scale, or evidence type.
2. Convert Chinese scientific phrasing into precise English retrieval terms used in literature.
3. For questions that ask to explain/prove/show whether a phenomenon occurs under a condition, treat the condition as a HARD constraint, not optional background.
4. Produce multiple query bundles that search different angles without losing the same hard constraints.
5. query_bundles MUST be English and should usually include:
   - exact_condition
   - numeric_condition when pH/ranges/numbers matter
   - domain_entity
   - mechanistic_variant

Return JSON only with these fields:
- original_question: original user question verbatim
- target_claim: one-sentence English statement of what must be supported
- evidence_type: usually "direct_support"
- question_language: "zh" or "en"
- intent_type: short label such as "evidence_request"
- analysis: 1-2 sentences describing the claim and hard constraints
- must_have_terms: list of English must-have retrieval terms
- optional_terms: list of helpful but non-essential English terms
- entity_terms_en: list of English domain entity terms
- condition_terms_en: list of English condition terms
- relation_terms_en: list of English relation/mechanism terms
- numeric_constraints: list of numeric constraints such as pH values/ranges
- query_variants: list of English search queries
- keyword_phrases: list of precise English phrases for FTS
- query_bundles: list of objects with fields bundle_name, query, keyword_phrases, required_terms
- must_match_groups: list of term groups; each group is OR internally but groups are jointly important
- diagnostic_notes: short notes for debugging
- mode: one of "keyword", "vector", "hybrid"
- top_k_vector: integer
- top_k_keyword: integer
"""

DOCUMENT_FILTER_SYSTEM_PROMPT = """You are a strict document-level evidence judge for scientific literature QA.
You are given exactly ONE candidate paper together with a few candidate passages from that paper.

Labels:
- direct_support: this paper contains passage(s) that directly support the target claim and hard constraints
- background_only: related paper, but the provided passages do not directly satisfy the required constraint(s)
- off_target: not useful for the target claim

Rules:
1. Judge ONLY the provided paper and ONLY the provided chunk_ids.
2. If label is direct_support or background_only, supporting_chunk_ids should point to the best supporting chunk_ids from this paper.
3. direct_support requires explicit support for the hard constraints, not just topical overlap.
4. Do not cite chunk_ids that are not present in the document candidate payload.
5. If label is direct_support, you MUST provide answer_line: a 2-3 sentence chunk-grounded explanation for this paper only.
6. answer_line should explain how the provided chunks directly support the user's question or target claim.
7. If the chunks mention specific entities, stages, timing, conditions, observations, or mechanisms, include those details.
8. Do not invent methods, experimental design, or evidence sources; mention them only if the provided chunks explicitly include them.
9. Avoid overly generic claims such as "this study supports the claim" without explaining the chunk-grounded evidence.
10. answer_line MUST use the same language as the user's question.
11. If label is background_only or off_target, answer_line must be an empty string.

Return JSON only with these fields:
- doc_id
- label
- supporting_chunk_ids
- matched_constraints
- missing_constraints
- reason
- answer_line
"""

RESEARCH_REPORT_SYSTEM_PROMPT = """You are a scientific literature research report writer.
You are given a user question and ranked evidence chunks.

Rules:
1. Use ONLY the provided chunks. Do not use outside knowledge.
2. Write in the same language as the user's question.
3. Respect the requested character range. For Chinese, one Chinese character counts as one character.
4. Cite evidence inline using the provided reference ids like [1], [2].
5. You do not need to use every chunk; use the best chunks that answer the question.
6. Do not force a fixed structure. Write naturally; use paragraphs or bullets only when they fit the evidence and question.
7. Do not cite reference ids that are not present in the provided chunks.
8. If the chunks are insufficient, state the limitation clearly and cite the closest evidence.

Return JSON only with these fields:
- report: the research report text
- used_ref_ids: list of numeric reference ids cited or used
"""

RESEARCH_REPORT_REPAIR_SYSTEM_PROMPT = """You revise a chunk-grounded research report to satisfy a length requirement.
Keep the same language, preserve the important claims and inline citations, and do not add evidence outside the provided chunks.

Return JSON only with these fields:
- report
- used_ref_ids
"""

NO_DIRECT_SUPPORT_ANSWER = "现有检索到的文献片段中，没有找到直接支持该问题的文献内容。"

EARLY_STOP_CONSECUTIVE_NON_SUPPORT = 20


def _contains_cjk(text: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff]", text))


def _query_dedup_key(text: str) -> str:
    normalized = re.sub(r"[^\w\u4e00-\u9fff]+", " ", text.lower())
    return " ".join(normalized.split())


def _diverse_hits(hits: list[SearchHit], max_per_doc: int = 3, total_limit: int = 20) -> list[SearchHit]:
    doc_counts: dict[str, int] = {}
    diverse: list[SearchHit] = []
    for hit in hits:
        doc_key = hit.doc_id or hit.source_path or hit.title or hit.chunk_id
        count = doc_counts.get(doc_key, 0)
        if count >= max_per_doc:
            continue
        doc_counts[doc_key] = count + 1
        diverse.append(hit)
        if len(diverse) >= total_limit:
            break
    return diverse


def _coerce_string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if isinstance(value, dict):
        flattened: list[str] = []
        for item in value.values():
            flattened.extend(_coerce_string_list(item))
        return dedupe_strings(flattened)
    if isinstance(value, list):
        flattened: list[str] = []
        for item in value:
            flattened.extend(_coerce_string_list(item))
        return dedupe_strings(flattened)
    text = str(value).strip()
    return [text] if text else []


def _coerce_string_groups(value: Any) -> list[list[str]]:
    if value is None:
        return []
    if isinstance(value, dict):
        groups = [_coerce_string_list(item) for item in value.values()]
        return [group for group in groups if group]
    if isinstance(value, list):
        groups: list[list[str]] = []
        for item in value:
            group = _coerce_string_list(item)
            if group:
                groups.append(group)
        return groups
    group = _coerce_string_list(value)
    return [group] if group else []


def _coerce_query_bundles(value: Any) -> list[dict[str, Any]]:
    raw_items: list[dict[str, Any]] = []
    if value is None:
        return []
    if isinstance(value, dict):
        for name, item in value.items():
            if isinstance(item, dict):
                payload = dict(item)
                payload.setdefault("bundle_name", str(payload.get("bundle_name") or name))
            else:
                payload = {"bundle_name": str(name), "query": str(item)}
            raw_items.append(payload)
    elif isinstance(value, list):
        for index, item in enumerate(value, start=1):
            if isinstance(item, dict):
                payload = dict(item)
                payload.setdefault("bundle_name", str(payload.get("bundle_name") or f"bundle_{index}"))
            else:
                payload = {"bundle_name": f"bundle_{index}", "query": str(item)}
            raw_items.append(payload)
    elif isinstance(value, str):
        raw_items.append({"bundle_name": "bundle_1", "query": value})

    normalized: list[dict[str, Any]] = []
    seen_queries: set[str] = set()
    for item in raw_items:
        query = str(item.get("query", "")).strip()
        if not query:
            continue
        query_key = _query_dedup_key(query)
        if query_key in seen_queries:
            continue
        seen_queries.add(query_key)
        normalized.append(
            {
                "bundle_name": str(item.get("bundle_name", "bundle")).strip() or "bundle",
                "query": query,
                "keyword_phrases": _coerce_string_list(item.get("keyword_phrases")),
                "required_terms": _coerce_string_list(item.get("required_terms")),
            }
        )
    return normalized


def _make_bundle(
    bundle_name: str,
    *,
    query_parts: list[str],
    keyword_phrases: list[str],
    required_terms: list[str],
) -> dict[str, Any] | None:
    query = " ".join(dedupe_strings(query_parts))
    if not query:
        return None
    return {
        "bundle_name": bundle_name,
        "query": query,
        "keyword_phrases": dedupe_strings(keyword_phrases),
        "required_terms": dedupe_strings(required_terms),
    }


def _default_query_bundles(
    *,
    entity_terms: list[str],
    condition_terms: list[str],
    relation_terms: list[str],
    numeric_constraints: list[str],
) -> list[dict[str, Any]]:
    bundles: list[dict[str, Any]] = []

    exact_condition = _make_bundle(
        "exact_condition",
        query_parts=condition_terms[:3] + entity_terms[:3],
        keyword_phrases=condition_terms[:3] + entity_terms[:2],
        required_terms=condition_terms[:3] + entity_terms[:2],
    )
    if exact_condition is not None:
        bundles.append(exact_condition)

    numeric_condition = _make_bundle(
        "numeric_condition",
        query_parts=numeric_constraints[:3] + entity_terms[:3],
        keyword_phrases=numeric_constraints[:3] + entity_terms[:2],
        required_terms=numeric_constraints[:3] + condition_terms[:2] + entity_terms[:2],
    )
    if numeric_condition is not None:
        bundles.append(numeric_condition)

    domain_entity = _make_bundle(
        "domain_entity",
        query_parts=entity_terms[:4] + relation_terms[:2] + condition_terms[:1],
        keyword_phrases=entity_terms[:4],
        required_terms=entity_terms[:3] + condition_terms[:1],
    )
    if domain_entity is not None:
        bundles.append(domain_entity)

    mechanistic_variant = _make_bundle(
        "mechanistic_variant",
        query_parts=relation_terms[:3] + entity_terms[:3] + condition_terms[:2],
        keyword_phrases=relation_terms[:3] + entity_terms[:2] + condition_terms[:1],
        required_terms=entity_terms[:2] + condition_terms[:2],
    )
    if mechanistic_variant is not None:
        bundles.append(mechanistic_variant)

    return bundles


def _merge_query_bundles(
    primary: list[dict[str, Any]],
    fallback: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    seen_queries: set[str] = set()
    for bundle in primary + fallback:
        query = str(bundle.get("query", "")).strip()
        if not query:
            continue
        key = _query_dedup_key(query)
        if key in seen_queries:
            continue
        seen_queries.add(key)
        merged.append(bundle)
    return merged[:8]


def _normalize_query_analysis_payload(payload: dict[str, Any], question: str, settings: Settings) -> dict[str, Any]:
    normalized = dict(payload)
    original_question = str(payload.get("original_question") or question).strip()
    target_claim = str(payload.get("target_claim", "")).strip()
    if not target_claim:
        raise ValueError("Missing target_claim in query analysis.")

    must_have_terms = _coerce_string_list(payload.get("must_have_terms"))
    optional_terms = _coerce_string_list(payload.get("optional_terms"))
    entity_terms = _coerce_string_list(payload.get("entity_terms_en"))
    condition_terms = _coerce_string_list(payload.get("condition_terms_en"))
    relation_terms = _coerce_string_list(payload.get("relation_terms_en"))
    numeric_constraints = _coerce_string_list(payload.get("numeric_constraints"))
    diagnostic_notes = _coerce_string_list(payload.get("diagnostic_notes"))
    query_variants = _coerce_string_list(payload.get("query_variants"))
    keyword_phrases = _coerce_string_list(payload.get("keyword_phrases"))

    if not entity_terms:
        entity_terms = must_have_terms[:6]

    query_bundles = _coerce_query_bundles(payload.get("query_bundles"))
    query_bundles = _merge_query_bundles(
        query_bundles,
        _default_query_bundles(
            entity_terms=entity_terms,
            condition_terms=condition_terms,
            relation_terms=relation_terms,
            numeric_constraints=numeric_constraints,
        ),
    )
    if not query_bundles:
        raise ValueError("Missing query_bundles in query analysis.")

    must_match_groups = _coerce_string_groups(payload.get("must_match_groups"))
    if not must_match_groups:
        if entity_terms:
            must_match_groups.append(entity_terms[:6])
        condition_group = dedupe_strings(condition_terms + numeric_constraints)
        if condition_group:
            must_match_groups.append(condition_group[:6])
        if relation_terms:
            must_match_groups.append(relation_terms[:6])

    query_variants = dedupe_strings(query_variants + [bundle["query"] for bundle in query_bundles])
    keyword_phrases = dedupe_strings(
        keyword_phrases
        + [phrase for bundle in query_bundles for phrase in bundle["keyword_phrases"]]
    )
    must_have_terms = dedupe_strings(
        must_have_terms + entity_terms + condition_terms + relation_terms + numeric_constraints
    )

    normalized["original_question"] = original_question
    normalized["target_claim"] = target_claim
    normalized["evidence_type"] = str(payload.get("evidence_type", "direct_support")).strip() or "direct_support"
    normalized["question_language"] = str(
        payload.get("question_language", "zh" if _contains_cjk(question) else "en")
    ).strip() or "unknown"
    normalized["intent_type"] = str(payload.get("intent_type", "literature_qa")).strip() or "literature_qa"
    normalized["analysis"] = str(payload.get("analysis", "")).strip()
    normalized["must_have_terms"] = must_have_terms
    normalized["optional_terms"] = optional_terms
    normalized["entity_terms_en"] = entity_terms
    normalized["condition_terms_en"] = condition_terms
    normalized["relation_terms_en"] = relation_terms
    normalized["numeric_constraints"] = numeric_constraints
    normalized["query_variants"] = query_variants
    normalized["keyword_phrases"] = keyword_phrases
    normalized["query_bundles"] = query_bundles
    normalized["must_match_groups"] = must_match_groups
    normalized["diagnostic_notes"] = diagnostic_notes

    mode = str(payload.get("mode", "hybrid")).strip().lower()
    normalized["mode"] = mode if mode in {"keyword", "vector", "hybrid"} else "hybrid"
    for key, default in (("top_k_vector", settings.top_k_vector), ("top_k_keyword", settings.top_k_keyword)):
        value = payload.get(key, default)
        try:
            normalized[key] = max(1, int(value))
        except (TypeError, ValueError):
            normalized[key] = default
    return normalized


def _normalize_document_assessment_payload(
    payload: dict[str, Any],
    *,
    document_candidate: DocumentCandidate,
) -> dict[str, Any]:
    normalized = dict(payload)
    label = str(payload.get("label", "off_target")).strip().lower()
    if label not in {"direct_support", "background_only", "off_target"}:
        label = "off_target"

    allowed_chunk_ids = [hit.chunk_id for hit in document_candidate.top_chunks]
    allowed_chunk_id_set = set(allowed_chunk_ids)
    supporting_chunk_ids = [
        chunk_id
        for chunk_id in _coerce_string_list(payload.get("supporting_chunk_ids"))
        if chunk_id in allowed_chunk_id_set
    ]
    if label in {"direct_support", "background_only"} and not supporting_chunk_ids and allowed_chunk_ids:
        supporting_chunk_ids = [allowed_chunk_ids[0]]

    normalized["doc_id"] = document_candidate.doc_id
    normalized["label"] = label
    normalized["supporting_chunk_ids"] = supporting_chunk_ids
    normalized["matched_constraints"] = _coerce_string_list(payload.get("matched_constraints"))
    normalized["missing_constraints"] = _coerce_string_list(payload.get("missing_constraints"))
    normalized["reason"] = str(payload.get("reason", "")).strip()
    answer_line = str(payload.get("answer_line", "")).strip()
    if label == "direct_support" and not answer_line:
        raise ValueError("Missing answer_line for direct_support document assessment.")
    normalized["answer_line"] = answer_line if label == "direct_support" else ""
    return normalized


def _sanitize_answer_line(text: str) -> str:
    cleaned = re.sub(r"^\s*\d+\s*[\.\)\、]\s*", "", text).strip()
    return " ".join(cleaned.split())


def _excerpt(text: str | None, limit: int = 800) -> str | None:
    if not text:
        return None
    cleaned = " ".join(text.split())
    if limit <= 0:
        return cleaned
    if len(cleaned) <= limit:
        return cleaned
    if limit <= 3:
        return cleaned[:limit]
    return cleaned[: limit - 3].rstrip() + "..."


def _estimate_rerank_tokens(text: str) -> int:
    compact = " ".join(text.split())
    if not compact:
        return 1
    ascii_chars = sum(1 for char in compact if ord(char) < 128)
    non_ascii_chars = len(compact) - ascii_chars
    return max(1, (ascii_chars + 3) // 4 + non_ascii_chars)


def _exception_raw_response(exc: Exception) -> str | None:
    raw_response = getattr(exc, "raw_response", None)
    return raw_response if isinstance(raw_response, str) else None


@dataclass(slots=True)
class _DocumentJudgeTaskResult:
    document_candidate: DocumentCandidate
    assessment: DocumentAssessment | None = None
    error: AgentStageError | None = None
    is_provider_pressure: bool = False


@dataclass(slots=True)
class _RerankBatch:
    original_indices: list[int]
    documents: list[str]


def _build_rerank_batches(
    *,
    documents: list[str],
    query: str,
    instruct: str | None,
    token_budget: int,
) -> list[_RerankBatch]:
    if not documents:
        return []

    request_overhead = _estimate_rerank_tokens(query) + _estimate_rerank_tokens(instruct or "") + 64
    available_budget = max(1, max(1, token_budget) - request_overhead)
    batches: list[_RerankBatch] = []
    current_indices: list[int] = []
    current_documents: list[str] = []
    current_tokens = 0

    for original_index, document in enumerate(documents):
        document_tokens = _estimate_rerank_tokens(document) + 16
        if current_documents and current_tokens + document_tokens > available_budget:
            batches.append(_RerankBatch(original_indices=current_indices, documents=current_documents))
            current_indices = []
            current_documents = []
            current_tokens = 0
        current_indices.append(original_index)
        current_documents.append(document)
        current_tokens += document_tokens

    if current_documents:
        batches.append(_RerankBatch(original_indices=current_indices, documents=current_documents))
    return batches


def _research_rerank_diagnostics(
    *,
    hits: list[SearchHit],
    settings: Settings,
    question: str,
    query_analysis: QueryAnalysis,
) -> dict[str, Any]:
    rerank_limit = min(len(hits), max(1, settings.research_rerank_chunk_limit))
    if rerank_limit <= 0:
        return {
            "rerank_documents": 0,
            "rerank_batches": 0,
            "rerank_request_token_budget": settings.rerank_request_token_budget,
        }
    rerank_query = query_analysis.target_claim.strip() or question
    documents = [_research_rerank_input(hit) for hit in hits[:rerank_limit]]
    batches = _build_rerank_batches(
        documents=documents,
        query=rerank_query,
        instruct=settings.rerank_instruct,
        token_budget=settings.rerank_request_token_budget,
    )
    return {
        "rerank_documents": rerank_limit,
        "rerank_batches": len(batches),
        "rerank_request_token_budget": settings.rerank_request_token_budget,
    }


def _should_stop_early(*, consecutive_non_support_count: int) -> bool:
    return consecutive_non_support_count >= EARLY_STOP_CONSECUTIVE_NON_SUPPORT


def _build_early_stop_reason() -> str:
    return f"连续 {EARLY_STOP_CONSECUTIVE_NON_SUPPORT} 篇文献都没有 direct_support。"


def _build_scan_status(
    *,
    stopped_early: bool,
    stopped_after_documents: int,
    final_judged_documents: int,
    total_documents: int,
    direct_documents: int,
    stop_reason: str | None,
) -> str:
    if stopped_early:
        reason = stop_reason or "触发了提前停止条件。"
        return (
            f"已在顺序提交第 {stopped_after_documents}/{total_documents} 篇文献后停止派发新文献；"
            f"尾部已派发文献已完成并计入最终结果；最终共判定 {final_judged_documents}/{total_documents} 篇文献；"
            f"已找到 {direct_documents} 篇 direct_support；触发原因：{reason}"
        )
    return f"已完成排序文献扫描，共判定 {final_judged_documents}/{total_documents} 篇文献。"


def _annotate_final_answer(
    answer: AgentAnswer,
    *,
    stopped_early: bool,
    stop_reason: str | None,
    stopped_after_documents: int,
    final_judged_documents: int,
    total_documents: int,
    direct_documents: int,
    stage_timings: StageTimings,
    performance_counters: PerformanceCounters,
) -> AgentAnswer:
    return answer.model_copy(
        update={
            "stopped_early": stopped_early,
            "stop_reason": stop_reason,
            "stopped_after_documents": stopped_after_documents,
            "total_documents": total_documents,
            "scan_status": _build_scan_status(
                stopped_early=stopped_early,
                stopped_after_documents=stopped_after_documents,
                final_judged_documents=final_judged_documents,
                total_documents=total_documents,
                direct_documents=direct_documents,
                stop_reason=stop_reason,
            ),
            "stage_timings": stage_timings,
            "performance_counters": performance_counters,
        }
    )


def _is_provider_pressure_stage_error(exc: AgentStageError) -> bool:
    combined = "\n".join(
        part for part in [exc.message, exc.raw_response] if isinstance(part, str) and part
    ).lower()
    provider_pressure_fragments = (
        "429",
        "timed out",
        "timeout",
        "too many requests",
        "rate limit",
        "engine is currently overloaded",
        "engine_overloaded_error",
        "overloaded",
        "temporarily unavailable",
        "temporary failure",
    )
    return any(fragment in combined for fragment in provider_pressure_fragments)


def _analysis_snapshot(query_analysis: QueryAnalysis | None) -> dict[str, Any] | None:
    if query_analysis is None:
        return None
    return {
        "target_claim": query_analysis.target_claim,
        "evidence_type": query_analysis.evidence_type,
        "entity_terms_en": query_analysis.entity_terms_en,
        "condition_terms_en": query_analysis.condition_terms_en,
        "relation_terms_en": query_analysis.relation_terms_en,
        "numeric_constraints": query_analysis.numeric_constraints,
        "query_bundles": [bundle.model_dump() for bundle in query_analysis.query_bundles],
        "must_match_groups": query_analysis.must_match_groups,
    }


def _prompt_analysis_snapshot(
    query_analysis: QueryAnalysis,
    *,
    include_bundles: bool,
) -> dict[str, Any]:
    snapshot: dict[str, Any] = {
        "target_claim": query_analysis.target_claim,
        "evidence_type": query_analysis.evidence_type,
        "question_language": query_analysis.question_language,
        "entity_terms_en": query_analysis.entity_terms_en[:5],
        "condition_terms_en": query_analysis.condition_terms_en[:6],
        "relation_terms_en": query_analysis.relation_terms_en[:5],
        "numeric_constraints": query_analysis.numeric_constraints[:4],
        "must_match_groups": [group[:5] for group in query_analysis.must_match_groups[:3]],
    }
    if include_bundles:
        snapshot["query_bundles"] = [
            {
                "bundle_name": bundle.bundle_name,
                "query": bundle.query,
                "required_terms": bundle.required_terms[:4],
            }
            for bundle in query_analysis.query_bundles[:5]
        ]
    return snapshot


def _search_hit_payload(hit: SearchHit, *, text_limit: int | None) -> dict[str, Any]:
    text = hit.text if text_limit is None or text_limit <= 0 else hit.text[:text_limit]
    return {
        "doc_id": hit.doc_id,
        "chunk_id": hit.chunk_id,
        "title": hit.title,
        "citation": hit.citation,
        "section_hint": hit.section_hint,
        "score_final": hit.score_final,
        "score_constraint": hit.score_constraint,
        "matched_constraints": hit.matched_constraints[:6],
        "missing_constraints": hit.missing_constraints[:6],
        "text": text,
    }


def _document_key_from_hit(hit: SearchHit) -> str:
    return (
        getattr(hit, "doc_id", "")
        or getattr(hit, "source_path", "")
        or getattr(hit, "title", "")
        or getattr(hit, "chunk_id", "")
    )


def _document_candidate_payload(
    candidate: DocumentCandidate,
    query_analysis: QueryAnalysis,
    *,
    chunk_limit: int,
    text_limit: int | None,
) -> dict[str, Any]:
    return {
        "doc_id": candidate.doc_id,
        "title": candidate.title,
        "source_path": candidate.source_path,
        "aggregate_score": candidate.aggregate_score,
        "matched_bundle_count": candidate.matched_bundle_count,
        "query_plan": _prompt_analysis_snapshot(query_analysis, include_bundles=False),
        "chunks": [
            _search_hit_payload(hit, text_limit=text_limit)
            for hit in candidate.top_chunks[:chunk_limit]
        ],
    }


def _build_merged_citation(hits: list[SearchHit]) -> str:
    if not hits:
        return ""
    title = hits[0].title
    if len(hits) == 1:
        return hits[0].citation
    parts: list[str] = []
    for hit in hits:
        if hit.page_start == hit.page_end:
            page_text = f"p. {hit.page_start}"
        else:
            page_text = f"pp. {hit.page_start}-{hit.page_end}"
        short_id = _shorten_chunk_id(hit.chunk_id)
        parts.append(f"{page_text}, chunk {short_id}")
    return f"{title} ({'; '.join(parts)})"


def _format_numbered_answer(answer_lines: list[str]) -> str:
    return "\n\n".join(f"{index}. {line}" for index, line in enumerate(answer_lines, start=1))


def _numbered_citations(citations: list[str]) -> list[str]:
    return [f"[{i + 1}] {citation}" for i, citation in enumerate(citations)]


def _research_hit_payload(hit: SearchHit, *, ref_id: int) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "ref_id": ref_id,
        "chunk_id": hit.chunk_id,
        "title": hit.title,
        "citation": hit.citation,
        "score_final": hit.score_final,
        "retrieval_source": hit.retrieval_source,
        "text": hit.text,
    }
    if hit.section_hint:
        payload["section_hint"] = hit.section_hint
    if hit.matched_constraints:
        payload["matched_constraints"] = hit.matched_constraints
    return payload


def _research_rerank_input(hit: SearchHit) -> str:
    parts = [f"Title: {hit.title}", f"Chunk: {hit.chunk_id}"]
    if hit.section_hint:
        parts.append(f"Section: {hit.section_hint}")
    parts.append(f"Evidence: {hit.text}")
    return "\n".join(parts).strip()


def _extract_report_ref_ids(report: str) -> list[int]:
    ids: list[int] = []
    seen: set[int] = set()
    for bracket_content in re.findall(r"\[([\d,\s]+)\]", report):
        for match in re.findall(r"\d+", bracket_content):
            ref_id = int(match)
            if ref_id in seen:
                continue
            seen.add(ref_id)
            ids.append(ref_id)
    return ids


def _coerce_ref_ids(value: Any, *, max_ref_id: int) -> list[int]:
    raw_values: list[Any]
    if value is None:
        raw_values = []
    elif isinstance(value, list):
        raw_values = value
    else:
        raw_values = [value]

    ref_ids: list[int] = []
    seen: set[int] = set()
    for item in raw_values:
        try:
            ref_id = int(str(item).strip().strip("[]"))
        except (TypeError, ValueError):
            continue
        if ref_id < 1 or ref_id > max_ref_id or ref_id in seen:
            continue
        seen.add(ref_id)
        ref_ids.append(ref_id)
    return ref_ids


def _normalize_research_report_payload(
    payload: dict[str, Any],
    *,
    max_ref_id: int,
) -> tuple[str, list[int]]:
    report = str(payload.get("report", "")).strip()
    if not report:
        raise ValueError("Missing report in research response.")
    used_ref_ids = _coerce_ref_ids(payload.get("used_ref_ids"), max_ref_id=max_ref_id)
    report_ref_ids = _extract_report_ref_ids(report)
    merged: list[int] = []
    seen: set[int] = set()
    for ref_id in report_ref_ids + used_ref_ids:
        if 1 <= ref_id <= max_ref_id and ref_id not in seen:
            seen.add(ref_id)
            merged.append(ref_id)
    return report, merged


def _research_report_length_status(report: str, *, min_chars: int) -> str | None:
    report_length = len(report)
    if min_chars > 0 and report_length < min_chars:
        return "short"
    return None


def _extract_final_citation_ids(report: str) -> list[int]:
    return _extract_report_ref_ids(report)


def _validate_final_research_citations(report: str, citations: list[str]) -> None:
    citation_count = len(citations)
    invalid_ids = [
        ref_id
        for ref_id in _extract_final_citation_ids(report)
        if ref_id < 1 or ref_id > citation_count
    ]
    if invalid_ids:
        invalid_text = ", ".join(str(ref_id) for ref_id in sorted(set(invalid_ids)))
        raise AgentStageError(
            stage="research_generate",
            message=f"Research report contains dangling citation id(s): {invalid_text}.",
            question="",
        )


def _collapse_repeated_citation_runs(text: str) -> str:
    def _replace(match: re.Match[str]) -> str:
        ids = re.findall(r"\[(\d+)\]", match.group(0))
        deduped: list[str] = []
        seen: set[str] = set()
        for ref_id in ids:
            if ref_id in seen:
                continue
            seen.add(ref_id)
            deduped.append(ref_id)
        return "".join(f"[{ref_id}]" for ref_id in deduped)

    return re.sub(r"(?:\[\d+\]\s*){2,}", _replace, text)


def _merge_research_report_citations(
    report: str,
    ref_ids: list[int],
    context_hits: list[SearchHit],
) -> tuple[str, list[str]]:
    ref_id_to_citation_id: dict[int, int] = {}
    hits_by_doc: dict[str, list[SearchHit]] = {}
    seen_chunks_by_doc: dict[str, set[str]] = {}

    for ref_id in ref_ids:
        if ref_id < 1 or ref_id > len(context_hits):
            continue
        hit = context_hits[ref_id - 1]
        doc_key = _document_key_from_hit(hit)
        if doc_key not in hits_by_doc:
            hits_by_doc[doc_key] = []
            seen_chunks_by_doc[doc_key] = set()
        ref_id_to_citation_id[ref_id] = list(hits_by_doc).index(doc_key) + 1
        if hit.chunk_id not in seen_chunks_by_doc[doc_key]:
            hits_by_doc[doc_key].append(hit)
            seen_chunks_by_doc[doc_key].add(hit.chunk_id)

    def _replace_ref_group(match: re.Match[str]) -> str:
        citation_ids: list[int] = []
        seen_citation_ids: set[int] = set()
        for raw_ref_id in re.findall(r"\d+", match.group(1)):
            citation_id = ref_id_to_citation_id.get(int(raw_ref_id))
            if citation_id is None or citation_id in seen_citation_ids:
                continue
            seen_citation_ids.add(citation_id)
            citation_ids.append(citation_id)
        return "".join(f"[{citation_id}]" for citation_id in citation_ids)

    rewritten_report = re.sub(r"\[([\d,\s]+)\]", _replace_ref_group, report)
    rewritten_report = _collapse_repeated_citation_runs(rewritten_report)
    citations = [
        f"[{index}] {_build_merged_citation(hits)}"
        for index, hits in enumerate(hits_by_doc.values(), start=1)
        if hits
    ]
    _validate_final_research_citations(rewritten_report, citations)
    return rewritten_report, citations


class AgentStageError(RuntimeError):
    def __init__(
        self,
        *,
        stage: str,
        message: str,
        question: str,
        query_analysis: QueryAnalysis | None = None,
        raw_response: str | None = None,
        extra_diagnostics: dict[str, Any] | None = None,
    ):
        super().__init__(message)
        self.stage = stage
        self.message = message
        self.question = question
        self.query_analysis = query_analysis
        self.raw_response = raw_response
        self.extra_diagnostics = extra_diagnostics or {}

    def diagnostic_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "stage": self.stage,
            "message": self.message,
            "question": self.question,
        }
        query_analysis = _analysis_snapshot(self.query_analysis)
        if query_analysis is not None:
            payload["query_analysis"] = query_analysis
        raw_response = _excerpt(self.raw_response)
        if raw_response is not None:
            payload["raw_response_excerpt"] = raw_response
        payload.update(self.extra_diagnostics)
        return payload


class ChatModelRequiredError(AgentStageError):
    def __init__(self, *, question: str):
        super().__init__(
            stage="setup",
            message="CHAT_API_KEY is required for ask/chat agent reasoning.",
            question=question,
        )


class DashScopeRequiredError(AgentStageError):
    def __init__(self, *, question: str, requirement: str):
        super().__init__(
            stage="setup",
            message=f"DASHSCOPE_API_KEY is required for {requirement}.",
            question=question,
        )


def _make_stage_error(
    *,
    stage: str,
    question: str,
    exc: Exception,
    query_analysis: QueryAnalysis | None = None,
    extra_diagnostics: dict[str, Any] | None = None,
) -> AgentStageError:
    message = str(exc).strip() or f"{stage} failed."
    return AgentStageError(
        stage=stage,
        message=message,
        question=question,
        query_analysis=query_analysis,
        raw_response=_exception_raw_response(exc),
        extra_diagnostics=extra_diagnostics,
    )


def _is_timeout_error(exc: Exception) -> bool:
    return "timed out" in str(exc).lower()


@dataclass(slots=True)
class AgenticAnswerer:
    retriever: Any
    settings: Settings
    llm_client: LLMClient | None = None
    rerank_client: Any | None = None

    def _hydrate_hits(self, hits: list[Any]) -> list[SearchHit]:
        if not hits:
            return []
        if isinstance(hits[0], SearchHit):
            return hits
        if self.retriever is None or not hasattr(self.retriever, "hydrate_hits"):
            raise RuntimeError("Retriever does not support candidate hydration.")
        return self.retriever.hydrate_hits(hits)

    async def analyze(self, question: str) -> QueryAnalysis:
        if self.llm_client is None:
            raise ChatModelRequiredError(question=question)
        try:
            payload = await self.llm_client.complete_json(
                ANALYZE_SYSTEM_PROMPT,
                (
                    "Question:\n"
                    f"{question}\n\n"
                    "Return JSON with fields: original_question, target_claim, evidence_type, "
                    "question_language, intent_type, analysis, must_have_terms, optional_terms, "
                    "entity_terms_en, condition_terms_en, relation_terms_en, numeric_constraints, "
                    "query_variants, keyword_phrases, query_bundles, must_match_groups, "
                    "diagnostic_notes, mode, top_k_vector, top_k_keyword."
                ),
            )
            return QueryAnalysis.model_validate(
                _normalize_query_analysis_payload(payload, question, self.settings)
            )
        except Exception as exc:
            raise _make_stage_error(stage="analyze", question=question, exc=exc) from exc

    async def judge_document(
        self,
        *,
        question: str,
        query_analysis: QueryAnalysis,
        document_candidate: DocumentCandidate,
    ) -> DocumentAssessment:
        if self.llm_client is None:
            raise ChatModelRequiredError(question=question)
        prompt = (
            "Question:\n"
            f"{question}\n\n"
            "Structured query plan:\n"
            f"{json.dumps(_prompt_analysis_snapshot(query_analysis, include_bundles=False), ensure_ascii=False)}\n\n"
            "Document candidate:\n"
            f"{json.dumps(_document_candidate_payload(document_candidate, query_analysis, chunk_limit=3, text_limit=None), ensure_ascii=False)}\n\n"
            "Return JSON with fields: doc_id, label, supporting_chunk_ids, matched_constraints, "
            "missing_constraints, reason, answer_line. For direct_support, answer_line must be a "
            "2-3 sentence chunk-grounded explanation for this paper."
        )
        last_exc: Exception | None = None
        for attempt_index in range(2):
            try:
                payload = await self.llm_client.complete_json(
                    DOCUMENT_FILTER_SYSTEM_PROMPT,
                    prompt,
                )
                return DocumentAssessment.model_validate(
                    _normalize_document_assessment_payload(
                        payload,
                        document_candidate=document_candidate,
                    )
                )
            except Exception as exc:
                last_exc = exc
                if attempt_index == 1 or (not _is_timeout_error(exc) and not isinstance(exc, ValueError)):
                    break
        assert last_exc is not None
        raise _make_stage_error(
            stage="judge_document",
            question=question,
            exc=last_exc,
            query_analysis=query_analysis,
        ) from last_exc

    def _recall_research_hits(
        self,
        *,
        question: str,
        query_analysis: QueryAnalysis,
        search_fn,
    ) -> tuple[list[SearchHit], list[str]]:
        del question
        used_queries: list[str] = []
        used_query_keys: set[str] = set()
        best_hits: dict[str, Any] = {}
        recall_limit = max(1, self.settings.research_rerank_chunk_limit)
        top_k_vector = max(query_analysis.top_k_vector or self.settings.top_k_vector, recall_limit)
        top_k_keyword = max(query_analysis.top_k_keyword or self.settings.top_k_keyword, recall_limit)

        for bundle in list(query_analysis.query_bundles):
            bundle_key = _query_dedup_key(bundle.query)
            if bundle_key in used_query_keys:
                continue
            used_query_keys.add(bundle_key)
            used_queries.append(bundle.query)

            keyword_fts_query = None
            if bundle.keyword_phrases:
                keyword_fts_query = build_fts_query_from_phrases(bundle.keyword_phrases)

            round_hits = search_fn(
                query=bundle.query,
                mode="hybrid",
                top_k_vector=top_k_vector,
                top_k_keyword=top_k_keyword,
                keyword_fts_query=keyword_fts_query,
            )
            if keyword_fts_query is not None:
                round_hits.extend(
                    search_fn(
                        query=bundle.query,
                        mode="hybrid",
                        top_k_vector=top_k_vector,
                        top_k_keyword=top_k_keyword,
                        keyword_fts_query=None,
                    )
                )

            for hit in round_hits:
                existing = best_hits.get(hit.chunk_id)
                if existing is None or hit.score_final > existing.score_final:
                    best_hits[hit.chunk_id] = hit

        ranked_hits = rerank_hits_for_query_plan(
            self._hydrate_hits(list(best_hits.values())),
            query_analysis,
        )
        return ranked_hits, used_queries

    def _rerank_research_hits(
        self,
        *,
        question: str,
        query_analysis: QueryAnalysis,
        hits: list[SearchHit],
    ) -> tuple[list[SearchHit], int]:
        if not hits:
            return [], 0
        rerank_limit = min(len(hits), max(1, self.settings.research_rerank_chunk_limit))
        prefix = [hit.model_copy(deep=True) for hit in hits[:rerank_limit]]
        suffix = hits[rerank_limit:]
        if self.rerank_client is None or len(prefix) <= 1:
            return prefix + suffix, len(prefix)

        rerank_query = query_analysis.target_claim.strip() or question
        documents = [_research_rerank_input(hit) for hit in prefix]
        rerank_batches = _build_rerank_batches(
            documents=documents,
            query=rerank_query,
            instruct=self.settings.rerank_instruct,
            token_budget=self.settings.rerank_request_token_budget,
        )
        score_by_index: dict[int, float] = {}
        for batch in rerank_batches:
            rerank_results = self.rerank_client.rerank(
                query=rerank_query,
                documents=batch.documents,
                top_n=len(batch.documents),
                instruct=self.settings.rerank_instruct,
            )
            for batch_index, score in rerank_results:
                if 0 <= batch_index < len(batch.original_indices):
                    score_by_index[batch.original_indices[batch_index]] = score

        ranked_pairs = sorted(
            enumerate(prefix),
            key=lambda item: (
                score_by_index.get(item[0], float("-inf")),
                item[1].score_final,
                item[1].score_constraint,
            ),
            reverse=True,
        )
        return [hit for _, hit in ranked_pairs] + suffix, len(prefix)

    def _select_research_context_hits(self, hits: list[SearchHit]) -> list[SearchHit]:
        max_chunks = max(1, self.settings.research_final_chunk_limit)
        token_budget = max(1, self.settings.research_context_token_budget)
        available_budget = max(1, token_budget - 2000)
        selected: list[SearchHit] = []
        used_tokens = 0
        for ref_id, hit in enumerate(hits[:max_chunks], start=1):
            payload_text = json.dumps(_research_hit_payload(hit, ref_id=ref_id), ensure_ascii=False)
            hit_tokens = _estimate_rerank_tokens(payload_text) + 16
            if selected and used_tokens + hit_tokens > available_budget:
                break
            if not selected and hit_tokens > available_budget:
                break
            selected.append(hit)
            used_tokens += hit_tokens
        return selected

    async def _generate_research_report(
        self,
        *,
        question: str,
        query_analysis: QueryAnalysis,
        context_hits: list[SearchHit],
    ) -> tuple[str, list[int]]:
        if self.llm_client is None:
            raise ChatModelRequiredError(question=question)
        if not context_hits:
            raise AgentStageError(
                stage="research_generate",
                message="No chunks fit the research context token budget.",
                question=question,
                query_analysis=query_analysis,
            )

        configured_min_chars = max(0, self.settings.research_report_min_chars)
        min_chars = configured_min_chars
        if min_chars > 0:
            length_instruction = (
                f"Write a research report of at least {min_chars} characters. "
                "There is no maximum length limit; decide the appropriate length based on the provided chunks. "
                "If writing Chinese, one Chinese character counts as one character."
            )
        else:
            length_instruction = (
                "Write a research report. There is no maximum length limit; decide the appropriate length "
                "based on the provided chunks."
            )
        evidence_chunks = [
            _research_hit_payload(hit, ref_id=index)
            for index, hit in enumerate(context_hits, start=1)
        ]
        prompt = (
            "Question:\n"
            f"{question}\n"
            f"{length_instruction}\n\n"
            "Evidence chunks:\n"
            f"{json.dumps(evidence_chunks, ensure_ascii=False)}"
        )

        payload = await self.llm_client.complete_json(RESEARCH_REPORT_SYSTEM_PROMPT, prompt)
        report, used_ref_ids = _normalize_research_report_payload(payload, max_ref_id=len(context_hits))
        length_status = _research_report_length_status(report, min_chars=min_chars)
        if length_status is None:
            return report, used_ref_ids

        repair_instruction = (
            f"The report below is {len(report)} characters long, shorter than the required "
            f"minimum of {min_chars} characters. Expand it using only the provided evidence chunks. "
            "There is no maximum length limit."
        )
        repair_prompt = (
            "Original question:\n"
            f"{question}\n\n"
            f"{repair_instruction}\n\n"
            "Allowed reference ids:\n"
            f"{list(range(1, len(context_hits) + 1))}\n\n"
            "Evidence chunks:\n"
            f"{json.dumps(evidence_chunks, ensure_ascii=False)}\n\n"
            "Draft report:\n"
            f"{report}\n\n"
            "Original used_ref_ids:\n"
            f"{used_ref_ids}"
        )
        repaired_payload = await self.llm_client.complete_json(
            RESEARCH_REPORT_REPAIR_SYSTEM_PROMPT,
            repair_prompt,
        )
        repaired_report, repaired_ref_ids = _normalize_research_report_payload(
            repaired_payload,
            max_ref_id=len(context_hits),
        )
        repaired_status = _research_report_length_status(repaired_report, min_chars=min_chars)
        if repaired_status is not None:
            raise AgentStageError(
                stage="research_generate",
                message=(
                    f"Research report is {len(repaired_report)} characters after repair, "
                    f"shorter than required minimum {min_chars}."
                ),
                question=question,
                query_analysis=query_analysis,
            )
        return repaired_report, repaired_ref_ids

    async def research(self, question: str, search_fn) -> ResearchAnswer:
        if self.llm_client is None:
            raise ChatModelRequiredError(question=question)

        total_started_at = perf_counter()
        analyze_started_at = perf_counter()
        query_analysis = await self.analyze(question)
        analyze_seconds = perf_counter() - analyze_started_at

        retrieve_started_at = perf_counter()
        recalled_hits, used_queries = self._recall_research_hits(
            question=question,
            query_analysis=query_analysis,
            search_fn=search_fn,
        )
        retrieve_seconds = perf_counter() - retrieve_started_at

        rerank_started_at = perf_counter()
        try:
            reranked_hits, chunks_reranked = self._rerank_research_hits(
                question=question,
                query_analysis=query_analysis,
                hits=recalled_hits,
            )
        except Exception as exc:
            raise _make_stage_error(
                stage="research_rerank",
                question=question,
                exc=exc,
                query_analysis=query_analysis,
                extra_diagnostics=_research_rerank_diagnostics(
                    hits=recalled_hits,
                    settings=self.settings,
                    question=question,
                    query_analysis=query_analysis,
                ),
            ) from exc
        rerank_seconds = perf_counter() - rerank_started_at

        context_hits = self._select_research_context_hits(reranked_hits)
        generate_started_at = perf_counter()
        try:
            report, used_ref_ids = await self._generate_research_report(
                question=question,
                query_analysis=query_analysis,
                context_hits=context_hits,
            )
        except AgentStageError:
            raise
        except Exception as exc:
            raise _make_stage_error(
                stage="research_generate",
                question=question,
                exc=exc,
                query_analysis=query_analysis,
            ) from exc
        generate_seconds = perf_counter() - generate_started_at
        report, citations = _merge_research_report_citations(report, used_ref_ids, context_hits)

        stage_timings = ResearchStageTimings(
            analyze_seconds=analyze_seconds,
            retrieve_seconds=retrieve_seconds,
            rerank_seconds=rerank_seconds,
            generate_seconds=generate_seconds,
            total_seconds=perf_counter() - total_started_at,
        )
        return ResearchAnswer(
            report=report,
            citations=citations,
            used_queries=used_queries,
            chunks_recalled=len(recalled_hits),
            chunks_reranked=chunks_reranked,
            chunks_in_context=len(context_hits),
            stage_timings=stage_timings,
        )

    def _recall_document_candidates(
        self,
        *,
        question: str,
        query_analysis: QueryAnalysis,
        search_fn,
    ) -> tuple[list[DocumentCandidate], list[str]]:
        used_queries: list[str] = []
        used_query_keys: set[str] = set()
        best_hits: dict[str, Any] = {}
        bundle_matches_by_doc: dict[str, set[str]] = {}
        recall_limit = max(1, self.settings.document_recall_limit)
        top_k_vector = max(query_analysis.top_k_vector or self.settings.top_k_vector, recall_limit)
        top_k_keyword = max(query_analysis.top_k_keyword or self.settings.top_k_keyword, recall_limit)

        for bundle in list(query_analysis.query_bundles):
            bundle_key = _query_dedup_key(bundle.query)
            if bundle_key in used_query_keys:
                continue
            used_query_keys.add(bundle_key)
            used_queries.append(bundle.query)

            keyword_fts_query = None
            if bundle.keyword_phrases:
                keyword_fts_query = build_fts_query_from_phrases(bundle.keyword_phrases)

            bundle_hits: dict[str, Any] = {}
            round_hits = search_fn(
                query=bundle.query,
                mode="hybrid",
                top_k_vector=top_k_vector,
                top_k_keyword=top_k_keyword,
                keyword_fts_query=keyword_fts_query,
            )
            if keyword_fts_query is not None:
                round_hits.extend(
                    search_fn(
                        query=bundle.query,
                        mode="hybrid",
                        top_k_vector=top_k_vector,
                        top_k_keyword=top_k_keyword,
                        keyword_fts_query=None,
                    )
                )

            for hit in round_hits:
                existing = best_hits.get(hit.chunk_id)
                if existing is None or hit.score_final > existing.score_final:
                    best_hits[hit.chunk_id] = hit

                bundle_existing = bundle_hits.get(hit.chunk_id)
                if bundle_existing is None or hit.score_final > bundle_existing.score_final:
                    bundle_hits[hit.chunk_id] = hit

            strong_doc_ids: list[str] = []
            seen_doc_ids: set[str] = set()
            for hit in sorted(bundle_hits.values(), key=lambda item: item.score_final, reverse=True):
                doc_key = _document_key_from_hit(hit)
                if doc_key in seen_doc_ids:
                    continue
                seen_doc_ids.add(doc_key)
                strong_doc_ids.append(doc_key)
                if len(strong_doc_ids) >= 20:
                    break
            for doc_id in strong_doc_ids:
                bundle_matches_by_doc.setdefault(doc_id, set()).add(bundle.bundle_name)

        ranked_hits = rerank_hits_for_query_plan(
            self._hydrate_hits(list(best_hits.values())),
            query_analysis,
        )
        grouped_hits: dict[str, list[SearchHit]] = {}
        for hit in ranked_hits:
            doc_key = _document_key_from_hit(hit)
            grouped_hits.setdefault(doc_key, []).append(hit)

        document_candidates: list[DocumentCandidate] = []
        for doc_key, hits in grouped_hits.items():
            sorted_hits = sorted(hits, key=lambda h: h.score_final, reverse=True)
            top_hits = sorted_hits[:3]
            aggregate_score = sum(hit.score_final for hit in top_hits) / len(top_hits)
            matched_bundle_count = len(bundle_matches_by_doc.get(doc_key, set()))
            best_hit = top_hits[0]
            document_candidates.append(
                DocumentCandidate(
                    doc_id=best_hit.doc_id or doc_key,
                    title=best_hit.title,
                    source_path=best_hit.source_path,
                    aggregate_score=aggregate_score,
                    matched_bundle_count=matched_bundle_count,
                    top_chunks=top_hits,
                )
            )

        document_candidates.sort(
            key=lambda candidate: (
                candidate.aggregate_score,
                candidate.matched_bundle_count,
                candidate.top_chunks[0].score_final if candidate.top_chunks else 0.0,
            ),
            reverse=True,
        )
        return document_candidates[:recall_limit], used_queries

    def _retrieve_document_candidates(
        self,
        *,
        question: str,
        query_analysis: QueryAnalysis,
        search_fn,
    ) -> tuple[list[DocumentCandidate], list[str]]:
        judge_limit = max(1, self.settings.document_judge_limit)
        document_candidates, used_queries = self._recall_document_candidates(
            question=question,
            query_analysis=query_analysis,
            search_fn=search_fn,
        )
        document_candidates = self._rerank_document_candidates(
            question=question,
            query_analysis=query_analysis,
            document_candidates=document_candidates,
        )
        return document_candidates[:judge_limit], used_queries

    def _document_rerank_input(self, candidate: DocumentCandidate) -> str:
        parts = [f"Title: {candidate.title}"]
        for index, hit in enumerate(
            candidate.top_chunks[: max(1, self.settings.rerank_document_chunk_limit)],
            start=1,
        ):
            parts.append(f"Chunk {index}: {hit.chunk_id}")
            if hit.section_hint:
                parts.append(f"Section: {hit.section_hint}")
            parts.append(f"Evidence: {hit.text}")
        document_input = "\n".join(parts).strip()
        if self.settings.rerank_document_text_limit > 0:
            document_input = _excerpt(document_input, limit=self.settings.rerank_document_text_limit) or ""
        return document_input or candidate.title

    def _rerank_document_candidates(
        self,
        *,
        question: str,
        query_analysis: QueryAnalysis,
        document_candidates: list[DocumentCandidate],
    ) -> list[DocumentCandidate]:
        if self.rerank_client is None or len(document_candidates) <= 1:
            return document_candidates

        rerank_limit = min(len(document_candidates), max(1, self.settings.document_recall_limit))
        prefix = [candidate.model_copy(deep=True) for candidate in document_candidates[:rerank_limit]]
        suffix = document_candidates[rerank_limit:]
        rerank_query = query_analysis.target_claim.strip() or question
        rerank_documents = [self._document_rerank_input(candidate) for candidate in prefix]
        rerank_batches = _build_rerank_batches(
            documents=rerank_documents,
            query=rerank_query,
            instruct=self.settings.rerank_instruct,
            token_budget=self.settings.rerank_request_token_budget,
        )

        score_by_index: dict[int, float] = {}
        for batch in rerank_batches:
            rerank_results = self.rerank_client.rerank(
                query=rerank_query,
                documents=batch.documents,
                top_n=len(batch.documents),
                instruct=self.settings.rerank_instruct,
            )
            for batch_index, score in rerank_results:
                if 0 <= batch_index < len(batch.original_indices):
                    score_by_index[batch.original_indices[batch_index]] = score

        ranked_pairs = sorted(
            enumerate(prefix),
            key=lambda item: (
                score_by_index.get(item[0], float("-inf")),
                item[1].aggregate_score,
                item[1].matched_bundle_count,
                item[1].top_chunks[0].score_final if item[1].top_chunks else 0.0,
            ),
            reverse=True,
        )
        ranked_prefix = [candidate for _, candidate in ranked_pairs]
        for original_index, candidate in ranked_pairs:
            candidate.rerank_score = score_by_index.get(original_index)
        return ranked_prefix + suffix

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
        del question, query_analysis, background_assessments
        candidate_by_doc = {candidate.doc_id: candidate for candidate in document_candidates}

        def support_hits_from_assessment(assessment: DocumentAssessment) -> list[SearchHit]:
            candidate = candidate_by_doc.get(assessment.doc_id)
            if candidate is None:
                return []
            chunk_map = {chunk.chunk_id: chunk for chunk in candidate.top_chunks}
            supporting_ids = assessment.supporting_chunk_ids or [chunk.chunk_id for chunk in candidate.top_chunks[:1]]
            supported_hits: list[SearchHit] = []
            for chunk_id in supporting_ids:
                chunk = chunk_map.get(chunk_id)
                if chunk is None:
                    continue
                hit = chunk.model_copy(deep=True)
                hit.matched_constraints = dedupe_strings(
                    hit.matched_constraints + assessment.matched_constraints
                )
                hit.missing_constraints = dedupe_strings(
                    hit.missing_constraints + assessment.missing_constraints
                )
                supported_hits.append(hit)
            if supported_hits:
                return supported_hits
            fallback_hits: list[SearchHit] = []
            for chunk in candidate.top_chunks[:1]:
                hit = chunk.model_copy(deep=True)
                hit.matched_constraints = dedupe_strings(
                    hit.matched_constraints + assessment.matched_constraints
                )
                hit.missing_constraints = dedupe_strings(
                    hit.missing_constraints + assessment.missing_constraints
                )
                fallback_hits.append(hit)
            return fallback_hits

        answer_lines: list[str] = []
        citations: list[str] = []
        for candidate in document_candidates:
            direct_assessment = direct_assessments.get(candidate.doc_id)
            if direct_assessment is None:
                continue
            supported_hits = sorted(
                support_hits_from_assessment(direct_assessment),
                key=lambda hit: hit.score_final,
                reverse=True,
            )
            answer_line = _sanitize_answer_line(direct_assessment.answer_line)
            if not answer_line or not supported_hits:
                continue
            answer_lines.append(answer_line)
            citations.append(_build_merged_citation(supported_hits))

        if not answer_lines:
            return AgentAnswer(
                answer=NO_DIRECT_SUPPORT_ANSWER,
                citations=[],
                used_queries=used_queries,
                rounds=1,
                confidence="low",
            )

        return AgentAnswer(
            answer=_format_numbered_answer(answer_lines),
            citations=_numbered_citations(citations),
            used_queries=used_queries,
            rounds=1,
            confidence="high",
        )

    async def answer_stream(self, question: str, search_fn) -> AsyncIterator[AnswerStreamEvent]:
        if self.llm_client is None:
            raise ChatModelRequiredError(question=question)

        total_started_at = perf_counter()
        analyze_started_at = perf_counter()
        query_analysis = await self.analyze(question)
        analyze_seconds = perf_counter() - analyze_started_at

        retrieve_started_at = perf_counter()
        if type(self)._retrieve_document_candidates is AgenticAnswerer._retrieve_document_candidates:
            recalled_candidates, used_queries = self._recall_document_candidates(
                question=question,
                query_analysis=query_analysis,
                search_fn=search_fn,
            )
            retrieve_seconds = perf_counter() - retrieve_started_at

            rerank_started_at = perf_counter()
            try:
                document_candidates = self._rerank_document_candidates(
                    question=question,
                    query_analysis=query_analysis,
                    document_candidates=recalled_candidates,
                )
            except Exception as exc:
                raise _make_stage_error(
                    stage="rerank",
                    question=question,
                    exc=exc,
                    query_analysis=query_analysis,
                ) from exc
            rerank_seconds = perf_counter() - rerank_started_at
            judge_limit = max(1, self.settings.document_judge_limit)
            document_candidates = document_candidates[:judge_limit]
        else:
            document_candidates, used_queries = self._retrieve_document_candidates(
                question=question,
                query_analysis=query_analysis,
                search_fn=search_fn,
            )
            retrieve_seconds = perf_counter() - retrieve_started_at
            rerank_seconds = 0.0

        judged_documents = 0
        direct_documents = 0
        background_documents = 0
        failed_documents = 0
        consecutive_non_support_count = 0
        stopped_early = False
        stop_reason: str | None = None
        stopped_after_documents = 0
        dispatch_stopped = False
        direct_assessments: dict[str, DocumentAssessment] = {}
        background_assessments: dict[str, DocumentAssessment] = {}
        latest_answer: AgentAnswer | None = None
        last_answer_direct_documents = 0
        initial_concurrency = max(1, self.settings.document_judge_initial_concurrency)
        target_concurrency = initial_concurrency
        provider_pressure_events = 0
        update_stride = max(1, self.settings.document_answer_update_stride)
        total_documents = len(document_candidates)
        judge_documents_total = total_documents
        next_dispatch_rank = 0
        next_commit_rank = 0
        results_by_rank: dict[int, _DocumentJudgeTaskResult] = {}
        judge_started_at = perf_counter()

        def build_stage_timings() -> StageTimings:
            judge_seconds = max(0.0, perf_counter() - judge_started_at)
            total_seconds = perf_counter() - total_started_at
            return StageTimings(
                analyze_seconds=analyze_seconds,
                retrieve_seconds=retrieve_seconds,
                rerank_seconds=rerank_seconds,
                judge_seconds=judge_seconds,
                total_seconds=total_seconds,
            )

        def build_performance_counters() -> PerformanceCounters:
            return PerformanceCounters(
                judge_documents_total=judge_documents_total,
                judge_concurrency_initial=initial_concurrency,
                judge_concurrency_final=max(1, target_concurrency),
                provider_pressure_events=provider_pressure_events,
            )

        def event_state() -> dict[str, Any]:
            return {
                "total_documents": total_documents,
                "judged_documents": judged_documents,
                "dispatched_documents": next_dispatch_rank,
                "in_flight_documents": len(task_to_rank),
                "active_judge_concurrency": max(1, target_concurrency),
                "concurrency_locked": target_concurrency < initial_concurrency,
                "dispatch_stopped": dispatch_stopped,
                "direct_documents": direct_documents,
                "background_documents": background_documents,
                "failed_documents": failed_documents,
                "stopped_early": stopped_early,
                "stop_reason": stop_reason,
                "stopped_after_documents": stopped_after_documents,
            }

        async def submit_candidate(rank: int) -> _DocumentJudgeTaskResult:
            document_candidate = document_candidates[rank]
            try:
                assessment = await self.judge_document(
                    question=question,
                    query_analysis=query_analysis,
                    document_candidate=document_candidate,
                )
                return _DocumentJudgeTaskResult(
                    document_candidate=document_candidate,
                    assessment=assessment,
                )
            except AgentStageError as exc:
                return _DocumentJudgeTaskResult(
                    document_candidate=document_candidate,
                    error=exc,
                    is_provider_pressure=_is_provider_pressure_stage_error(exc),
                )

        task_to_rank: dict[asyncio.Task[_DocumentJudgeTaskResult], int] = {}

        def dispatch_rank(rank: int) -> None:
            task = asyncio.create_task(submit_candidate(rank))
            task_to_rank[task] = rank

        def maybe_dispatch() -> None:
            nonlocal next_dispatch_rank
            while (
                not dispatch_stopped
                and next_dispatch_rank < total_documents
                and next_dispatch_rank - next_commit_rank < max(1, target_concurrency)
            ):
                dispatch_rank(next_dispatch_rank)
                next_dispatch_rank += 1

        yield AnswerStreamEvent(
            event="scan_started",
            **event_state(),
        )

        maybe_dispatch()

        while task_to_rank or next_commit_rank < next_dispatch_rank:
            while next_commit_rank in results_by_rank:
                result = results_by_rank.pop(next_commit_rank)
                document_candidate = result.document_candidate
                assessment = result.assessment
                error = result.error

                if error is not None and result.is_provider_pressure:
                    target_concurrency = max(1, target_concurrency - 1)
                    provider_pressure_events += 1
                    dispatch_rank(next_commit_rank)
                    break

                judged_documents += 1

                if error is not None:
                    failed_documents += 1
                    consecutive_non_support_count += 1
                elif assessment is not None and assessment.label == "direct_support":
                    direct_assessments[assessment.doc_id] = assessment
                    direct_documents = len(direct_assessments)
                    consecutive_non_support_count = 0
                elif assessment is not None and assessment.label == "background_only":
                    background_assessments[assessment.doc_id] = assessment
                    background_documents = len(background_assessments)
                    consecutive_non_support_count += 1
                else:
                    consecutive_non_support_count += 1

                yield AnswerStreamEvent(
                    event="document_judged",
                    document=document_candidate,
                    assessment=assessment,
                    error=error.message if error is not None else None,
                    **event_state(),
                )

                if (
                    not dispatch_stopped
                    and _should_stop_early(
                        consecutive_non_support_count=consecutive_non_support_count,
                    )
                ):
                    stopped_early = True
                    dispatch_stopped = True
                    stop_reason = _build_early_stop_reason()
                    stopped_after_documents = judged_documents

                should_update_answer = (
                    assessment is not None
                    and assessment.label == "direct_support"
                    and (
                        latest_answer is None
                        or direct_documents - last_answer_direct_documents >= update_stride
                    )
                )
                if should_update_answer:
                    try:
                        latest_answer = await self._build_answer_from_document_assessments(
                            question=question,
                            query_analysis=query_analysis,
                            document_candidates=document_candidates,
                            direct_assessments=direct_assessments,
                            background_assessments=background_assessments,
                            used_queries=used_queries,
                        )
                    except AgentStageError:
                        pass
                    else:
                        emitted_event = "first_answer_emitted" if last_answer_direct_documents == 0 else "answer_updated"
                        last_answer_direct_documents = direct_documents
                        yield AnswerStreamEvent(
                            event=emitted_event,
                            answer=latest_answer,
                            **event_state(),
                        )

                next_commit_rank += 1
                maybe_dispatch()

            if not task_to_rank:
                break

            done_tasks, _ = await asyncio.wait(tuple(task_to_rank), return_when=asyncio.FIRST_COMPLETED)
            for task in done_tasks:
                rank = task_to_rank.pop(task)
                results_by_rank[rank] = task.result()

        if not stopped_after_documents:
            stopped_after_documents = judged_documents
        final_answer = await self._build_answer_from_document_assessments(
            question=question,
            query_analysis=query_analysis,
            document_candidates=document_candidates,
            direct_assessments=direct_assessments,
            background_assessments=background_assessments,
            used_queries=used_queries,
        )
        stage_timings = build_stage_timings()
        performance_counters = build_performance_counters()
        final_answer = _annotate_final_answer(
            final_answer,
            stopped_early=stopped_early,
            stop_reason=stop_reason,
            stopped_after_documents=stopped_after_documents,
            final_judged_documents=judged_documents,
            total_documents=total_documents,
            direct_documents=direct_documents,
            stage_timings=stage_timings,
            performance_counters=performance_counters,
        )
        yield AnswerStreamEvent(
            event="scan_completed",
            answer=final_answer,
            stage_timings=stage_timings,
            performance_counters=performance_counters,
            **event_state(),
        )

    async def answer(self, question: str, search_fn) -> AgentAnswer:
        final_answer: AgentAnswer | None = None
        async for event in self.answer_stream(question, search_fn):
            if event.event == "scan_completed":
                final_answer = event.answer
        if final_answer is None:
            raise AgentStageError(
                stage="scan_completed",
                message="The answer stream completed without a final answer.",
                question=question,
            )
        return final_answer
