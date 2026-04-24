from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

from agentic_rag.builder.store import SQLiteIndexStore, build_citation
from agentic_rag.core.config import Settings
from agentic_rag.core.models import QueryAnalysis, SearchHit


def fts_query_from_text(text: str) -> str:
    tokens = re.findall(r"[0-9A-Za-z\u4e00-\u9fff]{2,}", text)
    deduped: list[str] = []
    seen: set[str] = set()
    for token in tokens:
        lowered = token.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        deduped.append(token)
    if not deduped:
        return '""'
    return " OR ".join(f'"{token}"' for token in deduped[:12])


def build_fts_query_from_phrases(phrases: list[str]) -> str:
    cleaned: list[str] = []
    for p in phrases:
        p = p.strip().replace('"', "")
        if len(p) >= 2:
            cleaned.append(p)
    if not cleaned:
        return '""'
    cleaned = cleaned[:6]
    if len(cleaned) == 1:
        return f'"{cleaned[0]}"'
    return " NEAR ".join(f'"{p}"' for p in cleaned)


def cosine_similarity(query_vector: np.ndarray, matrix: np.ndarray, norms: np.ndarray | None = None) -> np.ndarray:
    if matrix.size == 0:
        return np.zeros((0,), dtype=np.float32)
    query = query_vector.astype(np.float32)
    query_norm = np.linalg.norm(query)
    matrix_norms = norms if norms is not None else np.linalg.norm(matrix, axis=1)
    safe_denominator = np.clip(matrix_norms * max(query_norm, 1e-8), 1e-8, None)
    return (matrix @ query) / safe_denominator


def _normalize_constraint_text(text: str) -> str:
    lowered = text.lower()
    lowered = re.sub(r"[^0-9a-zA-Z\u4e00-\u9fff\s\.\-]", " ", lowered)
    return " ".join(lowered.split())


def _dedupe_in_order(values: list[str]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for value in values:
        cleaned = value.strip()
        if not cleaned:
            continue
        key = cleaned.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(cleaned)
    return deduped


def _extract_ph_values(text: str) -> set[str]:
    return set(re.findall(r"p\s*h\s*([0-9]+(?:\.[0-9]+)?)", text.lower()))


def _matches_constraint_term(raw_text: str, normalized_text: str, term: str) -> bool:
    cleaned_term = term.strip()
    if not cleaned_term:
        return False
    normalized_term = _normalize_constraint_text(cleaned_term)
    if not normalized_term:
        return False
    if normalized_term in normalized_text:
        return True
    compact_raw = raw_text.lower().replace(" ", "")
    compact_term = normalized_term.replace(" ", "")
    if compact_term and compact_term in compact_raw:
        return True
    if "ph" in compact_term:
        match = re.search(r"ph([0-9]+(?:\.[0-9]+)?)", compact_term)
        if match and match.group(1) in _extract_ph_values(raw_text):
            return True
    return False


def rerank_hits_for_query_plan(hits: list[SearchHit], analysis: QueryAnalysis) -> list[SearchHit]:
    ranked_hits: list[tuple[tuple[int, int, float], SearchHit]] = []
    condition_terms = _dedupe_in_order(analysis.condition_terms_en + analysis.numeric_constraints)
    must_groups = [group for group in analysis.must_match_groups if group]

    for original_hit in hits:
        hit = original_hit.model_copy(deep=True)
        raw_text = "\n".join(filter(None, [hit.title, hit.section_hint or "", hit.text]))
        normalized_text = _normalize_constraint_text(raw_text)

        entity_matches = [
            term for term in analysis.entity_terms_en if _matches_constraint_term(raw_text, normalized_text, term)
        ]
        condition_matches = [
            term for term in condition_terms if _matches_constraint_term(raw_text, normalized_text, term)
        ]
        relation_matches = [
            term for term in analysis.relation_terms_en if _matches_constraint_term(raw_text, normalized_text, term)
        ]

        matched_constraints = _dedupe_in_order(entity_matches + condition_matches + relation_matches)
        missing_constraints: list[str] = []
        matched_group_count = 0
        for group in must_groups:
            group_matches = [term for term in group if _matches_constraint_term(raw_text, normalized_text, term)]
            if group_matches:
                matched_group_count += 1
                matched_constraints.extend(group_matches)
            else:
                missing_constraints.append(" | ".join(group[:3]))

        matched_constraints = _dedupe_in_order(matched_constraints)
        hit.matched_constraints = matched_constraints
        hit.missing_constraints = _dedupe_in_order(missing_constraints)

        entity_hit = bool(entity_matches)
        condition_hit = bool(condition_matches)
        relation_hit = bool(relation_matches)

        group_score = matched_group_count / len(must_groups) if must_groups else 0.0
        constraint_score = 0.0
        if entity_hit:
            constraint_score += 0.25
        if condition_hit:
            constraint_score += 0.35
        if relation_hit:
            constraint_score += 0.1
        constraint_score += 0.3 * group_score
        if entity_hit and condition_hit:
            constraint_score += 0.2
        if must_groups and matched_group_count == len(must_groups):
            constraint_score += 0.1

        constraint_score = min(constraint_score, 1.0)
        if condition_terms and not condition_hit:
            constraint_score *= 0.2
        if analysis.entity_terms_en and not entity_hit:
            constraint_score *= 0.35

        hit.score_constraint = constraint_score
        combined_score = hit.score_final * 0.55 + constraint_score * 0.45
        if entity_hit and condition_hit:
            combined_score += 0.08
        if condition_terms and not condition_hit:
            combined_score -= 0.12
        hit.score_final = max(0.0, min(combined_score, 1.0))

        priority = (
            int(entity_hit and condition_hit),
            int(condition_hit),
            hit.score_final,
        )
        ranked_hits.append((priority, hit))

    ranked_hits.sort(key=lambda item: item[0], reverse=True)
    return [hit for _, hit in ranked_hits]


@dataclass(slots=True)
class RetrievedHit:
    doc_id: str
    chunk_id: str
    row_index: int
    score_vector: float | None = None
    score_keyword: float | None = None
    score_final: float = 0.0
    retrieval_source: str = "vector"


@dataclass(slots=True)
class HybridRetriever:
    store: SQLiteIndexStore
    settings: Settings

    def _top_indices(self, scores: np.ndarray, limit: int | None) -> np.ndarray:
        if scores.size == 0:
            return np.zeros((0,), dtype=np.int64)
        if limit is None or limit >= scores.size:
            return np.argsort(scores)[::-1]
        if limit <= 0:
            return np.zeros((0,), dtype=np.int64)
        partition = np.argpartition(scores, -limit)[-limit:]
        return partition[np.argsort(scores[partition])[::-1]]

    def hydrate_hit(self, hit: RetrievedHit) -> SearchHit:
        rows, _ = self.store.fetch_all_vectors()
        row = rows[hit.row_index]
        return SearchHit(
            doc_id=str(row["doc_id"]),
            chunk_id=str(row["chunk_id"]),
            score_vector=hit.score_vector,
            score_keyword=hit.score_keyword,
            score_final=hit.score_final,
            retrieval_source=hit.retrieval_source,
            text=str(row["text"]),
            citation=build_citation(
                title=str(row["title"]),
                page_start=int(row["page_start"]),
                page_end=int(row["page_end"]),
                chunk_id=str(row["chunk_id"]),
            ),
            title=str(row["title"]),
            source_path=str(row["source_path"]),
            page_start=int(row["page_start"]),
            page_end=int(row["page_end"]),
            section_hint=row["section_hint"],
        )

    def hydrate_hits(self, hits: list[RetrievedHit]) -> list[SearchHit]:
        return [self.hydrate_hit(hit) for hit in hits]

    def vector_candidates(self, query_vector: np.ndarray, limit: int | None) -> list[RetrievedHit]:
        rows, matrix = self.store.fetch_all_vectors()
        similarities = cosine_similarity(query_vector, matrix)
        if similarities.size == 0:
            return []
        hits: list[RetrievedHit] = []
        for index in self._top_indices(similarities, limit):
            row_index = int(index)
            row = rows[row_index]
            score_vector = float((similarities[row_index] + 1.0) / 2.0)
            hits.append(
                RetrievedHit(
                    doc_id=str(row["doc_id"]),
                    chunk_id=str(row["chunk_id"]),
                    row_index=row_index,
                    score_vector=score_vector,
                    score_final=score_vector,
                    retrieval_source="vector",
                )
            )
        return hits

    def vector_search(self, query_vector: np.ndarray, limit: int | None) -> list[SearchHit]:
        return self.hydrate_hits(self.vector_candidates(query_vector, limit=limit))

    def keyword_candidates(
        self,
        query: str,
        limit: int | None,
        *,
        fts_query: str | None = None,
    ) -> list[RetrievedHit]:
        q = fts_query if fts_query is not None else fts_query_from_text(query)
        rows = self.store.keyword_rows(q, limit=limit)
        vector_rows, _ = self.store.fetch_all_vectors()
        row_index_by_chunk_id = {str(row["chunk_id"]): index for index, row in enumerate(vector_rows)}
        hits: list[RetrievedHit] = []
        for row in rows:
            chunk_id = str(row["chunk_id"])
            row_index = row_index_by_chunk_id.get(chunk_id)
            if row_index is None:
                continue
            score_keyword = 1.0 / (1.0 + max(float(row["bm25_score"]), 0.0))
            metadata = vector_rows[row_index]
            hits.append(
                RetrievedHit(
                    doc_id=str(metadata["doc_id"]),
                    chunk_id=chunk_id,
                    row_index=row_index,
                    score_keyword=score_keyword,
                    score_final=score_keyword,
                    retrieval_source="keyword",
                )
            )
        return hits

    def keyword_search(
        self,
        query: str,
        limit: int | None,
        *,
        fts_query: str | None = None,
    ) -> list[SearchHit]:
        return self.hydrate_hits(self.keyword_candidates(query, limit=limit, fts_query=fts_query))

    def hybrid_candidates(
        self,
        query: str,
        query_vector: np.ndarray,
        *,
        top_k_vector: int | None,
        top_k_keyword: int | None,
        limit: int | None,
        keyword_fts_query: str | None = None,
    ) -> list[RetrievedHit]:
        vector_hits = self.vector_candidates(query_vector, limit=top_k_vector)
        keyword_hits = self.keyword_candidates(query, limit=top_k_keyword, fts_query=keyword_fts_query)
        by_id: dict[str, RetrievedHit] = {}

        def ensure(hit: RetrievedHit) -> RetrievedHit:
            existing = by_id.get(hit.chunk_id)
            if existing is None:
                by_id[hit.chunk_id] = RetrievedHit(
                    doc_id=hit.doc_id,
                    chunk_id=hit.chunk_id,
                    row_index=hit.row_index,
                    score_vector=hit.score_vector,
                    score_keyword=hit.score_keyword,
                    score_final=hit.score_final,
                    retrieval_source=hit.retrieval_source,
                )
                return by_id[hit.chunk_id]
            return existing

        for hit in vector_hits:
            target = ensure(hit)
            target.score_vector = hit.score_vector
            target.retrieval_source = "vector"

        for hit in keyword_hits:
            target = ensure(hit)
            target.score_keyword = hit.score_keyword
            if target.retrieval_source == "vector":
                target.retrieval_source = "hybrid"
            else:
                target.retrieval_source = "keyword"

        vector_ranks = {hit.chunk_id: rank for rank, hit in enumerate(vector_hits, start=1)}
        keyword_ranks = {hit.chunk_id: rank for rank, hit in enumerate(keyword_hits, start=1)}

        rrf_k = self.settings.rrf_k
        co_occurrence_bonus = self.settings.hybrid_co_occurrence_bonus
        max_rrf = 2.0 / (rrf_k + 1) + co_occurrence_bonus

        for target in by_id.values():
            rrf_score = 0.0
            if target.chunk_id in vector_ranks:
                rrf_score += 1.0 / (rrf_k + vector_ranks[target.chunk_id])
            if target.chunk_id in keyword_ranks:
                rrf_score += 1.0 / (rrf_k + keyword_ranks[target.chunk_id])
            if target.score_vector is not None and target.score_keyword is not None:
                rrf_score += co_occurrence_bonus
                target.retrieval_source = "hybrid"
            target.score_final = min(rrf_score / max_rrf, 1.0)

        result = sorted(by_id.values(), key=lambda hit: hit.score_final, reverse=True)
        if limit is not None:
            result = result[:limit]
        return result

    def hybrid_search(
        self,
        query: str,
        query_vector: np.ndarray,
        *,
        top_k_vector: int | None,
        top_k_keyword: int | None,
        limit: int | None,
        keyword_fts_query: str | None = None,
    ) -> list[SearchHit]:
        return self.hydrate_hits(
            self.hybrid_candidates(
                query=query,
                query_vector=query_vector,
                top_k_vector=top_k_vector,
                top_k_keyword=top_k_keyword,
                limit=limit,
                keyword_fts_query=keyword_fts_query,
            )
        )
