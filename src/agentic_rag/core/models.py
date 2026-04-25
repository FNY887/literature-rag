from __future__ import annotations

from pydantic import BaseModel, Field


class QueryBundle(BaseModel):
    bundle_name: str
    query: str
    keyword_phrases: list[str] = Field(default_factory=list)
    required_terms: list[str] = Field(default_factory=list)


class ChunkRecord(BaseModel):
    chunk_id: str
    doc_id: str
    title: str
    source_path: str
    page_start: int
    page_end: int
    text: str
    section_hint: str | None = None
    keywords_hint: str = ""
    normalized_text: str = ""
    block_start: int = 0
    block_end: int = 0


class SearchHit(BaseModel):
    doc_id: str = ""
    chunk_id: str
    score_vector: float | None = None
    score_keyword: float | None = None
    score_constraint: float = 0.0
    score_final: float
    retrieval_source: str
    text: str
    citation: str
    title: str
    source_path: str
    page_start: int
    page_end: int
    section_hint: str | None = None
    matched_constraints: list[str] = Field(default_factory=list)
    missing_constraints: list[str] = Field(default_factory=list)


class DocumentCandidate(BaseModel):
    doc_id: str
    title: str
    source_path: str
    aggregate_score: float
    rerank_score: float | None = None
    matched_bundle_count: int = 0
    top_chunks: list[SearchHit] = Field(default_factory=list)


class DocumentAssessment(BaseModel):
    doc_id: str
    label: str = "off_target"
    supporting_chunk_ids: list[str] = Field(default_factory=list)
    matched_constraints: list[str] = Field(default_factory=list)
    missing_constraints: list[str] = Field(default_factory=list)
    reason: str = ""
    answer_line: str = ""


class StageTimings(BaseModel):
    analyze_seconds: float = 0.0
    retrieve_seconds: float = 0.0
    rerank_seconds: float = 0.0
    judge_seconds: float = 0.0
    total_seconds: float = 0.0


class ResearchStageTimings(BaseModel):
    analyze_seconds: float = 0.0
    retrieve_seconds: float = 0.0
    rerank_seconds: float = 0.0
    generate_seconds: float = 0.0
    total_seconds: float = 0.0


class PerformanceCounters(BaseModel):
    judge_documents_total: int = 0
    judge_concurrency_initial: int = 1
    judge_concurrency_final: int = 1
    provider_pressure_events: int = 0


class AgentAnswer(BaseModel):
    answer: str
    citations: list[str]
    used_queries: list[str]
    rounds: int
    confidence: str
    evidence_summary: list[str] = Field(default_factory=list)
    stopped_early: bool = False
    stop_reason: str | None = None
    stopped_after_documents: int | None = None
    total_documents: int | None = None
    scan_status: str = ""
    stage_timings: StageTimings = Field(default_factory=StageTimings)
    performance_counters: PerformanceCounters = Field(default_factory=PerformanceCounters)


class ResearchAnswer(BaseModel):
    report: str
    citations: list[str]
    used_queries: list[str]
    chunks_recalled: int
    chunks_reranked: int
    chunks_in_context: int
    stage_timings: ResearchStageTimings = Field(default_factory=ResearchStageTimings)


class AnswerStreamEvent(BaseModel):
    event: str
    total_documents: int = 0
    judged_documents: int = 0
    dispatched_documents: int = 0
    in_flight_documents: int = 0
    active_judge_concurrency: int = 1
    concurrency_locked: bool = False
    dispatch_stopped: bool = False
    direct_documents: int = 0
    background_documents: int = 0
    failed_documents: int = 0
    stopped_early: bool = False
    stop_reason: str | None = None
    stopped_after_documents: int = 0
    document: DocumentCandidate | None = None
    assessment: DocumentAssessment | None = None
    answer: AgentAnswer | None = None
    error: str | None = None
    stage_timings: StageTimings | None = None
    performance_counters: PerformanceCounters | None = None


class BuildStats(BaseModel):
    processed_files: int = 0
    skipped_files: int = 0
    duplicates_skipped: int = 0
    failed_files: list[str] = Field(default_factory=list)
    chunks_indexed: int = 0
    artifacts_written: int = 0
    cleaning_profile: str = "strict_body"
    index_path: str


class DeleteStats(BaseModel):
    deleted: bool = False
    doc_id: str = ""
    title: str = ""
    source_path: str = ""
    source_deleted: bool = False
    matched_by: str = ""
    chunks_deleted: int = 0
    artifacts_deleted: int = 0
    index_path: str


class QueryAnalysis(BaseModel):
    original_question: str = ""
    target_claim: str = ""
    evidence_type: str = "direct_support"
    question_language: str = "unknown"
    intent_type: str = "literature_qa"
    analysis: str = ""
    must_have_terms: list[str] = Field(default_factory=list)
    optional_terms: list[str] = Field(default_factory=list)
    entity_terms_en: list[str] = Field(default_factory=list)
    condition_terms_en: list[str] = Field(default_factory=list)
    relation_terms_en: list[str] = Field(default_factory=list)
    numeric_constraints: list[str] = Field(default_factory=list)
    query_variants: list[str] = Field(default_factory=list)
    keyword_phrases: list[str] = Field(default_factory=list)
    query_bundles: list[QueryBundle] = Field(default_factory=list)
    must_match_groups: list[list[str]] = Field(default_factory=list)
    diagnostic_notes: list[str] = Field(default_factory=list)
    mode: str = "hybrid"
    top_k_vector: int | None = None
    top_k_keyword: int | None = None
