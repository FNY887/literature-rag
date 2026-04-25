from __future__ import annotations

import os
from dataclasses import dataclass, replace
from pathlib import Path

from dotenv import load_dotenv

_module_dir = Path(__file__).resolve().parent
for _parent in (_module_dir, *_module_dir.parents):
    _env_candidate = _parent / ".env"
    if _env_candidate.exists():
        load_dotenv(_env_candidate)
        break


@dataclass(slots=True)
class Settings:
    data_dir: Path = Path("literature")
    index_dir: Path = Path(".rag_store")
    index_path: Path = Path(".rag_store/literature_rag.sqlite3")
    cleaned_artifacts_dir: Path = Path(".rag_store/chunks")
    dashscope_api_key: str | None = os.getenv("DASHSCOPE_API_KEY")
    dashscope_base_url: str = os.getenv(
        "DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    dashscope_rerank_base_url: str = os.getenv(
        "DASHSCOPE_RERANK_BASE_URL", "https://dashscope.aliyuncs.com/compatible-api/v1"
    )
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "text-embedding-v4")
    embedding_dimensions: int = int(os.getenv("EMBEDDING_DIMENSIONS", "2048"))
    rerank_model: str = os.getenv("RERANK_MODEL", "qwen3-rerank")
    rerank_timeout_seconds: float = float(os.getenv("RERANK_TIMEOUT_SECONDS", "60"))
    rerank_max_retries: int = int(os.getenv("RERANK_MAX_RETRIES", "3"))
    rerank_retry_base_delay_seconds: float = float(os.getenv("RERANK_RETRY_BASE_DELAY_SECONDS", "1.0"))
    rerank_retry_max_delay_seconds: float = float(os.getenv("RERANK_RETRY_MAX_DELAY_SECONDS", "8.0"))
    document_recall_limit: int = int(
        os.getenv("DOCUMENT_RECALL_LIMIT", os.getenv("RERANK_CANDIDATE_LIMIT", "300"))
    )
    document_judge_limit: int = int(os.getenv("DOCUMENT_JUDGE_LIMIT", "100"))
    rerank_document_chunk_limit: int = int(os.getenv("RERANK_DOCUMENT_CHUNK_LIMIT", "3"))
    rerank_document_text_limit: int = int(os.getenv("RERANK_DOCUMENT_TEXT_LIMIT", "0"))
    rerank_request_token_budget: int = int(os.getenv("RERANK_REQUEST_TOKEN_BUDGET", "3600"))
    rerank_instruct: str = os.getenv(
        "RERANK_INSTRUCT",
        "Given a scientific literature query, retrieve papers whose passages directly answer or support the constrained query. Prefer papers matching the hard conditions over general background.",
    )
    chat_api_key: str | None = os.getenv("CHAT_API_KEY")
    chat_base_url: str = os.getenv("CHAT_BASE_URL", "https://api.deepseek.com")
    chat_model: str = os.getenv("CHAT_MODEL", "deepseek-chat")
    chat_timeout_seconds: float = float(os.getenv("CHAT_TIMEOUT_SECONDS", "300"))
    chat_max_retries: int = int(os.getenv("CHAT_MAX_RETRIES", "4"))
    chat_retry_base_delay_seconds: float = float(os.getenv("CHAT_RETRY_BASE_DELAY_SECONDS", "1.0"))
    chat_retry_max_delay_seconds: float = float(os.getenv("CHAT_RETRY_MAX_DELAY_SECONDS", "8.0"))
    document_judge_initial_concurrency: int = int(os.getenv("DOCUMENT_JUDGE_INITIAL_CONCURRENCY", "5"))
    document_answer_update_stride: int = int(os.getenv("DOCUMENT_ANSWER_UPDATE_STRIDE", "2"))
    research_rerank_chunk_limit: int = int(os.getenv("RESEARCH_RERANK_CHUNK_LIMIT", "500"))
    research_final_chunk_limit: int = int(os.getenv("RESEARCH_FINAL_CHUNK_LIMIT", "100"))
    research_context_token_budget: int = int(os.getenv("RESEARCH_CONTEXT_TOKEN_BUDGET", "120000"))
    research_report_min_chars: int = int(os.getenv("RESEARCH_REPORT_MIN_CHARS", "500"))
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "2200"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "100"))
    top_k_vector: int = int(os.getenv("TOP_K_VECTOR", "30"))
    top_k_keyword: int = int(os.getenv("TOP_K_KEYWORD", "30"))
    embedding_batch_size: int = int(os.getenv("EMBEDDING_BATCH_SIZE", "10"))
    hybrid_vector_weight: float = float(os.getenv("HYBRID_VECTOR_WEIGHT", "0.55"))
    hybrid_keyword_weight: float = float(os.getenv("HYBRID_KEYWORD_WEIGHT", "0.45"))
    rrf_k: float = float(os.getenv("RRF_K", "60.0"))
    hybrid_co_occurrence_bonus: float = float(os.getenv("HYBRID_CO_OCCURRENCE_BONUS", "0.15"))

    def resolved(self, root: str | Path | None = None) -> "Settings":
        base_dir = Path(root) if root is not None else Path.cwd()
        data_dir = self.data_dir if self.data_dir.is_absolute() else base_dir / self.data_dir
        index_dir = self.index_dir if self.index_dir.is_absolute() else base_dir / self.index_dir
        index_path = self.index_path if self.index_path.is_absolute() else base_dir / self.index_path
        cleaned_artifacts_dir = (
            self.cleaned_artifacts_dir
            if self.cleaned_artifacts_dir.is_absolute()
            else base_dir / self.cleaned_artifacts_dir
        )
        return replace(
            self,
            data_dir=data_dir,
            index_dir=index_dir,
            index_path=index_path,
            cleaned_artifacts_dir=cleaned_artifacts_dir,
        )

    def ensure_index_dir(self) -> None:
        self.index_path.parent.mkdir(parents=True, exist_ok=True)

    def ensure_cleaned_artifacts_dir(self) -> None:
        self.cleaned_artifacts_dir.mkdir(parents=True, exist_ok=True)


def get_settings(root: str | Path | None = None) -> Settings:
    return Settings().resolved(root=root)
