# CLAUDE.md — Agentic Literature RAG

This file is for Claude (not for other agents). It captures project context, architectural constraints, and collaboration preferences that should persist across sessions.

## 1. Project Overview

A local agentic RAG system for scientific literature **Markdown corpora**.

- **Input**: `.md` files (not JSON)
- **Index**: SQLite + FTS5 + DashScope embeddings (2048-dim, text-embedding-v4)
- **Retrieval**: Hybrid (vector + keyword), RRF fusion, qwen3-rerank. Agent mode returns **all chunks** (no truncation) so every document can be scored; document rerank uses top-3 chunks and token-budgeted batches.
- **Reasoning**: OpenAI-compatible chat model for query analysis, document judgment, and research report generation
- **Answer lines**: direct-support answer lines are 2-3 sentence chunk-grounded explanations, not single generic claims.
- **Design philosophy**: Build and query are physically separated modules. No chat session history. No complex JSON parsing.
- **Document scoring**: documents are ranked by the **average of their top-3 chunk scores** after hybrid fusion, then reranked, then judged.

## 2. Architecture

```
agentic_rag/
├── builder/          # Offline indexing
│   ├── chunker.py    # Markdown → semantic chunks
│   ├── embedder.py   # Embedder Protocol + DashScopeEmbeddingClient
│   └── store.py      # SQLiteIndexStore
├── query/            # Online retrieval
│   ├── retriever.py  # HybridRetriever (vector + keyword + RRF)
│   ├── reranker.py   # Reranker Protocol + DashScopeRerankClient
│   └── agent.py      # AgenticAnswerer (analyze → judge(write) → deterministic assemble)
├── core/             # Shared
│   ├── models.py     # Pydantic data models
│   ├── config.py     # Settings (reads .env via python-dotenv)
│   ├── llm.py        # LLMClient Protocol + OpenAICompatibleChatClient
│   └── utils.py      # Retry, dedupe, extract_json_block
├── cli.py            # Typer CLI: build, add, interactive ask
└── __init__.py
```

## 3. Hard Rules

### API Keys
- **NEVER** write real API keys into source code, tests, README, or CLAUDE.md
- The repository may track an empty `.env` template. Once real keys are filled in, do not commit local `.env` modifications.
- Current keys: `DASHSCOPE_API_KEY` for embedding/rerank and `API_KEY` for the OpenAI-compatible chat model

### Input Format
- Only `.md` Markdown files. No JSON parsing.
- `# ` heading = document title. Formal section headings become `section_hint`.
- The chunker now removes authors / affiliations / `ARTICLE INFO` / `Keywords` / image Markdown / references tail sections, keeps figure captions, and builds section-anchored semantic chunks with `Title + Abstract` as the first chunk.

### Module Boundaries
- `builder/` knows nothing about LLM reasoning or reranking
- `query/` knows nothing about chunking or embedding generation (except through Protocols)
- `core/` has zero cross-module imports

### Storage Truth
- SQLite is still the only source of truth for retrieval and answering.
- In addition, builder now exports inspection-only chunk artifacts to `.rag_store/chunks/` as `<doc_id>.chunks.json` and `<doc_id>.chunks.md`.
- `chunk_fts` virtual table for FTS5, `chunks` for vectors+metadata.

## 4. Defaults Locked

| Setting | Value |
|---------|-------|
| EMBEDDING_MODEL | text-embedding-v4 |
| EMBEDDING_DIMENSIONS | 2048 |
| RERANK_MODEL | qwen3-rerank |
| RERANK_DOCUMENT_CHUNK_LIMIT | 3 |
| RERANK_DOCUMENT_TEXT_LIMIT | 0 |
| RERANK_REQUEST_TOKEN_BUDGET | 3600 |
| MODEL | deepseek-chat |
| CHUNK_SIZE | 2200 |
| CHUNK_OVERLAP | 100 |
| TOP_K_VECTOR / KEYWORD | 30 |
| DOCUMENT_RECALL_LIMIT | 300 |
| DOCUMENT_JUDGE_LIMIT | 100 |
| DOCUMENT_JUDGE_INITIAL_CONCURRENCY | 5 |
| EARLY_STOP_CONSECUTIVE_NON_SUPPORT | 20 |
| RRF_K | 60.0 |
| HYBRID_CO_OCCURRENCE_BONUS | 0.15 |

## 5. Extension Points

Three Protocol classes for swapping backends:

```python
class Embedder(Protocol):
    def embed_texts(self, texts: list[str]) -> list[np.ndarray]: ...
    def embed_query(self, query: str) -> np.ndarray: ...

class Reranker(Protocol):
    def rerank(self, *, query: str, documents: list[str], top_n: int | None = None) -> list[tuple[int, float]]: ...

class LLMClient(Protocol):
    async def complete_json(self, system_prompt: str, user_prompt: str) -> dict[str, Any]: ...
```

Inject via `embedder=`, `reranker=`, `llm=` kwargs in the Python API. `answer()` is awaited and `answer_stream()` is consumed with `async for`.

## 6. CLI Commands

```bash
# Build index from a directory of Markdown files
literature-rag build ./papers [--index-path .rag_store/db.sqlite3] [--rebuild]

# Add new papers incrementally from a directory
literature-rag add ./new_papers [--index-path .rag_store/db.sqlite3]

# Interactive query mode (replaces old search/ask)
literature-rag [--index-path .rag_store/db.sqlite3]
```

In interactive mode, the CLI prompts for a question, runs the full agentic pipeline, and outputs:
- Numbered answer lines (one per paper)
- Inline `[N] citation` after each line
- A numbered reference list
- Scan status summary

## 7. Testing

- Run: `pytest -q`
- All tests use monkeypatched `FakeEmbedder` / fake LLM clients — no real API calls in tests
- Key test files:
  - `test_builder_chunker.py` — Markdown parsing
  - `test_builder.py` — build_index + add_documents incremental
  - `test_query.py` — search + answer end-to-end
  - `test_query_agent.py` — AgenticAnswerer logic
  - `test_cli.py` — CLI integration
  - `test_core_utils.py` — shared utilities

## 8. When Modifying

- **Parser changes**: affect `builder/chunker.py` only. No JSON anything.
- **Embedding changes**: update `builder/embedder.py`, `core/config.py`, `core/models.py` (if needed), and `tests/`.
- **Retrieval changes**: update `query/retriever.py` and `query/__init__.py`.
- **Agent loop changes**: never regress to one-shot retrieval. Preserve the analyze → retrieve → judge(write) → deterministic assemble pipeline.
- **Judge provider pressure**: already-dispatched documents must be retried at the same rank with lower concurrency until success or a non-transient error.
- **New config**: add to `core/config.py` with env var fallback. Update `README.md`.
- **Dependencies**: add to `pyproject.toml` and `pip install`.
