# Agentic Literature RAG

`agentic-literature-rag` is a local agentic RAG package for Markdown literature
corpora. It splits the workflow into two independent modules:

1. **Builder** (`agentic_rag.builder`): chunk Markdown files, generate embeddings,
   persist a hybrid index (SQLite + FTS5 + vectors), and export cleaned/chunked artifacts for inspection under `.rag_store/chunks/`.
2. **Query** (`agentic_rag.query`): analyze questions, run hybrid retrieval,
   either grade evidence per paper for literature search or synthesize a chunk-grounded research report.

The design is not "one-shot vector retrieval then generate". Instead, an agent
plans retrieval, scores **all** chunks without truncation, aggregates documents
by the average of their top-3 chunk scores, re-ranks the top 300 documents,
judges evidence per paper on the top 100, and stops early when no more
direct-support papers are found.

The optional deep research route keeps the same retrieval planner, re-ranks the
top 500 chunks, sends up to 100 complete chunks within a context budget, and
asks the chat model to write a cited research report of at least 500 characters, with no fixed upper limit.

## Installation

```bash
pip install -e .
```

Requires Python >=3.11.

## Configuration

All API keys and tunables are read from environment variables.

### Embeddings & Rerank

- `DASHSCOPE_API_KEY`: required for real embedding generation and reranking
- `DASHSCOPE_BASE_URL`: defaults to `https://dashscope.aliyuncs.com/compatible-mode/v1`
- `EMBEDDING_MODEL`: defaults to `text-embedding-v4`
- `EMBEDDING_DIMENSIONS`: defaults to `2048`
- `RERANK_MODEL`: defaults to `qwen3-rerank`
- `RERANK_DOCUMENT_CHUNK_LIMIT`: defaults to `3`
- `RERANK_DOCUMENT_TEXT_LIMIT`: defaults to `0` (no truncation)
- `RERANK_REQUEST_TOKEN_BUDGET`: defaults to `3600` for batched rerank requests

### Agent reasoning

- `CHAT_API_KEY`: required for OpenAI-compatible ask agent reasoning
- `CHAT_BASE_URL`: defaults to `https://api.deepseek.com`
- `CHAT_MODEL`: defaults to `deepseek-chat`
- Common compatible endpoints:
  - DeepSeek: `CHAT_BASE_URL=https://api.deepseek.com`
  - DashScope compatible mode: `CHAT_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1`
  - OpenAI: `CHAT_BASE_URL=https://api.openai.com/v1`

### Chunking & retrieval

- `CHUNK_SIZE`: defaults to `2200`
- `CHUNK_OVERLAP`: defaults to `100`
- Chunk artifacts are exported by default to `.rag_store/chunks`
- `TOP_K_VECTOR`: defaults to `30`
- `TOP_K_KEYWORD`: defaults to `30`
- `DOCUMENT_RECALL_LIMIT`: defaults to `300`
- `DOCUMENT_JUDGE_LIMIT`: defaults to `100`
- `DOCUMENT_JUDGE_INITIAL_CONCURRENCY`: defaults to `5`; provider pressure retries already-dispatched documents with lower concurrency
- `RESEARCH_RERANK_CHUNK_LIMIT`: defaults to `500`
- `RESEARCH_FINAL_CHUNK_LIMIT`: defaults to `100`
- `RESEARCH_CONTEXT_TOKEN_BUDGET`: defaults to `120000`
- `RESEARCH_REPORT_MIN_CHARS`: defaults to `500`; Python character count, so one Chinese character counts as one

## CLI

```bash
# Rebuild the vector database from a directory of Markdown files
literature-rag build ./papers

# Add new papers incrementally from a directory
literature-rag add ./new_papers

# Interactive mode selection: literature search or deep research
literature-rag

# Show command help
literature-rag --help
```

Default behavior:

- `literature-rag build <dir>` clears the current vector database and rebuilds it from the Markdown files under `<dir>`.
- `literature-rag add <dir>` keeps the existing vector database and adds new Markdown papers from `<dir>`.
- `literature-rag` loads the default database path `.rag_store/literature_rag.sqlite3`, then shows an up/down-key menu for `文献检索` or `深度研究`.
- `literature-rag --help` shows the available commands and usage.

Literature search mode prompts for a question, runs the document-level agentic pipeline, and outputs:
1. Numbered answer lines (one per paper)
2. Numbered reference list (multi-chunk papers list all supporting chunks)
3. Scan status summary

Deep research mode prompts for a research question, runs chunk-level rerank, and outputs:
1. A research report of at least `RESEARCH_REPORT_MIN_CHARS`, with length decided by the chat model from the provided chunks
2. A reference list for cited chunks
3. Recall/rerank/context chunk counts and timing summary

## Python API

### Builder

```python
from agentic_rag.builder import build_index, add_documents

# Build from a directory
build_index(
    source_dir="literature",
    index_path=".rag_store/literature_rag.sqlite3",
)

# Add more papers later
add_documents(
    paths=["new_paper1.md", "new_paper2.md"],
    index_path=".rag_store/literature_rag.sqlite3",
)
```

### Query

```python
import asyncio

from agentic_rag.query import answer, answer_stream, research, search

# Search
hits = search(
    query="prenucleation cluster mediated calcium phosphate nucleation",
    mode="hybrid",
    top_k=5,
)

async def main():
    result = await answer(
        question="What evidence links prenucleation clusters to collagen mineralization?",
    )
    print(result.answer)
    print(result.citations)

    async for event in answer_stream(
        question="What evidence links prenucleation clusters to collagen mineralization?",
    ):
        print(event.event)

asyncio.run(main())

# Deep research report
report = asyncio.run(
    research(
        question="summarize evidence for collagen intrafibrillar mineralization",
    )
)
print(report.report)
```

## Architecture

```
agentic_rag/
├── builder/          # Offline indexing
│   ├── chunker.py    # Markdown → semantic chunks
│   ├── embedder.py   # Embedding client (DashScope)
│   └── store.py      # SQLite + FTS5 + vectors
├── query/            # Online retrieval
│   ├── retriever.py  # Hybrid search (vector + keyword + RRF)
│   ├── reranker.py   # Document reranker (DashScope)
│   └── agent.py      # Evidence judge + deterministic answer assembly
├── core/             # Shared
│   ├── models.py     # Pydantic data models
│   ├── config.py     # Settings
│   ├── llm.py        # OpenAI-compatible chat LLM client
│   └── utils.py      # Retry, dedupe, JSON extraction
└── cli.py            # Typer CLI
```

### Pluggable backends

The package defines three `Protocol` classes for swapping backends:

- `Embedder`: `embed_texts()` / `embed_query()`
- `Reranker`: `rerank()`
- `LLMClient`: `await complete_json()`

Inject your own implementations via the `embedder=`, `reranker=`, and `llm=`
keyword arguments in the Python API.

## Input format

Each literature file should be a plain Markdown (`.md`) file:

```markdown
# Paper Title

## Abstract
The abstract text goes here...

## Introduction
Introduction paragraphs...

## Methods
Methods paragraphs...
```

The chunker is tuned for standard English journal-paper Markdown:

- It uses the first `# ` heading as the document title and keeps formal section headings as `section_hint`.
- It removes front matter such as authors, affiliations, `ARTICLE INFO`, `Keywords`, received/accepted metadata, image Markdown, and everything after `References` / `Bibliography` / similar tail sections.
- It preserves `Abstract`, `Statement of significance` / `Highlights`, body paragraphs, and figure captions.
- It builds section-anchored semantic chunks: the first chunk is `Title + Abstract`, and body chunks start from `section heading + first paragraph`, merging short follow-on paragraphs when needed.

After build/add runs, the cleaned and chunked result of each paper is also written to `.rag_store/chunks/`:

- `<doc_id>.chunks.json` — structured chunk metadata and text
- `<doc_id>.chunks.md` — human-readable chunk preview for manual inspection
