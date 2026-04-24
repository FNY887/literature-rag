from __future__ import annotations

import json
from pathlib import Path
import shutil

from agentic_rag.core.models import ChunkRecord


def _artifact_paths(artifacts_dir: Path, doc_id: str) -> tuple[Path, Path]:
    return (
        artifacts_dir / f"{doc_id}.chunks.json",
        artifacts_dir / f"{doc_id}.chunks.md",
    )


def clear_chunk_artifacts(artifacts_dir: Path) -> None:
    if artifacts_dir.exists():
        shutil.rmtree(artifacts_dir)


def delete_chunk_artifacts(artifacts_dir: Path, doc_id: str) -> int:
    deleted = 0
    for path in _artifact_paths(artifacts_dir, doc_id):
        try:
            path.unlink()
        except FileNotFoundError:
            continue
        deleted += 1
    return deleted


def chunk_artifacts_exist(artifacts_dir: Path, doc_id: str) -> bool:
    return all(path.exists() for path in _artifact_paths(artifacts_dir, doc_id))


def write_chunk_artifacts(
    *,
    artifacts_dir: Path,
    doc_id: str,
    title: str,
    source_path: str,
    file_hash: str,
    chunks: list[ChunkRecord],
    chunk_size: int,
    chunk_overlap: int,
) -> int:
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    json_path, markdown_path = _artifact_paths(artifacts_dir, doc_id)

    payload = {
        "doc_id": doc_id,
        "title": title,
        "source_path": source_path,
        "file_hash": file_hash,
        "cleaning_profile": "markdown_chunked",
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "chunk_count": len(chunks),
        "chunks": [
            {
                "chunk_id": chunk.chunk_id,
                "section_hint": chunk.section_hint,
                "page_start": chunk.page_start,
                "page_end": chunk.page_end,
                "block_start": chunk.block_start,
                "block_end": chunk.block_end,
                "keywords_hint": chunk.keywords_hint,
                "text": chunk.text,
            }
            for chunk in chunks
        ],
    }
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        f"# {title}",
        "",
        f"- doc_id: `{doc_id}`",
        f"- source_path: `{source_path}`",
        f"- file_hash: `{file_hash}`",
        f"- chunk_count: `{len(chunks)}`",
        f"- chunk_size: `{chunk_size}`",
        f"- chunk_overlap: `{chunk_overlap}`",
        "",
    ]
    for index, chunk in enumerate(chunks, start=1):
        lines.extend(
            [
                f"## Chunk {index:04d}",
                "",
                f"- chunk_id: `{chunk.chunk_id}`",
                f"- section_hint: `{chunk.section_hint or ''}`",
                f"- keywords_hint: `{chunk.keywords_hint}`",
                "",
                chunk.text,
                "",
            ]
        )
    markdown_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return 2
