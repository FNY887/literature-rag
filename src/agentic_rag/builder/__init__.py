from __future__ import annotations

import shutil
from pathlib import Path
from typing import Protocol

import numpy as np

from agentic_rag.core.config import Settings, get_settings
from agentic_rag.core.models import BuildStats, DeleteStats
from agentic_rag.core.utils import normalize_title

from .artifacts import chunk_artifacts_exist, clear_chunk_artifacts, delete_chunk_artifacts, write_chunk_artifacts
from .chunker import chunk_markdown
from .embedder import DashScopeEmbeddingClient, chunk_embedding_input
from .store import SQLiteIndexStore


class Embedder(Protocol):
    def embed_texts(self, texts: list[str]) -> list[np.ndarray]: ...
    def embed_query(self, query: str) -> np.ndarray: ...


def _resolve_settings(
    settings: Settings | None,
    *,
    source_dir: str | Path | None = None,
    index_path: str | Path | None = None,
) -> Settings:
    resolved = settings.resolved() if settings is not None else get_settings()
    if source_dir is not None:
        resolved.data_dir = Path(source_dir) if Path(source_dir).is_absolute() else Path.cwd() / source_dir
    if index_path is not None:
        resolved.index_path = Path(index_path) if Path(index_path).is_absolute() else Path.cwd() / index_path
    resolved.index_dir = resolved.index_path.parent
    return resolved


def _collect_markdown_paths(paths: list[str | Path]) -> list[Path]:
    collected: list[Path] = []
    seen: set[str] = set()

    for raw_path in sorted(Path(path) for path in paths):
        if raw_path.is_dir():
            candidates = sorted(path for path in raw_path.rglob("*.md") if path.is_file())
        elif raw_path.is_file() and raw_path.suffix.lower() == ".md":
            candidates = [raw_path]
        else:
            continue

        for candidate in candidates:
            resolved_key = str(candidate.resolve())
            if resolved_key in seen:
                continue
            seen.add(resolved_key)
            collected.append(candidate)
    return collected


def _remove_sidecar_storage(index_path: Path) -> None:
    shutil.rmtree(Path(f"{index_path}.sidecar"), ignore_errors=True)


def _remove_index_storage(index_path: Path) -> None:
    for path in (
        index_path,
        Path(f"{index_path}-wal"),
        Path(f"{index_path}-shm"),
    ):
        try:
            path.unlink()
        except FileNotFoundError:
            continue
    _remove_sidecar_storage(index_path)


def _normalized_title(title: str) -> str:
    normalized = normalize_title(title)
    if normalized:
        return normalized
    raise ValueError("Markdown paper must define a valid level-1 title heading.")


def _record_failed_file(stats: BuildStats, path: Path, reason: str) -> None:
    stats.failed_files.append(f"{path}: {reason}")


def _delete_query_keys(title: str) -> set[str]:
    cleaned = title.strip()
    keys = {normalize_title(cleaned)}
    basename = Path(cleaned).name
    if basename:
        keys.add(normalize_title(basename))
        keys.add(normalize_title(Path(basename).stem))
    return {key for key in keys if key}


def _find_delete_matches(store: SQLiteIndexStore, title: str) -> list[tuple[object, str]]:
    cleaned = title.strip()
    basename = Path(cleaned).name if cleaned else cleaned
    basename_lower = basename.lower()
    keys = _delete_query_keys(cleaned)
    matches: dict[str, tuple[object, str]] = {}

    for row in store.list_documents():
        doc_id = str(row["doc_id"])
        source_path = Path(str(row["source_path"]))
        matched_by = ""
        if str(row["normalized_title"]) in keys:
            matched_by = "title"
        elif source_path.name.lower() == basename_lower:
            matched_by = "filename"
        elif normalize_title(source_path.stem) in keys:
            matched_by = "filename"

        if matched_by:
            previous = matches.get(doc_id)
            if previous is None or previous[1] != "title":
                matches[doc_id] = (row, matched_by)

    return list(matches.values())


def _delete_source_markdown(source_path: str) -> bool:
    path = Path(source_path)
    if not path.exists():
        return False
    if not path.is_file() or path.suffix.lower() != ".md":
        raise RuntimeError(f"拒绝删除非 Markdown 源文件：{path}")
    path.unlink()
    return True


def _process_file(
    path: Path,
    store: SQLiteIndexStore,
    embedder: Embedder,
    artifacts_dir: Path,
    chunk_size: int,
    chunk_overlap: int,
    stats: BuildStats,
) -> None:
    try:
        doc_id, title, file_hash, chunks = chunk_markdown(
            path, chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        normalized_title = _normalized_title(title)
    except ValueError as exc:
        _record_failed_file(stats, path, str(exc))
        return
    state = store.get_document_state(str(path))
    title_owner = store.get_document_by_normalized_title(normalized_title)

    if state is not None and str(state["file_hash"]) == file_hash:
        doc_id = str(state["doc_id"])
        if not chunk_artifacts_exist(artifacts_dir, doc_id):
            stats.artifacts_written += write_chunk_artifacts(
                artifacts_dir=artifacts_dir,
                doc_id=doc_id,
                title=str(state["title"]),
                source_path=str(path),
                file_hash=file_hash,
                chunks=store.fetch_document_chunks(doc_id),
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
        stats.skipped_files += 1
        return

    if title_owner is not None and str(title_owner["source_path"]) != str(path):
        if state is not None:
            raise RuntimeError(
                "Document title conflicts with an existing indexed paper from a different source path. "
                "Resolve the duplicate title or rebuild with unique inputs."
            )
        stats.duplicates_skipped += 1
        return

    if chunks:
        try:
            vectors = embedder.embed_texts([chunk_embedding_input(chunk) for chunk in chunks])
        except Exception as exc:
            _record_failed_file(stats, path, f"embedding failed: {exc}")
            return
        store.replace_document(
            doc_id=doc_id,
            source_path=str(path),
            title=title,
            normalized_title=normalized_title,
            file_hash=file_hash,
            chunks=chunks,
            vectors=vectors,
        )
    else:
        store.replace_document(
            doc_id=doc_id,
            source_path=str(path),
            title=title,
            normalized_title=normalized_title,
            file_hash=file_hash,
            chunks=[],
            vectors=[],
        )
    stats.artifacts_written += write_chunk_artifacts(
        artifacts_dir=artifacts_dir,
        doc_id=doc_id,
        title=title,
        source_path=str(path),
        file_hash=file_hash,
        chunks=chunks,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    stats.processed_files += 1
    stats.chunks_indexed += len(chunks)


def build_index(
    *,
    source_dir: str | Path,
    index_path: str | Path,
    settings: Settings | None = None,
    rebuild: bool = False,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    embedder: Embedder | None = None,
) -> BuildStats:
    resolved = _resolve_settings(settings, source_dir=source_dir, index_path=index_path)
    resolved.ensure_index_dir()
    resolved.ensure_cleaned_artifacts_dir()
    if rebuild:
        _remove_index_storage(resolved.index_path)
        clear_chunk_artifacts(resolved.cleaned_artifacts_dir)
        resolved.ensure_cleaned_artifacts_dir()
    else:
        _remove_sidecar_storage(resolved.index_path)

    store = SQLiteIndexStore(resolved.index_path)
    store.ensure_schema()

    embedding_client = embedder or DashScopeEmbeddingClient(resolved)
    stats = BuildStats(index_path=str(resolved.index_path), cleaning_profile="markdown")
    size = chunk_size if chunk_size is not None else resolved.chunk_size
    overlap = chunk_overlap if chunk_overlap is not None else resolved.chunk_overlap

    for path in _collect_markdown_paths([resolved.data_dir]):
        _process_file(
            path,
            store,
            embedding_client,
            resolved.cleaned_artifacts_dir,
            size,
            overlap,
            stats,
        )

    return stats


def delete_document(
    *,
    title: str,
    index_path: str | Path,
    settings: Settings | None = None,
) -> DeleteStats:
    resolved = _resolve_settings(settings, index_path=index_path)
    if not resolved.index_path.exists():
        raise RuntimeError(
            f"向量库不存在：{resolved.index_path}。请先运行 literature-rag build <存放Markdown文献的目录>。"
        )

    _remove_sidecar_storage(resolved.index_path)
    store = SQLiteIndexStore(resolved.index_path)
    store.ensure_schema()
    matches = _find_delete_matches(store, title)
    if not matches:
        raise RuntimeError(f"未找到要删除的文献：{title}")
    if len(matches) > 1:
        candidates = "; ".join(
            f"{row['title']} ({Path(str(row['source_path'])).name})"
            for row, _ in matches
        )
        raise RuntimeError(f"删除目标不唯一，请输入更完整的论文标题或文件名：{candidates}")

    row, matched_by = matches[0]
    source_deleted = _delete_source_markdown(str(row["source_path"]))
    deleted = store.delete_document_by_doc_id(str(row["doc_id"]))
    if deleted is None:
        raise RuntimeError(f"未找到要删除的文献：{title}")
    deleted_row, chunks_deleted = deleted
    artifacts_deleted = delete_chunk_artifacts(resolved.cleaned_artifacts_dir, str(deleted_row["doc_id"]))

    return DeleteStats(
        deleted=True,
        doc_id=str(deleted_row["doc_id"]),
        title=str(deleted_row["title"]),
        source_path=str(deleted_row["source_path"]),
        source_deleted=source_deleted,
        matched_by=matched_by,
        chunks_deleted=chunks_deleted,
        artifacts_deleted=artifacts_deleted,
        index_path=str(resolved.index_path),
    )


def add_documents(
    *,
    paths: list[str | Path],
    index_path: str | Path,
    settings: Settings | None = None,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    embedder: Embedder | None = None,
) -> BuildStats:
    resolved = _resolve_settings(settings, index_path=index_path)
    resolved.ensure_index_dir()
    resolved.ensure_cleaned_artifacts_dir()
    _remove_sidecar_storage(resolved.index_path)
    store = SQLiteIndexStore(resolved.index_path)
    store.ensure_schema()

    embedding_client = embedder or DashScopeEmbeddingClient(resolved)
    stats = BuildStats(index_path=str(resolved.index_path), cleaning_profile="markdown")
    size = chunk_size if chunk_size is not None else resolved.chunk_size
    overlap = chunk_overlap if chunk_overlap is not None else resolved.chunk_overlap

    for path in _collect_markdown_paths(paths):
        _process_file(
            path,
            store,
            embedding_client,
            resolved.cleaned_artifacts_dir,
            size,
            overlap,
            stats,
        )

    return stats
