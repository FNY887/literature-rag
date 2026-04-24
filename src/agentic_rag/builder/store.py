from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import ClassVar

import numpy as np

from agentic_rag.core.models import ChunkRecord, SearchHit


def vector_to_blob(vector: np.ndarray) -> bytes:
    return np.asarray(vector, dtype=np.float32).tobytes()


def blob_to_vector(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32)


def _shorten_chunk_id(chunk_id: str) -> str:
    if ":" not in chunk_id:
        return chunk_id
    prefix, segment = chunk_id.rsplit(":", 1)
    if "-" in prefix:
        short_hash = prefix.rsplit("-", 1)[-1]
        return f"{short_hash}:{segment}"
    return chunk_id


def build_citation(title: str, page_start: int, page_end: int, chunk_id: str) -> str:
    short_id = _shorten_chunk_id(chunk_id)
    if page_start == page_end:
        page_text = f"p. {page_start}"
    else:
        page_text = f"pp. {page_start}-{page_end}"
    return f"{title} ({page_text}, chunk {short_id})"


@dataclass(slots=True)
class SQLiteIndexStore:
    path: str | Path
    _vector_cache: ClassVar[
        dict[str, tuple[tuple[tuple[int, int] | None, tuple[int, int] | None], list[sqlite3.Row], np.ndarray]]
    ] = {}
    _vector_cache_lock: ClassVar[Lock] = Lock()

    def __post_init__(self) -> None:
        self.path = str(self.path)

    def _path_signature(self, path: Path) -> tuple[int, int] | None:
        try:
            stat = path.stat()
        except FileNotFoundError:
            return None
        return (stat.st_mtime_ns, stat.st_size)

    def _vector_cache_signature(self) -> tuple[tuple[int, int] | None, tuple[int, int] | None]:
        db_path = Path(self.path)
        wal_path = Path(f"{self.path}-wal")
        return (self._path_signature(db_path), self._path_signature(wal_path))

    def _invalidate_vector_cache(self) -> None:
        with self._vector_cache_lock:
            self._vector_cache.pop(self.path, None)

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        return connection

    def database_signature(self) -> dict[str, list[int] | None]:
        def _serialize(signature: tuple[int, int] | None) -> list[int] | None:
            if signature is None:
                return None
            return [int(signature[0]), int(signature[1])]

        db_path = Path(self.path)
        wal_path = Path(f"{self.path}-wal")
        return {
            "db": _serialize(self._path_signature(db_path)),
            "wal": _serialize(self._path_signature(wal_path)),
        }

    def ensure_schema(self) -> None:
        with self.connect() as connection:
            existing_documents = connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'documents'"
            ).fetchone()
            if existing_documents is not None:
                columns = {
                    row["name"]
                    for row in connection.execute("PRAGMA table_info(documents)").fetchall()
                }
                if "normalized_title" not in columns:
                    raise RuntimeError(
                        "Index schema is outdated and requires a rebuild. Run build with rebuild=True."
                    )
            connection.executescript(
                """
                PRAGMA journal_mode=WAL;

                CREATE TABLE IF NOT EXISTS documents (
                    doc_id TEXT PRIMARY KEY,
                    source_path TEXT NOT NULL UNIQUE,
                    title TEXT NOT NULL,
                    normalized_title TEXT NOT NULL UNIQUE,
                    file_hash TEXT NOT NULL,
                    chunk_count INTEGER NOT NULL,
                    indexed_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS chunks (
                    chunk_id TEXT PRIMARY KEY,
                    doc_id TEXT NOT NULL,
                    title TEXT NOT NULL,
                    source_path TEXT NOT NULL,
                    page_start INTEGER NOT NULL,
                    page_end INTEGER NOT NULL,
                    block_start INTEGER NOT NULL,
                    block_end INTEGER NOT NULL,
                    section_hint TEXT,
                    text TEXT NOT NULL,
                    normalized_text TEXT NOT NULL,
                    keywords_hint TEXT NOT NULL,
                    vector_dim INTEGER NOT NULL,
                    vector BLOB NOT NULL
                );

                CREATE VIRTUAL TABLE IF NOT EXISTS chunk_fts USING fts5(
                    chunk_id UNINDEXED,
                    title,
                    text,
                    keywords_hint,
                    source_path,
                    tokenize = 'unicode61'
                );
                """
            )
            connection.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS idx_documents_normalized_title ON documents(normalized_title)"
            )

    def clear(self) -> None:
        with self.connect() as connection:
            connection.execute("DELETE FROM chunk_fts")
            connection.execute("DELETE FROM chunks")
            connection.execute("DELETE FROM documents")
        self._invalidate_vector_cache()

    def checkpoint(self) -> None:
        with self.connect() as connection:
            connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")

    def get_document_state(self, source_path: str) -> sqlite3.Row | None:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT doc_id, title, normalized_title, file_hash FROM documents WHERE source_path = ?",
                (source_path,),
            ).fetchone()
        return row

    def get_document_by_normalized_title(self, normalized_title: str) -> sqlite3.Row | None:
        with self.connect() as connection:
            row = connection.execute(
                """
                SELECT doc_id, source_path, title, normalized_title, file_hash
                FROM documents
                WHERE normalized_title = ?
                """,
                (normalized_title,),
            ).fetchone()
        return row

    def list_documents(self) -> list[sqlite3.Row]:
        with self.connect() as connection:
            return connection.execute(
                """
                SELECT doc_id, source_path, title, normalized_title, file_hash, chunk_count
                FROM documents
                ORDER BY title, source_path
                """
            ).fetchall()

    def _delete_documents(self, connection: sqlite3.Connection, doc_ids: set[str]) -> None:
        if not doc_ids:
            return
        for doc_id in doc_ids:
            existing_chunk_ids = connection.execute(
                "SELECT chunk_id FROM chunks WHERE doc_id = ?",
                (doc_id,),
            ).fetchall()
            for row in existing_chunk_ids:
                connection.execute(
                    "DELETE FROM chunk_fts WHERE chunk_id = ?",
                    (row["chunk_id"],),
                )
            connection.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))
            connection.execute("DELETE FROM documents WHERE doc_id = ?", (doc_id,))

    def delete_document_by_doc_id(self, doc_id: str) -> tuple[sqlite3.Row, int] | None:
        with self.connect() as connection:
            row = connection.execute(
                """
                SELECT doc_id, source_path, title, normalized_title, file_hash, chunk_count
                FROM documents
                WHERE doc_id = ?
                """,
                (doc_id,),
            ).fetchone()
            if row is None:
                return None
            chunk_count = connection.execute(
                "SELECT COUNT(*) FROM chunks WHERE doc_id = ?",
                (doc_id,),
            ).fetchone()[0]
            self._delete_documents(connection, {doc_id})
        self._invalidate_vector_cache()
        return row, int(chunk_count)

    def replace_document(
        self,
        *,
        doc_id: str,
        source_path: str,
        title: str,
        normalized_title: str,
        file_hash: str,
        chunks: list[ChunkRecord],
        vectors: list[np.ndarray],
    ) -> None:
        with self.connect() as connection:
            existing_rows = connection.execute(
                """
                SELECT doc_id
                FROM documents
                WHERE doc_id = ? OR source_path = ? OR normalized_title = ?
                """,
                (doc_id, source_path, normalized_title),
            ).fetchall()
            self._delete_documents(connection, {str(row["doc_id"]) for row in existing_rows})

            for chunk, vector in zip(chunks, vectors, strict=True):
                connection.execute(
                    """
                    INSERT INTO chunks (
                        chunk_id, doc_id, title, source_path, page_start, page_end,
                        block_start, block_end, section_hint, text, normalized_text,
                        keywords_hint, vector_dim, vector
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        chunk.chunk_id,
                        chunk.doc_id,
                        chunk.title,
                        chunk.source_path,
                        chunk.page_start,
                        chunk.page_end,
                        chunk.block_start,
                        chunk.block_end,
                        chunk.section_hint,
                        chunk.text,
                        chunk.normalized_text,
                        chunk.keywords_hint,
                        int(vector.shape[0]),
                        vector_to_blob(vector),
                    ),
                )
                connection.execute(
                    """
                    INSERT INTO chunk_fts (chunk_id, title, text, keywords_hint, source_path)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        chunk.chunk_id,
                        chunk.title,
                        chunk.text,
                        chunk.keywords_hint,
                        chunk.source_path,
                    ),
                )

            connection.execute(
                """
                INSERT INTO documents (doc_id, source_path, title, normalized_title, file_hash, chunk_count)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (doc_id, source_path, title, normalized_title, file_hash, len(chunks)),
            )
        self._invalidate_vector_cache()

    def fetch_chunk(self, chunk_id: str) -> sqlite3.Row:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT * FROM chunks WHERE chunk_id = ?",
                (chunk_id,),
            ).fetchone()
        if row is None:
            raise KeyError(f"Chunk not found: {chunk_id}")
        return row

    def fetch_document_chunks(self, doc_id: str) -> list[ChunkRecord]:
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT *
                FROM chunks
                WHERE doc_id = ?
                ORDER BY chunk_id
                """,
                (doc_id,),
            ).fetchall()
        return [
            ChunkRecord(
                chunk_id=row["chunk_id"],
                doc_id=row["doc_id"],
                title=row["title"],
                source_path=row["source_path"],
                page_start=int(row["page_start"]),
                page_end=int(row["page_end"]),
                text=row["text"],
                section_hint=row["section_hint"],
                keywords_hint=row["keywords_hint"],
                normalized_text=row["normalized_text"],
                block_start=int(row["block_start"]),
                block_end=int(row["block_end"]),
            )
            for row in rows
        ]

    def fetch_all_vectors(self) -> tuple[list[sqlite3.Row], np.ndarray]:
        signature = self._vector_cache_signature()
        with self._vector_cache_lock:
            cached = self._vector_cache.get(self.path)
            if cached is not None and cached[0] == signature:
                return cached[1], cached[2]

        with self.connect() as connection:
            rows = connection.execute("SELECT * FROM chunks").fetchall()
        if not rows:
            matrix = np.zeros((0, 0), dtype=np.float32)
        else:
            vectors = [blob_to_vector(row["vector"]) for row in rows]
            matrix = np.vstack(vectors).astype(np.float32)

        with self._vector_cache_lock:
            self._vector_cache[self.path] = (self._vector_cache_signature(), rows, matrix)
        return rows, matrix

    def keyword_rows(self, query: str, limit: int | None) -> list[sqlite3.Row]:
        sql = """
            SELECT
                chunk_id,
                bm25(chunk_fts) AS bm25_score
            FROM chunk_fts
            WHERE chunk_fts MATCH ?
            ORDER BY bm25_score
        """
        params: tuple = (query,)
        if limit is not None:
            sql += " LIMIT ?"
            params = (query, limit)
        with self.connect() as connection:
            return connection.execute(sql, params).fetchall()

    def keyword_search(self, query: str, limit: int | None) -> list[SearchHit]:
        sql = """
            SELECT
                c.doc_id,
                c.chunk_id,
                c.title,
                c.source_path,
                c.page_start,
                c.page_end,
                c.section_hint,
                c.text,
                bm25(chunk_fts) AS bm25_score
            FROM chunk_fts
            JOIN chunks c ON c.chunk_id = chunk_fts.chunk_id
            WHERE chunk_fts MATCH ?
            ORDER BY bm25_score
        """
        params: tuple = (query,)
        if limit is not None:
            sql += " LIMIT ?"
            params = (query, limit)
        with self.connect() as connection:
            rows = connection.execute(sql, params).fetchall()
        hits: list[SearchHit] = []
        for row in rows:
            score_keyword = 1.0 / (1.0 + max(float(row["bm25_score"]), 0.0))
            hits.append(
                SearchHit(
                    doc_id=row["doc_id"],
                    chunk_id=row["chunk_id"],
                    score_keyword=score_keyword,
                    score_vector=None,
                    score_final=score_keyword,
                    retrieval_source="keyword",
                    text=row["text"],
                    citation=build_citation(
                        title=row["title"],
                        page_start=int(row["page_start"]),
                        page_end=int(row["page_end"]),
                        chunk_id=row["chunk_id"],
                    ),
                    title=row["title"],
                    source_path=row["source_path"],
                    page_start=int(row["page_start"]),
                    page_end=int(row["page_end"]),
                    section_hint=row["section_hint"],
                )
            )
        return hits
