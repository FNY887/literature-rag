from pathlib import Path
import json

import numpy as np
import pytest

from agentic_rag.builder import add_documents, build_index, delete_document
from agentic_rag.builder.chunker import chunk_markdown
from agentic_rag.builder.store import SQLiteIndexStore, build_citation
from agentic_rag.core.config import get_settings


class FakeEmbedder:
    def embed_texts(self, texts: list[str]) -> list[np.ndarray]:
        return [self._embed(text) for text in texts]

    def embed_query(self, query: str) -> np.ndarray:
        return self._embed(query)

    def _embed(self, text: str) -> np.ndarray:
        return np.asarray([0.5, 0.5], dtype=np.float32)


class FailingEmbedder:
    def embed_texts(self, texts: list[str]) -> list[np.ndarray]:
        raise RuntimeError("embedding input too long")

    def embed_query(self, query: str) -> np.ndarray:
        raise RuntimeError("not used")


def _write_sample_md(source_dir: Path, name: str, content: str) -> None:
    (source_dir / name).write_text(content, encoding="utf-8")


def test_build_index_and_incremental(tmp_path: Path):
    source_dir = tmp_path / "literature"
    source_dir.mkdir()
    index_path = tmp_path / "index.sqlite3"

    _write_sample_md(
        source_dir,
        "paper1.md",
        "# Paper One\n\n## Abstract\n\nThis is about calcium phosphate mineralization under alkaline conditions.\n",
    )
    _write_sample_md(
        source_dir,
        "paper2.md",
        "# Paper Two\n\n## Abstract\n\nThis discusses water dynamics in nanoconfinement.\n",
    )

    settings = get_settings()
    settings.dashscope_api_key = None
    settings.cleaned_artifacts_dir = tmp_path / "chunk_artifacts"
    embedder = FakeEmbedder()

    first = build_index(
        source_dir=source_dir,
        index_path=index_path,
        settings=settings,
        embedder=embedder,
    )
    assert first.processed_files == 2
    assert first.chunks_indexed >= 2
    assert first.artifacts_written == 4

    doc_id, _, _, _ = chunk_markdown(source_dir / "paper1.md")
    json_artifact = settings.cleaned_artifacts_dir / f"{doc_id}.chunks.json"
    markdown_artifact = settings.cleaned_artifacts_dir / f"{doc_id}.chunks.md"
    assert json_artifact.exists()
    assert markdown_artifact.exists()
    payload = json.loads(json_artifact.read_text(encoding="utf-8"))
    assert payload["doc_id"] == doc_id
    assert payload["chunk_count"] >= 1

    second = build_index(
        source_dir=source_dir,
        index_path=index_path,
        settings=settings,
        embedder=embedder,
    )
    assert second.skipped_files == 2
    assert second.processed_files == 0
    assert second.artifacts_written == 0


def test_build_index_restores_missing_artifacts_for_skipped_documents(tmp_path: Path):
    source_dir = tmp_path / "literature"
    source_dir.mkdir()
    index_path = tmp_path / "index.sqlite3"

    _write_sample_md(
        source_dir,
        "paper1.md",
        "# Paper One\n\n## Abstract\n\nThis is about calcium phosphate mineralization under alkaline conditions.\n",
    )
    _write_sample_md(
        source_dir,
        "paper2.md",
        "# Paper Two\n\n## Abstract\n\nThis discusses water dynamics in nanoconfinement.\n",
    )

    settings = get_settings()
    settings.dashscope_api_key = None
    settings.cleaned_artifacts_dir = tmp_path / "chunk_artifacts"
    embedder = FakeEmbedder()

    build_index(
        source_dir=source_dir,
        index_path=index_path,
        settings=settings,
        embedder=embedder,
    )

    doc_id, _, _, _ = chunk_markdown(source_dir / "paper1.md")
    json_artifact = settings.cleaned_artifacts_dir / f"{doc_id}.chunks.json"
    markdown_artifact = settings.cleaned_artifacts_dir / f"{doc_id}.chunks.md"
    json_artifact.unlink()
    markdown_artifact.unlink()

    result = build_index(
        source_dir=source_dir,
        index_path=index_path,
        settings=settings,
        embedder=embedder,
    )

    assert result.skipped_files == 2
    assert result.processed_files == 0
    assert result.artifacts_written == 2
    assert json_artifact.exists()
    assert markdown_artifact.exists()
    payload = json.loads(json_artifact.read_text(encoding="utf-8"))
    assert payload["title"] == "Paper One"
    assert payload["chunk_count"] >= 1


def test_add_documents(tmp_path: Path):
    source_dir = tmp_path / "literature"
    source_dir.mkdir()
    index_path = tmp_path / "index.sqlite3"

    _write_sample_md(
        source_dir,
        "paper1.md",
        "# Paper One\n\n## Abstract\n\nFirst paper.\n",
    )

    settings = get_settings()
    settings.dashscope_api_key = None
    settings.cleaned_artifacts_dir = tmp_path / "chunk_artifacts"
    embedder = FakeEmbedder()

    build_index(
        source_dir=source_dir,
        index_path=index_path,
        settings=settings,
        embedder=embedder,
    )

    new_dir = tmp_path / "new_papers"
    new_dir.mkdir()
    (new_dir / "nested").mkdir()
    (new_dir / "nested" / "new_paper.md").write_text(
        "# New Paper\n\n## Abstract\n\nA new paper.\n",
        encoding="utf-8",
    )
    (new_dir / "duplicate.md").write_text(
        "# Paper One\n\n## Abstract\n\nDuplicate title should be skipped.\n",
        encoding="utf-8",
    )

    result = add_documents(
        paths=[new_dir],
        index_path=index_path,
        settings=settings,
        embedder=embedder,
    )
    assert result.processed_files == 1
    assert result.artifacts_written == 2
    assert result.duplicates_skipped == 1


def test_add_documents_skips_invalid_markdown_and_records_failed_files(tmp_path: Path):
    source_dir = tmp_path / "literature"
    source_dir.mkdir()
    index_path = tmp_path / "index.sqlite3"

    _write_sample_md(
        source_dir,
        "paper1.md",
        "# Paper One\n\n## Abstract\n\nFirst paper.\n",
    )

    settings = get_settings()
    settings.dashscope_api_key = None
    settings.cleaned_artifacts_dir = tmp_path / "chunk_artifacts"
    embedder = FakeEmbedder()

    build_index(
        source_dir=source_dir,
        index_path=index_path,
        settings=settings,
        embedder=embedder,
    )

    new_dir = tmp_path / "new_papers"
    new_dir.mkdir()
    _write_sample_md(
        new_dir,
        "valid.md",
        "# Valid Added Paper\n\n## Abstract\n\nThis valid add paper should be indexed.\n",
    )
    _write_sample_md(
        new_dir,
        "invalid.md",
        "# Collagen mineralization in hydrated fibrils under nanoconfinement\n\n"
        "# Calcium phosphate nucleation in hydrated collagen fibrils under confinement\n\n"
        "## Abstract\n\nThis add file is ambiguous and should be skipped.\n",
    )

    result = add_documents(
        paths=[new_dir],
        index_path=index_path,
        settings=settings,
        embedder=embedder,
    )

    assert result.processed_files == 1
    assert len(result.failed_files) == 1
    assert "invalid.md" in result.failed_files[0]


def test_build_index_skips_invalid_markdown_and_records_failed_files(tmp_path: Path):
    source_dir = tmp_path / "literature"
    source_dir.mkdir()
    index_path = tmp_path / "index.sqlite3"

    _write_sample_md(
        source_dir,
        "valid.md",
        "# Valid Paper\n\n## Abstract\n\nThis valid paper should still be indexed.\n",
    )
    _write_sample_md(
        source_dir,
        "invalid.md",
        "# Collagen mineralization in hydrated fibrils under nanoconfinement\n\n"
        "# Calcium phosphate nucleation in hydrated collagen fibrils under confinement\n\n"
        "## Abstract\n\nThis file is ambiguous and should be skipped instead of aborting the build.\n",
    )

    settings = get_settings()
    settings.dashscope_api_key = None
    settings.cleaned_artifacts_dir = tmp_path / "chunk_artifacts"
    embedder = FakeEmbedder()

    result = build_index(
        source_dir=source_dir,
        index_path=index_path,
        settings=settings,
        embedder=embedder,
    )

    assert result.processed_files == 1
    assert len(result.failed_files) == 1
    assert "invalid.md" in result.failed_files[0]
    assert "multiple plausible level-1 title headings" in result.failed_files[0]

    store = SQLiteIndexStore(index_path)
    with store.connect() as connection:
        titles = [row["title"] for row in connection.execute("SELECT title FROM documents ORDER BY title").fetchall()]

    assert titles == ["Valid Paper"]


def test_build_index_records_embedding_failure_without_artifacts(tmp_path: Path):
    source_dir = tmp_path / "literature"
    source_dir.mkdir()
    index_path = tmp_path / "index.sqlite3"
    _write_sample_md(
        source_dir,
        "paper1.md",
        "# Paper One\n\n## Abstract\n\nThis paper should fail during embedding generation.\n",
    )

    settings = get_settings()
    settings.dashscope_api_key = None
    settings.cleaned_artifacts_dir = tmp_path / "chunk_artifacts"

    result = build_index(
        source_dir=source_dir,
        index_path=index_path,
        settings=settings,
        embedder=FailingEmbedder(),
    )

    assert result.processed_files == 0
    assert len(result.failed_files) == 1
    assert "embedding failed" in result.failed_files[0]
    assert list(settings.cleaned_artifacts_dir.iterdir()) == []
    store = SQLiteIndexStore(index_path)
    with store.connect() as connection:
        document_count = connection.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        chunk_count = connection.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    assert document_count == 0
    assert chunk_count == 0


def test_build_index_rebuild_clears_removed_documents(tmp_path: Path):
    source_dir = tmp_path / "literature"
    source_dir.mkdir()
    index_path = tmp_path / "index.sqlite3"

    _write_sample_md(
        source_dir,
        "paper1.md",
        "# Paper One\n\n## Abstract\n\nFirst paper.\n",
    )
    _write_sample_md(
        source_dir,
        "paper2.md",
        "# Paper Two\n\n## Abstract\n\nSecond paper.\n",
    )

    settings = get_settings()
    settings.dashscope_api_key = None
    settings.cleaned_artifacts_dir = tmp_path / "chunk_artifacts"
    embedder = FakeEmbedder()

    build_index(
        source_dir=source_dir,
        index_path=index_path,
        settings=settings,
        embedder=embedder,
    )

    (source_dir / "paper2.md").unlink()
    rebuilt = build_index(
        source_dir=source_dir,
        index_path=index_path,
        settings=settings,
        embedder=embedder,
        rebuild=True,
    )

    store = SQLiteIndexStore(index_path)
    with store.connect() as connection:
        document_count = connection.execute("SELECT COUNT(*) FROM documents").fetchone()[0]

    assert rebuilt.processed_files == 1
    assert document_count == 1
    paper1_doc_id, _, _, _ = chunk_markdown(source_dir / "paper1.md")
    artifact_files = sorted(path.name for path in settings.cleaned_artifacts_dir.iterdir())
    assert len(artifact_files) == 2
    assert artifact_files == [f"{paper1_doc_id}.chunks.json", f"{paper1_doc_id}.chunks.md"]


def test_delete_document_by_markdown_filename_removes_index_and_artifacts(tmp_path: Path):
    source_dir = tmp_path / "literature"
    source_dir.mkdir()
    index_path = tmp_path / "index.sqlite3"

    _write_sample_md(
        source_dir,
        "paper1.md",
        "# Paper One\n\n## Abstract\n\nFirst paper about mineralization.\n",
    )
    _write_sample_md(
        source_dir,
        "paper2.md",
        "# Paper Two\n\n## Abstract\n\nSecond paper about hydration.\n",
    )

    settings = get_settings()
    settings.dashscope_api_key = None
    settings.cleaned_artifacts_dir = tmp_path / "chunk_artifacts"
    embedder = FakeEmbedder()

    build_index(
        source_dir=source_dir,
        index_path=index_path,
        settings=settings,
        embedder=embedder,
    )
    paper1_doc_id, _, _, _ = chunk_markdown(source_dir / "paper1.md")
    assert (settings.cleaned_artifacts_dir / f"{paper1_doc_id}.chunks.json").exists()
    assert (settings.cleaned_artifacts_dir / f"{paper1_doc_id}.chunks.md").exists()

    store = SQLiteIndexStore(index_path)
    rows_before, _ = store.fetch_all_vectors()
    assert any(row["doc_id"] == paper1_doc_id for row in rows_before)

    stats = delete_document(title="paper1.md", index_path=index_path, settings=settings)

    assert stats.deleted is True
    assert stats.title == "Paper One"
    assert stats.matched_by == "filename"
    assert stats.source_deleted is True
    assert stats.chunks_deleted >= 1
    assert stats.artifacts_deleted == 2
    assert not (source_dir / "paper1.md").exists()
    assert (source_dir / "paper2.md").exists()
    assert not (settings.cleaned_artifacts_dir / f"{paper1_doc_id}.chunks.json").exists()
    assert not (settings.cleaned_artifacts_dir / f"{paper1_doc_id}.chunks.md").exists()

    rows_after, _ = store.fetch_all_vectors()
    assert all(row["doc_id"] != paper1_doc_id for row in rows_after)
    with store.connect() as connection:
        titles = [row["title"] for row in connection.execute("SELECT title FROM documents ORDER BY title").fetchall()]
        chunk_count = connection.execute("SELECT COUNT(*) FROM chunks WHERE doc_id = ?", (paper1_doc_id,)).fetchone()[0]
        fts_count = connection.execute(
            "SELECT COUNT(*) FROM chunk_fts WHERE chunk_id LIKE ?",
            (f"{paper1_doc_id}:%",),
        ).fetchone()[0]

    assert titles == ["Paper Two"]
    assert chunk_count == 0
    assert fts_count == 0


def test_build_index_uses_selected_paper_title_for_documents_and_citations(tmp_path: Path):
    source_dir = tmp_path / "literature"
    source_dir.mkdir()
    index_path = tmp_path / "index.sqlite3"
    source_path = source_dir / "paper.md"
    source_path.write_text(
        "# Journal of Materials Chemistry B\n\n"
        "# Bacterial S-layer protein inspired multifunctional peptide\n\n"
        "## Abstract\n\n"
        "This abstract paragraph should be indexed under the real paper title rather than the journal masthead.\n",
        encoding="utf-8",
    )

    settings = get_settings()
    settings.dashscope_api_key = None
    settings.cleaned_artifacts_dir = tmp_path / "chunk_artifacts"
    embedder = FakeEmbedder()

    build_index(source_dir=source_dir, index_path=index_path, settings=settings, embedder=embedder)

    store = SQLiteIndexStore(index_path)
    with store.connect() as connection:
        row = connection.execute(
            "SELECT title, normalized_title FROM documents WHERE source_path = ?",
            (str(source_path),),
        ).fetchone()
        chunk_row = connection.execute("SELECT title, page_start, page_end, chunk_id FROM chunks").fetchone()

    assert row is not None
    assert row["title"] == "Bacterial S-layer protein inspired multifunctional peptide"
    assert row["normalized_title"] == "bacterial s layer protein inspired multifunctional peptide"
    assert chunk_row is not None
    assert chunk_row["title"] == "Bacterial S-layer protein inspired multifunctional peptide"
    assert build_citation(
        title=chunk_row["title"],
        page_start=int(chunk_row["page_start"]),
        page_end=int(chunk_row["page_end"]),
        chunk_id=chunk_row["chunk_id"],
    ).startswith("Bacterial S-layer protein inspired multifunctional peptide (p. 0, chunk ")


def test_build_index_uses_repaired_accepted_front_matter_title_for_documents_and_citations(tmp_path: Path):
    source_dir = tmp_path / "literature"
    source_dir.mkdir()
    index_path = tmp_path / "index.sqlite3"
    source_path = source_dir / "paper.md"
    paper_title = (
        "Crystallization of citrate-stabilized amorphous calcium phosphate to nanocrystalline apatite: "
        "a surface-mediated transformation"
    )
    source_path.write_text(
        "Deposited via The University of York.\n\n"
        "White Rose Research Online URL for this paper: https://eprints.whiterose.ac.uk/id/eprint/98103/\n\n"
        "Version: Accepted Version\n\n"
        "# Article:\n\n"
        "Chatzipanagis, Konstantinos et al. (2016) Crystallization of citrate-stabilized amorphous calcium "
        "phosphate to nanocrystalline apatite: a surface-mediated transformation.\n\n"
        "https://doi.org/10.1039/C6CE00521G\n\n"
        "# Reuse\n\n"
        "Items deposited in White Rose Research Online are protected by copyright.\n\n"
        "# Takedown\n\n"
        "If you consider content in White Rose Research Online to be in breach of UK law, please notify us.\n\n"
        "# CrystEngComm\n\n"
        "Accepted Manuscript\n\n"
        "This article can be cited before page numbers have been issued.\n\n"
        "# Crystallization of citrate-stabilized amorphous calcium phosphate to\n\n"
        "# nanocrystalline apatite: a surface-mediated transformation\n\n"
        "Received 00th January 20xx, Accepted 00th January 20xx\n\n"
        "# ABSTRACT\n\n"
        "This work explores the mechanisms underlying the crystallization of citrate-functionalized amorphous "
        "calcium phosphate in relevant aqueous media.\n\n"
        "# INTRODUCTION\n\n"
        "Many aspects of the mechanisms underlying the formation of nanocrystalline apatite remain under debate.\n",
        encoding="utf-8",
    )

    settings = get_settings()
    settings.dashscope_api_key = None
    settings.cleaned_artifacts_dir = tmp_path / "chunk_artifacts"
    embedder = FakeEmbedder()

    build_index(source_dir=source_dir, index_path=index_path, settings=settings, embedder=embedder)

    store = SQLiteIndexStore(index_path)
    with store.connect() as connection:
        row = connection.execute(
            "SELECT title, normalized_title FROM documents WHERE source_path = ?",
            (str(source_path),),
        ).fetchone()
        chunk_row = connection.execute("SELECT title, text, page_start, page_end, chunk_id FROM chunks").fetchone()

    assert row is not None
    assert row["title"] == paper_title
    assert row["normalized_title"] == (
        "crystallization of citrate stabilized amorphous calcium phosphate to nanocrystalline apatite "
        "a surface mediated transformation"
    )
    assert chunk_row is not None
    assert chunk_row["title"] == paper_title
    assert "White Rose Research Online" not in chunk_row["text"]
    assert "Accepted Manuscript" not in chunk_row["text"]
    assert build_citation(
        title=chunk_row["title"],
        page_start=int(chunk_row["page_start"]),
        page_end=int(chunk_row["page_end"]),
        chunk_id=chunk_row["chunk_id"],
    ).startswith(f"{paper_title} (p. 0, chunk ")
