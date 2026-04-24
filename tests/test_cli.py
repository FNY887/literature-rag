from pathlib import Path
from types import SimpleNamespace

import numpy as np
from typer.testing import CliRunner

from agentic_rag.cli import app
from agentic_rag.core.config import get_settings
from agentic_rag.core.models import AgentAnswer, PerformanceCounters, StageTimings

runner = CliRunner()


class FakeEmbedder:
    def embed_texts(self, texts: list[str]) -> list[np.ndarray]:
        return [np.asarray([0.5, 0.5], dtype=np.float32) for _ in texts]

    def embed_query(self, query: str) -> np.ndarray:
        return np.asarray([0.5, 0.5], dtype=np.float32)


def test_cli_help_renders():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert result.stdout.strip().splitlines() == [
        "literature-rag: 进入提问模式，检索并回答文献问题。",
        "literature-rag build <dir>: 清空旧库，并用目录中的 Markdown 文献重建向量库。",
        "literature-rag add <dir>: 将目录中的 Markdown 文献递归加入现有向量库。",
        "literature-rag delete --title \"论文标题或md文件名\": 从向量库删除一篇文献。",
    ]


def test_index_build(tmp_path: Path, monkeypatch):
    source_dir = tmp_path / "literature"
    source_dir.mkdir()
    index_path = tmp_path / "index.sqlite3"

    (source_dir / "paper1.md").write_text(
        "# Paper One\n\n## Abstract\n\nThis is the abstract.\n",
        encoding="utf-8",
    )

    settings = get_settings()
    settings.dashscope_api_key = None
    monkeypatch.setattr("agentic_rag.builder.DashScopeEmbeddingClient", lambda s: FakeEmbedder())
    spinner_messages: list[str] = []
    monkeypatch.setattr(
        "agentic_rag.cli._run_with_spinner",
        lambda message, fn: (spinner_messages.append(message), fn())[1],
    )

    result = runner.invoke(app, [
        "build",
        str(source_dir),
        "--index-path", str(index_path),
    ])
    assert result.exit_code == 0
    assert "processed_files" in result.output
    assert spinner_messages == ["正在重建向量库"]

    (source_dir / "paper2.md").write_text(
        "# Paper Two\n\n## Abstract\n\nThis is the second abstract.\n",
        encoding="utf-8",
    )
    rebuilt = runner.invoke(app, [
        "build",
        str(source_dir),
        "--index-path", str(index_path),
    ])
    assert rebuilt.exit_code == 0
    assert "\"processed_files\": 2" in rebuilt.output
    assert spinner_messages == ["正在重建向量库", "正在重建向量库"]


def test_add_command(tmp_path: Path, monkeypatch):
    source_dir = tmp_path / "literature"
    source_dir.mkdir()
    index_path = tmp_path / "index.sqlite3"

    (source_dir / "paper1.md").write_text(
        "# Paper One\n\n## Abstract\n\nFirst paper.\n",
        encoding="utf-8",
    )

    monkeypatch.setattr("agentic_rag.builder.DashScopeEmbeddingClient", lambda s: FakeEmbedder())
    spinner_messages: list[str] = []
    monkeypatch.setattr(
        "agentic_rag.cli._run_with_spinner",
        lambda message, fn: (spinner_messages.append(message), fn())[1],
    )

    runner.invoke(app, [
        "build",
        str(source_dir),
        "--index-path", str(index_path),
    ])

    new_dir = tmp_path / "new_papers"
    new_dir.mkdir()
    (new_dir / "new_paper.md").write_text(
        "# New Paper\n\n## Abstract\n\nA new paper.\n",
        encoding="utf-8",
    )

    result = runner.invoke(app, [
        "add",
        str(new_dir),
        "--index-path", str(index_path),
    ])
    assert result.exit_code == 0
    assert "processed_files" in result.output
    assert spinner_messages == ["正在重建向量库", "正在添加文献"]


def test_delete_command(tmp_path: Path, monkeypatch):
    source_dir = tmp_path / "literature"
    source_dir.mkdir()
    index_path = tmp_path / "index.sqlite3"

    (source_dir / "paper1.md").write_text(
        "# Paper One\n\n## Abstract\n\nFirst paper.\n",
        encoding="utf-8",
    )

    monkeypatch.setattr("agentic_rag.builder.DashScopeEmbeddingClient", lambda s: FakeEmbedder())
    spinner_messages: list[str] = []
    monkeypatch.setattr(
        "agentic_rag.cli._run_with_spinner",
        lambda message, fn: (spinner_messages.append(message), fn())[1],
    )

    runner.invoke(app, [
        "build",
        str(source_dir),
        "--index-path", str(index_path),
    ])

    result = runner.invoke(app, [
        "delete",
        "--title", "paper1.md",
        "--index-path", str(index_path),
    ])

    assert result.exit_code == 0
    assert "\"deleted\": true" in result.output
    assert "\"title\": \"Paper One\"" in result.output
    assert "\"source_deleted\": true" in result.output
    assert "\"matched_by\": \"filename\"" in result.output
    assert not (source_dir / "paper1.md").exists()
    assert spinner_messages == ["正在重建向量库", "正在删除文献"]


def test_build_prints_failed_files(tmp_path: Path, monkeypatch):
    source_dir = tmp_path / "literature"
    source_dir.mkdir()
    index_path = tmp_path / "index.sqlite3"

    (source_dir / "valid.md").write_text(
        "# Valid Paper\n\n## Abstract\n\nThis is the abstract.\n",
        encoding="utf-8",
    )
    (source_dir / "invalid.md").write_text(
        "# Collagen mineralization in hydrated fibrils under nanoconfinement\n\n"
        "# Calcium phosphate nucleation in hydrated collagen fibrils under confinement\n\n"
        "## Abstract\n\nThis file should be skipped.\n",
        encoding="utf-8",
    )

    settings = get_settings()
    settings.dashscope_api_key = None
    monkeypatch.setattr("agentic_rag.builder.DashScopeEmbeddingClient", lambda s: FakeEmbedder())
    monkeypatch.setattr("agentic_rag.cli._run_with_spinner", lambda message, fn: fn())

    result = runner.invoke(app, [
        "build",
        str(source_dir),
        "--index-path", str(index_path),
    ])

    assert result.exit_code == 0
    assert "\"processed_files\": 1" in result.output
    assert "\"failed_files\":" in result.output
    assert "未成功建库的 Markdown 文件：" in result.output
    assert "invalid.md" in result.output


def test_add_prints_failed_files(tmp_path: Path, monkeypatch):
    source_dir = tmp_path / "literature"
    source_dir.mkdir()
    index_path = tmp_path / "index.sqlite3"

    (source_dir / "paper1.md").write_text(
        "# Paper One\n\n## Abstract\n\nFirst paper.\n",
        encoding="utf-8",
    )

    new_dir = tmp_path / "new_papers"
    new_dir.mkdir()
    (new_dir / "valid.md").write_text(
        "# Valid Added Paper\n\n## Abstract\n\nA valid added paper.\n",
        encoding="utf-8",
    )
    (new_dir / "invalid.md").write_text(
        "# Collagen mineralization in hydrated fibrils under nanoconfinement\n\n"
        "# Calcium phosphate nucleation in hydrated collagen fibrils under confinement\n\n"
        "## Abstract\n\nThis file should be skipped.\n",
        encoding="utf-8",
    )

    settings = get_settings()
    settings.dashscope_api_key = None
    monkeypatch.setattr("agentic_rag.builder.DashScopeEmbeddingClient", lambda s: FakeEmbedder())
    monkeypatch.setattr("agentic_rag.cli._run_with_spinner", lambda message, fn: fn())

    runner.invoke(app, [
        "build",
        str(source_dir),
        "--index-path", str(index_path),
    ])
    result = runner.invoke(app, [
        "add",
        str(new_dir),
        "--index-path", str(index_path),
    ])

    assert result.exit_code == 0
    assert "\"processed_files\": 1" in result.output
    assert "\"failed_files\":" in result.output
    assert "未成功建库的 Markdown 文件：" in result.output
    assert "invalid.md" in result.output


def test_interactive_query_prints_timing_summary(monkeypatch):
    warmup_calls: list[str] = []

    def fake_warm_index(*, index_path: str):
        warmup_calls.append(index_path)
        return SimpleNamespace(chunks_loaded=2, vector_dimensions=2)

    async def fake_run_answer_stream(question: str, index_path: str):
        assert question == "What evidence?"
        assert index_path == "dummy.sqlite3"
        return AgentAnswer(
            answer="1. Paper One directly supports the claim.",
            citations=["[1] Paper One (p. 1, chunk demo:0001)"],
            used_queries=["test"],
            rounds=1,
            confidence="high",
            scan_status="已完成排序文献扫描，共判定 1/1 篇文献。",
            stage_timings=StageTimings(
                analyze_seconds=0.1,
                retrieve_seconds=0.2,
                rerank_seconds=0.3,
                judge_seconds=0.4,
                total_seconds=1.0,
            ),
            performance_counters=PerformanceCounters(
                judge_documents_total=1,
                judge_concurrency_initial=5,
                judge_concurrency_final=1,
                provider_pressure_events=1,
            ),
        )

    monkeypatch.setattr("agentic_rag.cli.warm_index", fake_warm_index)
    monkeypatch.setattr("agentic_rag.cli._run_answer_stream", fake_run_answer_stream)
    monkeypatch.setattr("agentic_rag.cli._spinning_indicator", lambda stop_event, message='正在检索': None)

    result = runner.invoke(
        app,
        ["--index-path", "dummy.sqlite3"],
        input="What evidence?\n\n",
    )

    assert result.exit_code == 0
    assert warmup_calls == ["dummy.sqlite3"]
    assert "向量库已加载：2 chunks，dim=2。" in result.stdout
    assert "耗时摘要：" in result.stdout
    assert "- analyze: 0.10s" in result.stdout
    assert "- synthesize:" not in result.stdout
    assert "Judge统计：候选 1 篇；并发 5 -> 1；provider pressure 1 次" in result.stdout
