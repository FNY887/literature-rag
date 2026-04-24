from __future__ import annotations

import asyncio
import itertools
import json
import sys
import threading
import time
from pathlib import Path

import typer

from agentic_rag.builder import add_documents, build_index, delete_document
from agentic_rag.core.config import get_settings
from agentic_rag.core.models import AgentAnswer
from agentic_rag.query import answer_stream, warm_index
from agentic_rag.query.agent import AgentStageError

CLI_HELP_TEXT = (
    "literature-rag: 进入提问模式，检索并回答文献问题。\n"
    "literature-rag build <dir>: 清空旧库，并用目录中的 Markdown 文献重建向量库。\n"
    "literature-rag add <dir>: 将目录中的 Markdown 文献递归加入现有向量库。\n"
    "literature-rag delete --title \"论文标题或md文件名\": 从向量库删除一篇文献。"
)

app = typer.Typer(
    add_completion=False,
    add_help_option=False,
    rich_markup_mode=None,
    help=CLI_HELP_TEXT,
)


def _print_agent_error(exc: AgentStageError) -> None:
    typer.echo(
        json.dumps(
            {
                "error": exc.message,
                "diagnostics": exc.diagnostic_payload(),
            },
            indent=2,
            ensure_ascii=False,
        ),
        err=True,
    )


def _spinning_indicator(stop_event: threading.Event, message: str = "正在检索") -> None:
    spinner = itertools.cycle(["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"])
    while not stop_event.is_set():
        sys.stdout.write(f"\r{next(spinner)} {message}...")
        sys.stdout.flush()
        time.sleep(0.1)
    sys.stdout.write("\r" + " " * (len(message) + 6) + "\r")
    sys.stdout.flush()


async def _run_answer_stream(question: str, index_path: str):
    result = None
    async for event in answer_stream(question=question, index_path=index_path):
        if event.event == "scan_completed":
            result = event.answer
            break
    return result


def _print_timing_summary(result: AgentAnswer) -> None:
    typer.echo("耗时摘要：")
    typer.echo(f"- analyze: {result.stage_timings.analyze_seconds:.2f}s")
    typer.echo(f"- retrieve: {result.stage_timings.retrieve_seconds:.2f}s")
    typer.echo(f"- rerank: {result.stage_timings.rerank_seconds:.2f}s")
    typer.echo(f"- judge: {result.stage_timings.judge_seconds:.2f}s")
    typer.echo(f"- total: {result.stage_timings.total_seconds:.2f}s")
    typer.echo()
    typer.echo(
        "Judge统计："
        f"候选 {result.performance_counters.judge_documents_total} 篇；"
        f"并发 {result.performance_counters.judge_concurrency_initial}"
        f" -> {result.performance_counters.judge_concurrency_final}；"
        f"provider pressure {result.performance_counters.provider_pressure_events} 次"
    )
    typer.echo()


def _print_simple_help() -> None:
    typer.echo(CLI_HELP_TEXT)


def _print_failed_files(stats) -> None:
    if not stats.failed_files:
        return
    typer.echo()
    typer.echo("未成功建库的 Markdown 文件：")
    for failed in stats.failed_files:
        typer.echo(f"- {failed}")


def _print_warmup_summary(stats) -> None:
    typer.echo(f"向量库已加载：{stats.chunks_loaded} chunks，dim={stats.vector_dimensions}。")


def _handle_root_help(ctx: typer.Context, param: typer.CallbackParam, value: bool) -> None:
    if not value or ctx.resilient_parsing:
        return
    _print_simple_help()
    raise typer.Exit()


def _run_with_spinner(message: str, fn):
    if not sys.stdout.isatty():
        return fn()

    stop_event = threading.Event()
    spinner_thread = threading.Thread(
        target=_spinning_indicator,
        args=(stop_event, message),
        daemon=True,
    )
    spinner_thread.start()
    try:
        return fn()
    finally:
        stop_event.set()
        spinner_thread.join()


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    index_path: str = typer.Option(".rag_store/literature_rag.sqlite3", help="SQLite index path."),
    help_: bool = typer.Option(
        False,
        "--help",
        callback=_handle_root_help,
        is_eager=True,
        expose_value=False,
    ),
) -> None:
    if ctx.invoked_subcommand is not None:
        return

    try:
        warmup_stats = _run_with_spinner(
            "正在加载向量库",
            lambda: warm_index(index_path=index_path),
        )
    except RuntimeError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc
    _print_warmup_summary(warmup_stats)

    while True:
        try:
            question = typer.prompt(
                "请输入您的问题（直接回车退出）",
                default="",
                show_default=False,
            )
        except (typer.Abort, EOFError, KeyboardInterrupt):
            break

        question = question.strip()
        if not question:
            break

        try:
            result = _run_with_spinner(
                "正在检索",
                lambda: asyncio.run(_run_answer_stream(question=question, index_path=index_path)),
            )
        except AgentStageError as exc:
            _print_agent_error(exc)
            continue

        if result is None:
            typer.echo("未能生成回答。", err=True)
            continue

        typer.echo(result.answer)
        typer.echo()

        if result.citations:
            typer.echo("参考文献：")
            for citation in result.citations:
                typer.echo(citation)
            typer.echo()

        if result.scan_status:
            typer.echo(result.scan_status)
            typer.echo()

        _print_timing_summary(result)


@app.command("build")
def build_index_command(
    source_dir: str = typer.Argument(
        ...,
        help="Directory containing Markdown papers. Rebuilds the default vector database from this directory.",
    ),
    index_path: str = typer.Option(".rag_store/literature_rag.sqlite3", help="Vector database path."),
    chunk_size: int | None = typer.Option(None, help="Override chunk size."),
    chunk_overlap: int | None = typer.Option(None, help="Override chunk overlap."),
) -> None:
    settings = get_settings()
    if chunk_size is not None:
        settings.chunk_size = chunk_size
    if chunk_overlap is not None:
        settings.chunk_overlap = chunk_overlap
    stats = _run_with_spinner(
        "正在重建向量库",
        lambda: build_index(
            source_dir=source_dir,
            index_path=index_path,
            settings=settings,
            rebuild=True,
        ),
    )
    typer.echo(json.dumps(stats.model_dump(), indent=2))
    _print_failed_files(stats)


@app.command("add")
def add_command(
    source_path: str = typer.Argument(
        ...,
        help="Markdown file or directory to add to the existing vector database (directories are scanned recursively).",
    ),
    index_path: str = typer.Option(".rag_store/literature_rag.sqlite3", help="Vector database path."),
    chunk_size: int | None = typer.Option(None, help="Override chunk size."),
    chunk_overlap: int | None = typer.Option(None, help="Override chunk overlap."),
) -> None:
    settings = get_settings()
    if chunk_size is not None:
        settings.chunk_size = chunk_size
    if chunk_overlap is not None:
        settings.chunk_overlap = chunk_overlap

    stats = _run_with_spinner(
        "正在添加文献",
        lambda: add_documents(
            paths=[source_path],
            index_path=index_path,
            settings=settings,
        ),
    )
    typer.echo(json.dumps(stats.model_dump(), indent=2))
    _print_failed_files(stats)


@app.command("delete")
def delete_command(
    title: str = typer.Option(
        ...,
        "--title",
        help="Paper title or original Markdown filename to delete.",
    ),
    index_path: str = typer.Option(".rag_store/literature_rag.sqlite3", help="Vector database path."),
) -> None:
    try:
        stats = _run_with_spinner(
            "正在删除文献",
            lambda: delete_document(
                title=title,
                index_path=index_path,
                settings=get_settings(),
            ),
        )
    except RuntimeError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(json.dumps(stats.model_dump(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    app()
