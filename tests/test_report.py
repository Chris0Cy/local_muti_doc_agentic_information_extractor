from __future__ import annotations

import io
from pathlib import Path

from rich.console import Console

from extractor.models import JudgeResult, RunReport, WorkerResult
from extractor.report import render_console, render_json, render_markdown

# Rich's Console.print()/Table.add_row() parse "[...]" as style markup by
# default; unmatched brackets in LLM-generated or externally-sourced text
# used to crash rendering entirely (see git history). This exact string
# reproduces that crash if escaping regresses.
BRACKET_BOMB = "See citation [1] and markdown [link](url), closing tag [/bold] here."


def make_report(question: str = "What happened?", answer: str = "It happened.") -> RunReport:
    worker_result = WorkerResult(
        doc_path=Path("doc.txt"),
        chunk_index=0,
        total_chunks=1,
        tier_used="tiny",
        found_relevant_info=True,
        answer_excerpt=answer,
        confidence="high",
    )
    judge_result = JudgeResult(final_answer_markdown=answer, unanswered=False, raw_response=answer)
    return RunReport(
        question=question,
        documents_scanned=1,
        documents_failed=[],
        worker_results=[worker_result],
        judge_result=judge_result,
        duration_s=1.0,
    )


def test_render_console_does_not_crash_on_bracket_content_in_answer():
    report = make_report(answer=BRACKET_BOMB)
    console = Console(file=io.StringIO())
    render_console(report, console)  # must not raise rich.errors.MarkupError


def test_render_console_does_not_crash_on_bracket_content_in_question():
    report = make_report(question=BRACKET_BOMB)
    console = Console(file=io.StringIO())
    render_console(report, console)


def test_render_console_does_not_crash_on_bracket_content_in_failures_and_errors():
    report = make_report()
    report.documents_failed = [(Path("bad.pdf"), BRACKET_BOMB)]
    report.worker_results[0].error = BRACKET_BOMB
    console = Console(file=io.StringIO())
    render_console(report, console)


def test_render_console_does_not_crash_on_unanswered_with_brackets():
    report = make_report()
    report.judge_result = JudgeResult(
        final_answer_markdown=BRACKET_BOMB, unanswered=True, raw_response=""
    )
    console = Console(file=io.StringIO())
    render_console(report, console)


def test_render_markdown_includes_answer_and_warnings():
    report = make_report(answer="The answer is 42.")
    report.documents_failed = [(Path("bad.pdf"), "corrupt file")]
    markdown = render_markdown(report)
    assert "The answer is 42." in markdown
    assert "corrupt file" in markdown
    assert "bad.pdf" in markdown


def test_render_json_round_trips_via_pydantic():
    report = make_report()
    json_str = render_json(report)
    assert '"question"' in json_str
    assert "It happened." in json_str
