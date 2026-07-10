"""Render a RunReport to console, markdown, or JSON."""

from __future__ import annotations

from rich.console import Console
from rich.markup import escape
from rich.table import Table

from extractor.models import RunReport


def render_console(report: RunReport, console: Console) -> None:
    # Rich's Console.print() and Table.add_row() parse "[...]" as style markup
    # by default. Any LLM-generated text, external error message, or the
    # user's own question can contain something that looks like a markup tag
    # (a numbered citation "[1]", a markdown link, a stray bracket) and would
    # otherwise crash rendering *after* the (expensive) pipeline run has
    # already completed. Everything that isn't a literal tag we wrote
    # ourselves must be escaped.
    console.print(f"\n[bold]Question:[/bold] {escape(report.question)}\n")

    if report.judge_result is None:
        console.print("[red]No judge result was produced.[/red]")
    elif report.judge_result.unanswered:
        console.print(f"[yellow]{escape(report.judge_result.final_answer_markdown)}[/yellow]")
    else:
        console.print(escape(report.judge_result.final_answer_markdown))

    errored = [r for r in report.worker_results if r.error]
    if report.documents_failed or errored:
        console.print(
            f"\n[dim]Scanned {report.documents_scanned} document(s) in {report.duration_s:.1f}s "
            f"({len(report.documents_failed)} failed extraction, {len(errored)} worker call(s) errored)[/dim]"
        )
        if report.documents_failed:
            table = Table(title="Extraction failures")
            table.add_column("File")
            table.add_column("Reason")
            for path, reason in report.documents_failed:
                table.add_row(escape(str(path)), escape(reason))
            console.print(table)
        if errored:
            table = Table(title="Worker call errors")
            table.add_column("File")
            table.add_column("Chunk")
            table.add_column("Error")
            for r in errored:
                table.add_row(
                    escape(str(r.doc_path)),
                    f"{r.chunk_index + 1}/{r.total_chunks}",
                    escape(r.error or ""),
                )
            console.print(table)
    else:
        console.print(
            f"\n[dim]Scanned {report.documents_scanned} document(s) in {report.duration_s:.1f}s, no failures.[/dim]"
        )


def render_markdown(report: RunReport) -> str:
    lines = [f"# Question\n\n{report.question}\n", "# Answer\n"]
    if report.judge_result is not None:
        lines.append(report.judge_result.final_answer_markdown)
    else:
        lines.append("_No judge result was produced._")

    errored = [r for r in report.worker_results if r.error]
    if report.documents_failed or errored:
        lines.append("\n# Warnings\n")
        for path, reason in report.documents_failed:
            lines.append(f"- Extraction failed for `{path}`: {reason}")
        for r in errored:
            lines.append(
                f"- Worker error on `{r.doc_path}` (chunk {r.chunk_index + 1}/{r.total_chunks}): {r.error}"
            )

    lines.append(
        f"\n---\n_Scanned {report.documents_scanned} document(s) in {report.duration_s:.1f}s._"
    )
    return "\n".join(lines)


def render_json(report: RunReport) -> str:
    return report.model_dump_json(indent=2)
