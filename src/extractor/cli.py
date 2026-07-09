"""CLI entry points for the extractor."""

from __future__ import annotations

import asyncio
import datetime as dt
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from extractor import __version__, pipeline, report
from extractor.config import load_config
from extractor.discovery import scan_folder
from extractor.extraction import extract_text
from extractor.extraction.base import ExtractionError
from extractor.lmstudio_client import LMStudioClient, LMStudioUnavailable
from extractor.models import Chunk
from extractor.sizing import assign_tier, estimate_tokens
from extractor.worker import ask_worker

app = typer.Typer(help="Local agentic multi-document information extractor.")
console = Console()


@app.command()
def version() -> None:
    """Print the extractor version."""
    typer.echo(__version__)


@app.command()
def inspect(
    folder: Path = typer.Option(
        ..., "--folder", exists=True, file_okay=False, help="Folder to scan."
    ),
    config_path: Path = typer.Option(None, "--config", help="Path to a config YAML file."),
) -> None:
    """Scan a folder and show each document's extracted size and assigned tier,
    with no LLM calls (useful for sanity-checking sizing/tiers before running `ask`)."""
    config = load_config(config_path)
    tiers = config.tiers_ascending
    paths = scan_folder(folder)

    table = Table(title=f"{len(paths)} document(s) found in {folder}")
    table.add_column("File")
    table.add_column("Method")
    table.add_column("Tokens", justify="right")
    table.add_column("Tier / chunks")

    failed: list[tuple[Path, str]] = []
    for path in paths:
        try:
            doc = extract_text(path)
        except ExtractionError as e:
            failed.append((path, str(e)))
            table.add_row(str(path.relative_to(folder)), "[red]FAILED[/red]", "-", str(e))
            continue

        tokens = estimate_tokens(doc.text)
        tier = assign_tier(doc.text, tiers, config.safety_margin)
        tier_label = tier.name if tier else "[yellow]oversized -> will be chunked[/yellow]"
        table.add_row(str(path.relative_to(folder)), doc.extraction_method, str(tokens), tier_label)

    console.print(table)
    if failed:
        console.print(f"\n[red]{len(failed)} document(s) failed extraction.[/red]")


@app.command(name="list-models")
def list_models(
    config_path: Path = typer.Option(None, "--config", help="Path to a config YAML file."),
) -> None:
    """List the models currently loaded in LM Studio."""
    config = load_config(config_path)

    async def _run() -> list[str]:
        client = LMStudioClient(config.lmstudio.base_url, config.lmstudio.request_timeout_s)
        try:
            return await client.list_models()
        finally:
            await client.aclose()

    try:
        models = asyncio.run(_run())
    except LMStudioUnavailable as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(code=1) from e

    for model_id in models:
        console.print(model_id)


@app.command(name="ask-one")
def ask_one(
    file: Path = typer.Option(
        ..., "--file", exists=True, dir_okay=False, help="Document to ask about."
    ),
    question: str = typer.Option(..., "--question", help="Question to ask about the document."),
    config_path: Path = typer.Option(None, "--config", help="Path to a config YAML file."),
) -> None:
    """Manual smoke test: ask a single question against a single document,
    using the tier its size assigns it to (oversized docs use the largest tier
    and only the first chunk is sent, for a quick check against a real LM Studio)."""
    config = load_config(config_path)
    tiers = config.tiers_ascending

    try:
        doc = extract_text(file)
    except ExtractionError as e:
        console.print(f"[red]Extraction failed: {e}[/red]")
        raise typer.Exit(code=1) from e

    tier = assign_tier(doc.text, tiers, config.safety_margin) or tiers[-1]
    console.print(f"[dim]Using tier '{tier.name}' ({tier.model_id})[/dim]")
    chunk = Chunk(doc_path=doc.path, chunk_index=0, total_chunks=1, text=doc.text, tier=tier)

    async def _run():
        client = LMStudioClient(config.lmstudio.base_url, config.lmstudio.request_timeout_s)
        try:
            await client.health_check()
            return await ask_worker(client, question, chunk)
        finally:
            await client.aclose()

    try:
        result = asyncio.run(_run())
    except LMStudioUnavailable as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(code=1) from e

    if result.error:
        console.print(f"[red]Worker error: {result.error}[/red]")
        raise typer.Exit(code=1)

    if result.found_relevant_info:
        console.print(
            f"[green]Relevant (confidence: {result.confidence}):[/green]\n{result.answer_excerpt}"
        )
    else:
        console.print("[yellow]No relevant information found in this document.[/yellow]")


@app.command()
def ask(
    question: str = typer.Argument(..., help="Question to ask across the documents."),
    folder: Path = typer.Option(
        ..., "--folder", exists=True, file_okay=False, help="Folder to search."
    ),
    config_path: Path = typer.Option(None, "--config", help="Path to a config YAML file."),
    save: bool = typer.Option(False, "--save", help="Save the report to config's save_path."),
) -> None:
    """Ask a question across every document in a folder and print one synthesized,
    cited answer, fanning the question out to per-document workers in parallel."""
    config = load_config(config_path)

    async def _run():
        client = LMStudioClient(config.lmstudio.base_url, config.lmstudio.request_timeout_s)
        try:
            return await pipeline.run(question, folder, config, client)
        finally:
            await client.aclose()

    try:
        run_report = asyncio.run(_run())
    except LMStudioUnavailable as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(code=1) from e

    if config.output_format == "json":
        console.print(report.render_json(run_report))
    elif config.output_format == "markdown":
        console.print(report.render_markdown(run_report))
    else:
        report.render_console(run_report, console)

    if save:
        if not config.save_path:
            console.print(
                "\n[yellow]--save was passed but no save_path is configured; skipping.[/yellow]"
            )
        else:
            save_path = Path(
                config.save_path.format(timestamp=dt.datetime.now().strftime("%Y%m%d-%H%M%S"))
            )
            save_path.parent.mkdir(parents=True, exist_ok=True)
            content = (
                report.render_json(run_report)
                if config.output_format == "json"
                else report.render_markdown(run_report)
            )
            save_path.write_text(content, encoding="utf-8")
            console.print(f"\n[dim]Report saved to {save_path}[/dim]")


if __name__ == "__main__":
    app()
