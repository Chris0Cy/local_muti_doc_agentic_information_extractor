"""Plain text and Markdown extraction."""

from __future__ import annotations

from pathlib import Path

from extractor.extraction.base import ExtractionError


def extract(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        raise ExtractionError(f"Could not read text file: {e}") from e
