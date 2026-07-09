"""HTML extraction."""

from __future__ import annotations

from pathlib import Path

from extractor.extraction.base import ExtractionError


def extract(path: Path) -> str:
    from bs4 import BeautifulSoup

    try:
        raw = path.read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        raise ExtractionError(f"Could not read .html file: {e}") from e

    soup = BeautifulSoup(raw, "lxml")
    for tag in soup(["script", "style"]):
        tag.decompose()

    text = soup.get_text(separator="\n")
    text = "\n".join(line.strip() for line in text.splitlines() if line.strip())
    if not text.strip():
        raise ExtractionError("No extractable text found in .html")
    return text
