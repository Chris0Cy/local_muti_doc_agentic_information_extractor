"""Universal fallback extraction via markitdown.

Used for suffixes with no dedicated extractor, and as a last resort when a
format's primary extractor(s) fail.
"""

from __future__ import annotations

from pathlib import Path

from extractor.extraction.base import ExtractionError


def extract(path: Path) -> str:
    from markitdown import MarkItDown

    try:
        result = MarkItDown().convert(str(path))
    except Exception as e:
        raise ExtractionError(f"markitdown could not convert file: {e}") from e

    text = result.text_content
    if not text or not text.strip():
        raise ExtractionError("markitdown extracted no text")
    return text
