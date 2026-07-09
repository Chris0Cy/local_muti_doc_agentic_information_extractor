"""Per-format text extraction, dispatched by file suffix."""

from __future__ import annotations

from pathlib import Path

from extractor.extraction import docx, email, fallback, html, pdf, pptx, text_md, xlsx_csv
from extractor.extraction.base import ExtractionError
from extractor.models import ExtractedDocument

_DISPATCH = {
    ".txt": ("text", text_md.extract),
    ".md": ("text", text_md.extract),
    ".markdown": ("text", text_md.extract),
    ".pdf": ("pdf", pdf.extract),
    ".docx": ("docx", docx.extract),
    ".pptx": ("pptx", pptx.extract),
    ".xlsx": ("xlsx", xlsx_csv.extract_xlsx),
    ".xlsm": ("xlsx", xlsx_csv.extract_xlsx),
    ".csv": ("csv", xlsx_csv.extract_csv),
    ".html": ("html", html.extract),
    ".htm": ("html", html.extract),
    ".eml": ("eml", email.extract_eml),
    ".msg": ("msg", email.extract_msg),
}


def extract_text(path: Path) -> ExtractedDocument:
    """Extract text from a document, dispatching by suffix with a markitdown fallback."""
    suffix = path.suffix.lower()
    warnings: list[str] = []

    dispatch_entry = _DISPATCH.get(suffix)
    if dispatch_entry is not None:
        method, extractor_fn = dispatch_entry
        try:
            text = extractor_fn(path)
            return ExtractedDocument(path=path, text=text, extraction_method=method)
        except ExtractionError as e:
            warnings.append(f"{method} extractor failed ({e}); trying markitdown fallback")

    try:
        text = fallback.extract(path)
        return ExtractedDocument(
            path=path, text=text, extraction_method="markitdown", warnings=warnings
        )
    except ExtractionError as e:
        warnings.append(f"markitdown fallback failed ({e})")
        raise ExtractionError("; ".join(warnings)) from e
