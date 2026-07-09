"""PDF extraction: pypdf primary, pdfplumber fallback."""

from __future__ import annotations

from pathlib import Path

from extractor.extraction.base import ExtractionError


def _extract_pypdf(path: Path) -> str:
    from pypdf import PdfReader
    from pypdf.errors import PdfReadError

    try:
        reader = PdfReader(str(path))
    except PdfReadError as e:
        raise ExtractionError(f"pypdf could not open PDF: {e}") from e

    if reader.is_encrypted:
        # Try an empty password (common for "restricted" but not truly locked PDFs).
        if reader.decrypt("") == 0:
            raise ExtractionError("PDF is password-protected")

    text = "\n\n".join(page.extract_text() or "" for page in reader.pages)
    if not text.strip():
        raise ExtractionError("pypdf extracted no text (possibly scanned/image-only PDF)")
    return text


def _extract_pdfplumber(path: Path) -> str:
    import pdfplumber

    try:
        with pdfplumber.open(str(path)) as pdf:
            text = "\n\n".join(page.extract_text() or "" for page in pdf.pages)
    except Exception as e:  # pdfplumber raises varied exceptions per failure mode
        raise ExtractionError(f"pdfplumber could not open PDF: {e}") from e

    if not text.strip():
        raise ExtractionError("pdfplumber extracted no text (possibly scanned/image-only PDF)")
    return text


def extract(path: Path) -> str:
    errors: list[str] = []
    for extractor_fn in (_extract_pypdf, _extract_pdfplumber):
        try:
            return extractor_fn(path)
        except ExtractionError as e:
            errors.append(str(e))
    raise ExtractionError("; ".join(errors))
