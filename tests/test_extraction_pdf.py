from __future__ import annotations

import pytest

from extractor.extraction import extract_text
from extractor.extraction.base import ExtractionError


def test_extracts_pdf(make_pdf):
    path = make_pdf("Hello from a PDF document.")
    doc = extract_text(path)
    assert doc.extraction_method in ("pdf", "markitdown")
    assert "Hello from a PDF document" in doc.text


def test_encrypted_pdf_raises_extraction_error(make_pdf, tmp_path):
    from pypdf import PdfReader, PdfWriter

    src = make_pdf("Secret content.")
    reader = PdfReader(str(src))
    writer = PdfWriter()
    for page in reader.pages:
        writer.add_page(page)
    writer.encrypt(user_password="secret123")
    encrypted_path = tmp_path / "encrypted.pdf"
    with encrypted_path.open("wb") as f:
        writer.write(f)

    with pytest.raises(ExtractionError):
        extract_text(encrypted_path)


def test_corrupt_pdf_raises_extraction_error(tmp_path):
    corrupt_path = tmp_path / "corrupt.pdf"
    # Random binary garbage (not decodable as text) so even the markitdown
    # fallback can't salvage anything from it.
    corrupt_path.write_bytes(bytes(range(256)) * 4)

    with pytest.raises(ExtractionError):
        extract_text(corrupt_path)
