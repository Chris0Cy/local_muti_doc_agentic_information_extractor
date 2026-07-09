from __future__ import annotations

from extractor.extraction import extract_text


def test_extracts_txt(make_txt):
    path = make_txt("Hello world.")
    doc = extract_text(path)
    assert doc.extraction_method == "text"
    assert "Hello world." in doc.text


def test_extracts_md(tmp_path):
    p = tmp_path / "sample.md"
    p.write_text("# Title\n\nSome **markdown** content.")
    doc = extract_text(p)
    assert doc.extraction_method == "text"
    assert "markdown" in doc.text
