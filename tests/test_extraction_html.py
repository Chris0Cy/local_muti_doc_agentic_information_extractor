from __future__ import annotations

from extractor.extraction import extract_text


def test_extracts_html_and_strips_scripts_styles(make_html):
    path = make_html("<h1>Title</h1><p>Hello from HTML.</p><script>alert('x')</script>")
    doc = extract_text(path)
    assert doc.extraction_method == "html"
    assert "Hello from HTML." in doc.text
    assert "alert" not in doc.text
