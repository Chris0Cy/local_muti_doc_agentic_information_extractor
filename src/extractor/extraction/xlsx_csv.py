"""Excel (.xlsx) and CSV extraction."""

from __future__ import annotations

import csv
from pathlib import Path

from extractor.extraction.base import ExtractionError


def extract_xlsx(path: Path) -> str:
    import openpyxl

    try:
        workbook = openpyxl.load_workbook(str(path), read_only=True, data_only=True)
    except Exception as e:
        raise ExtractionError(f"openpyxl could not open file: {e}") from e

    sheets_text: list[str] = []
    for sheet in workbook.worksheets:
        rows_text = []
        for row in sheet.iter_rows(values_only=True):
            cells = ["" if c is None else str(c) for c in row]
            if any(cells):
                rows_text.append(" | ".join(cells))
        if rows_text:
            sheets_text.append(f"--- Sheet: {sheet.title} ---\n" + "\n".join(rows_text))

    text = "\n\n".join(sheets_text)
    if not text.strip():
        raise ExtractionError("No extractable data found in .xlsx")
    return text


def extract_csv(path: Path) -> str:
    try:
        with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
            reader = csv.reader(f)
            rows = [" | ".join(row) for row in reader if any(cell.strip() for cell in row)]
    except OSError as e:
        raise ExtractionError(f"Could not read .csv file: {e}") from e

    text = "\n".join(rows)
    if not text.strip():
        raise ExtractionError("No extractable data found in .csv")
    return text
