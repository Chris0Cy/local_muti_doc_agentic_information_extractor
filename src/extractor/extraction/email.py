"""Email extraction: stdlib for .eml, extract-msg for Outlook .msg."""

from __future__ import annotations

from pathlib import Path

from extractor.extraction.base import ExtractionError


def extract_eml(path: Path) -> str:
    from email import message_from_bytes
    from email.policy import default as default_policy

    try:
        raw = path.read_bytes()
        msg = message_from_bytes(raw, policy=default_policy)
    except OSError as e:
        raise ExtractionError(f"Could not read .eml file: {e}") from e

    header_lines = [f"{h}: {msg[h]}" for h in ("From", "To", "Subject", "Date") if msg[h]]

    body_part = msg.get_body(preferencelist=("plain", "html"))
    body = body_part.get_content() if body_part else ""

    text = "\n".join(header_lines) + "\n\n" + body
    if not text.strip():
        raise ExtractionError("No extractable content found in .eml")
    return text


def extract_msg(path: Path) -> str:
    import extract_msg

    try:
        msg = extract_msg.openMsg(str(path))
    except Exception as e:
        raise ExtractionError(f"extract-msg could not open .msg file: {e}") from e

    header_lines = [
        f"{label}: {value}"
        for label, value in (
            ("From", msg.sender),
            ("To", msg.to),
            ("Subject", msg.subject),
            ("Date", msg.date),
        )
        if value
    ]
    body = msg.body or ""

    text = "\n".join(header_lines) + "\n\n" + body
    if not text.strip():
        raise ExtractionError("No extractable content found in .msg")
    return text
