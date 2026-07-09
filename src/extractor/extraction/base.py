"""Shared extraction error type."""

from __future__ import annotations


class ExtractionError(Exception):
    """Raised when a document's text could not be extracted."""
