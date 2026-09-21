"""Text normalization utilities for radiology reports."""

from __future__ import annotations

import re
from typing import Iterable, List


def normalize_report_text(text: str | None) -> str:
    """Normalize whitespace while preserving readable section boundaries."""

    if not text:
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_sentences(text: str) -> List[str]:
    """Split report text into coarse evidence spans."""

    normalized = normalize_report_text(text)
    parts = re.split(r"(?<=[.!?])\s+|\n+", normalized)
    return [part.strip() for part in parts if part.strip()]


def contains_any(text: str, terms: Iterable[str]) -> bool:
    """Case-insensitive phrase search helper."""

    lower = text.lower()
    return any(term.lower() in lower for term in terms)
