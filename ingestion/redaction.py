"""PII redaction for ingested policy documents.

Provides a PIIRedactor class that scrubs personal identifiers before text is
embedded or stored.  Regex patterns cover Singapore-specific formats (NRIC,
matric number, local mobile).  If presidio_analyzer / presidio_anonymizer are
installed, an NLP-based NER pass is also applied on top.
"""

from __future__ import annotations

import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    # NUS matriculation number  A1234567B
    ("NUS_MATRIC", re.compile(r"\bA\d{7}[A-Z]\b")),
    # Singapore NRIC / FIN  S1234567D  T/F/G variants
    ("NRIC", re.compile(r"\b[STFG]\d{7}[A-Z]\b")),
    # Singapore mobile — local 8-digit starting with 8 or 9
    ("SG_MOBILE_LOCAL", re.compile(r"\b[89]\d{7}\b")),
    # Singapore mobile — international prefix +65 or 65
    ("SG_MOBILE_INTL", re.compile(r"(?:\+65|65)\s?[89]\d{7}\b")),
    # Generic email address
    ("EMAIL", re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b")),
]

_REPLACEMENT = "[REDACTED]"


# ---------------------------------------------------------------------------
# Presidio helper (optional dependency)
# ---------------------------------------------------------------------------

def _try_load_presidio() -> Optional[object]:
    """Return a configured Presidio AnalyzerEngine, or None if unavailable."""
    try:
        from presidio_analyzer import AnalyzerEngine  # type: ignore
        engine = AnalyzerEngine()
        logger.debug("Presidio NLP engine loaded successfully.")
        return engine
    except ImportError:
        logger.warning(
            "presidio_analyzer / presidio_anonymizer not installed — "
            "falling back to regex-only PII redaction.  "
            "Install with: pip install presidio-analyzer presidio-anonymizer spacy "
            "&& python -m spacy download en_core_web_lg"
        )
        return None


# ---------------------------------------------------------------------------
# PIIRedactor
# ---------------------------------------------------------------------------

class PIIRedactor:
    """Redacts PII from plain text using regex and (optionally) Presidio NER.

    Usage::

        redactor = PIIRedactor()
        clean = redactor.redact(raw_text)
    """

    # Presidio entity types to redact when the NLP engine is available.
    _PRESIDIO_ENTITIES = [
        "PERSON",
        "PHONE_NUMBER",
        "EMAIL_ADDRESS",
        "LOCATION",
        "NRP",          # Nationality / Religion / Political group
        "IN_PAN",       # Catch-all for national IDs some models detect
        "SG_NRIC_FIN",  # Presidio custom recogniser if installed
    ]

    def __init__(self) -> None:
        self._analyzer = _try_load_presidio()
        self._use_presidio = self._analyzer is not None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def redact(self, text: str) -> str:
        """Return *text* with all detected PII replaced by ``[REDACTED]``.

        The method:
        1. Applies all regex patterns (always).
        2. If Presidio is available, runs NLP-based NER over the *original*
           text to find spans missed by regex, then merges the replacements.

        The original values are never logged — only the redaction count is.
        """
        if not text:
            return text

        redaction_count = 0

        # --- Pass 1: Presidio NER (operates on original text for better span accuracy) ---
        presidio_spans: list[tuple[int, int]] = []
        if self._use_presidio:
            try:
                results = self._analyzer.analyze(  # type: ignore[union-attr]
                    text=text,
                    entities=self._PRESIDIO_ENTITIES,
                    language="en",
                )
                presidio_spans = [(r.start, r.end) for r in results]
            except Exception as exc:  # pragma: no cover
                logger.warning("Presidio analysis failed (will use regex only): %s", exc)

        # Replace Presidio spans back-to-front so indices stay valid.
        if presidio_spans:
            # Sort descending by start position; deduplicate overlapping spans.
            presidio_spans = _merge_spans(sorted(presidio_spans, key=lambda s: s[0]))
            for start, end in reversed(presidio_spans):
                text = text[:start] + _REPLACEMENT + text[end:]
                redaction_count += 1

        # --- Pass 2: Regex patterns ---
        for label, pattern in _PATTERNS:
            new_text, n = pattern.subn(_REPLACEMENT, text)
            if n:
                logger.debug("Regex pattern %s matched %d occurrence(s).", label, n)
                redaction_count += n
                text = new_text

        if redaction_count:
            logger.info("Redacted %d PII item(s) from text.", redaction_count)
        else:
            logger.debug("No PII detected in text.")

        return text


# ---------------------------------------------------------------------------
# Span utilities
# ---------------------------------------------------------------------------

def _merge_spans(spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Merge overlapping or adjacent spans (sorted ascending by start)."""
    if not spans:
        return []
    merged: list[tuple[int, int]] = [spans[0]]
    for start, end in spans[1:]:
        prev_start, prev_end = merged[-1]
        if start <= prev_end:
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))
    return merged


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_redactor_instance: Optional[PIIRedactor] = None


def get_redactor() -> PIIRedactor:
    """Return the module-level PIIRedactor singleton (lazy-initialised)."""
    global _redactor_instance
    if _redactor_instance is None:
        _redactor_instance = PIIRedactor()
    return _redactor_instance
