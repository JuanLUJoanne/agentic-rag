"""
Security primitives: prompt-injection detection, PII redaction, audit logging.

Three components:

  InputSanitizer  — rejects inputs that match known prompt-injection patterns.
                    Raises ``PromptInjectionDetected`` so callers can handle it
                    at the workflow level without catching a generic exception.

  PIIDetector     — detects and redacts email addresses, phone numbers, and
                    credit-card numbers using regular expressions.

  AuditLogger     — appends a JSONL record for every completed query.  The file
                    is append-only; each entry is a valid JSON object followed
                    by a newline so it can be streamed or grepped without
                    loading the full file.
"""
from __future__ import annotations

import json
import re
import uuid
from datetime import UTC, datetime
from pathlib import Path

import structlog

logger = structlog.get_logger()

# ── Prompt-injection patterns ──────────────────────────────────────────────

_INJECTION_PATTERNS: list[str] = [
    r"ignore\s+(previous\s+)?instructions",
    r"system\s+prompt",
    r"you\s+are\s+now",
    r"disregard\s+(all\s+)?previous",
    r"forget\s+(all\s+)?previous",
    r"act\s+as\s+(?:if\s+you\s+are|a)\b",
]

# ── PII patterns ───────────────────────────────────────────────────────────

_PII_PATTERNS: dict[str, re.Pattern[str]] = {
    "email": re.compile(
        r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"
    ),
    "phone": re.compile(
        r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"
    ),
    "credit_card": re.compile(
        r"\b(?:\d{4}[-\s]?){3}\d{4}\b"
    ),
}


class PromptInjectionDetected(Exception):
    """Raised when a prompt-injection pattern is found in user input."""

    def __init__(self, matched: str) -> None:
        super().__init__(f"Prompt injection detected: {matched!r}")
        self.matched = matched


class InputSanitizer:
    """
    Detects prompt injection in user-supplied text.

    Patterns are compiled once at instantiation. Raises
    ``PromptInjectionDetected`` on the first match; returns the original text
    unchanged when clean.
    """

    def __init__(self, patterns: list[str] | None = None) -> None:
        raw = patterns if patterns is not None else _INJECTION_PATTERNS
        self._compiled = [re.compile(p, re.IGNORECASE) for p in raw]

    def sanitize(self, text: str) -> str:
        """Return ``text`` unchanged, or raise PromptInjectionDetected."""
        for pattern in self._compiled:
            m = pattern.search(text)
            if m:
                logger.warning(
                    "injection_detected",
                    matched=m.group(0),
                    pattern=pattern.pattern,
                )
                raise PromptInjectionDetected(m.group(0))
        return text


class PIIDetector:
    """
    Detects and redacts PII (email, phone, credit card) via regex.

    ``detect`` returns a mapping of PII type → list of matched strings.
    ``redact`` returns a copy of the text with all matches replaced by
    ``[REDACTED]``.
    """

    def detect(self, text: str) -> dict[str, list[str]]:
        found: dict[str, list[str]] = {}
        for name, pattern in _PII_PATTERNS.items():
            matches = pattern.findall(text)
            if matches:
                found[name] = matches
        if found:
            logger.warning(
                "pii_found",
                types=list(found.keys()),
                total_matches=sum(len(v) for v in found.values()),
            )
        return found

    def redact(self, text: str) -> str:
        """Replace all PII occurrences with ``[REDACTED]``."""
        result = text
        for pattern in _PII_PATTERNS.values():
            result = pattern.sub("[REDACTED]", result)
        return result


class AuditLogger:
    """
    Append-only JSONL audit log.

    Each call to ``log`` appends one JSON line with a unique event_id,
    UTC timestamp, and all provided fields.  The parent directory is
    created automatically if it does not exist.
    """

    def __init__(self, path: str | Path = "data/audit.jsonl") -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def log(
        self,
        *,
        query_id: str | None = None,
        input_text: str,
        output_text: str,
        model: str,
        cost: float,
        agents_used: list[str],
    ) -> None:
        entry = {
            "event_id": str(uuid.uuid4()),
            "timestamp": datetime.now(UTC).isoformat(),
            "query_id": query_id or str(uuid.uuid4()),
            "input": input_text,
            "output": output_text,
            "model": model,
            "cost": cost,
            "agents_used": agents_used,
        }
        with self._path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry) + "\n")
        logger.info(
            "audit_logged",
            query_id=entry["query_id"],
            model=model,
            cost=cost,
            agents=len(agents_used),
        )

    def log_pii_event(
        self,
        *,
        event_type: str,
        query_id: str | None = None,
        pii_types: list[str],
        pii_count: int,
        action_taken: str,
        redaction_style: str = "placeholder",
    ) -> None:
        """
        Append a PII-specific compliance event to the audit log.

        ``event_type`` should be one of:
          - ``pii_detected_ingestion``   — PII found in raw user input
          - ``pii_redacted_pre_llm``     — PII redacted before LLM call
          - ``pii_found_in_output``      — PII detected in LLM response
          - ``pii_detection_failed``     — detector error (log and continue)

        Each entry includes: timestamp, query_id, pii_types, pii_count,
        action_taken, and redaction_style for compliance reporting.
        """
        entry = {
            "event_id": str(uuid.uuid4()),
            "timestamp": datetime.now(UTC).isoformat(),
            "query_id": query_id or str(uuid.uuid4()),
            "event_type": event_type,
            "pii_types": pii_types,
            "pii_count": pii_count,
            "action_taken": action_taken,
            "redaction_style": redaction_style,
        }
        with self._path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry) + "\n")
        logger.info(
            "pii_audit_logged",
            event_type=event_type,
            pii_count=pii_count,
            query_id=entry["query_id"],
        )
