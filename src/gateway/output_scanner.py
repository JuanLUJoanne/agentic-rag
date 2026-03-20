"""
Output PII scanner — Layer 3 of the PII compliance pipeline.

Scans LLM-generated text *after* generation and before returning it to the
caller.  This catches two classes of problems:

  1. Context leakage — the LLM surfaces PII that was present in retrieved
     document chunks (e.g., a support ticket containing a customer email).
  2. Hallucinated PII — the LLM fabricates real-looking personal data
     (names, phone numbers, addresses) that did not appear in the input.

Both are treated identically: detect → redact → log to audit trail.

Usage:
    scanner = OutputScanner()
    clean_text, entities = scanner.scan(llm_output, query_id="q-123")
"""
from __future__ import annotations

import structlog

from src.gateway.pii_detector import PIIDetector, PIIEntity

logger = structlog.get_logger()


class OutputScanner:
    """
    Post-generation PII scanner.

    Wraps a :class:`PIIDetector` and an optional :class:`AuditLogger` so that
    every scanned output is recorded in the compliance audit trail.

    Parameters
    ----------
    audit_logger:
        If provided, a ``pii_found_in_output`` event is appended to the audit
        log whenever PII is detected.  Pass ``None`` to disable audit logging
        (useful in unit tests that don't need the JSONL file).
    confidence_threshold:
        Forwarded to the underlying :class:`PIIDetector`.
    allow_list:
        Forwarded to the underlying :class:`PIIDetector`.
    """

    def __init__(
        self,
        audit_logger=None,  # AuditLogger | None — avoid circular import
        confidence_threshold: float = 0.7,
        allow_list: list[str] | None = None,
    ) -> None:
        self._audit = audit_logger
        self._detector = PIIDetector(
            confidence_threshold=confidence_threshold,
            allow_list=allow_list,
        )

    def scan(
        self,
        text: str,
        query_id: str | None = None,
    ) -> tuple[str, list[PIIEntity]]:
        """
        Scan ``text`` for PII.  Returns ``(clean_text, entities_found)``.

        If no PII is found ``clean_text == text`` and ``entities_found`` is
        an empty list.  If PII is found it is redacted in ``clean_text`` and
        the detected entities are returned for inspection.
        """
        redacted, entities = self._detector.redact(text)

        if entities:
            pii_types = list({e.entity_type for e in entities})
            logger.warning(
                "pii_found_in_output",
                pii_types=pii_types,
                count=len(entities),
                query_id=query_id,
            )
            if self._audit is not None:
                try:
                    self._audit.log_pii_event(
                        event_type="pii_found_in_output",
                        query_id=query_id,
                        pii_types=pii_types,
                        pii_count=len(entities),
                        action_taken="redacted",
                    )
                except Exception as exc:
                    logger.warning("output_scan_audit_failed", reason=str(exc)[:80])

        return redacted, entities
