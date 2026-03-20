"""
Tests for Layer 2 (PIIGuardrail) and Layer 3 (OutputScanner) of the PII
compliance pipeline, plus AuditLogger.log_pii_event.

All tests pass without Presidio installed (regex fallback path).
"""
from __future__ import annotations

import json

from src.gateway.guardrails import PIIGuardrail
from src.gateway.output_scanner import OutputScanner
from src.gateway.security import AuditLogger


def test_pre_llm_check_redacts_pii_in_context() -> None:
    """
    PIIGuardrail in default mode (strict_mode=False) should allow the request
    but return a redacted prompt with PII replaced by placeholders.
    """
    guard = PIIGuardrail(strict_mode=False)
    prompt = "User query: what is my balance? Context: email alice@example.com"
    result = guard.check(prompt, query_id="q-test-1")

    assert result.allowed
    assert result.pii_count >= 1
    assert "alice@example.com" not in result.redacted_text
    assert "[EMAIL_1]" in result.redacted_text
    assert "EMAIL" in result.pii_types


def test_strict_mode_blocks_request() -> None:
    """
    PIIGuardrail with strict_mode=True must block requests that contain PII
    and set allowed=False.
    """
    guard = PIIGuardrail(strict_mode=True)
    result = guard.check("Call me at 555-123-4567 to discuss the contract.")
    assert not result.allowed
    assert result.pii_count >= 1
    assert "strict_mode" in result.reason


def test_output_scanner_catches_pii_in_generation() -> None:
    """
    OutputScanner must detect and redact PII that appears in LLM-generated text
    before the response reaches the caller.
    """
    scanner = OutputScanner()
    raw_output = "According to our records, contact bob@internal.org for follow-up."
    clean, entities = scanner.scan(raw_output, query_id="q-test-2")

    assert "bob@internal.org" not in clean
    assert any(e.entity_type == "EMAIL" for e in entities)


def test_audit_log_records_pii_events(tmp_path) -> None:
    """
    AuditLogger.log_pii_event must write a valid JSONL entry with all required
    PII compliance fields.
    """
    audit = AuditLogger(path=tmp_path / "pii_audit.jsonl")
    audit.log_pii_event(
        event_type="pii_redacted_pre_llm",
        query_id="q-compliance-1",
        pii_types=["EMAIL", "PHONE_NUMBER"],
        pii_count=2,
        action_taken="redacted",
        redaction_style="placeholder",
    )

    line = (tmp_path / "pii_audit.jsonl").read_text(encoding="utf-8").strip()
    entry = json.loads(line)

    assert entry["event_type"] == "pii_redacted_pre_llm"
    assert entry["query_id"] == "q-compliance-1"
    assert set(entry["pii_types"]) == {"EMAIL", "PHONE_NUMBER"}
    assert entry["pii_count"] == 2
    assert entry["action_taken"] == "redacted"
    assert entry["redaction_style"] == "placeholder"
    assert "event_id" in entry
    assert "timestamp" in entry


def test_no_pii_passes_through_unchanged() -> None:
    """
    A prompt with no PII should pass through the PIIGuardrail unchanged:
    pii_count=0, redacted_text equals original text, and allowed=True.
    """
    guard = PIIGuardrail()
    clean_prompt = "What is retrieval augmented generation and how does it work?"
    result = guard.check(clean_prompt)

    assert result.allowed
    assert result.pii_count == 0
    assert result.redacted_text == clean_prompt
    assert result.pii_types == []
