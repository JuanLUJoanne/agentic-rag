"""
Tests for the enhanced PIIDetector (src/gateway/pii_detector.py).

All tests pass with regex-only fallback (no Presidio required) so the CI
pipeline stays green without spaCy model downloads.  When Presidio is
installed the same tests validate the NER path.
"""
from __future__ import annotations

from src.gateway.pii_detector import PIIDetector


def test_detect_email() -> None:
    """Email addresses should be detected with entity_type='EMAIL'."""
    detector = PIIDetector()
    entities = detector.detect("Contact us at user@example.com for help.")
    email_entities = [e for e in entities if e.entity_type == "EMAIL"]
    assert len(email_entities) >= 1
    assert any("user@example.com" in e.text for e in email_entities)


def test_detect_phone() -> None:
    """US-format phone numbers should be detected."""
    detector = PIIDetector()
    entities = detector.detect("Call me at 555-867-5309 anytime.")
    phone_entities = [e for e in entities if e.entity_type == "PHONE_NUMBER"]
    assert len(phone_entities) >= 1


def test_detect_credit_card() -> None:
    """16-digit credit card numbers should be detected."""
    detector = PIIDetector()
    entities = detector.detect("My card: 4111-1111-1111-1111.")
    cc_entities = [e for e in entities if e.entity_type == "CREDIT_CARD"]
    assert len(cc_entities) >= 1


def test_detect_person_name() -> None:
    """Two consecutive Title-Case words should be detected as PERSON."""
    # Default threshold=0.7; regex PERSON score=0.75 → passes
    detector = PIIDetector(confidence_threshold=0.7)
    entities = detector.detect("The patient is John Smith.")
    person_entities = [e for e in entities if e.entity_type == "PERSON"]
    assert len(person_entities) >= 1
    assert any("John Smith" in e.text for e in person_entities)


def test_detect_australian_medicare() -> None:
    """Australian Medicare numbers (10 digits, first digit 2-6) should be detected."""
    detector = PIIDetector()
    entities = detector.detect("Medicare number: 2123456789")
    medicare_entities = [e for e in entities if e.entity_type == "AU_MEDICARE"]
    assert len(medicare_entities) >= 1
    assert medicare_entities[0].text == "2123456789"


def test_redact_replaces_with_placeholder() -> None:
    """PIIDetector.redact should replace PII with '[TYPE_N]' placeholders."""
    detector = PIIDetector(redaction_style="placeholder")
    redacted, entities = detector.redact("Email alice@example.org now.")
    assert "alice@example.org" not in redacted
    assert "[EMAIL_1]" in redacted
    assert len(entities) >= 1


def test_redaction_map_created_correctly() -> None:
    """redaction_map must map each placeholder back to the original value."""
    detector = PIIDetector()
    original_email = "bob@test.com"
    redacted, entities = detector.redact(f"Contact {original_email} ASAP.")
    rmap = detector.redaction_map
    # At least one placeholder should resolve to the original email
    assert original_email in rmap.values()
    placeholder = next(k for k, v in rmap.items() if v == original_email)
    assert placeholder in redacted


def test_confidence_threshold_filters_low_confidence() -> None:
    """
    PERSON heuristic has score=0.75; raising threshold to 0.8 must filter it out.
    """
    # Threshold above regex PERSON score (0.75) → PERSON entities filtered
    detector = PIIDetector(confidence_threshold=0.8)
    entities = detector.detect("Meeting with John Smith tomorrow.")
    person_entities = [e for e in entities if e.entity_type == "PERSON"]
    assert len(person_entities) == 0


def test_allow_list_skips_company_names() -> None:
    """
    Exact-text values in allow_list must not be flagged even if they match a
    pattern.  'Bill Gates' matches the PERSON heuristic but should be ignored
    when added to the allow list.
    """
    detector = PIIDetector(allow_list=["Bill Gates"])
    entities = detector.detect("Contact Bill Gates at Microsoft for licensing.")
    blocked_entities = [e for e in entities if "Bill Gates" in e.text]
    assert len(blocked_entities) == 0
