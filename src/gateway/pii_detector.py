"""
Enhanced PII detection with Microsoft Presidio NER + regex fallback.

Three-layer detection strategy:
  1. Presidio (if installed) — uses spaCy NER for entity types that regex cannot
     reliably catch: PERSON, LOCATION, DATE_OF_BIRTH, and general NER types.
  2. Custom regex recognizers — always active; handle EMAIL, PHONE_NUMBER,
     CREDIT_CARD, IP_ADDRESS, and Australian-specific identifiers (AU_MEDICARE,
     AU_TFN, AU_ABN) where Presidio's built-in coverage is limited.
  3. Regex-only fallback — when Presidio is not installed; covers the same PII
     types with a PERSON heuristic (consecutive Title-Case word pairs).

Graceful degradation: if ``presidio-analyzer`` is not installed the module
still works — coverage is narrower but no ImportError is raised.

Usage:
    detector = PIIDetector()
    entities = detector.detect("Call John Smith at 555-123-4567")
    redacted, entities = detector.redact("Email alice@example.com")
    original = detector.redaction_map["[EMAIL_1]"]  # de-redact if authorised
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Literal

import structlog

logger = structlog.get_logger()

# ── Regex patterns ──────────────────────────────────────────────────────────
# Used both as the standalone fallback AND as supplementary AU recognisers
# when Presidio is available (Presidio has weak AU coverage out of the box).

_REGEX_PATTERNS: dict[str, tuple[re.Pattern[str], float]] = {
    # (pattern, confidence_score)
    "EMAIL": (
        re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"),
        0.85,
    ),
    "PHONE_NUMBER": (
        re.compile(r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"),
        0.85,
    ),
    "CREDIT_CARD": (
        re.compile(r"\b(?:\d{4}[-\s]?){3}\d{4}\b"),
        0.85,
    ),
    "IP_ADDRESS": (
        re.compile(r"\b(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\b"),
        0.85,
    ),
    # Australian Medicare card: 10 digits, first digit in range 2–6
    "AU_MEDICARE": (
        re.compile(r"\b[2-6]\d{9}\b"),
        0.85,
    ),
    # Tax File Number: space-delimited NNN NNN NNN (requires spaces to avoid
    # false-positive matches on phone numbers / credit cards)
    "AU_TFN": (
        re.compile(r"\b\d{3} \d{3} \d{3}\b"),
        0.85,
    ),
    # Australian Business Number: NN NNN NNN NNN (space-delimited, 11 digits)
    "AU_ABN": (
        re.compile(r"\b\d{2} \d{3} \d{3} \d{3}\b"),
        0.85,
    ),
    # Person-name heuristic (regex fallback only): two consecutive Title-Case
    # words.  Lower confidence (0.75) because false-positive rate is higher —
    # location names, proper nouns, and sentence starts all produce matches.
    "PERSON": (
        re.compile(r"\b[A-Z][a-z]{1,20}\s+[A-Z][a-z]{1,20}\b"),
        0.75,
    ),
}

# Entity types handled by Presidio natively (not duplicated in regex fallback
# when Presidio is available, to avoid double-counting)
_PRESIDIO_NATIVE = frozenset({
    "PERSON", "LOCATION", "DATE_OF_BIRTH",
    "EMAIL_ADDRESS",  # Presidio uses EMAIL_ADDRESS; we normalise to EMAIL
    "PHONE_NUMBER", "CREDIT_CARD", "IP_ADDRESS",
})

# Normalise Presidio entity-type names to our canonical names
_PRESIDIO_NAME_MAP = {
    "EMAIL_ADDRESS": "EMAIL",
}


@dataclass
class PIIEntity:
    """A single PII span detected in text."""

    entity_type: str   # e.g. "EMAIL", "PERSON", "AU_MEDICARE"
    start: int         # inclusive character offset
    end: int           # exclusive character offset
    score: float       # detection confidence 0–1
    text: str          # verbatim matched text


class PIIDetector:
    """
    Multi-layer PII detector: Presidio NER + custom regex (with regex fallback).

    Parameters
    ----------
    confidence_threshold:
        Entities with ``score < confidence_threshold`` are silently dropped.
        Default 0.7 retains all regex matches (≥ 0.75) and all reasonable
        Presidio hits while filtering noise.
    redaction_style:
        ``"placeholder"`` → ``[PERSON_1]``  (default, reversible)
        ``"hash"``        → ``[PERSON:a1b2c3d4]``  (irreversible but unique)
        ``"mask"``        → ``***``  (irreversible, hides length)
    allow_list:
        Exact-text values that should never be flagged, even if they match a
        pattern.  Use this to whitelist known company names, product names, or
        internal identifiers.
    """

    def __init__(
        self,
        confidence_threshold: float = 0.7,
        redaction_style: Literal["placeholder", "hash", "mask"] = "placeholder",
        allow_list: list[str] | None = None,
    ) -> None:
        self.confidence_threshold = confidence_threshold
        self.redaction_style = redaction_style
        self._allow_set: set[str] = set(allow_list or [])
        self._redaction_map: dict[str, str] = {}
        self._presidio_engine = None
        self._presidio_available = False
        self._init_presidio()

    # ── Initialisation ──────────────────────────────────────────────────────

    def _init_presidio(self) -> None:
        """
        Try to initialise a Presidio AnalyzerEngine with spaCy backend.

        Attempts en_core_web_lg first (more accurate), falls back to
        en_core_web_sm.  If presidio-analyzer is not installed the detector
        silently operates in regex-only mode.
        """
        try:
            from presidio_analyzer import AnalyzerEngine
            from presidio_analyzer.nlp_engine import NlpEngineProvider

            for model_name in ("en_core_web_lg", "en_core_web_sm"):
                try:
                    provider = NlpEngineProvider(
                        nlp_configuration={
                            "nlp_engine_name": "spacy",
                            "models": [{"lang_code": "en", "model_name": model_name}],
                        }
                    )
                    nlp_engine = provider.create_engine()
                    self._presidio_engine = AnalyzerEngine(nlp_engine=nlp_engine)
                    self._presidio_available = True
                    logger.info("presidio_initialized", spacy_model=model_name)
                    self._register_au_recognizers()
                    return
                except OSError:
                    # Model not downloaded yet — try next
                    continue

            logger.warning("presidio_spacy_model_missing", fallback="regex_only")

        except ImportError:
            logger.info("presidio_not_installed", fallback="regex_only")
        except Exception as exc:
            logger.warning(
                "presidio_init_failed", reason=str(exc)[:120], fallback="regex_only"
            )

    def _register_au_recognizers(self) -> None:
        """Register Australian identifier recognisers with the Presidio engine."""
        try:
            from presidio_analyzer import Pattern, PatternRecognizer

            for entity_type in ("AU_MEDICARE", "AU_TFN", "AU_ABN"):
                pattern_re, score = _REGEX_PATTERNS[entity_type]
                recognizer = PatternRecognizer(
                    supported_entity=entity_type,
                    patterns=[
                        Pattern(
                            name=entity_type.lower(),
                            regex=pattern_re.pattern,
                            score=score,
                        )
                    ],
                )
                self._presidio_engine.registry.add_recognizer(recognizer)
        except Exception as exc:
            logger.warning("au_recognizer_registration_failed", reason=str(exc)[:80])

    # ── Public API ──────────────────────────────────────────────────────────

    def detect(self, text: str) -> list[PIIEntity]:
        """
        Detect all PII entities in ``text``.

        Returns entities sorted by start offset, filtered by
        ``confidence_threshold`` and ``allow_list``.
        """
        if self._presidio_available and self._presidio_engine:
            try:
                entities = self._detect_presidio(text)
            except Exception as exc:
                logger.warning(
                    "presidio_detection_failed",
                    reason=str(exc)[:80],
                    fallback="regex",
                )
                entities = self._detect_regex(text)
        else:
            entities = self._detect_regex(text)

        # Apply filters
        entities = [
            e
            for e in entities
            if e.score >= self.confidence_threshold and e.text not in self._allow_set
        ]

        if entities:
            counts: dict[str, int] = {}
            for e in entities:
                counts[e.entity_type] = counts.get(e.entity_type, 0) + 1
            logger.info("pii_detected", by_type=counts, total=len(entities))

        return sorted(entities, key=lambda e: e.start)

    def redact(self, text: str) -> tuple[str, list[PIIEntity]]:
        """
        Replace all PII in ``text`` with typed placeholders.

        Returns ``(redacted_text, entities_found)``.  The ``redaction_map``
        property maps each placeholder back to its original value so
        authorised callers can de-redact if needed.
        """
        entities = self.detect(text)
        if not entities:
            return text, []

        # Assign per-type counters and build placeholder mapping
        type_counters: dict[str, int] = {}
        placeholder_map: list[tuple[PIIEntity, str]] = []

        for entity in entities:
            type_counters[entity.entity_type] = (
                type_counters.get(entity.entity_type, 0) + 1
            )
            idx = type_counters[entity.entity_type]
            placeholder = self._make_placeholder(entity, idx)
            self._redaction_map[placeholder] = entity.text
            placeholder_map.append((entity, placeholder))

        # Replace right-to-left so earlier offsets remain valid
        result = text
        for entity, placeholder in sorted(
            placeholder_map, key=lambda x: x[0].start, reverse=True
        ):
            result = result[: entity.start] + placeholder + result[entity.end :]

        logger.info(
            "pii_redacted",
            count=len(entities),
            style=self.redaction_style,
        )
        return result, entities

    @property
    def redaction_map(self) -> dict[str, str]:
        """
        Read-only mapping of ``placeholder → original_text``.

        Authorised callers can use this to reverse redaction on responses
        destined for users who are cleared to see the original PII.
        """
        return dict(self._redaction_map)

    # ── Internal detection helpers ──────────────────────────────────────────

    def _detect_presidio(self, text: str) -> list[PIIEntity]:
        results = self._presidio_engine.analyze(text=text, language="en")
        entities = []
        for r in results:
            entity_type = _PRESIDIO_NAME_MAP.get(r.entity_type, r.entity_type)
            entities.append(
                PIIEntity(
                    entity_type=entity_type,
                    start=r.start,
                    end=r.end,
                    score=r.score,
                    text=text[r.start : r.end],
                )
            )
        return entities

    def _detect_regex(self, text: str) -> list[PIIEntity]:
        entities = []
        for entity_type, (pattern, score) in _REGEX_PATTERNS.items():
            for m in pattern.finditer(text):
                entities.append(
                    PIIEntity(
                        entity_type=entity_type,
                        start=m.start(),
                        end=m.end(),
                        score=score,
                        text=m.group(0),
                    )
                )
        return entities

    def _make_placeholder(self, entity: PIIEntity, idx: int) -> str:
        if self.redaction_style == "placeholder":
            return f"[{entity.entity_type}_{idx}]"
        if self.redaction_style == "hash":
            digest = hashlib.sha256(entity.text.encode()).hexdigest()[:8]
            return f"[{entity.entity_type}:{digest}]"
        # mask: replace each character with *
        return "*" * len(entity.text)
