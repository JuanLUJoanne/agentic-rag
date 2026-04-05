"""
A/B testing framework for RAG pipeline experiments.

Supports deterministic variant assignment (same entity always gets the same
variant), outcome recording, and per-variant statistical summaries.

Usage example::

    registry = get_ab_registry()

    # Assign a user/query to a variant
    variant = registry.get("retrieval_strategy").assign("user-123")

    # Record an outcome metric
    registry.get("retrieval_strategy").record_outcome(variant, "mrr", 0.85)

    # Get summary stats
    print(registry.all_summaries())
"""
from __future__ import annotations

import hashlib
from collections import defaultdict
from typing import TypedDict

import structlog

logger = structlog.get_logger()


class ABVariant(TypedDict):
    name: str
    weight: float  # 0.0 – 1.0; all weights in a test must sum to 1.0


class ABTest:
    """
    A named A/B test with weighted variant assignment.

    Variant assignment is deterministic: the same ``entity_id`` always
    maps to the same variant for a given test, so results are reproducible
    and users don't switch groups across requests.
    """

    def __init__(self, name: str, variants: list[ABVariant]) -> None:
        if not variants:
            raise ValueError("ABTest requires at least one variant")
        total_weight = sum(v["weight"] for v in variants)
        if abs(total_weight - 1.0) > 1e-6:
            raise ValueError(
                f"Variant weights must sum to 1.0, got {total_weight:.6f}"
            )
        self.name = name
        self.variants = variants
        # {variant_name: {metric: {"sum": float, "count": int}}}
        self._outcomes: dict[str, dict[str, dict[str, float]]] = defaultdict(
            lambda: defaultdict(lambda: {"sum": 0.0, "count": 0})
        )

    def assign(self, entity_id: str) -> str:
        """
        Deterministically assign an entity to a variant.

        Uses sha256(test_name + entity_id) % 100 mapped against cumulative
        weight buckets.
        """
        digest = hashlib.sha256(f"{self.name}{entity_id}".encode()).hexdigest()
        bucket = int(digest, 16) % 100  # 0 – 99

        cumulative = 0.0
        for variant in self.variants:
            cumulative += variant["weight"] * 100
            if bucket < cumulative:
                return variant["name"]
        # Fallback to last variant (handles floating-point edge cases)
        return self.variants[-1]["name"]

    def record_outcome(self, variant: str, metric: str, value: float) -> None:
        """Record a metric value for a variant."""
        self._outcomes[variant][metric]["sum"] += value
        self._outcomes[variant][metric]["count"] += 1
        logger.debug(
            "ab_outcome_recorded",
            test=self.name,
            variant=variant,
            metric=metric,
            value=value,
        )

    def summary(self) -> dict:
        """
        Return per-variant stats.

        Shape: ``{variant_name: {metric_name: {mean, count, sum}}}``
        """
        result: dict[str, dict[str, dict]] = {}
        for variant_name, metrics in self._outcomes.items():
            result[variant_name] = {}
            for metric_name, stats in metrics.items():
                count = stats["count"]
                total = stats["sum"]
                result[variant_name][metric_name] = {
                    "mean": total / count if count > 0 else 0.0,
                    "count": count,
                    "sum": total,
                }
        return result


class ABTestRegistry:
    """Registry for named A/B tests."""

    def __init__(self) -> None:
        self._tests: dict[str, ABTest] = {}

    def register(self, test: ABTest) -> None:
        """Register a test. Overwrites any existing test with the same name."""
        self._tests[test.name] = test
        logger.info("ab_test_registered", name=test.name, variants=[v["name"] for v in test.variants])

    def get(self, name: str) -> ABTest | None:
        """Return a registered test by name, or None."""
        return self._tests.get(name)

    def all_summaries(self) -> dict:
        """Return summaries for all registered tests."""
        return {name: test.summary() for name, test in self._tests.items()}


# ── Module-level singleton ──────────────────────────────────────────────────

_ab_registry: ABTestRegistry | None = None


def get_ab_registry() -> ABTestRegistry:
    global _ab_registry
    if _ab_registry is None:
        _ab_registry = ABTestRegistry()
        # Pre-register standard tests
        _ab_registry.register(
            ABTest(
                name="retrieval_strategy",
                variants=[
                    ABVariant(name="rrf_only", weight=0.5),
                    ABVariant(name="rrf_mmr", weight=0.5),
                ],
            )
        )
        _ab_registry.register(
            ABTest(
                name="reranking",
                variants=[
                    ABVariant(name="none", weight=0.5),
                    ABVariant(name="litm", weight=0.5),
                ],
            )
        )
    return _ab_registry
