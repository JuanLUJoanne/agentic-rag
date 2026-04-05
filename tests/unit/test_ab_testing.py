"""
Tests for A/B testing framework (src/eval/ab_testing.py).
"""
from __future__ import annotations

import hashlib
import uuid

import pytest

from src.eval.ab_testing import ABTest, ABTestRegistry, ABVariant, get_ab_registry


@pytest.fixture
def fresh_registry():
    """Return a fresh registry (not the module singleton) for isolation."""
    return ABTestRegistry()


@pytest.fixture
def two_variant_test():
    return ABTest(
        name="test_exp",
        variants=[
            ABVariant(name="control", weight=0.5),
            ABVariant(name="treatment", weight=0.5),
        ],
    )


class TestABAssignment:
    def test_assignment_is_deterministic(self, two_variant_test):
        """Same entity_id always gets the same variant."""
        entity_id = "user-abc-123"
        first = two_variant_test.assign(entity_id)
        for _ in range(10):
            assert two_variant_test.assign(entity_id) == first

    def test_assignment_distributes_roughly(self, two_variant_test):
        """1000 random IDs should distribute ~50/50 (within 45–55%)."""
        counts: dict[str, int] = {"control": 0, "treatment": 0}
        for _ in range(1000):
            entity_id = str(uuid.uuid4())
            variant = two_variant_test.assign(entity_id)
            counts[variant] += 1

        for name, count in counts.items():
            assert 400 <= count <= 600, (
                f"Variant '{name}' got {count}/1000 assignments — expected ~500 (±100)"
            )

    def test_variant_names_only(self, two_variant_test):
        """assign always returns a name that is in the variant list."""
        valid_names = {v["name"] for v in two_variant_test.variants}
        for _ in range(100):
            result = two_variant_test.assign(str(uuid.uuid4()))
            assert result in valid_names


class TestABOutcomes:
    def test_record_and_summary(self, two_variant_test):
        """Record outcomes and verify summary stats."""
        two_variant_test.record_outcome("control", "mrr", 0.60)
        two_variant_test.record_outcome("control", "mrr", 0.80)
        two_variant_test.record_outcome("treatment", "mrr", 0.90)

        summary = two_variant_test.summary()

        control_mrr = summary["control"]["mrr"]
        assert control_mrr["count"] == 2
        assert control_mrr["sum"] == pytest.approx(1.40)
        assert control_mrr["mean"] == pytest.approx(0.70)

        treatment_mrr = summary["treatment"]["mrr"]
        assert treatment_mrr["count"] == 1
        assert treatment_mrr["mean"] == pytest.approx(0.90)

    def test_summary_empty_before_records(self, two_variant_test):
        """Summary is empty before any outcomes are recorded."""
        assert two_variant_test.summary() == {}


class TestABRegistry:
    def test_registry_get_returns_registered(self, fresh_registry, two_variant_test):
        """register + get round-trip."""
        fresh_registry.register(two_variant_test)
        result = fresh_registry.get("test_exp")
        assert result is two_variant_test

    def test_registry_get_unknown_returns_none(self, fresh_registry):
        assert fresh_registry.get("nonexistent") is None

    def test_all_summaries(self, fresh_registry, two_variant_test):
        fresh_registry.register(two_variant_test)
        two_variant_test.record_outcome("control", "clicks", 5.0)
        summaries = fresh_registry.all_summaries()
        assert "test_exp" in summaries
        assert "control" in summaries["test_exp"]


class TestPreregisteredTests:
    def test_preregistered_tests_exist(self):
        """get_ab_registry() should have retrieval_strategy and reranking pre-registered."""
        registry = get_ab_registry()
        assert registry.get("retrieval_strategy") is not None
        assert registry.get("reranking") is not None

    def test_retrieval_strategy_variants(self):
        test = get_ab_registry().get("retrieval_strategy")
        variant_names = {v["name"] for v in test.variants}
        assert "rrf_only" in variant_names
        assert "rrf_mmr" in variant_names

    def test_reranking_variants(self):
        test = get_ab_registry().get("reranking")
        variant_names = {v["name"] for v in test.variants}
        assert "none" in variant_names
        assert "litm" in variant_names

    def test_preregistered_assignment_works(self):
        """Pre-registered tests can assign entities without error."""
        registry = get_ab_registry()
        for test_name in ("retrieval_strategy", "reranking"):
            test = registry.get(test_name)
            variant = test.assign("some-query-id")
            valid = {v["name"] for v in test.variants}
            assert variant in valid
