"""Unit tests for CostTracker."""
from __future__ import annotations

from decimal import Decimal

import pytest

from src.gateway.cost_tracker import BudgetExceededError, CostTracker

# ── Helpers ────────────────────────────────────────────────────────────────────

def _gpt4o_cost(input_t: int, output_t: int) -> float:
    """Expected cost for gpt-4o at published rates."""
    return input_t * 5.00 / 1_000_000 + output_t * 15.00 / 1_000_000


def _mini_cost(input_t: int, output_t: int) -> float:
    """Expected cost for gpt-4o-mini at published rates."""
    return input_t * 0.15 / 1_000_000 + output_t * 0.60 / 1_000_000


# ── Tests ──────────────────────────────────────────────────────────────────────


def test_record_and_total() -> None:
    """record_usage should return cost and accumulate total_cost."""
    tracker = CostTracker()
    cost = tracker.record_usage("gpt-4o-mini", 1_000, 500)

    expected = _mini_cost(1_000, 500)
    assert float(cost) == pytest.approx(expected, rel=1e-6)
    assert float(tracker.total_cost) == pytest.approx(expected, rel=1e-6)


def test_multiple_records_accumulate() -> None:
    """Successive calls should sum into total_cost."""
    tracker = CostTracker()
    tracker.record_usage("gpt-4o-mini", 100, 50)
    tracker.record_usage("gpt-4o-mini", 100, 50)

    expected = 2 * _mini_cost(100, 50)
    assert float(tracker.total_cost) == pytest.approx(expected, rel=1e-6)


def test_per_query_tracking() -> None:
    """per_query_cost should sum only calls tagged with the same query_id."""
    tracker = CostTracker()
    q = "my-query-id"
    tracker.record_usage("gpt-4o-mini", 100, 50, query_id=q)
    tracker.record_usage("gpt-4o-mini", 100, 50, query_id=q)
    tracker.record_usage("gpt-4o-mini", 100, 50, query_id="other")

    expected_q = 2 * _mini_cost(100, 50)
    assert float(tracker.per_query_cost(q)) == pytest.approx(expected_q, rel=1e-6)
    # Untagged query
    assert float(tracker.per_query_cost("unknown")) == 0.0


def test_budget_exceeded_raises() -> None:
    """record_usage must raise BudgetExceededError when the limit is reached."""
    tracker = CostTracker(budget=0.001)  # $0.001 — tiny budget

    with pytest.raises(BudgetExceededError) as exc_info:
        # gpt-4o at 1000 tokens each ≈ $0.02 >> $0.001
        tracker.record_usage("gpt-4o", 1_000, 1_000)

    err = exc_info.value
    assert err.total >= Decimal("0.001")
    assert err.budget == Decimal("0.001")


def test_budget_exceeded_records_before_raising() -> None:
    """Usage must be recorded even when BudgetExceededError is raised."""
    tracker = CostTracker(budget=0.001)
    with pytest.raises(BudgetExceededError):
        tracker.record_usage("gpt-4o", 1_000, 1_000)
    # Cost was still recorded
    assert tracker.total_cost > 0


def test_remaining_budget_decreases() -> None:
    """remaining_budget must decrease as usage is recorded."""
    tracker = CostTracker(budget=1.0)
    before = tracker.remaining_budget

    tracker.record_usage("gpt-4o-mini", 100, 50)

    assert tracker.remaining_budget < before


def test_remaining_budget_never_negative() -> None:
    """remaining_budget should be clamped to 0 when exceeded."""
    tracker = CostTracker(budget=0.001)
    try:
        tracker.record_usage("gpt-4o", 1_000, 1_000)
    except BudgetExceededError:
        pass
    assert tracker.remaining_budget == Decimal("0")


def test_summary_by_model() -> None:
    """summary_by_model should return separate entries per model."""
    tracker = CostTracker()
    tracker.record_usage("gpt-4o", 100, 100)
    tracker.record_usage("gpt-4o-mini", 100, 100)

    summary = tracker.summary_by_model()

    assert "gpt-4o" in summary
    assert "gpt-4o-mini" in summary
    # gpt-4o is more expensive
    assert summary["gpt-4o"]["total_cost"] > summary["gpt-4o-mini"]["total_cost"]
    assert summary["gpt-4o"]["input_tokens"] == 100
    assert summary["gpt-4o"]["output_tokens"] == 100
    assert summary["gpt-4o-mini"]["input_tokens"] == 100


def test_summary_by_model_accumulates_tokens() -> None:
    """Multiple calls for the same model must sum token counts."""
    tracker = CostTracker()
    tracker.record_usage("gpt-4o-mini", 100, 50)
    tracker.record_usage("gpt-4o-mini", 200, 80)

    summary = tracker.summary_by_model()
    assert summary["gpt-4o-mini"]["input_tokens"] == 300
    assert summary["gpt-4o-mini"]["output_tokens"] == 130


def test_unknown_model_uses_default_pricing() -> None:
    """An unrecognised model_id should fall back to default pricing."""
    tracker = CostTracker()
    cost = tracker.record_usage("my-local-model", 1_000, 1_000)
    assert float(cost) > 0  # pricing was applied


def test_decimal_precision() -> None:
    """Costs computed via Decimal must not drift from expected value."""
    tracker = CostTracker()
    cost = tracker.record_usage("gpt-4o", 1, 1)
    expected = Decimal("1") * Decimal("5.00") / Decimal("1000000") + \
               Decimal("1") * Decimal("15.00") / Decimal("1000000")
    assert cost == expected
