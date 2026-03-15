"""Unit tests for CostGuardrail and InputGuardrail."""
from __future__ import annotations

from src.gateway.cost_tracker import CostTracker
from src.gateway.guardrails import CostGuardrail, GuardrailResult, InputGuardrail

# ── CostGuardrail ──────────────────────────────────────────────────────────────


def _guardrail(
    per_request: float = 0.05,
    per_query: float = 0.10,
    total_budget: float = 10.0,
    anomaly_multiplier: float = 3.0,
) -> tuple[CostTracker, CostGuardrail]:
    tracker = CostTracker(budget=total_budget)
    guard = CostGuardrail(
        tracker=tracker,
        per_request_limit=per_request,
        per_query_limit=per_query,
        total_budget=total_budget,
        anomaly_multiplier=anomaly_multiplier,
    )
    return tracker, guard


def test_normal_call_passes() -> None:
    """A cheap, small request should pass all four checks."""
    _, guard = _guardrail()
    result = guard.check("gpt-4o-mini", estimated_tokens=100)
    assert result.allowed
    assert result.reason == "ok"


def test_guardrail_result_bool_true() -> None:
    """GuardrailResult with allowed=True should be truthy."""
    r = GuardrailResult(allowed=True, reason="ok")
    assert bool(r) is True


def test_guardrail_result_bool_false() -> None:
    """GuardrailResult with allowed=False should be falsy."""
    r = GuardrailResult(allowed=False, reason="blocked")
    assert bool(r) is False


def test_per_request_blocks_expensive_call() -> None:
    """
    gpt-4o at 10,000 tokens costs ~$0.10 (> $0.05 per-request limit).

    Estimate: 5000 input × $5/1M + 5000 output × $15/1M = $0.025 + $0.075 = $0.10
    """
    _, guard = _guardrail(per_request=0.05)
    result = guard.check("gpt-4o", estimated_tokens=10_000)
    assert not result.allowed
    assert "per_request" in result.reason


def test_per_request_allows_cheap_call() -> None:
    """gpt-4o-mini at 100 tokens is well under the per-request limit."""
    _, guard = _guardrail(per_request=0.05)
    result = guard.check("gpt-4o-mini", estimated_tokens=100)
    assert result.allowed


def test_per_query_blocks_after_threshold() -> None:
    """
    Cumulative query cost exceeding per_query_limit must be blocked.

    gpt-4o-mini 1000 tokens ≈ $0.000375 per call.
    We record many calls under the same query_id until we approach the limit,
    then the next check should fail.
    """
    tracker, guard = _guardrail(per_query=0.001)
    q = "test-query"

    # Record usage until we're close to the $0.001 limit
    # Each call: 500 * 0.15/1M + 500 * 0.60/1M = 0.0003750
    for _ in range(3):
        tracker.record_usage("gpt-4o-mini", 500, 500, query_id=q)

    # Query cost is now > $0.001; next request should be blocked
    result = guard.check("gpt-4o-mini", estimated_tokens=100, query_id=q)
    assert not result.allowed
    assert "per_query" in result.reason


def test_anomaly_detects_spike() -> None:
    """A request 4× larger than the rolling average should be blocked."""
    _, guard = _guardrail(anomaly_multiplier=3.0)

    # Build a rolling average with small requests (100 tokens each)
    for _ in range(10):
        r = guard.check("gpt-4o-mini", estimated_tokens=100)
        assert r.allowed, "Normal requests should pass"

    # Now send a spike: 100 tokens × 40 = 4000 tokens → 40× the rolling avg
    spike_result = guard.check("gpt-4o-mini", estimated_tokens=4_000)
    assert not spike_result.allowed
    assert "anomaly" in spike_result.reason


def test_anomaly_no_history_passes() -> None:
    """Without rolling history, anomaly detection must not fire."""
    _, guard = _guardrail()
    # First ever request — no rolling average yet
    result = guard.check("gpt-4o-mini", estimated_tokens=5_000)
    # May be blocked by per_request, but NOT by anomaly
    if not result.allowed:
        assert "per_request" in result.reason or "per_query" in result.reason
        assert "anomaly" not in result.reason


def test_total_budget_blocks_when_near_limit() -> None:
    """When total_cost exceeds the budget ceiling, new requests are blocked."""
    tracker, guard = _guardrail(total_budget=0.01)

    # Exhaust the budget — loop until BudgetExceededError is raised
    # (each gpt-4o-mini 1000+1000-token call ≈ $0.00075; ~14 calls reach $0.01)
    while True:
        try:
            tracker.record_usage("gpt-4o-mini", 1_000, 1_000)
        except Exception:
            break  # BudgetExceededError — tracker total is now >= budget

    # Guardrail total_cost + any new estimate >= total_budget → blocked
    result = guard.check("gpt-4o-mini", estimated_tokens=100)
    assert not result.allowed


# ── InputGuardrail ─────────────────────────────────────────────────────────────


def test_input_guardrail_normal_passes() -> None:
    """A normal short query with low iteration count should pass."""
    guard = InputGuardrail()
    result = guard.check("What is machine learning?", iteration_count=0)
    assert result.allowed


def test_input_guardrail_too_long() -> None:
    """Queries over 10,000 characters should be rejected."""
    guard = InputGuardrail(max_query_length=100)
    result = guard.check("x" * 101)
    assert not result.allowed
    assert "too long" in result.reason


def test_input_guardrail_max_iterations() -> None:
    """Iteration count over the limit should be rejected."""
    guard = InputGuardrail(max_iterations=5)
    result = guard.check("some query", iteration_count=6)
    assert not result.allowed
    assert "iterations" in result.reason


def test_input_guardrail_at_boundary() -> None:
    """Exactly at the boundaries should still pass."""
    guard = InputGuardrail(max_query_length=10, max_iterations=3)
    result = guard.check("a" * 10, iteration_count=3)
    assert result.allowed
