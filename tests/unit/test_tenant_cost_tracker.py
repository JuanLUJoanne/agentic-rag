"""
Tests for per-tenant cost tracking (src/gateway/cost_tracker.py).
"""
from __future__ import annotations

import pytest

from src.gateway.cost_tracker import TenantBudgetExceededError, TenantCostTracker


class TestTenantCostTracker:
    def _tracker(self, global_budget: float = 100.0) -> TenantCostTracker:
        return TenantCostTracker(global_budget=global_budget)

    def test_record_accumulates_cost(self):
        """Costs add up per tenant across multiple records."""
        tracker = self._tracker()
        tracker.record("acme", "gpt-4o-mini", input_tokens=1_000_000, output_tokens=0)
        tracker.record("acme", "gpt-4o-mini", input_tokens=1_000_000, output_tokens=0)

        summary = tracker.tenant_summary("acme")
        # gpt-4o-mini input: $0.15 / 1M tokens * 2M = $0.30
        assert abs(summary["total_cost"] - 0.30) < 1e-6, f"Expected ~0.30, got {summary['total_cost']}"

    def test_budget_exceeded_raises(self):
        """Recording past budget raises TenantBudgetExceededError."""
        tracker = self._tracker()
        tracker.set_budget("acme", 0.10)

        # First record: 1M input tokens for gpt-4o-mini = $0.15 > $0.10 budget
        with pytest.raises(TenantBudgetExceededError) as exc_info:
            tracker.record("acme", "gpt-4o-mini", input_tokens=1_000_000, output_tokens=0)

        err = exc_info.value
        assert err.tenant_id == "acme"
        assert err.cost > 0.10
        assert err.budget == pytest.approx(0.10)

    def test_tenant_isolation(self):
        """Tenant A's costs do not affect tenant B's summary."""
        tracker = self._tracker()
        tracker.record("tenantA", "gpt-4o-mini", input_tokens=1_000_000, output_tokens=0)

        summary_b = tracker.tenant_summary("tenantB")
        assert summary_b["total_cost"] == 0.0, "Tenant B should have zero cost"

    def test_all_tenants_summary_lists_all(self):
        """all_tenants_summary returns entries for every known tenant."""
        tracker = self._tracker()
        tracker.record("alpha", "gpt-4o-mini", input_tokens=100, output_tokens=50)
        tracker.record("beta", "gpt-4o-mini", input_tokens=200, output_tokens=100)
        tracker.set_budget("gamma", 5.0)  # gamma has budget but no spend yet

        summaries = tracker.all_tenants_summary()
        tenant_ids = {s["tenant_id"] for s in summaries}
        assert {"alpha", "beta", "gamma"}.issubset(tenant_ids)

    def test_tenant_summary_over_budget_flag(self):
        """over_budget flag is True when tenant has exceeded their budget."""
        tracker = self._tracker()
        tracker.set_budget("acme", 0.001)  # very small budget

        try:
            tracker.record("acme", "gpt-4o-mini", input_tokens=1_000_000, output_tokens=0)
        except TenantBudgetExceededError:
            pass

        summary = tracker.tenant_summary("acme")
        assert summary["over_budget"] is True
        assert summary["remaining"] == 0.0
