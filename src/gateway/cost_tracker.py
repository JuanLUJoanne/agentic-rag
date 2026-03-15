"""
Cost tracker for LLM API usage.

Uses Python's ``Decimal`` for exact arithmetic — floating-point errors
accumulate noticeably over hundreds of requests, especially when comparing
against a budget threshold.

Pricing is per-1M-token (OpenAI published rates as of 2025):
  gpt-4o       : $5.00 input / $15.00 output
  gpt-4o-mini  : $0.15 input / $0.60 output
"""
from __future__ import annotations

from collections import defaultdict
from decimal import Decimal

import structlog

logger = structlog.get_logger()

# Pricing per 1 M tokens
_PRICING: dict[str, dict[str, Decimal]] = {
    "gpt-4o": {
        "input": Decimal("5.00"),
        "output": Decimal("15.00"),
    },
    "gpt-4o-mini": {
        "input": Decimal("0.15"),
        "output": Decimal("0.60"),
    },
    "default": {
        "input": Decimal("1.00"),
        "output": Decimal("3.00"),
    },
}

_MILLION = Decimal("1000000")

_BUDGET_WARNING_THRESHOLD = Decimal("0.8")  # warn at 80 % utilisation


class BudgetExceededError(Exception):
    """Raised when cumulative cost meets or exceeds the configured budget."""

    def __init__(self, total: Decimal, budget: Decimal) -> None:
        super().__init__(f"Budget exceeded: ${total:.6f} >= ${budget:.6f}")
        self.total = total
        self.budget = budget


class CostTracker:
    """
    Records per-model and per-query LLM usage costs.

    All arithmetic uses ``Decimal`` to avoid floating-point drift.
    ``BudgetExceededError`` is raised *after* recording the usage that pushed
    the total over the limit — the cost is tracked regardless.
    """

    def __init__(self, budget: float = 10.0) -> None:
        self._budget = Decimal(str(budget))
        self._total: Decimal = Decimal("0")
        self._by_model: dict[str, Decimal] = defaultdict(lambda: Decimal("0"))
        self._by_query: dict[str, Decimal] = defaultdict(lambda: Decimal("0"))
        self._input_tokens: dict[str, int] = defaultdict(int)
        self._output_tokens: dict[str, int] = defaultdict(int)

    # ── Recording ───────────────────────────────────────────────────────────

    def record_usage(
        self,
        model_id: str,
        input_tokens: int,
        output_tokens: int,
        query_id: str | None = None,
    ) -> Decimal:
        """
        Record token usage and return the cost for this request.

        Raises BudgetExceededError if the running total reaches the budget.
        The usage is still recorded before the raise so callers can read the
        total even after catching the error.
        """
        pricing = _PRICING.get(model_id, _PRICING["default"])
        cost = (
            Decimal(str(input_tokens)) * pricing["input"] / _MILLION
            + Decimal(str(output_tokens)) * pricing["output"] / _MILLION
        )

        self._total += cost
        self._by_model[model_id] += cost
        self._input_tokens[model_id] += input_tokens
        self._output_tokens[model_id] += output_tokens

        if query_id:
            self._by_query[query_id] += cost

        utilisation = self._total / self._budget if self._budget else Decimal("0")
        logger.info(
            "usage_recorded",
            model=model_id,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost=float(cost),
            total=float(self._total),
            budget_pct=round(float(utilisation), 3),
        )

        if utilisation >= _BUDGET_WARNING_THRESHOLD:
            logger.warning(
                "budget_warning",
                total=float(self._total),
                budget=float(self._budget),
                pct=float(utilisation),
            )

        if self._total >= self._budget:
            logger.error(
                "budget_exceeded",
                total=float(self._total),
                budget=float(self._budget),
            )
            raise BudgetExceededError(self._total, self._budget)

        return cost

    # ── Properties ──────────────────────────────────────────────────────────

    @property
    def total_cost(self) -> Decimal:
        return self._total

    @property
    def remaining_budget(self) -> Decimal:
        return max(Decimal("0"), self._budget - self._total)

    # ── Queries ─────────────────────────────────────────────────────────────

    def per_query_cost(self, query_id: str) -> Decimal:
        """Return the accumulated cost for a specific query_id."""
        return self._by_query.get(query_id, Decimal("0"))

    def summary_by_model(self) -> dict[str, dict]:
        """Return per-model totals: cost, input_tokens, output_tokens."""
        return {
            model: {
                "total_cost": float(cost),
                "input_tokens": self._input_tokens[model],
                "output_tokens": self._output_tokens[model],
            }
            for model, cost in self._by_model.items()
        }


# ── Module-level singleton ──────────────────────────────────────────────────

_default_tracker: CostTracker | None = None


def get_default_tracker() -> CostTracker:
    global _default_tracker
    if _default_tracker is None:
        _default_tracker = CostTracker()
    return _default_tracker
