"""
Guardrails for LLM API requests.

Two guard types:
  CostGuardrail  — blocks requests that would exceed cost budgets or look
                   anomalous compared to recent usage.
  InputGuardrail — blocks requests whose input characteristics (length,
                   iteration depth) are outside safe bounds.

Both return a ``GuardrailResult`` — callers decide what to do on a block
rather than having the guardrail raise an exception, making it easy to
log, retry with fewer tokens, or fall back gracefully.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from decimal import Decimal

import structlog

from src.gateway.cost_tracker import _MILLION, _PRICING, CostTracker

logger = structlog.get_logger()


@dataclass
class GuardrailResult:
    """Result returned by every guardrail check."""

    allowed: bool
    reason: str

    def __bool__(self) -> bool:  # allows ``if guardrail.check(...):``
        return self.allowed


class CostGuardrail:
    """
    Four-layer cost guard:

    1. per_request  — estimated cost for this single call.
    2. per_query    — cumulative cost for one logical query (multi-step).
    3. anomaly      — current estimate is >N× the rolling average.
    4. total_budget — global budget ceiling across all queries.

    Token cost is estimated by splitting ``estimated_tokens`` 50/50 between
    input and output, which is a reasonable approximation when the split is
    unknown at request time.
    """

    def __init__(
        self,
        tracker: CostTracker,
        per_request_limit: float = 0.05,
        per_query_limit: float = 0.10,
        total_budget: float = 10.0,
        anomaly_multiplier: float = 3.0,
        rolling_window: int = 20,
    ) -> None:
        self._tracker = tracker
        self._per_request = Decimal(str(per_request_limit))
        self._per_query = Decimal(str(per_query_limit))
        self._total_budget = Decimal(str(total_budget))
        self._anomaly_multiplier = Decimal(str(anomaly_multiplier))
        self._rolling: deque[Decimal] = deque(maxlen=rolling_window)

    def _estimate_cost(self, model_id: str, estimated_tokens: int) -> Decimal:
        pricing = _PRICING.get(model_id, _PRICING["default"])
        half = Decimal(str(max(1, estimated_tokens // 2)))
        return half * pricing["input"] / _MILLION + half * pricing["output"] / _MILLION

    def check(
        self,
        model_id: str,
        estimated_tokens: int,
        query_id: str | None = None,
    ) -> GuardrailResult:
        """
        Check all cost layers in order of cheapness-to-compute.

        Does NOT record usage — call ``CostTracker.record_usage`` separately
        after the actual LLM call completes.
        """
        estimated = self._estimate_cost(model_id, estimated_tokens)

        # 1. Per-request limit
        if estimated > self._per_request:
            reason = (
                f"per_request limit: estimated ${estimated:.6f} > ${self._per_request}"
            )
            logger.warning(
                "guardrail_triggered",
                level="per_request",
                reason=reason,
                estimated=float(estimated),
            )
            return GuardrailResult(allowed=False, reason=reason)

        # 2. Per-query cumulative limit
        if query_id is not None:
            query_so_far = self._tracker.per_query_cost(query_id)
            if query_so_far + estimated > self._per_query:
                reason = (
                    f"per_query limit: ${float(query_so_far):.6f} spent + "
                    f"${float(estimated):.6f} estimated > ${float(self._per_query):.6f}"
                )
                logger.warning(
                    "guardrail_triggered",
                    level="per_query",
                    reason=reason,
                    query_id=query_id,
                    query_so_far=float(query_so_far),
                )
                return GuardrailResult(allowed=False, reason=reason)

        # 3. Anomaly detection (only when we have rolling history)
        if self._rolling:
            rolling_sum = sum(self._rolling)
            avg = rolling_sum / Decimal(str(len(self._rolling)))
            if avg > Decimal("0") and estimated > self._anomaly_multiplier * avg:
                reason = (
                    f"anomaly: ${float(estimated):.6f} is "
                    f"{float(self._anomaly_multiplier):.1f}× rolling avg "
                    f"${float(avg):.6f}"
                )
                logger.warning(
                    "guardrail_triggered",
                    level="anomaly",
                    reason=reason,
                    estimated=float(estimated),
                    rolling_avg=float(avg),
                )
                return GuardrailResult(allowed=False, reason=reason)

        # 4. Total budget ceiling
        if self._tracker.total_cost + estimated >= self._total_budget:
            reason = (
                f"total budget limit: ${float(self._tracker.total_cost):.6f} + "
                f"${float(estimated):.6f} >= ${float(self._total_budget):.6f}"
            )
            logger.warning(
                "guardrail_triggered",
                level="total_budget",
                reason=reason,
            )
            return GuardrailResult(allowed=False, reason=reason)

        # All checks passed — update rolling window
        self._rolling.append(estimated)
        return GuardrailResult(allowed=True, reason="ok")


class InputGuardrail:
    """
    Validates request characteristics before any LLM call.

    Checks:
      - query length (prevents enormous context injections)
      - iteration depth (prevents runaway agent loops)
    """

    def __init__(
        self,
        max_query_length: int = 10_000,
        max_iterations: int = 10,
    ) -> None:
        self._max_query_length = max_query_length
        self._max_iterations = max_iterations

    def check(
        self,
        query: str,
        iteration_count: int = 0,
    ) -> GuardrailResult:
        if len(query) > self._max_query_length:
            reason = f"query too long: {len(query)} chars > {self._max_query_length}"
            logger.warning(
                "guardrail_triggered",
                level="input_length",
                reason=reason,
                length=len(query),
            )
            return GuardrailResult(allowed=False, reason=reason)

        if iteration_count > self._max_iterations:
            reason = f"max iterations exceeded: {iteration_count} > {self._max_iterations}"
            logger.warning(
                "guardrail_triggered",
                level="max_iterations",
                reason=reason,
                iterations=iteration_count,
            )
            return GuardrailResult(allowed=False, reason=reason)

        return GuardrailResult(allowed=True, reason="ok")
