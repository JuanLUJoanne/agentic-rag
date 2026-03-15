"""
Evaluation drift detection.

Detects when pipeline quality regresses across prompt versions or deployments
by comparing current eval results against a saved baseline.

Design:
  - Baselines are saved to a JSON file (``data/drift_baseline.json`` by default)
    so they survive process restarts and can be committed to the repo.
  - ``detect_drift`` fires an alert if any metric dimension drops more than
    ``threshold`` (default 5 percentage points) below the baseline.
  - structlog events allow alert routing to Slack / PagerDuty via a log sink.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import structlog

from src.eval.ragas_eval import RAGEvalResult

logger = structlog.get_logger()

_METRIC_DIMENSIONS = [
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
    "citation_accuracy",
]


@dataclass
class DriftBaseline:
    prompt_version: str
    dimension_scores: dict[str, float]
    created_at: str = field(
        default_factory=lambda: datetime.now(UTC).isoformat()
    )


@dataclass
class DriftReport:
    """Result of a drift check against a saved baseline."""

    baseline_version: str
    current_scores: dict[str, float]
    per_dimension_deltas: dict[str, float]
    alert_triggered: bool
    degraded_dimensions: list[str]
    threshold: float


def _average_scores(results: list[RAGEvalResult]) -> dict[str, float]:
    """Compute mean over each metric dimension across a result list."""
    if not results:
        return {d: 0.0 for d in _METRIC_DIMENSIONS}
    return {
        dim: sum(getattr(r, dim, 0.0) for r in results) / len(results)
        for dim in _METRIC_DIMENSIONS
    }


class DriftDetector:
    """
    Saves baselines and detects metric regressions across deployments.

    Usage::

        detector = DriftDetector()
        detector.save_baseline("v1.0", eval_results)

        # After a new deployment:
        report = detector.detect_drift(new_results, threshold=0.05)
        if report.alert_triggered:
            notify_oncall(report)
    """

    def __init__(self, baseline_path: str | Path = "data/drift_baseline.json") -> None:
        self._path = Path(baseline_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)

    # ── Persistence ─────────────────────────────────────────────────────────

    def save_baseline(self, version: str, results: list[RAGEvalResult]) -> None:
        """Compute average scores and persist as the current baseline."""
        baseline = DriftBaseline(
            prompt_version=version,
            dimension_scores=_average_scores(results),
        )
        # Load existing baselines or start fresh
        all_baselines: dict = {}
        if self._path.exists():
            try:
                all_baselines = json.loads(self._path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                all_baselines = {}

        all_baselines[version] = {
            "prompt_version": baseline.prompt_version,
            "dimension_scores": baseline.dimension_scores,
            "created_at": baseline.created_at,
        }
        self._path.write_text(
            json.dumps(all_baselines, indent=2), encoding="utf-8"
        )
        logger.info(
            "baseline_saved",
            version=version,
            scores={k: round(v, 3) for k, v in baseline.dimension_scores.items()},
        )

    def load_baseline(self, version: str | None = None) -> DriftBaseline | None:
        """Load a named baseline (or the most recently saved one if None)."""
        if not self._path.exists():
            return None
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None

        if not data:
            return None

        if version is None:
            # Pick the most recently created
            version = max(data, key=lambda v: data[v].get("created_at", ""))

        entry = data.get(version)
        if entry is None:
            return None

        return DriftBaseline(
            prompt_version=entry["prompt_version"],
            dimension_scores=entry["dimension_scores"],
            created_at=entry.get("created_at", ""),
        )

    # ── Detection ────────────────────────────────────────────────────────────

    def detect_drift(
        self,
        current: list[RAGEvalResult],
        version: str | None = None,
        threshold: float = 0.05,
    ) -> DriftReport:
        """
        Compare current results against a saved baseline.

        Returns a ``DriftReport``.  ``alert_triggered`` is True when any
        metric dimension has dropped by more than ``threshold``.
        """
        baseline = self.load_baseline(version)
        if baseline is None:
            # No baseline exists yet — auto-save this run and report stable
            logger.warning("drift_no_baseline", action="auto_saving_as_baseline")
            tmp_version = version or "auto"
            self.save_baseline(tmp_version, current)
            current_scores = _average_scores(current)
            return DriftReport(
                baseline_version=tmp_version,
                current_scores=current_scores,
                per_dimension_deltas={d: 0.0 for d in _METRIC_DIMENSIONS},
                alert_triggered=False,
                degraded_dimensions=[],
                threshold=threshold,
            )

        current_scores = _average_scores(current)
        deltas: dict[str, float] = {}
        degraded: list[str] = []

        for dim in _METRIC_DIMENSIONS:
            baseline_score = baseline.dimension_scores.get(dim, 0.0)
            current_score = current_scores.get(dim, 0.0)
            delta = current_score - baseline_score
            deltas[dim] = round(delta, 4)
            if delta < -threshold:
                degraded.append(dim)

        alert = len(degraded) > 0

        if alert:
            logger.warning(
                "drift_alert",
                baseline_version=baseline.prompt_version,
                degraded_dimensions=degraded,
                deltas={k: round(v, 3) for k, v in deltas.items()},
                threshold=threshold,
            )
        else:
            logger.info(
                "drift_check_passed",
                baseline_version=baseline.prompt_version,
                deltas={k: round(v, 3) for k, v in deltas.items()},
            )

        return DriftReport(
            baseline_version=baseline.prompt_version,
            current_scores=current_scores,
            per_dimension_deltas=deltas,
            alert_triggered=alert,
            degraded_dimensions=degraded,
            threshold=threshold,
        )
