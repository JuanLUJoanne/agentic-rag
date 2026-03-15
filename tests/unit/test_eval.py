"""Unit tests for eval: RAGEvaluator, DriftDetector, ComparativeEvaluator."""
from __future__ import annotations

from src.eval.drift_detector import DriftDetector
from src.eval.ragas_eval import RAGEvalResult, RAGEvaluator

# ── RAGEvaluator ───────────────────────────────────────────────────────────────


def test_evaluate_single_returns_valid_result() -> None:
    evaluator = RAGEvaluator()
    result = evaluator.evaluate_single(
        query="What is machine learning?",
        answer="Machine learning is a branch of AI that enables computers to learn.",
        contexts=["Machine learning enables systems to learn from data."],
        ground_truth="Machine learning is a type of artificial intelligence.",
    )
    assert isinstance(result, RAGEvalResult)
    assert 0.0 <= result.faithfulness <= 1.0
    assert 0.0 <= result.answer_relevancy <= 1.0
    assert 0.0 <= result.context_precision <= 1.0
    assert 0.0 <= result.context_recall <= 1.0
    assert 0.0 <= result.citation_accuracy <= 1.0


def test_evaluate_single_no_contexts() -> None:
    """Evaluation without context should still return valid (partial) scores."""
    evaluator = RAGEvaluator()
    result = evaluator.evaluate_single("What is X?", "X is Y.", contexts=[])
    assert isinstance(result, RAGEvalResult)


def test_evaluate_single_empty_answer() -> None:
    """Empty answer should produce low scores but not crash."""
    evaluator = RAGEvaluator()
    result = evaluator.evaluate_single("q", "", contexts=["some context"])
    # faithfulness returns 0.5 (unknown) when answer is empty
    assert 0.0 <= result.faithfulness <= 0.5
    assert result.answer_relevancy == 0.0


def test_citation_accuracy_high_when_citations_present() -> None:
    evaluator = RAGEvaluator()
    result = evaluator.evaluate_single(
        "q", "The answer [1] is here.", contexts=["context"]
    )
    assert result.citation_accuracy == 1.0


def test_citation_accuracy_low_when_no_citations() -> None:
    evaluator = RAGEvaluator()
    result = evaluator.evaluate_single(
        "q", "The answer is here.", contexts=["context"]
    )
    assert result.citation_accuracy < 1.0


# ── DriftDetector ──────────────────────────────────────────────────────────────


def _make_result(score: float) -> RAGEvalResult:
    return RAGEvalResult(
        query="q",
        answer="a",
        faithfulness=score,
        answer_relevancy=score,
        context_precision=score,
        context_recall=score,
        citation_accuracy=score,
    )


def test_drift_detection_triggers_on_quality_drop(tmp_path) -> None:
    """A metric drop beyond threshold should trigger an alert."""
    detector = DriftDetector(baseline_path=tmp_path / "drift.json")

    # Save baseline with high scores
    baseline_results = [_make_result(0.9)]
    detector.save_baseline("v1", baseline_results)

    # Evaluate with significantly lower scores
    degraded_results = [_make_result(0.7)]  # 0.2 drop > threshold 0.05
    report = detector.detect_drift(degraded_results, version="v1", threshold=0.05)

    assert report.alert_triggered
    assert len(report.degraded_dimensions) > 0


def test_drift_passes_when_stable(tmp_path) -> None:
    """Small score fluctuations within the threshold should not alert."""
    detector = DriftDetector(baseline_path=tmp_path / "drift.json")
    baseline_results = [_make_result(0.8)]
    detector.save_baseline("v1", baseline_results)

    # Tiny variation — within threshold
    stable_results = [_make_result(0.81)]
    report = detector.detect_drift(stable_results, version="v1", threshold=0.05)

    assert not report.alert_triggered
    assert report.degraded_dimensions == []


def test_drift_no_baseline_auto_saves(tmp_path) -> None:
    """When no baseline exists, detect_drift should auto-save and return stable."""
    detector = DriftDetector(baseline_path=tmp_path / "drift.json")
    results = [_make_result(0.75)]
    report = detector.detect_drift(results, threshold=0.05)

    assert not report.alert_triggered
    # Baseline should now exist
    assert (tmp_path / "drift.json").exists()


def test_drift_report_has_all_dimensions(tmp_path) -> None:
    """DriftReport must contain deltas for all RAGAS metric dimensions."""
    detector = DriftDetector(baseline_path=tmp_path / "drift.json")
    results = [_make_result(0.8)]
    detector.save_baseline("v1", results)
    report = detector.detect_drift([_make_result(0.8)], version="v1")

    expected_dims = {
        "faithfulness", "answer_relevancy",
        "context_precision", "context_recall", "citation_accuracy",
    }
    assert set(report.per_dimension_deltas.keys()) >= expected_dims


# ── ComparativeEvaluator ───────────────────────────────────────────────────────


async def test_comparative_eval_runs_both_modes() -> None:
    """ComparativeEvaluator must produce results for simple and multi_agent."""
    from src.eval.comparative_eval import ComparativeEvaluator

    evaluator = ComparativeEvaluator()
    report = await evaluator.run(["What is RAG?"])

    assert len(report.simple_results) == 1
    assert len(report.multi_agent_results) == 1
    assert isinstance(report.simple_results[0], RAGEvalResult)
    assert isinstance(report.multi_agent_results[0], RAGEvalResult)


async def test_comparative_eval_has_per_metric_table() -> None:
    """Report must include a per_metric comparison dict."""
    from src.eval.comparative_eval import ComparativeEvaluator

    evaluator = ComparativeEvaluator()
    report = await evaluator.run(["What is machine learning?"])

    assert "faithfulness" in report.per_metric
    assert "simple" in report.per_metric["faithfulness"]
    assert "multi_agent" in report.per_metric["faithfulness"]
    assert "cost" in report.per_metric
    assert "latency_s" in report.per_metric


async def test_comparative_eval_has_winners() -> None:
    """Each metric must have a declared winner."""
    from src.eval.comparative_eval import ComparativeEvaluator

    evaluator = ComparativeEvaluator()
    report = await evaluator.run(["Explain transformers."])

    expected_metrics = {
        "faithfulness", "answer_relevancy",
        "context_precision", "context_recall", "citation_accuracy",
        "cost", "latency_s",
    }
    assert set(report.winner.keys()) >= expected_metrics
    for metric, winner in report.winner.items():
        assert winner in ("simple", "multi_agent"), f"unexpected winner {winner!r} for {metric}"
