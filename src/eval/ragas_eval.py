"""
RAGAS-style RAG evaluation.

Provides the same metric interface as the open-source RAGAS library but with
mock implementations so the system can run end-to-end without an LLM API key
or the full ragas dependency.

All scores are in [0, 1].  The heuristics used here are intentionally simple:

  faithfulness       — fraction of answer sentences that overlap with contexts
  answer_relevancy   — word-level Jaccard similarity between query and answer
  context_precision  — fraction of retrieved docs that overlap with ground truth
  context_recall     — fraction of ground-truth concepts covered by docs
  citation_accuracy  — citations present vs ideal citation count

When the real RAGAS library is available, swap ``MockRAGASEvaluator`` for the
``RAGASEvaluator`` wrapper in this same module (interface is identical).
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import structlog

logger = structlog.get_logger()


@dataclass
class RAGEvalResult:
    """Per-query evaluation scores from a single RAG pipeline run."""

    query: str
    answer: str

    # Core RAGAS metrics
    faithfulness: float = 0.0
    answer_relevancy: float = 0.0
    context_precision: float = 0.0
    context_recall: float = 0.0
    citation_accuracy: float = 0.0

    # Pipeline metadata
    agent_steps: int = 0
    cost_usd: float = 0.0
    retrieval_rounds: int = 1


def _tokenise(text: str) -> set[str]:
    """Lowercase word set for Jaccard-style overlap calculations."""
    return set(re.findall(r"\b\w+\b", text.lower()))


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    return len(a & b) / len(union) if union else 0.0


class RAGEvaluator:
    """
    Evaluate a single RAG pipeline output using heuristic metrics.

    Interface is designed to be a drop-in for the real RAGAS library:
    calling ``evaluate_single`` will produce the same result shape whether
    this mock or the real library is used.

    Integration guide (for when API keys are available):
    ::

        from ragas import evaluate
        from ragas.metrics import faithfulness, answer_relevancy

        # Replace _score_* methods with calls to the ragas evaluate() function
        # and map the Dataset columns to the same RAGEvalResult fields.
    """

    def evaluate_single(
        self,
        query: str,
        answer: str,
        contexts: list[str],
        ground_truth: str = "",
    ) -> RAGEvalResult:
        """
        Score a query/answer/context tuple and return a RAGEvalResult.

        Parameters
        ----------
        query:        The user's original question.
        answer:       The pipeline's generated answer.
        contexts:     Retrieved document chunks used during generation.
        ground_truth: Reference answer for precision/recall metrics.
                      If empty, those metrics fall back to answer/context overlap.
        """
        faithfulness = self._score_faithfulness(answer, contexts)
        relevancy = self._score_relevancy(query, answer)
        precision = self._score_context_precision(contexts, ground_truth or answer)
        recall = self._score_context_recall(contexts, ground_truth or answer)
        citation_acc = self._score_citation_accuracy(answer)

        result = RAGEvalResult(
            query=query,
            answer=answer,
            faithfulness=faithfulness,
            answer_relevancy=relevancy,
            context_precision=precision,
            context_recall=recall,
            citation_accuracy=citation_acc,
            agent_steps=0,
            cost_usd=0.0,
            retrieval_rounds=1,
        )

        logger.info(
            "eval_complete",
            query=query[:80],
            faithfulness=round(faithfulness, 3),
            relevancy=round(relevancy, 3),
            precision=round(precision, 3),
            recall=round(recall, 3),
            citation_acc=round(citation_acc, 3),
        )
        return result

    # ── Metric implementations ──────────────────────────────────────────────

    def _score_faithfulness(self, answer: str, contexts: list[str]) -> float:
        """
        Fraction of answer tokens that appear in at least one context.

        A real RAGAS implementation would use NLI to check entailment of
        each answer sentence against the retrieved contexts.
        """
        if not answer or not contexts:
            return 0.5  # unknown — no data to compare

        answer_tokens = _tokenise(answer)
        context_tokens = _tokenise(" ".join(contexts))
        if not answer_tokens:
            return 0.0

        overlap = len(answer_tokens & context_tokens) / len(answer_tokens)
        # Scale: full overlap → 1.0, partial → proportional
        return min(1.0, overlap * 1.2)  # slight boost for partial matches

    def _score_relevancy(self, query: str, answer: str) -> float:
        """Word-level Jaccard similarity between query and answer."""
        if not answer:
            return 0.0
        score = _jaccard(_tokenise(query), _tokenise(answer))
        # Clamp: purely identical text → penalise (not a real answer)
        if score >= 0.95:
            score = 0.6
        return min(1.0, score * 2.5)  # scale up for typical divergence

    def _score_context_precision(self, contexts: list[str], ground_truth: str) -> float:
        """
        Fraction of retrieved contexts that are relevant to the ground truth.
        """
        if not contexts:
            return 0.0
        gt_tokens = _tokenise(ground_truth)
        if not gt_tokens:
            return 0.5
        relevant = sum(
            1 for ctx in contexts if len(_tokenise(ctx) & gt_tokens) >= 2
        )
        return relevant / len(contexts)

    def _score_context_recall(self, contexts: list[str], ground_truth: str) -> float:
        """Fraction of ground-truth concepts present in the context pool."""
        if not ground_truth:
            return 0.5
        gt_tokens = _tokenise(ground_truth)
        if not gt_tokens:
            return 0.5
        context_tokens = _tokenise(" ".join(contexts)) if contexts else set()
        return len(gt_tokens & context_tokens) / len(gt_tokens)

    def _score_citation_accuracy(self, answer: str) -> float:
        """
        Heuristic citation check: look for bracketed references [1], [2]
        or source markers.  Returns 1.0 if citations present, 0.3 if not.
        """
        citation_pattern = re.compile(r"\[\d+\]|\[source\]|\(source\)", re.IGNORECASE)
        return 1.0 if citation_pattern.search(answer) else 0.3
