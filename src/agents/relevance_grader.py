"""
Relevance grader — Reflection pattern (part 1 of 2).

Grading documents before generation prevents the model from hallucinating
details from off-topic documents it was forced to attend to. A document
the model never sees cannot contaminate the answer.
"""
from __future__ import annotations

import json
from typing import Literal

import structlog

from src.graph.state import AgentState
from src.utils.llm import DummyLLM, get_llm

logger = structlog.get_logger()

_GRADE_PROMPT = """\
You are a strict relevance grader. Score the document's relevance to the query.
Return valid JSON with exactly two keys: "relevant" (bool) and "score" (float 0.0–1.0).

Query: {query}

Document:
{content}

JSON:"""


async def _grade_single_doc(llm, query: str, doc: dict) -> tuple[bool, float]:
    """
    Grade one document.

    Returns (is_relevant, confidence_score). On parse failure we treat the
    document as relevant to avoid silently discarding valid context.
    """
    if isinstance(llm, DummyLLM):
        return True, 0.8

    prompt = _GRADE_PROMPT.format(query=query, content=doc.get("content", "")[:500])
    response = await llm.ainvoke(prompt)

    try:
        data = json.loads(response.content.strip())
        return bool(data.get("relevant", True)), float(data.get("score", 0.5))
    except (json.JSONDecodeError, KeyError, ValueError):
        # Treat parse failure as relevant — better to over-retrieve than under-retrieve
        return True, 0.5


def _aggregate_status(scores: list[float]) -> Literal["all_relevant", "partial", "none"]:
    """Bucket a list of relevance scores into a routing signal."""
    if not scores:
        return "none"
    relevant_count = sum(1 for s in scores if s >= 0.5)
    if relevant_count == 0:
        return "none"
    if relevant_count == len(scores):
        return "all_relevant"
    return "partial"


async def grade_documents(state: AgentState) -> dict:
    """
    LangGraph node: grade retrieved docs and route based on quality.

    Filtering irrelevant documents here means the generator sees only
    signal, not noise. The aggregated status drives the conditional edge:
    all_relevant → generate, partial → rewrite, none → web_search.
    """
    docs = state.get("retrieved_docs", [])
    query = state["query"]
    llm = get_llm()

    if not docs:
        logger.info("grading_complete", status="none", doc_count=0)
        return {
            "docs_relevant": "none",
            "relevance_scores": [],
            "retrieved_docs": [],
            "agent_trace": [{"node": "grade_documents", "status": "none"}],
        }

    grades: list[tuple[dict, bool, float]] = []
    for doc in docs:
        is_relevant, score = await _grade_single_doc(llm, query, doc)
        grades.append((doc, is_relevant, score))

    filtered_docs = [doc for doc, relevant, _ in grades if relevant]
    scores = [score for _, _, score in grades]

    # Derive status from the filtered set, not raw scores
    if not filtered_docs:
        status: Literal["all_relevant", "partial", "none"] = "none"
    elif len(filtered_docs) == len(docs):
        status = "all_relevant"
    else:
        status = "partial"

    logger.info(
        "grading_complete",
        status=status,
        total_docs=len(docs),
        relevant_docs=len(filtered_docs),
    )

    return {
        "docs_relevant": status,
        "relevance_scores": scores,
        "retrieved_docs": filtered_docs,
        "agent_trace": [
            {"node": "grade_documents", "status": status, "relevant_count": len(filtered_docs)}
        ],
    }
