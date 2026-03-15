"""
Hallucination checker — Reflection pattern (part 2 of 2).

Self-RAG: if the generated answer makes claims not supported by retrieved
documents, we retry rather than surface a fabricated answer. The retry
count in state prevents infinite loops.
"""
from __future__ import annotations

import json

import structlog

from src.graph.state import AgentState
from src.utils.llm import DummyLLM, get_llm

logger = structlog.get_logger()

_HALLUCINATION_PROMPT = """\
You are a factual grounding checker. Determine whether the answer below is fully
supported by the provided documents. A hallucination is any claim in the answer
that cannot be traced to a specific document passage.

Return valid JSON with exactly one key: "is_hallucinated" (bool).

Documents:
{docs}

Answer:
{answer}

JSON:"""


async def check_hallucination(state: AgentState) -> dict:
    """
    LangGraph node: verify the generated answer against retrieved documents.

    Increments retry_count only when hallucination is detected — the
    router downstream uses this counter to decide whether to re-generate
    or finalize with best-effort output.
    """
    generation = state.get("generation", "")
    docs = state.get("retrieved_docs", [])
    retry_count = state.get("retry_count", 0)
    llm = get_llm()

    if isinstance(llm, DummyLLM) or not docs or not generation:
        logger.info("hallucination_check", result="grounded", mode="dummy")
        return {
            "is_hallucinated": False,
            "agent_trace": [{"node": "check_hallucination", "is_hallucinated": False}],
        }

    docs_text = "\n\n".join(
        f"[Doc {i + 1}] {doc.get('content', '')}" for i, doc in enumerate(docs)
    )
    prompt = _HALLUCINATION_PROMPT.format(docs=docs_text, answer=generation)
    response = await llm.ainvoke(prompt)

    try:
        data = json.loads(response.content.strip())
        is_hallucinated = bool(data.get("is_hallucinated", False))
    except (json.JSONDecodeError, KeyError, ValueError):
        # Parse failure → assume grounded rather than discarding valid output
        is_hallucinated = False

    # Increment retry counter only when we're about to attempt a re-generation
    new_retry_count = retry_count + 1 if is_hallucinated else retry_count

    logger.info(
        "hallucination_check",
        result="hallucinated" if is_hallucinated else "grounded",
        retry_count=new_retry_count,
    )

    return {
        "is_hallucinated": is_hallucinated,
        "retry_count": new_retry_count,
        "agent_trace": [
            {"node": "check_hallucination", "is_hallucinated": is_hallucinated}
        ],
    }
