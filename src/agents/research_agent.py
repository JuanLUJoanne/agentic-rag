"""
Research Agent — Tool Use pattern (specialist implementation).

Wraps the ParallelRetriever + relevance grader into a single agent skill.
Checks QueryMemory first: if a high-quality answer already exists for this
query, skip retrieval entirely and surface the cached answer directly.
"""
from __future__ import annotations

import structlog

from src.agents.base import BaseAgent
from src.agents.relevance_grader import grade_documents
from src.agents.retriever import retrieve
from src.graph.state import AgentState
from src.retrieval.memory import get_default_memory

logger = structlog.get_logger()


class ResearchAgent(BaseAgent):
    name = "research"
    skills = ["document_retrieval", "web_search", "query_decomposition", "graph_traversal"]
    description = "Finds and retrieves relevant documents using hybrid search"

    async def execute(self, state: AgentState) -> dict:
        """
        Check QueryMemory, then run ParallelRetriever + relevance grader.

        Memory-first design: if we already answered this exact query with
        high confidence, returning the cached answer skips three LLM calls
        (retrieval grading, generation, hallucination check) and their cost.
        """
        memory = get_default_memory()
        memory_result = await memory.recall(state["query"])

        if memory_result:
            logger.info(
                "research_agent_memory_hit",
                query=state["query"][:80],
                score=memory_result.eval_score,
            )
            return {
                "generation": memory_result.answer,
                "citations": memory_result.citations,
                "final_answer": memory_result.answer,
                "retrieved_docs": [],
                "docs_relevant": "all_relevant",
                "retrieval_strategy": "memory",
                "agent_trace": [
                    {
                        "node": "research_agent",
                        "source": "memory",
                        "score": memory_result.eval_score,
                    }
                ],
            }

        # ── Normal retrieval path ──────────────────────────────────────────────
        retrieval_result = await retrieve(state)

        grader_state = {**state, **retrieval_result}
        grading_result = await grade_documents(grader_state)

        docs_found = len(grading_result.get("retrieved_docs", []))
        logger.info(
            "research_agent_complete",
            docs_found=docs_found,
            docs_relevant=grading_result.get("docs_relevant"),
        )

        return {
            "retrieved_docs": grading_result["retrieved_docs"],
            "relevance_scores": grading_result["relevance_scores"],
            "docs_relevant": grading_result["docs_relevant"],
            "retrieval_strategy": retrieval_result.get("retrieval_strategy", "parallel_simple"),
            "agent_trace": [{"node": "research_agent", "docs_found": docs_found}],
        }
