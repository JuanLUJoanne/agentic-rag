"""Shared fixtures for unit tests."""
from __future__ import annotations

import pytest


@pytest.fixture
def sample_docs() -> list[dict]:
    return [
        {
            "id": "doc_1",
            "content": "LangGraph builds stateful, multi-actor LLM applications.",
            "source": "langgraph_docs",
            "score": 0.9,
        },
        {
            "id": "doc_2",
            "content": "RAG retrieves relevant context to ground language model generation.",
            "source": "rag_survey",
            "score": 0.8,
        },
    ]


@pytest.fixture
def base_state(sample_docs: list[dict]) -> dict:
    """Minimal valid AgentState dict for most unit tests."""
    return {
        "query": "What is LangGraph?",
        "chat_history": [],
        "query_type": "simple",
        "sub_queries": ["What is LangGraph?"],
        "retrieved_docs": sample_docs,
        "graph_context": [],
        "relevance_scores": [],
        "retrieval_strategy": "dense",
        "generation": "",
        "citations": [],
        "docs_relevant": None,
        "is_hallucinated": None,
        "answer_quality": None,
        "retry_count": 0,
        "max_retries": 2,
        "should_rewrite_query": False,
        "final_answer": None,
        "cost_so_far": 0.0,
        "agent_trace": [],
    }
