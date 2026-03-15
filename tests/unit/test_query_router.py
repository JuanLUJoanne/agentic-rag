"""Unit tests for the heuristic query router."""
from __future__ import annotations

import pytest

from src.agents.query_router import classify, query_router

# ── classify() pure-function tests ────────────────────────────────────────────


def test_simple_short_question():
    route = classify("What is LangGraph?")
    assert route.type == "simple"
    assert route.sub_queries == ["What is LangGraph?"]
    assert route.strategy == "dense"


def test_simple_returns_original_as_sub_query():
    route = classify("What is a vector database?")
    assert len(route.sub_queries) == 1
    assert route.sub_queries[0] == "What is a vector database?"


def test_complex_compare_keyword():
    route = classify("Compare RAG and fine-tuning for domain adaptation")
    assert route.type == "complex"
    assert route.strategy == "hybrid"


def test_complex_contrast_keyword():
    route = classify("What is the contrast between BM25 and dense retrieval?")
    assert route.type == "complex"


def test_complex_difference_keyword():
    route = classify("What is the difference between BM25 and dense retrieval?")
    assert route.type == "complex"


def test_complex_long_query():
    query = (
        "Explain how corrective RAG works and describe all the steps involved "
        "in the query rewriting and re-retrieval process in detail"
    )
    route = classify(query)
    assert route.type == "complex"


def test_ambiguous_single_word():
    route = classify("RAG")
    assert route.type == "ambiguous"


def test_ambiguous_two_words():
    route = classify("vector embeddings")
    assert route.type == "ambiguous"


def test_sub_queries_generated_for_conjunction():
    route = classify("What is BM25 and how does dense retrieval work")
    assert route.type == "complex"
    assert len(route.sub_queries) >= 2


def test_conjunction_split_content():
    route = classify("What is BM25 and how does dense retrieval work")
    # Both parts should be non-empty strings
    for sq in route.sub_queries:
        assert sq.strip()


def test_strategy_hybrid_for_complex():
    route = classify("Compare BM25 and dense retrieval methods for RAG")
    assert route.strategy == "hybrid"


def test_strategy_dense_for_simple():
    route = classify("What is RAG?")
    assert route.strategy == "dense"


# ── query_router node test ─────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_query_router_node_simple():
    state = {
        "query": "What is LangGraph?",
        "chat_history": [],
        "query_type": None,
        "sub_queries": [],
        "retrieved_docs": [],
        "graph_context": [],
        "relevance_scores": [],
        "retrieval_strategy": "",
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
    result = await query_router(state)
    assert result["query_type"] == "simple"
    assert len(result["sub_queries"]) >= 1
    assert result["retrieval_strategy"] == "dense"


@pytest.mark.asyncio
async def test_query_router_node_complex():
    state = {
        "query": "Compare BM25 and dense retrieval for long-tail queries",
        "chat_history": [],
        "query_type": None,
        "sub_queries": [],
        "retrieved_docs": [],
        "graph_context": [],
        "relevance_scores": [],
        "retrieval_strategy": "",
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
    result = await query_router(state)
    assert result["query_type"] == "complex"
    assert result["retrieval_strategy"] == "hybrid"
