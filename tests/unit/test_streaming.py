"""Unit tests for SSE streaming."""
from __future__ import annotations

import json
from collections.abc import AsyncIterator

from src.api.streaming import StreamEvent, _node_to_events, stream_query

# ── StreamEvent model ──────────────────────────────────────────────────────────


def test_stream_event_serialises_to_json() -> None:
    """StreamEvent.model_dump_json() must produce valid JSON."""
    event = StreamEvent(
        event_type="agent_complete",
        agent="retrieve",
        data={"docs": 5},
        timestamp=1234567890.0,
    )
    payload = json.loads(event.model_dump_json())
    assert payload["event_type"] == "agent_complete"
    assert payload["agent"] == "retrieve"
    assert payload["data"]["docs"] == 5


def test_stream_event_none_agent() -> None:
    """agent field can be None for non-agent events."""
    event = StreamEvent(event_type="done", agent=None, data={}, timestamp=0.0)
    assert event.agent is None


# ── Node-to-event mapping ──────────────────────────────────────────────────────


def test_supervisor_node_maps_to_supervisor_decision() -> None:
    decision = {"next_agent": "research", "reasoning": "needs docs"}
    events = _node_to_events("supervisor", {"supervisor_decision": decision})
    assert len(events) == 1
    assert events[0].event_type == "supervisor_decision"
    assert events[0].data["decision"] == decision


def test_specialist_agent_maps_to_agent_complete() -> None:
    for node in ("research_agent", "analysis_agent", "quality_agent"):
        events = _node_to_events(node, {"retrieved_docs": [{"id": "1"}]})
        types = {e.event_type for e in events}
        assert "agent_complete" in types, f"{node} should emit agent_complete"


def test_generate_node_emits_answer_chunk() -> None:
    events = _node_to_events("generate", {"generation": "The answer is 42."})
    types = [e.event_type for e in events]
    assert "answer_chunk" in types


def test_finalize_node_emits_answer_chunk_when_answer_set() -> None:
    events = _node_to_events("finalize", {"final_answer": "Final answer here."})
    types = [e.event_type for e in events]
    assert "answer_chunk" in types


def test_audit_log_node_emits_no_events() -> None:
    """audit_log is a silent node — should produce no stream events."""
    events = _node_to_events("audit_log", {})
    assert events == []


def test_unknown_node_emits_agent_complete() -> None:
    events = _node_to_events("some_unknown_node", {})
    assert len(events) == 1
    assert events[0].event_type == "agent_complete"
    assert events[0].agent == "some_unknown_node"


# ── Full stream_query integration ──────────────────────────────────────────────


async def _collect_events(gen: AsyncIterator[str]) -> list[dict]:
    """Drain an SSE generator and parse all data lines."""
    events = []
    async for line in gen:
        if line.startswith("data: "):
            events.append(json.loads(line[6:]))
    return events


async def test_stream_ends_with_done_event() -> None:
    """The very last event in every stream must be type 'done'."""
    events = await _collect_events(stream_query("What is machine learning?"))
    assert len(events) > 0
    assert events[-1]["event_type"] == "done"


async def test_stream_done_event_has_answer() -> None:
    """The done event data must contain an 'answer' key."""
    events = await _collect_events(stream_query("What is machine learning?"))
    done = events[-1]
    assert "answer" in done["data"]


async def test_stream_produces_agent_complete_events() -> None:
    """Simple workflow must emit at least one agent_complete event."""
    events = await _collect_events(stream_query("Tell me about databases."))
    types = {e["event_type"] for e in events}
    assert "agent_complete" in types


async def test_stream_multi_agent_produces_supervisor_decision() -> None:
    """Multi-agent mode must emit a supervisor_decision event."""
    events = await _collect_events(
        stream_query("What is the difference between RAG and fine-tuning?", mode="multi_agent")
    )
    types = {e["event_type"] for e in events}
    assert "supervisor_decision" in types
    assert "done" in types


async def test_stream_sse_format_correct() -> None:
    """Every yielded string must start with 'data: ' and end with '\\n\\n'."""
    gen = stream_query("Hello?")
    count = 0
    async for chunk in gen:
        assert chunk.startswith("data: "), f"Bad SSE format: {chunk!r}"
        assert chunk.endswith("\n\n"), f"Missing SSE terminator: {chunk!r}"
        count += 1
    assert count > 0


async def test_error_mid_stream_emits_error_event(monkeypatch) -> None:
    """An exception inside the workflow must produce an 'error' event, not raise."""
    from src.graph import simple_workflow

    async def _crashing_stream(*args, **kwargs):
        # Yield nothing — just raise immediately
        raise RuntimeError("simulated workflow failure")
        yield  # make it a generator

    monkeypatch.setattr(simple_workflow.graph, "astream", _crashing_stream)

    events = await _collect_events(stream_query("test query"))
    assert any(e["event_type"] == "error" for e in events), (
        f"Expected error event in {[e['event_type'] for e in events]}"
    )
    # No 'done' event after an error
    assert events[-1]["event_type"] == "error"
