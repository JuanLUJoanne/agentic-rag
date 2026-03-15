"""
SSE streaming for RAG workflow progress.

Bridges LangGraph's ``astream(stream_mode="updates")`` to Server-Sent Events
so the client can render incremental progress without polling.

Each LangGraph node update is mapped to a ``StreamEvent`` whose ``event_type``
tells the client what happened:

  supervisor_decision — supervisor chose the next agent
  agent_start         — an agent node is about to execute
  agent_complete      — an agent node finished; data carries its outputs
  answer_chunk        — a generation fragment is ready
  error               — the workflow raised an unhandled exception
  done                — stream finished; data carries the full result

The caller is responsible for wrapping this generator in a FastAPI
``StreamingResponse(media_type="text/event-stream")``.
"""
from __future__ import annotations

import time
import uuid
from collections.abc import AsyncIterator
from typing import Literal

import structlog
from pydantic import BaseModel

logger = structlog.get_logger()

# ── Event model ────────────────────────────────────────────────────────────────


class StreamEvent(BaseModel):
    event_type: Literal[
        "supervisor_decision",
        "agent_start",
        "agent_complete",
        "answer_chunk",
        "error",
        "done",
    ]
    agent: str | None = None
    data: dict
    timestamp: float


# ── Node → event mapping ───────────────────────────────────────────────────────

_SUPERVISOR_NODES = {"supervisor"}
_SPECIALIST_AGENTS = {"research_agent", "analysis_agent", "quality_agent"}
_GENERATION_NODES = {"generate"}       # gate_generate is registered as "generate"
_ANSWER_NODES = {"finalize"}


def _node_to_events(node_name: str, state_delta: dict) -> list[StreamEvent]:
    """
    Map a single LangGraph node update to a list of StreamEvents.

    Returns an empty list for nodes that don't produce user-visible events
    (e.g. audit_log) to avoid cluttering the stream.
    """
    now = time.time()
    events: list[StreamEvent] = []

    if node_name in _SUPERVISOR_NODES:
        decision = state_delta.get("supervisor_decision") or {}
        events.append(
            StreamEvent(
                event_type="supervisor_decision",
                agent=node_name,
                data={"decision": decision},
                timestamp=now,
            )
        )
        return events

    if node_name in _SPECIALIST_AGENTS:
        events.append(
            StreamEvent(
                event_type="agent_complete",
                agent=node_name,
                data={
                    "docs_retrieved": len(state_delta.get("retrieved_docs", [])),
                    "generation": state_delta.get("generation", ""),
                    "quality": state_delta.get("answer_quality"),
                },
                timestamp=now,
            )
        )
        return events

    if node_name in _GENERATION_NODES:
        generation = state_delta.get("generation", "")
        if generation:
            events.append(
                StreamEvent(
                    event_type="answer_chunk",
                    agent=None,
                    data={"chunk": generation},
                    timestamp=now,
                )
            )
        # Also emit agent_complete so the client knows the node ran
        events.append(
            StreamEvent(
                event_type="agent_complete",
                agent=node_name,
                data={"has_generation": bool(generation)},
                timestamp=now,
            )
        )
        return events

    if node_name in _ANSWER_NODES:
        final_answer = state_delta.get("final_answer", "")
        if final_answer:
            events.append(
                StreamEvent(
                    event_type="answer_chunk",
                    agent=None,
                    data={"chunk": final_answer},
                    timestamp=now,
                )
            )
        events.append(
            StreamEvent(
                event_type="agent_complete",
                agent=node_name,
                data={"has_answer": bool(final_answer)},
                timestamp=now,
            )
        )
        return events

    if node_name == "audit_log":
        # Silent node — no event emitted
        return []

    # Default: agent_complete for all other nodes
    events.append(
        StreamEvent(
            event_type="agent_complete",
            agent=node_name,
            data={},
            timestamp=now,
        )
    )
    return events


# ── Public streaming function ──────────────────────────────────────────────────


async def stream_query(
    query: str,
    mode: str = "simple",
    config: dict | None = None,
) -> AsyncIterator[str]:
    """
    Async generator that runs a RAG workflow and yields SSE-formatted strings.

    Each yielded string is a complete SSE data line: ``data: <json>\\n\\n``.
    The final event always has ``event_type="done"`` and carries the full
    result summary.  On unhandled exceptions an ``event_type="error"`` event
    is emitted before the generator exits.

    Usage in FastAPI::

        return StreamingResponse(
            stream_query(request.query, request.mode),
            media_type="text/event-stream",
        )
    """
    # Import here to avoid circular imports at module load
    from src.graph.multi_agent_workflow import (
        get_initial_supervisor_state,
    )
    from src.graph.multi_agent_workflow import graph as multi_agent_graph
    from src.graph.simple_workflow import get_initial_state
    from src.graph.simple_workflow import graph as simple_graph

    thread_id = str(uuid.uuid4())
    cfg = config or {"configurable": {"thread_id": thread_id}}

    if mode == "multi_agent":
        initial_state = get_initial_supervisor_state(query, mode="multi_agent")
        g = multi_agent_graph
    else:
        initial_state = get_initial_state(query)
        g = simple_graph

    # Track result fields for the final done event
    final_answer: str = ""
    citations: list = []
    cost: float = 0.0
    agents_used: list[str] = []

    logger.info("stream_query_start", query=query[:80], mode=mode, thread_id=thread_id)

    try:
        async for update in g.astream(initial_state, config=cfg, stream_mode="updates"):
            for node_name, state_delta in update.items():
                # Accumulate result fields
                if state_delta.get("final_answer"):
                    final_answer = state_delta["final_answer"]
                if state_delta.get("citations"):
                    citations = state_delta["citations"]
                if "cost_so_far" in state_delta:
                    cost = state_delta["cost_so_far"]
                if state_delta.get("agents_called"):
                    agents_used.extend(state_delta["agents_called"])

                for event in _node_to_events(node_name, state_delta):
                    yield f"data: {event.model_dump_json()}\n\n"

    except Exception as exc:
        logger.error("stream_query_error", error=str(exc), query=query[:80])
        error_event = StreamEvent(
            event_type="error",
            agent=None,
            data={"error": str(exc)},
            timestamp=time.time(),
        )
        yield f"data: {error_event.model_dump_json()}\n\n"
        return

    done_event = StreamEvent(
        event_type="done",
        agent=None,
        data={
            "answer": final_answer,
            "citations": citations,
            "cost": cost,
            "agents_used": agents_used,
        },
        timestamp=time.time(),
    )
    yield f"data: {done_event.model_dump_json()}\n\n"
    logger.info("stream_query_done", query=query[:80], mode=mode)
