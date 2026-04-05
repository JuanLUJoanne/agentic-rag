"""
Request deduplication for identical concurrent queries.

RequestDeduplicator.get_or_run(key, coro_factory):
  - If a request with the same key is already in flight, awaits the
    existing Future (shares the result).
  - Otherwise runs coro_factory(), stores the result, and cleans up.
  - Thread-safe via asyncio.Lock.

get_deduplicator() — process-wide singleton.
"""
from __future__ import annotations

import asyncio
from typing import Any, Callable, Coroutine

_instance: RequestDeduplicator | None = None


class RequestDeduplicator:
    """Coalesce concurrent identical requests into a single execution."""

    def __init__(self) -> None:
        self._inflight: dict[str, asyncio.Future] = {}
        self._lock = asyncio.Lock()

    async def get_or_run(
        self,
        key: str,
        coro_factory: Callable[[], Coroutine[Any, Any, Any]],
    ) -> Any:
        """
        Execute coro_factory() for the first caller; subsequent callers
        with the same key share the result via a Future.
        """
        async with self._lock:
            if key in self._inflight:
                # Another coroutine is already running — wait for it
                future = self._inflight[key]
                is_owner = False
            else:
                # We are the first caller — create the Future
                loop = asyncio.get_event_loop()
                future = loop.create_future()
                self._inflight[key] = future
                is_owner = True

        if not is_owner:
            return await asyncio.shield(future)

        # Owner: execute the coroutine and settle the future
        try:
            result = await coro_factory()
            future.set_result(result)
            return result
        except Exception as exc:
            future.set_exception(exc)
            raise
        finally:
            async with self._lock:
                self._inflight.pop(key, None)


def get_deduplicator() -> RequestDeduplicator:
    """Return the process-wide RequestDeduplicator singleton."""
    global _instance
    if _instance is None:
        _instance = RequestDeduplicator()
    return _instance
