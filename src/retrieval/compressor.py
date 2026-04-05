"""
Contextual compressor — LLM-based sentence extraction.

Given a query and a retrieved document, asks an LLM to extract only the
sentences that are directly relevant to the query.  This shrinks the
context passed to the downstream answer-generation model, reducing token
cost and improving signal-to-noise.

Feature flag: ``COMPRESSION_ENABLED=true`` (default off).

DummyLLM guard: if ``OPENAI_API_KEY`` starts with a placeholder prefix
(sk-test, sk-xxx, sk-fake, sk-placeholder, sk-dummy) compression is
skipped and documents are returned unchanged — keeping CI green without
API keys.
"""
from __future__ import annotations

import asyncio
import os
from dataclasses import replace
from typing import Any

import structlog

from src.retrieval.models import SearchResult

logger = structlog.get_logger()

_PLACEHOLDER_PREFIXES = ("sk-xxx", "sk-test", "sk-fake", "sk-placeholder", "sk-dummy")

_COMPRESS_PROMPT = (
    'Extract only the sentences from the following document that are directly relevant to answering: "{query}"\n'
    "If nothing is relevant, return the first 2 sentences.\n"
    "Document: {content}"
)

_SEMAPHORE_SIZE = 3

_compressor_singleton: "ContextualCompressor | None" = None


def _is_dummy_mode() -> bool:
    api_key = os.getenv("OPENAI_API_KEY", "")
    return not api_key or any(api_key.startswith(p) for p in _PLACEHOLDER_PREFIXES)


class ContextualCompressor:
    """
    LLM-backed sentence extractor that reduces document context to only
    query-relevant sentences.

    Parameters
    ----------
    llm_client:
        Optional OpenAI ``AsyncOpenAI`` client.  If *None*, obtained from
        ``src/utils/llm.py`` via ``get_llm``.  Accepts any object with an
        ``async`` ``chat.completions.create`` method *or* a LangChain
        ``BaseChatModel`` with ``ainvoke``.
    """

    def __init__(self, llm_client: Any = None) -> None:
        self._client = llm_client  # None → resolved lazily

    def _get_client(self) -> Any:
        if self._client is None:
            from src.utils.llm import get_llm
            self._client = get_llm()
        return self._client

    async def compress(self, query: str, doc: SearchResult) -> SearchResult:
        """
        Return a new ``SearchResult`` whose ``content`` contains only the
        sentences relevant to *query*.

        If running in DummyLLM mode the original document is returned
        unchanged (no LLM call is made).
        """
        if _is_dummy_mode():
            logger.debug("compression_skipped_dummy_mode", doc_id=doc.doc_id)
            return doc

        prompt = _COMPRESS_PROMPT.format(query=query, content=doc.content)
        client = self._get_client()

        try:
            # Support both LangChain BaseChatModel (ainvoke) and raw OpenAI
            # async client (chat.completions.create).
            if hasattr(client, "ainvoke"):
                from langchain_core.messages import HumanMessage
                response = await client.ainvoke([HumanMessage(content=prompt)])
                compressed_text = response.content.strip()
            else:
                response = await client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                )
                compressed_text = response.choices[0].message.content.strip()
        except Exception as exc:
            logger.warning("compression_error", doc_id=doc.doc_id, error=str(exc))
            return doc

        new_metadata = {
            **doc.metadata,
            "compressed": True,
            "original_length": len(doc.content),
        }
        return SearchResult(
            doc_id=doc.doc_id,
            content=compressed_text,
            score=doc.score,
            source=doc.source,
            metadata=new_metadata,
        )

    async def compress_batch(
        self,
        query: str,
        docs: list[SearchResult],
    ) -> list[SearchResult]:
        """
        Compress all docs concurrently, with a semaphore of 3 to avoid
        hitting OpenAI rate limits.
        """
        sem = asyncio.Semaphore(_SEMAPHORE_SIZE)

        async def _guarded(doc: SearchResult) -> SearchResult:
            async with sem:
                return await self.compress(query, doc)

        return list(await asyncio.gather(*[_guarded(doc) for doc in docs]))


def get_compressor() -> ContextualCompressor:
    """Return the process-wide singleton ``ContextualCompressor``."""
    global _compressor_singleton
    if _compressor_singleton is None:
        _compressor_singleton = ContextualCompressor()
    return _compressor_singleton
