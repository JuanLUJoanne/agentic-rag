"""
Multi-granularity document chunker.

Produces sentence-level and paragraph-level chunks from raw document text,
linking each sentence chunk to its parent paragraph chunk via parent_id.

Strategy — two granularities, not three:
  sentence-level  → precise retrieval signal (small, focused)
  paragraph-level → sufficient context for the LLM (no document-level noise)

Document-level is intentionally omitted: it adds little recall beyond
paragraph-level and dilutes the precision gained by sentence retrieval.

Storage layout::

    paragraph chunk:  chunk_id="para_{doc_id}_{p_idx}"   parent_id=None
    sentence chunk:   chunk_id="sent_{doc_id}_{p_idx}_{s_idx}"
                      parent_id="para_{doc_id}_{p_idx}"

A single dict lookup (sentence_id → paragraph content) expands a sentence
hit to its parent paragraph at query time — no second index scan needed.

Sentence splitting uses a regex lookbehind on [.!?] followed by whitespace.
This is intentionally simple: it works well for clean corpus text and avoids
requiring NLTK as a hard dependency. For corpora with abbreviations (Dr.,
e.g., etc.) replace _split_sentences with nltk.sent_tokenize.
"""
from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class Chunk:
    chunk_id: str
    content: str
    granularity: str        # "sentence" | "paragraph"
    source_doc_id: str
    parent_id: str | None   # sentence → paragraph chunk_id; paragraph → None


# Split on whitespace that immediately follows a sentence-ending punctuation mark.
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")

# Ignore fragments below this word count — avoids indexing sentence stubs.
_MIN_SENTENCE_TOKENS = 3


def _split_sentences(text: str) -> list[str]:
    parts = _SENTENCE_RE.split(text.strip())
    return [p.strip() for p in parts if len(p.split()) >= _MIN_SENTENCE_TOKENS]


def _split_paragraphs(text: str) -> list[str]:
    """Split on blank lines; return the full text as one block when none exist."""
    blocks = [b.strip() for b in re.split(r"\n\s*\n", text) if b.strip()]
    return blocks if blocks else [text.strip()]


def chunk_document(doc_id: str, text: str) -> list[Chunk]:
    """
    Return sentence + paragraph :class:`Chunk` objects for *text*.

    Paragraphs are returned first, then their child sentences — so a caller
    can build the paragraph store in a single pass without two passes.

    Example (two-sentence single-paragraph doc)::

        chunks = chunk_document("doc_01", "BERT was introduced in 2018. It changed NLP.")
        # → Chunk(chunk_id="para_doc_01_0", granularity="paragraph", parent_id=None)
        # → Chunk(chunk_id="sent_doc_01_0_0", granularity="sentence", parent_id="para_doc_01_0")
        # → Chunk(chunk_id="sent_doc_01_0_1", granularity="sentence", parent_id="para_doc_01_0")
    """
    chunks: list[Chunk] = []
    paragraphs = _split_paragraphs(text)

    for p_idx, para_text in enumerate(paragraphs):
        para_id = f"para_{doc_id}_{p_idx}"
        chunks.append(
            Chunk(
                chunk_id=para_id,
                content=para_text,
                granularity="paragraph",
                source_doc_id=doc_id,
                parent_id=None,
            )
        )
        for s_idx, sent_text in enumerate(_split_sentences(para_text)):
            chunks.append(
                Chunk(
                    chunk_id=f"sent_{doc_id}_{p_idx}_{s_idx}",
                    content=sent_text,
                    granularity="sentence",
                    source_doc_id=doc_id,
                    parent_id=para_id,
                )
            )

    return chunks
