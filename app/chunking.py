"""Four chunking strategies, run in parallel over the corpus and all indexed
together (each chunk tagged with its strategy + metadata). This is the
'vast chunking' requirement: rather than betting on one splitter, we let
retrieval fuse across all of them.

Strategies:
  1. FIXED       — fixed token window with overlap. Cheap recall floor.
  2. SEMANTIC     — split on embedding-similarity breakpoints between
                    sentences, so a chunk boundary falls where topic
                    actually shifts rather than at a token count.
  3. PASSAGE      — respects MSMARCO's native passage boundaries; don't
                    re-split what's already a coherent retrieval unit.
  4. HIERARCHICAL — small child chunks for precise matching, each pointing
                    at a larger parent chunk that's substituted in at
                    generation time for more context ("small-to-big").
"""
from __future__ import annotations

import re
import uuid
from typing import Iterable

import numpy as np

from app.schemas import Chunk, ChunkStrategy

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?।])\s+")  # includes Hindi danda ।


def _sentences(text: str) -> list[str]:
    sents = [s.strip() for s in _SENTENCE_SPLIT_RE.split(text) if s.strip()]
    return sents or [text.strip()]


def _tok_count(text: str) -> int:
    return max(1, len(text.split()))


def chunk_fixed_overlap(
    doc_id: str, text: str, window_tokens: int = 220, overlap_ratio: float = 0.2
) -> list[Chunk]:
    words = text.split()
    step = max(1, int(window_tokens * (1 - overlap_ratio)))
    chunks: list[Chunk] = []
    pos = 0
    for i, start in enumerate(range(0, len(words), step)):
        window = words[start : start + window_tokens]
        if not window:
            break
        chunk_text = " ".join(window)
        chunks.append(
            Chunk(
                chunk_id=str(uuid.uuid4()),
                doc_id=doc_id,
                text=chunk_text,
                strategy=ChunkStrategy.FIXED,
                position=pos,
                token_count=_tok_count(chunk_text),
            )
        )
        pos += 1
        if start + window_tokens >= len(words):
            break
    return chunks


def chunk_semantic(
    doc_id: str, text: str, embed_fn, similarity_drop_threshold: float = 0.35
) -> list[Chunk]:
    """Group consecutive sentences until embedding similarity between the
    running group and the next sentence drops below threshold, i.e. a
    topic-shift boundary. `embed_fn` is injected so this module has no
    hard dependency on a specific embedding backend.
    """
    sents = _sentences(text)
    if len(sents) <= 1:
        return [
            Chunk(
                chunk_id=str(uuid.uuid4()),
                doc_id=doc_id,
                text=text,
                strategy=ChunkStrategy.SEMANTIC,
                position=0,
                token_count=_tok_count(text),
            )
        ]

    embeddings = embed_fn(sents)
    chunks: list[Chunk] = []
    current = [sents[0]]
    current_vec = embeddings[0]
    pos = 0

    def cos_sim(a, b) -> float:
        denom = (np.linalg.norm(a) * np.linalg.norm(b)) or 1e-9
        return float(np.dot(a, b) / denom)

    for sent, vec in zip(sents[1:], embeddings[1:]):
        sim = cos_sim(current_vec, vec)
        if sim < similarity_drop_threshold:
            chunk_text = " ".join(current)
            chunks.append(
                Chunk(
                    chunk_id=str(uuid.uuid4()),
                    doc_id=doc_id,
                    text=chunk_text,
                    strategy=ChunkStrategy.SEMANTIC,
                    position=pos,
                    token_count=_tok_count(chunk_text),
                )
            )
            pos += 1
            current = [sent]
            current_vec = vec
        else:
            current.append(sent)
            current_vec = (current_vec + vec) / 2.0

    if current:
        chunk_text = " ".join(current)
        chunks.append(
            Chunk(
                chunk_id=str(uuid.uuid4()),
                doc_id=doc_id,
                text=chunk_text,
                strategy=ChunkStrategy.SEMANTIC,
                position=pos,
                token_count=_tok_count(chunk_text),
            )
        )
    return chunks


def chunk_passage_native(doc_id: str, passages: Iterable[str]) -> list[Chunk]:
    """MSMARCO-XI is already structured as (query, passage) pairs — treat
    each provided passage as one coherent chunk instead of re-splitting it.
    """
    chunks = []
    for i, passage in enumerate(passages):
        passage = passage.strip()
        if not passage:
            continue
        chunks.append(
            Chunk(
                chunk_id=str(uuid.uuid4()),
                doc_id=doc_id,
                text=passage,
                strategy=ChunkStrategy.PASSAGE,
                position=i,
                token_count=_tok_count(passage),
            )
        )
    return chunks


def chunk_hierarchical(
    doc_id: str, text: str, child_tokens: int = 90, parent_tokens: int = 300
) -> list[Chunk]:
    """Small child chunks (precise embedding match) each pointing at a
    larger parent chunk (substituted in at generation time for context).
    Parents are also emitted as Chunk objects so the vector store /
    retrieval layer can dereference parent_id -> full parent text.
    """
    words = text.split()
    chunks: list[Chunk] = []

    parents: list[tuple[str, str]] = []  # (parent_id, parent_text)
    for p_start in range(0, len(words), parent_tokens):
        p_words = words[p_start : p_start + parent_tokens]
        if not p_words:
            continue
        parent_id = str(uuid.uuid4())
        parent_text = " ".join(p_words)
        parents.append((parent_id, parent_text))
        chunks.append(
            Chunk(
                chunk_id=parent_id,
                doc_id=doc_id,
                text=parent_text,
                strategy=ChunkStrategy.HIERARCHICAL,
                position=p_start // parent_tokens,
                token_count=_tok_count(parent_text),
            )
        )

    pos = 0
    for parent_id, parent_text in parents:
        p_words = parent_text.split()
        for c_start in range(0, len(p_words), child_tokens):
            c_words = p_words[c_start : c_start + child_tokens]
            if not c_words:
                continue
            child_text = " ".join(c_words)
            chunks.append(
                Chunk(
                    chunk_id=str(uuid.uuid4()),
                    doc_id=doc_id,
                    parent_id=parent_id,
                    text=child_text,
                    strategy=ChunkStrategy.HIERARCHICAL,
                    position=pos,
                    token_count=_tok_count(child_text),
                )
            )
            pos += 1
    return chunks


def chunk_all_strategies(
    doc_id: str, text: str, embed_fn, passages: Iterable[str] | None = None
) -> list[Chunk]:
    """Run every strategy over one document and return the union. Retrieval
    fuses across all of them at query time (see app/retrieval.py).
    """
    out: list[Chunk] = []
    out += chunk_fixed_overlap(doc_id, text)
    out += chunk_semantic(doc_id, text, embed_fn)
    out += chunk_hierarchical(doc_id, text)
    if passages:
        out += chunk_passage_native(doc_id, passages)
    return out
