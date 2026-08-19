"""Hybrid retrieval layer: dense (Qdrant) + sparse (BM25) search,
fused with Reciprocal Rank Fusion.

The multilingual embedding model is used for dense retrieval.
BM25 provides lexical matching. The cross-encoder reranker is
disabled because the previous English MS-MARCO reranker was
mis-ranking Hindi queries and pushing relevant Hindi passages down.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from rank_bm25 import BM25Okapi

from app.config import settings
from app.embeddings import embed_query
from app.schemas import Chunk, ChunkStrategy, RetrievedChunk
from app.vector_store import dense_search


@dataclass
class BM25Index:
    bm25: BM25Okapi
    chunks: list[Chunk]


_bm25_index: BM25Index | None = None


def build_bm25_index(chunks: list[Chunk]) -> None:
    """Build the BM25 sparse index once after indexing."""
    global _bm25_index

    tokenized = [c.text.lower().split() for c in chunks]
    _bm25_index = BM25Index(
        bm25=BM25Okapi(tokenized),
        chunks=chunks,
    )


def _sparse_search(
    query: str,
    top_k: int,
) -> list[tuple[Chunk, float]]:
    if _bm25_index is None:
        return []

    scores = _bm25_index.bm25.get_scores(
        query.lower().split()
    )

    ranked = sorted(
        zip(_bm25_index.chunks, scores),
        key=lambda x: x[1],
        reverse=True,
    )[:top_k]

    return ranked


def _reciprocal_rank_fusion(
    dense_ranked: list[tuple[str, float]],
    sparse_ranked: list[tuple[str, float]],
    k: int = 60,
) -> dict[str, float]:
    scores: dict[str, float] = {}

    for rank, (chunk_id, _) in enumerate(dense_ranked):
        scores[chunk_id] = (
            scores.get(chunk_id, 0.0)
            + 1.0 / (k + rank + 1)
        )

    for rank, (chunk_id, _) in enumerate(sparse_ranked):
        scores[chunk_id] = (
            scores.get(chunk_id, 0.0)
            + 1.0 / (k + rank + 1)
        )

    return scores


def rerank(
    query: str,
    candidates: list[RetrievedChunk],
    top_k: int,
) -> list[RetrievedChunk]:
    """
    Reranking is intentionally disabled.

    The previous reranker:
        cross-encoder/ms-marco-MiniLM-L-6-v2

    is English-focused and was incorrectly ranking Hindi passages.
    We therefore preserve the RRF ranking produced by the multilingual
    dense + sparse retrieval system.
    """
    return candidates[:top_k]


def retrieve(query: str) -> list[RetrievedChunk]:
    """
    Full hybrid retrieval:

        multilingual dense search
              +
        BM25 sparse search
              +
        Reciprocal Rank Fusion

    No English cross-encoder reranking is applied.
    """

    q_vec = embed_query(query)

    # Dense retrieval
    dense_hits = dense_search(
        q_vec,
        top_k=settings.top_k_dense,
    )

    dense_ranked = [
        (hit.id, hit.score)
        for hit in dense_hits
    ]

    dense_by_id = {
        hit.id: hit
        for hit in dense_hits
    }

    # Sparse retrieval
    sparse_hits = _sparse_search(
        query,
        top_k=settings.top_k_sparse,
    )

    sparse_ranked = [
        (c.chunk_id, score)
        for c, score in sparse_hits
    ]

    sparse_by_id = {
        c.chunk_id: (c, score)
        for c, score in sparse_hits
    }

    # RRF fusion
    fused = _reciprocal_rank_fusion(
        dense_ranked,
        sparse_ranked,
        k=settings.rrf_k,
    )

    top_ids = sorted(
        fused,
        key=fused.get,
        reverse=True,
    )[: settings.top_k_fused]

    candidates: list[RetrievedChunk] = []

    for chunk_id in top_ids:

        if chunk_id in dense_by_id:
            hit = dense_by_id[chunk_id]
            payload = hit.payload

            chunk = Chunk(
                chunk_id=chunk_id,
                doc_id=payload["doc_id"],
                parent_id=payload.get("parent_id"),
                text=payload["text"],
                strategy=ChunkStrategy(payload["strategy"]),
                position=payload["position"],
                token_count=payload["token_count"],
                language=payload.get("language"),
            )

            dense_score = hit.score

        else:
            chunk, _ = sparse_by_id[chunk_id]
            dense_score = 0.0

        sparse_score = sparse_by_id.get(
            chunk_id,
            (None, 0.0),
        )[1]

        candidates.append(
            RetrievedChunk(
                chunk=chunk,
                dense_score=dense_score,
                sparse_score=sparse_score,
                fused_score=fused[chunk_id],
            )
        )

    return rerank(
        query,
        candidates,
        top_k=settings.top_k_final,
    )


def resolve_hierarchical_context(
    retrieved: list[RetrievedChunk],
) -> list[RetrievedChunk]:
    """
    Small-to-big retrieval.

    If a retrieved chunk is a hierarchical child,
    replace its text with the parent's fuller text
    while keeping the child's retrieval scores.
    """

    from app.vector_store import get_by_id

    resolved = []

    for rc in retrieved:

        if (
            rc.chunk.strategy == ChunkStrategy.HIERARCHICAL
            and rc.chunk.parent_id
        ):
            parent = get_by_id(
                rc.chunk.parent_id
            )

            if parent:
                rc.chunk.text = parent.payload["text"]

        resolved.append(rc)

    return resolved
