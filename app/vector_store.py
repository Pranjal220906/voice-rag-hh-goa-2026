"""Qdrant wrapper: index creation, upsert, dense search with payload
metadata (strategy, parent_id, doc_id) so retrieval can filter/boost and
dereference hierarchical parents.
"""
from __future__ import annotations

from functools import lru_cache

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

from app.config import settings
from app.schemas import Chunk


@lru_cache(maxsize=1)
def client() -> QdrantClient:
    return QdrantClient(url=settings.qdrant_url)


def ensure_collection() -> None:
    c = client()
    existing = [col.name for col in c.get_collections().collections]
    if settings.qdrant_collection in existing:
        return
    c.create_collection(
        collection_name=settings.qdrant_collection,
        vectors_config=qm.VectorParams(
            size=settings.embedding_dim, distance=qm.Distance.COSINE
        ),
    )


def upsert_chunks(chunks: list[Chunk], vectors) -> None:
    ensure_collection()
    points = [
        qm.PointStruct(
            id=chunk.chunk_id,
            vector=vector.tolist(),
            payload={
                "doc_id": chunk.doc_id,
                "parent_id": chunk.parent_id,
                "text": chunk.text,
                "strategy": chunk.strategy.value,
                "position": chunk.position,
                "token_count": chunk.token_count,
                "language": chunk.language,
            },
        )
        for chunk, vector in zip(chunks, vectors)
    ]
    client().upsert(collection_name=settings.qdrant_collection, points=points)


def dense_search(query_vector, top_k: int, strategy_filter: str | None = None):
    q_filter = None
    if strategy_filter:
        q_filter = qm.Filter(
            must=[qm.FieldCondition(key="strategy", match=qm.MatchValue(value=strategy_filter))]
        )
    return client().search(
        collection_name=settings.qdrant_collection,
        query_vector=query_vector.tolist(),
        limit=top_k,
        query_filter=q_filter,
        with_payload=True,
    )


def get_by_id(chunk_id: str):
    result = client().retrieve(
        collection_name=settings.qdrant_collection, ids=[chunk_id], with_payload=True
    )
    return result[0] if result else None
