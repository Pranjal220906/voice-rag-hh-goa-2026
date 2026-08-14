"""Thin wrapper around a multilingual sentence-embedding model. Kept as a
single choke point so the model can be swapped (e.g. for a distilled/faster
model to hit the latency budget) without touching chunking/retrieval code.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np

from app.config import settings


@lru_cache(maxsize=1)
def _model():
    # Imported lazily so environments that only run the API (and not
    # index-building) don't pay the import cost.
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(settings.embedding_model)


def embed_texts(texts: list[str]) -> np.ndarray:
    if not texts:
        return np.zeros((0, settings.embedding_dim), dtype=np.float32)
    vecs = _model().encode(
        texts,
        normalize_embeddings=True,
        batch_size=32,
        show_progress_bar=False,
    )
    return np.asarray(vecs, dtype=np.float32)


def embed_query(text: str) -> np.ndarray:
    # e5-family models expect a "query:" / "passage:" prefix convention.
    return embed_texts([f"query: {text}"])[0]


def embed_passages(texts: list[str]) -> np.ndarray:
    return embed_texts([f"passage: {t}" for t in texts])
