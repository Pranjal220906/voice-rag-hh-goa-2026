"""Lightweight multilingual E5 embedding wrapper."""

from __future__ import annotations

import os
from functools import lru_cache

import numpy as np

# Keep CPU memory/thread usage low on small Render instances.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

from app.config import settings


@lru_cache(maxsize=1)
def _model():
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(
        settings.embedding_model,
        device="cpu",
    )

    # Avoid unnecessary long sequence allocations.
    model.max_seq_length = 256

    return model


def embed_texts(texts: list[str]) -> np.ndarray:
    if not texts:
        return np.zeros((0, settings.embedding_dim), dtype=np.float32)

    vecs = _model().encode(
        texts,
        normalize_embeddings=True,
        batch_size=1,
        show_progress_bar=False,
        convert_to_numpy=True,
    )

    return np.asarray(vecs, dtype=np.float32)


def embed_query(text: str) -> np.ndarray:
    return embed_texts([f"query: {text}"])[0]


def embed_passages(texts: list[str]) -> np.ndarray:
    return embed_texts([f"passage: {t}" for t in texts])
