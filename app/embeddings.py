"""Thin wrapper around the multilingual sentence-embedding model."""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from app.config import settings


@lru_cache(maxsize=1)
def _model():
    print("[EMBED] START MODEL LOAD", flush=True)

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(settings.embedding_model)

    print("[EMBED] MODEL LOADED", flush=True)

    return model


def embed_texts(texts: list[str]) -> np.ndarray:
    if not texts:
        return np.zeros(
            (0, settings.embedding_dim),
            dtype=np.float32,
        )

    print(
        f"[EMBED] START ENCODE: {len(texts)} text(s)",
        flush=True,
    )

    model = _model()

    print("[EMBED] GOT MODEL", flush=True)

    vecs = model.encode(
        texts,
        normalize_embeddings=True,
        batch_size=1,
        show_progress_bar=False,
        convert_to_numpy=True,
    )

    print("[EMBED] ENCODE COMPLETE", flush=True)

    return np.asarray(vecs, dtype=np.float32)


def embed_query(text: str) -> np.ndarray:
    print("[EMBED] QUERY EMBEDDING START", flush=True)

    result = embed_texts([f"query: {text}"])[0]

    print("[EMBED] QUERY EMBEDDING COMPLETE", flush=True)

    return result


def embed_passages(texts: list[str]) -> np.ndarray:
    return embed_texts(
        [f"passage: {t}" for t in texts]
    )
