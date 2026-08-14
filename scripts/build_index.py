"""Loads ai4bharat/MSMARCO-XI, runs every chunking strategy over each
document, embeds all resulting chunks, and upserts them into Qdrant. Also
builds the BM25 sparse index and the corpus centroid used by the off-topic
guardrail — both saved to disk so the API process can load them without
re-embedding the whole corpus on startup.

Usage:
    python scripts/build_index.py --limit 20000 --split train
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
from datasets import load_dataset

sys.path.append(str(Path(__file__).resolve().parents[1]))

from app.chunking import chunk_all_strategies  # noqa: E402
from app.embeddings import embed_passages  # noqa: E402
from app.guardrails import set_corpus_centroid  # noqa: E402
from app.retrieval import build_bm25_index  # noqa: E402
from app.vector_store import upsert_chunks  # noqa: E402

ARTIFACT_DIR = Path(__file__).resolve().parents[1] / "data"


def main(dataset: str, split: str, limit: int, batch_size: int) -> None:
    print(f"Loading {dataset} [{split}] (streaming, limit={limit})...")
    ds = load_dataset(dataset, split=split, streaming=True)

    all_chunks = []
    for i, row in enumerate(ds):
        if i >= limit:
            break
        doc_id = str(row.get("id", i))
        # MSMARCO-style rows typically carry a query + a list of passages;
        # field names vary by config, so this is defensive.
        passages = row.get("passages") or row.get("passage_text") or []
        if isinstance(passages, dict):
            passages = passages.get("passage_text", [])
        text = " ".join(passages) if passages else str(row.get("passage", ""))
        if not text.strip():
            continue

        chunks = chunk_all_strategies(
            doc_id, text, embed_fn=lambda ts: embed_passages(ts), passages=passages
        )
        all_chunks.extend(chunks)

        if (i + 1) % 500 == 0:
            print(f"  processed {i + 1} documents, {len(all_chunks)} chunks so far")

    print(f"Total chunks across all strategies: {len(all_chunks)}")

    print("Embedding and upserting into Qdrant in batches...")
    all_vectors = []
    for start in range(0, len(all_chunks), batch_size):
        batch = all_chunks[start : start + batch_size]
        vecs = embed_passages([c.text for c in batch])
        upsert_chunks(batch, vecs)
        all_vectors.append(vecs)
        print(f"  indexed {min(start + batch_size, len(all_chunks))}/{len(all_chunks)}")

    print("Building BM25 sparse index...")
    build_bm25_index(all_chunks)
    ARTIFACT_DIR.mkdir(exist_ok=True)
    with open(ARTIFACT_DIR / "bm25_chunks.pkl", "wb") as f:
        pickle.dump(all_chunks, f)

    print("Computing corpus centroid for off-topic guardrail...")
    stacked = np.vstack(all_vectors) if all_vectors else np.zeros((1, 768))
    centroid = stacked.mean(axis=0)
    np.save(ARTIFACT_DIR / "corpus_centroid.npy", centroid)
    set_corpus_centroid(centroid)

    print("Done. Index ready.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="ai4bharat/MSMARCO-XI")
    parser.add_argument("--split", default="train")
    parser.add_argument("--limit", type=int, default=20000)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()
    main(args.dataset, args.split, args.limit, args.batch_size)
