"""Loads ai4bharat/MSMARCO-XI (from a locally downloaded parquet file — HF's
streaming API proved unreliable on flaky connections, so we read a file
that's already sitting on disk via curl instead), runs every chunking
strategy over each document, embeds all resulting chunks, and upserts them
into Qdrant. Also builds the BM25 sparse index and the corpus centroid used
by the off-topic guardrail — both saved to disk so the API process can load
them without re-embedding the whole corpus on startup.

Usage:
    python scripts/build_index.py --local-file data/hinval.parquet --limit 500
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

sys.path.append(str(Path(__file__).resolve().parents[1]))

from app.chunking import chunk_all_strategies  # noqa: E402
from app.embeddings import embed_passages  # noqa: E402
from app.guardrails import set_corpus_centroid  # noqa: E402
from app.retrieval import build_bm25_index  # noqa: E402
from app.vector_store import upsert_chunks  # noqa: E402

ARTIFACT_DIR = Path(__file__).resolve().parents[1] / "data"

# MSMARCO-XI's actual schema (confirmed from the parquet file directly):
# query, Answer, Eng_Query, Eng_Answer, query_id, query_type,
# passages.English_passages / passages.Translated_passages / passages.is_selected
QUERY_COL = "query"
QUERY_ID_COL = "query_id"
PASSAGES_COL = "passages"  # struct column: {English_passages, Translated_passages, is_selected}


def iter_local_rows(path: str, limit: int, batch_size: int = 200):
    """Stream rows out of a local parquet file in batches, so we never load
    the whole (multi-GB) file into memory — only pull the columns we need.
    `passages` is a nested struct column; to_pylist() resolves it into a
    plain dict per row, so we pull Translated_passages (falling back to
    English_passages) out of that dict rather than guessing a dotted name.
    """
    pf = pq.ParquetFile(path)
    available = set(pf.schema_arrow.names)
    columns = [c for c in (QUERY_COL, QUERY_ID_COL, PASSAGES_COL) if c in available]

    count = 0
    for batch in pf.iter_batches(batch_size=batch_size, columns=columns):
        rows = batch.to_pylist()
        for row in rows:
            if count >= limit:
                return
            passages_struct = row.get(PASSAGES_COL) or {}
            passages = (
                passages_struct.get("Translated_passages")
                or passages_struct.get("English_passages")
                or []
            )
            yield {
                "id": row.get(QUERY_ID_COL, count),
                "query": row.get(QUERY_COL, ""),
                "passages": passages,
            }
            count += 1


def main(local_file: str, limit: int, batch_size: int) -> None:
    print(f"Reading local file {local_file} (limit={limit})...")

    all_chunks = []
    for i, row in enumerate(iter_local_rows(local_file, limit)):
        doc_id = str(row["id"])
        passages = row["passages"]
        text = " ".join(p for p in passages if p) if passages else ""
        if not text.strip():
            continue

        chunks = chunk_all_strategies(
            doc_id, text, embed_fn=lambda ts: embed_passages(ts), passages=passages
        )
        all_chunks.extend(chunks)

        if (i + 1) % 100 == 0:
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
    stacked = np.vstack(all_vectors) if all_vectors else np.zeros((1, 384))
    centroid = stacked.mean(axis=0)
    np.save(ARTIFACT_DIR / "corpus_centroid.npy", centroid)
    set_corpus_centroid(centroid)

    print("Done. Index ready.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--local-file",
        default="data/hinval.parquet",
        help="Path to a locally downloaded MSMARCO-XI parquet file",
    )
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    main(args.local_file, args.limit, args.batch_size)
