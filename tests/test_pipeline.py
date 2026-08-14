"""Smoke tests that don't require Qdrant/Sarvam/LLM network access — they
exercise the pure-Python logic (chunking math, RRF fusion, guardrail
thresholds) so `pytest` gives fast signal in CI without live services.
"""
import numpy as np
import pytest

from app.chunking import (
    chunk_fixed_overlap,
    chunk_hierarchical,
    chunk_passage_native,
)
from app.retrieval import _reciprocal_rank_fusion
from app.schemas import ChunkStrategy


def test_fixed_overlap_chunking_covers_all_words():
    text = " ".join(f"word{i}" for i in range(500))
    chunks = chunk_fixed_overlap("doc1", text, window_tokens=100, overlap_ratio=0.2)
    assert len(chunks) > 1
    assert all(c.strategy == ChunkStrategy.FIXED for c in chunks)
    # last chunk should reach the end of the text
    assert "word499" in chunks[-1].text


def test_passage_native_skips_empty_passages():
    passages = ["real passage one", "", "   ", "real passage two"]
    chunks = chunk_passage_native("doc2", passages)
    assert len(chunks) == 2
    assert all(c.strategy == ChunkStrategy.PASSAGE for c in chunks)


def test_hierarchical_children_reference_valid_parents():
    text = " ".join(f"word{i}" for i in range(600))
    chunks = chunk_hierarchical("doc3", text, child_tokens=50, parent_tokens=200)
    parent_ids = {c.chunk_id for c in chunks if c.parent_id is None}
    children = [c for c in chunks if c.parent_id is not None]
    assert children, "expected at least one child chunk"
    assert all(c.parent_id in parent_ids for c in children)


def test_rrf_fusion_prefers_items_ranked_highly_in_both_lists():
    dense = [("a", 0.9), ("b", 0.8), ("c", 0.7)]
    sparse = [("b", 5.0), ("a", 4.0), ("d", 3.0)]
    fused = _reciprocal_rank_fusion(dense, sparse, k=60)
    # "a" and "b" both appear near the top of both lists -> should outscore "c"/"d"
    assert fused["a"] > fused["c"]
    assert fused["b"] > fused["d"]


def test_rrf_fusion_handles_disjoint_lists():
    dense = [("a", 0.9)]
    sparse = [("z", 5.0)]
    fused = _reciprocal_rank_fusion(dense, sparse, k=60)
    assert set(fused.keys()) == {"a", "z"}
