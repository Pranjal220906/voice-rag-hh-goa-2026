"""Three guardrail checkpoints:

  1. input_guardrail    — before retrieval: off-topic query? unsafe content?
  2. grounding_guardrail — after retrieval, before generation: is the
                            retrieved context actually relevant enough to
                            answer from? If not, refuse rather than guess.
  3. hallucination_guardrail — after generation: does the answer's content
                            actually overlap with the chunks it cited?

Each returns a GuardrailFlag so the harness can log *why* it refused, not
just that it refused.
"""
from __future__ import annotations

import re

import numpy as np

from app.config import settings
from app.embeddings import embed_query, embed_texts
from app.schemas import GuardrailFlag, RetrievedChunk

# Small, illustrative deny-list for unsafe input. In production this would
# be a moderation model/API call, not a regex.
_UNSAFE_PATTERNS = [
    r"\bhow (to|do i) (make|build) (a )?(bomb|explosive|weapon)\b",
    r"\bself[- ]harm\b",
    r"\bkill (myself|someone)\b",
]

# Corpus-centroid embedding, set once after indexing (see scripts/build_index.py)
_corpus_centroid: np.ndarray | None = None


def set_corpus_centroid(vec: np.ndarray) -> None:
    global _corpus_centroid
    _corpus_centroid = vec


def _cos_sim(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) or 1e-9
    return float(np.dot(a, b) / denom)


def input_guardrail(query: str) -> GuardrailFlag:
    lowered = query.lower()
    for pattern in _UNSAFE_PATTERNS:
        if re.search(pattern, lowered):
            return GuardrailFlag(
                stage="input", triggered=True, reason="unsafe_content_pattern"
            )

    if _corpus_centroid is not None:
        q_vec = embed_query(query)
        sim = _cos_sim(q_vec, _corpus_centroid)
        if sim < settings.off_topic_similarity_threshold:
            return GuardrailFlag(
                stage="input",
                triggered=True,
                reason="off_topic_low_corpus_similarity",
                score=sim,
            )

    return GuardrailFlag(stage="input", triggered=False)


def grounding_guardrail(query: str, retrieved: list[RetrievedChunk]) -> GuardrailFlag:
    if not retrieved:
        return GuardrailFlag(stage="grounding", triggered=True, reason="no_chunks_retrieved")

    best_score = max((rc.rerank_score or rc.fused_score) for rc in retrieved)
    # rerank_score from a cross-encoder isn't 0-1 bounded the same way as
    # fused_score; normalize loosely via sigmoid for a consistent gate.
    normalized = 1 / (1 + np.exp(-best_score)) if best_score is not None else 0.0

    if normalized < settings.min_relevance_score:
        return GuardrailFlag(
            stage="grounding",
            triggered=True,
            reason="retrieved_context_not_relevant_enough",
            score=normalized,
        )
    return GuardrailFlag(stage="grounding", triggered=False, score=normalized)


def hallucination_guardrail(
    answer: str, cited_chunk_ids: list[str], retrieved: list[RetrievedChunk]
) -> GuardrailFlag:
    if not answer.strip():
        return GuardrailFlag(stage="hallucination", triggered=True, reason="empty_answer")

    cited_texts = [
        rc.chunk.text for rc in retrieved if rc.chunk.chunk_id in set(cited_chunk_ids)
    ]
    if not cited_texts:
        return GuardrailFlag(
            stage="hallucination", triggered=True, reason="no_valid_citations"
        )

    # Embedding-overlap check: does the answer sit semantically close to
    # what it claims to cite? Cheap proxy for entailment; swap for an NLI
    # model if you want a stricter check.
    answer_vec = embed_texts([answer])[0]
    cited_vecs = embed_texts(cited_texts)
    max_overlap = max(_cos_sim(answer_vec, v) for v in cited_vecs)

    if max_overlap < settings.hallucination_overlap_threshold:
        return GuardrailFlag(
            stage="hallucination",
            triggered=True,
            reason="answer_not_supported_by_cited_chunks",
            score=max_overlap,
        )
    return GuardrailFlag(stage="hallucination", triggered=False, score=max_overlap)
