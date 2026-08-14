"""Typed contracts for every stage of the pipeline. The harness passes these
between stages instead of raw dicts/strings so failures are structural, not
'the LLM forgot a field'.
"""
from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class ChunkStrategy(str, Enum):
    FIXED = "fixed_overlap"
    SEMANTIC = "semantic_breakpoint"
    PASSAGE = "passage_native"
    HIERARCHICAL = "hierarchical_parent_child"


class Chunk(BaseModel):
    chunk_id: str
    doc_id: str
    parent_id: Optional[str] = None  # for hierarchical chunks
    text: str
    strategy: ChunkStrategy
    position: int
    language: Optional[str] = None
    token_count: int


class STTResult(BaseModel):
    transcript: str
    confidence: float
    language_detected: Optional[str] = None
    raw_provider_response: Optional[dict] = None


class RetrievedChunk(BaseModel):
    chunk: Chunk
    dense_score: float = 0.0
    sparse_score: float = 0.0
    fused_score: float = 0.0
    rerank_score: Optional[float] = None


class GuardrailFlag(BaseModel):
    stage: str  # "input" | "grounding" | "hallucination"
    triggered: bool
    reason: str = ""
    score: Optional[float] = None


class PipelineResponse(BaseModel):
    query: str
    transcript: Optional[str] = None
    stt_confidence: Optional[float] = None
    answer: Optional[str] = None
    sources: list[RetrievedChunk] = Field(default_factory=list)
    refused: bool = False
    refusal_reason: Optional[str] = None
    guardrail_flags: list[GuardrailFlag] = Field(default_factory=list)
    stage_latencies_ms: dict[str, float] = Field(default_factory=dict)
    total_latency_ms: float = 0.0
