"""The harness. Every stage is timed independently (stage_latencies_ms),
every failure has an explicit, structured fallback, and the pipeline never
raises out to the API layer uncaught — it always returns a PipelineResponse,
with `refused=True` and a reason when it can't (or shouldn't) answer.
"""
from __future__ import annotations

import time
from contextlib import contextmanager

from app.generation import GenerationError, generate_answer
from app.guardrails import (
    grounding_guardrail,
    hallucination_guardrail,
    input_guardrail,
)
from app.retrieval import resolve_hierarchical_context, retrieve
from app.schemas import PipelineResponse
from app.stt import LowConfidenceTranscript, STTError, transcribe


@contextmanager
def _timed(response: PipelineResponse, stage: str):
    start = time.perf_counter()
    try:
        yield
    finally:
        response.stage_latencies_ms[stage] = (time.perf_counter() - start) * 1000.0


async def run_pipeline(
    *, audio_bytes: bytes | None = None, text_query: str | None = None
) -> PipelineResponse:
    """Entry point for both voice and text queries (text bypasses STT — useful
    for the latency benchmark and for a text-input fallback in the UI).
    """
    t_start = time.perf_counter()
    response = PipelineResponse(query=text_query or "")

    # ---- Stage 1: STT (voice path only) -----------------------------------
    if audio_bytes is not None:
        with _timed(response, "stt"):
            try:
                stt_result = await transcribe(audio_bytes)
                response.transcript = stt_result.transcript
                response.stt_confidence = stt_result.confidence
                response.query = stt_result.transcript
            except LowConfidenceTranscript as e:
                response.refused = True
                response.refusal_reason = (
                    "Could not transcribe your question confidently — "
                    "please try again in a quieter environment."
                )
                response.stt_confidence = e.confidence
                response.total_latency_ms = (time.perf_counter() - t_start) * 1000.0
                return response
            except STTError as e:
                response.refused = True
                response.refusal_reason = f"Speech-to-text unavailable: {e}"
                response.total_latency_ms = (time.perf_counter() - t_start) * 1000.0
                return response
    else:
        response.query = text_query or ""

    # ---- Stage 2: input guardrail -------------------------------------------
    with _timed(response, "input_guardrail"):
        flag = input_guardrail(response.query)
        response.guardrail_flags.append(flag)
    if flag.triggered:
        response.refused = True
        response.refusal_reason = f"Query rejected at input stage: {flag.reason}"
        response.total_latency_ms = (time.perf_counter() - t_start) * 1000.0
        return response

    # ---- Stage 3: retrieval (chunking already happened offline at index time)
    with _timed(response, "retrieval"):
        try:
            retrieved = retrieve(response.query)
            retrieved = resolve_hierarchical_context(retrieved)
        except Exception as e:  # noqa: BLE001 - harness boundary, must not crash
            response.refused = True
            response.refusal_reason = f"Retrieval failed: {e}"
            response.total_latency_ms = (time.perf_counter() - t_start) * 1000.0
            return response
    response.sources = retrieved

    # ---- Stage 4: grounding guardrail (pre-generation) -----------------------
    with _timed(response, "grounding_guardrail"):
        g_flag = grounding_guardrail(response.query, retrieved)
        response.guardrail_flags.append(g_flag)
    if g_flag.triggered:
        response.refused = True
        response.refusal_reason = (
            "I don't have enough relevant information in the dataset to "
            f"answer that confidently ({g_flag.reason})."
        )
        response.total_latency_ms = (time.perf_counter() - t_start) * 1000.0
        return response

    # ---- Stage 5: answer generation ------------------------------------------
    with _timed(response, "generation"):
        try:
            result = await generate_answer(response.query, retrieved)
        except GenerationError as e:
            response.refused = True
            response.refusal_reason = f"Answer generation failed: {e}"
            response.total_latency_ms = (time.perf_counter() - t_start) * 1000.0
            return response

    if not result.get("answerable", False):
        response.refused = True
        response.refusal_reason = result.get("answer") or "Model determined it could not answer from context."
        response.total_latency_ms = (time.perf_counter() - t_start) * 1000.0
        return response

    response.answer = result.get("answer", "")
    cited_ids = result.get("cited_chunk_ids", [])

    # ---- Stage 6: hallucination guardrail (post-generation) ------------------
    with _timed(response, "hallucination_guardrail"):
        h_flag = hallucination_guardrail(response.answer, cited_ids, retrieved)
        response.guardrail_flags.append(h_flag)
    if h_flag.triggered:
        response.refused = True
        response.refusal_reason = (
            f"Generated answer failed the groundedness check ({h_flag.reason}) "
            "and was withheld rather than shown."
        )
        response.answer = None
        response.total_latency_ms = (time.perf_counter() - t_start) * 1000.0
        return response

    response.total_latency_ms = (time.perf_counter() - t_start) * 1000.0
    return response
