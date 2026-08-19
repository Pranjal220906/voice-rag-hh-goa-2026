"""The harness for the RAG pipeline."""

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
    print(f"[PIPELINE] START {stage}", flush=True)

    try:
        yield
    finally:
        elapsed = (time.perf_counter() - start) * 1000.0
        response.stage_latencies_ms[stage] = elapsed
        print(f"[PIPELINE] END {stage}: {elapsed:.0f} ms", flush=True)


async def run_pipeline(
    *, audio_bytes: bytes | None = None, text_query: str | None = None
) -> PipelineResponse:

    print("[PIPELINE] REQUEST START", flush=True)

    t_start = time.perf_counter()
    response = PipelineResponse(query=text_query or "")

    # STT
    if audio_bytes is not None:
        print("[PIPELINE] Voice request", flush=True)

        with _timed(response, "stt"):
            try:
                stt_result = await transcribe(audio_bytes)
                response.transcript = stt_result.transcript
                response.stt_confidence = stt_result.confidence
                response.query = stt_result.transcript

            except LowConfidenceTranscript as e:
                response.refused = True
                response.refusal_reason = (
                    "Could not transcribe your question confidently."
                )
                response.stt_confidence = e.confidence
                response.total_latency_ms = (
                    time.perf_counter() - t_start
                ) * 1000.0
                return response

            except STTError as e:
                response.refused = True
                response.refusal_reason = f"Speech-to-text unavailable: {e}"
                response.total_latency_ms = (
                    time.perf_counter() - t_start
                ) * 1000.0
                return response

    else:
        response.query = text_query or ""

    print(f"[PIPELINE] QUERY: {response.query}", flush=True)

    # Input guardrail
    with _timed(response, "input_guardrail"):
        flag = input_guardrail(response.query)
        response.guardrail_flags.append(flag)

    if flag.triggered:
        response.refused = True
        response.refusal_reason = (
            f"Query rejected at input stage: {flag.reason}"
        )
        response.total_latency_ms = (
            time.perf_counter() - t_start
        ) * 1000.0
        return response

    # Retrieval
    print("[PIPELINE] BEFORE RETRIEVAL", flush=True)

    with _timed(response, "retrieval"):
        try:
            retrieved = retrieve(response.query)

            print(
                f"[PIPELINE] RETRIEVED {len(retrieved)} CHUNKS",
                flush=True,
            )

            retrieved = resolve_hierarchical_context(retrieved)

            print(
                "[PIPELINE] HIERARCHICAL CONTEXT RESOLVED",
                flush=True,
            )

        except Exception as e:
            print(
                f"[PIPELINE] RETRIEVAL ERROR: {repr(e)}",
                flush=True,
            )

            response.refused = True
            response.refusal_reason = f"Retrieval failed: {e}"
            response.total_latency_ms = (
                time.perf_counter() - t_start
            ) * 1000.0
            return response

    response.sources = retrieved

    # Grounding
    print("[PIPELINE] BEFORE GROUNDING", flush=True)

    with _timed(response, "grounding_guardrail"):
        g_flag = grounding_guardrail(
            response.query,
            retrieved,
        )
        response.guardrail_flags.append(g_flag)

    if g_flag.triggered:
        response.refused = True
        response.refusal_reason = (
            "I don't have enough relevant information in the dataset "
            f"to answer that confidently ({g_flag.reason})."
        )
        response.total_latency_ms = (
            time.perf_counter() - t_start
        ) * 1000.0
        return response

    # Generation
    print("[PIPELINE] BEFORE GENERATION", flush=True)

    with _timed(response, "generation"):
        try:
            result = await generate_answer(
                response.query,
                retrieved,
            )

            print("[PIPELINE] GENERATION COMPLETE", flush=True)

        except GenerationError as e:
            print(
                f"[PIPELINE] GENERATION ERROR: {repr(e)}",
                flush=True,
            )

            response.refused = True
            response.refusal_reason = (
                f"Answer generation failed: {e}"
            )
            response.total_latency_ms = (
                time.perf_counter() - t_start
            ) * 1000.0
            return response

    if not result.get("answerable", False):
        response.refused = True
        response.refusal_reason = (
            result.get("answer")
            or "Model determined it could not answer from context."
        )
        response.total_latency_ms = (
            time.perf_counter() - t_start
        ) * 1000.0
        return response

    response.answer = result.get("answer", "")
    cited_ids = result.get("cited_chunk_ids", [])

    # Hallucination guardrail
    print("[PIPELINE] BEFORE HALLUCINATION CHECK", flush=True)

    with _timed(response, "hallucination_guardrail"):
        h_flag = hallucination_guardrail(
            response.answer,
            cited_ids,
            retrieved,
        )
        response.guardrail_flags.append(h_flag)

    if h_flag.triggered:
        response.refused = True
        response.refusal_reason = (
            f"Generated answer failed the groundedness check "
            f"({h_flag.reason}) and was withheld rather than shown."
        )
        response.answer = None
        response.total_latency_ms = (
            time.perf_counter() - t_start
        ) * 1000.0
        return response

    response.total_latency_ms = (
        time.perf_counter() - t_start
    ) * 1000.0

    print(
        f"[PIPELINE] REQUEST COMPLETE: "
        f"{response.total_latency_ms:.0f} ms",
        flush=True,
    )

    return response
