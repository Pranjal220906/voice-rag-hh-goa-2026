"""Answer generation: takes the query + retrieved chunks, forces the model
into structured JSON output with explicit chunk-id citations, so the
hallucination guardrail has something concrete to check against.
"""
from __future__ import annotations

import json

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from app.config import settings
from app.schemas import RetrievedChunk

SYSTEM_PROMPT = """You are a grounded question-answering assistant.
You will be given a user question and a set of retrieved context passages,
each with a chunk_id. Rules:
1. Answer ONLY using information present in the provided passages.
2. Every factual sentence in your answer must be traceable to at least one
   chunk_id - list the chunk_ids you relied on.
3. If the passages do not contain enough information to answer, set
   "answerable" to false and explain why in "answer" instead of guessing.
4. Respond with ONLY a JSON object, no other text, no markdown fences:
   {"answerable": bool, "answer": str, "cited_chunk_ids": [str, ...]}
"""


def _build_user_prompt(query: str, chunks: list[RetrievedChunk]) -> str:
    context_blocks = "\n\n".join(
        f"[chunk_id={rc.chunk.chunk_id}]\n{rc.chunk.text}" for rc in chunks
    )
    return f"Question: {query}\n\nContext:\n{context_blocks}"


class GenerationError(Exception):
    pass


@retry(
    stop=stop_after_attempt(2),
    wait=wait_exponential(multiplier=0.5, min=0.5, max=3.0),
    reraise=True,
)
async def _call_groq(system: str, user: str) -> str:
    """Groq's free-tier API - OpenAI-compatible chat completions endpoint,
    no billing/card required to get started.
    """
    if not settings.llm_api_key:
        raise GenerationError("LLM_API_KEY (Groq key) is not configured")
    async with httpx.AsyncClient(timeout=settings.stage_timeout_seconds) as client:
        resp = await client.post(
            settings.groq_api_url,
            headers={
                "Authorization": f"Bearer {settings.llm_api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": settings.llm_model,
                "max_tokens": settings.max_answer_tokens,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "response_format": {"type": "json_object"},
            },
        )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]


async def generate_answer(query: str, chunks: list[RetrievedChunk]) -> dict:
    """Returns {"answerable": bool, "answer": str, "cited_chunk_ids": [...]}.
    Raises GenerationError on unrecoverable failure (caught by the harness
    -> falls back to a structured refusal, never an unhandled crash).
    """
    user_prompt = _build_user_prompt(query, chunks)
    raw = await _call_groq(SYSTEM_PROMPT, user_prompt)

    raw = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as e:
        raise GenerationError(f"Model did not return valid JSON: {e}") from e

    parsed.setdefault("answerable", False)
    parsed.setdefault("answer", "")
    parsed.setdefault("cited_chunk_ids", [])
    return parsed
