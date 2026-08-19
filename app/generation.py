"""Answer generation using Groq.

The model receives the user question and retrieved passages and must
return a small JSON object. JSON is requested through the prompt rather
than the response_format API parameter so this works reliably across
Groq models.
"""

from __future__ import annotations

import json

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from app.config import settings
from app.schemas import RetrievedChunk


SYSTEM_PROMPT = """You are a grounded question-answering assistant.

You will be given a user question and a set of retrieved context passages.
Each passage has a chunk_id.

Rules:
1. Answer ONLY using information present in the provided passages.
2. Do not use outside knowledge.
3. Every factual statement in the answer must be supported by the passages.
4. If the passages contain enough information, set "answerable" to true.
5. If the passages do not contain enough information, set "answerable" to false
   and explain briefly in "answer". Do not guess.
6. Return ONLY one valid JSON object.
7. Do not use markdown fences.
8. The JSON object MUST have exactly this structure:

{
  "answerable": true,
  "answer": "your answer here",
  "cited_chunk_ids": ["chunk-id-1"]
}

If the question cannot be answered from the passages, use:

{
  "answerable": false,
  "answer": "The provided passages do not contain enough information to answer the question.",
  "cited_chunk_ids": []
}
"""


def _build_user_prompt(query: str, chunks: list[RetrievedChunk]) -> str:
    context_blocks = "\n\n".join(
        f"[chunk_id={rc.chunk.chunk_id}]\n{rc.chunk.text}"
        for rc in chunks
    )

    return (
        f"Question: {query}\n\n"
        f"Retrieved context:\n{context_blocks}\n\n"
        "Now answer the question using ONLY the retrieved context. "
        "Return ONLY valid JSON."
    )


class GenerationError(Exception):
    """Raised when Groq generation fails or returns invalid data."""


@retry(
    stop=stop_after_attempt(2),
    wait=wait_exponential(multiplier=0.5, min=0.5, max=3.0),
    reraise=True,
)
async def _call_groq(system: str, user: str) -> str:
    """Call Groq's OpenAI-compatible chat completion endpoint."""

    if not settings.llm_api_key:
        raise GenerationError(
            "LLM_API_KEY (Groq API key) is not configured"
        )

    async with httpx.AsyncClient(
        timeout=settings.stage_timeout_seconds
    ) as client:

        payload = {
            "model": settings.llm_model,
            "max_tokens": settings.max_answer_tokens,
            "messages": [
                {
                    "role": "system",
                    "content": system,
                },
                {
                    "role": "user",
                    "content": user,
                },
            ],
            "temperature": 0,
        }

        resp = await client.post(
            settings.groq_api_url,
            headers={
                "Authorization": f"Bearer {settings.llm_api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
        )

        if resp.status_code >= 400:
            error_text = resp.text[:2000]

            raise GenerationError(
                f"Groq API error {resp.status_code}: {error_text}"
            )

        try:
            data = resp.json()
        except Exception as e:
            raise GenerationError(
                f"Groq returned invalid JSON response: {resp.text[:1000]}"
            ) from e

        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as e:
            raise GenerationError(
                f"Unexpected Groq response format: {data}"
            ) from e


def _clean_json_response(raw: str) -> str:
    """Remove accidental markdown fences or surrounding whitespace."""

    raw = raw.strip()

    if raw.startswith("```json"):
        raw = raw[len("```json"):].strip()

    if raw.startswith("```"):
        raw = raw[len("```"):].strip()

    if raw.endswith("```"):
        raw = raw[:-3].strip()

    return raw


def _validate_result(parsed: dict, chunks: list[RetrievedChunk]) -> dict:
    """Validate and normalize the model's JSON response."""

    if not isinstance(parsed, dict):
        raise GenerationError("Groq response JSON is not an object")

    answerable = parsed.get("answerable", False)
    answer = parsed.get("answer", "")
    cited_chunk_ids = parsed.get("cited_chunk_ids", [])

    if not isinstance(answerable, bool):
        answerable = bool(answerable)

    if not isinstance(answer, str):
        answer = str(answer)

    if not isinstance(cited_chunk_ids, list):
        cited_chunk_ids = []

    cited_chunk_ids = [
        str(chunk_id)
        for chunk_id in cited_chunk_ids
        if chunk_id
    ]

    valid_chunk_ids = {
        rc.chunk.chunk_id
        for rc in chunks
    }

    cited_chunk_ids = [
        chunk_id
        for chunk_id in cited_chunk_ids
        if chunk_id in valid_chunk_ids
    ]

    return {
        "answerable": answerable,
        "answer": answer,
        "cited_chunk_ids": cited_chunk_ids,
    }


async def generate_answer(
    query: str,
    chunks: list[RetrievedChunk],
) -> dict:
    """Generate a grounded answer from retrieved chunks."""

    if not chunks:
        return {
            "answerable": False,
            "answer": (
                "The retrieval system did not return any relevant "
                "passages for this question."
            ),
            "cited_chunk_ids": [],
        }

    user_prompt = _build_user_prompt(query, chunks)

    raw = await _call_groq(
        SYSTEM_PROMPT,
        user_prompt,
    )

    raw = _clean_json_response(raw)

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as e:
        raise GenerationError(
            f"Groq did not return valid JSON. Raw response: {raw[:2000]}"
        ) from e

    return _validate_result(parsed, chunks)
