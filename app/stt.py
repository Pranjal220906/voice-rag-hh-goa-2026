"""Speech-to-text via Sarvam, wrapped for the harness: retries, timeout,
and a confidence gate so garbage transcripts never reach retrieval.
"""
from __future__ import annotations

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from app.config import settings
from app.schemas import STTResult


class STTError(Exception):
    pass


class LowConfidenceTranscript(STTError):
    def __init__(self, transcript: str, confidence: float):
        self.transcript = transcript
        self.confidence = confidence
        super().__init__(
            f"Transcript confidence {confidence:.2f} below threshold "
            f"{settings.stt_min_confidence:.2f}"
        )


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=0.3, min=0.3, max=2.0),
    reraise=True,
)
async def _call_sarvam(audio_bytes: bytes, filename: str) -> dict:
    if not settings.sarvam_api_key:
        raise STTError("SARVAM_API_KEY is not configured")

    async with httpx.AsyncClient(timeout=settings.stage_timeout_seconds) as client:
        resp = await client.post(
            settings.sarvam_stt_url,
            headers={"api-subscription-key": settings.sarvam_api_key},
            files={"file": (filename, audio_bytes, "audio/wav")},
            data={"model": "saarika:v2"},
        )
        resp.raise_for_status()
        return resp.json()


async def transcribe(audio_bytes: bytes, filename: str = "query.wav") -> STTResult:
    """Transcribe audio and enforce a minimum confidence bar.

    Raises LowConfidenceTranscript (caught by the harness -> guardrail
    response asking the user to repeat themselves) rather than silently
    feeding a bad transcript into retrieval.
    """
    payload = await _call_sarvam(audio_bytes, filename)

    transcript = payload.get("transcript", "").strip()
    # Sarvam doesn't always return a scalar confidence; fall back to a proxy
    # (non-empty + no [inaudible]-style markers) if the field is absent.
    confidence = payload.get("confidence")
    if confidence is None:
        confidence = 0.0 if not transcript else 0.75

    result = STTResult(
        transcript=transcript,
        confidence=float(confidence),
        language_detected=payload.get("language_code"),
        raw_provider_response=payload,
    )

    if result.confidence < settings.stt_min_confidence or not transcript:
        raise LowConfidenceTranscript(transcript, result.confidence)

    return result
