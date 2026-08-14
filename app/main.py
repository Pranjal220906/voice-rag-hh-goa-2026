from __future__ import annotations

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from app.pipeline import run_pipeline
from app.schemas import PipelineResponse

app = FastAPI(title="Voice-Enabled RAG — HH Goa 2026")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten for production deploy
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/ask/voice", response_model=PipelineResponse)
async def ask_voice(file: UploadFile = File(...)):
    audio_bytes = await file.read()
    return await run_pipeline(audio_bytes=audio_bytes)


@app.post("/ask/text", response_model=PipelineResponse)
async def ask_text(query: str = Form(...)):
    return await run_pipeline(text_query=query)
