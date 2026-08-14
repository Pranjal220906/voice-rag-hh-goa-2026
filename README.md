# Voice-Enabled RAG — HH Goa 2026 (Task 2)

End-to-end pipeline: **voice → Sarvam STT → guardrails → hybrid multi-strategy retrieval
(Qdrant) → reranking → grounded answer generation → hallucination/groundedness guardrail
→ structured output**, built on [`ai4bharat/MSMARCO-XI`](https://huggingface.co/datasets/ai4bharat/MSMARCO-XI).

## Why this design

- **STT: Sarvam**, not ElevenLabs — MSMARCO-XI is an Indian-language dataset; Sarvam's
  `saarika` model is tuned for Indian languages/accents, which matters far more here than
  raw transcription polish.
- **Chunking is not a single strategy.** We build and index four chunking strategies in
  parallel (see `app/chunking.py`), tag every chunk with which strategy produced it plus
  a parent-document pointer, and fuse retrieval results across them at query time.
- **The 200ms budget is scoped honestly.** "Chunking + vector DB retrieval" (embedding
  the query, hybrid search, fusion, rerank) is what we hold to <200ms and benchmark at
  P50/P70/P100 in `scripts/benchmark_latency.py`. STT and LLM generation are measured and
  reported *separately* — claiming a sub-200ms number that silently excludes an LLM call
  over the network would be misleading, and we'd rather show real numbers than fake ones.
- **This is a harness, not a prompt.** `app/pipeline.py` orchestrates every stage as a
  typed, retried, independently-failable step with a structured output contract
  (`app/schemas.py`), including an explicit `refused: bool` field.
- **Guardrails run at three points**: on the input (off-topic / unsafe / low-confidence
  transcript), pre-generation (is retrieved context actually relevant to the query?), and
  post-generation (is the answer's content actually supported by the retrieved chunks?).

## Project layout

```
app/
  main.py         FastAPI app — /ask (voice or text), /health
  config.py       env-driven settings
  schemas.py      pydantic request/response contracts
  stt.py          Sarvam STT client, retry + confidence gate
  chunking.py     4 chunking strategies + a run-all indexer helper
  embeddings.py   embedding model wrapper (multilingual e5)
  vector_store.py Qdrant client wrapper (dense + payload filtering)
  retrieval.py    hybrid dense+BM25 retrieval, RRF fusion, reranking
  generation.py   LLM answer generation, forced chunk citations, JSON mode
  guardrails.py   input filter, grounding-relevance gate, hallucination check
  pipeline.py     the harness: wires every stage together with retries/fallbacks
scripts/
  build_index.py       loads MSMARCO-XI, runs all chunking strategies, embeds, indexes
  benchmark_latency.py measures P50/P70/P100 per stage across N real queries
frontend/
  index.html      minimal mic-capture + query UI, calls /ask
tests/
  test_pipeline.py smoke tests for each stage with a fake vector store
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1. Bring up Qdrant (local, low-latency)
docker run -p 6333:6333 qdrant/qdrant

# 2. Set secrets
export SARVAM_API_KEY=...
export LLM_API_KEY=...          # your provider of choice, see app/generation.py
export QDRANT_URL=http://localhost:6333

# 3. Build the index from MSMARCO-XI (samples N passages by default, see --help)
python scripts/build_index.py --dataset ai4bharat/MSMARCO-XI --limit 20000

# 4. Run the API
uvicorn app.main:app --reload --port 8000

# 5. Benchmark latency
python scripts/benchmark_latency.py --n 100 --out latency_report.csv
```

Open `frontend/index.html` (or serve it statically) to talk to the pipeline from a browser.

## Submission checklist (not code, just so nobody forgets)

- [ ] GitHub repo link (push this)
- [ ] Live working link (deploy `app/` + a small managed Qdrant instance)
- [ ] `latency_report.csv` → P50/P70/P100 numbers, retrieval-stage AND full-pipeline
- [ ] 90s team/process video
- [ ] End-to-end demo video
- [ ] Both videos posted on Instagram, X, and LinkedIn by **every** team member, each post
      tagged `#RAGInGoa`; at least one Instagram account public
- [ ] Fill the submission form — no resubmissions allowed, so submit only when final
