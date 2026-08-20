"""Central, env-driven configuration. Nothing here should be hardcoded secrets."""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # STT
    sarvam_api_key: str = ""
    sarvam_stt_url: str = "https://api.sarvam.ai/speech-to-text"
    stt_min_confidence: float = 0.55

    # Vector store
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str = ""
    qdrant_collection: str = "msmarco_xi_chunks"

    # Embeddings
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    embedding_dim: int = 384

    # Retrieval
    top_k_dense: int = 20
    top_k_sparse: int = 20
    top_k_fused: int = 8
    top_k_final: int = 4
    rrf_k: int = 60  # reciprocal rank fusion constant
    min_relevance_score: float = 0.35  # pre-generation grounding gate

    # Generation — Groq's free tier (OpenAI-compatible endpoint), no card needed
    llm_provider: str = "groq"
    llm_model: str = "openai/gpt-oss-20b"
    llm_api_key: str = ""
    groq_api_url: str = "https://api.groq.com/openai/v1/chat/completions"
    max_answer_tokens: int = 800

    # Guardrails
    hallucination_overlap_threshold: float = 0.4
    off_topic_similarity_threshold: float = 0.28

    # Harness
    max_retries: int = 2
    stage_timeout_seconds: float = 8.0


settings = Settings()
