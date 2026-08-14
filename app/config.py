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
    qdrant_collection: str = "msmarco_xi_chunks"

    # Embeddings
    embedding_model: str = "intfloat/multilingual-e5-base"
    embedding_dim: int = 768

    # Retrieval
    top_k_dense: int = 20
    top_k_sparse: int = 20
    top_k_fused: int = 8
    top_k_final: int = 4
    rrf_k: int = 60  # reciprocal rank fusion constant
    min_relevance_score: float = 0.35  # pre-generation grounding gate

    # Generation
    llm_provider: str = "anthropic"  # or "openai"
    llm_model: str = "claude-sonnet-4-6"
    llm_api_key: str = ""
    max_answer_tokens: int = 400

    # Guardrails
    hallucination_overlap_threshold: float = 0.4
    off_topic_similarity_threshold: float = 0.28

    # Harness
    max_retries: int = 2
    stage_timeout_seconds: float = 8.0


settings = Settings()
