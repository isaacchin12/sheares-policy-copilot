"""
LLM settings — loaded from environment / .env file.

Import the singleton with:
    from llm.settings import get_settings
    cfg = get_settings()
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # ── Provider selection ────────────────────────────────────────────────────
    llm_provider: Literal["anthropic", "azure_openai"] = "anthropic"

    # ── Anthropic ─────────────────────────────────────────────────────────────
    anthropic_api_key: str = ""

    # ── Azure OpenAI ──────────────────────────────────────────────────────────
    azure_openai_api_key: str = ""
    azure_openai_endpoint: str = ""
    azure_openai_chat_deployment: str = "gpt-4o"
    azure_openai_embedding_deployment: str = "text-embedding-3-large"
    azure_openai_api_version: str = "2024-02-01"

    # ── Qdrant ────────────────────────────────────────────────────────────────
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "sheares_policy"

    # ── Data paths ────────────────────────────────────────────────────────────
    corpus_path: str = "data/synthetic"

    # ── Logging ───────────────────────────────────────────────────────────────
    log_level: str = "INFO"

    # ── Langfuse (observability) ──────────────────────────────────────────────
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "http://localhost:3000"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached singleton — call freely; .env is parsed only once."""
    return Settings()
