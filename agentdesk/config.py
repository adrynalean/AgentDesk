"""Centralized configuration, loaded from environment / .env.

Uses pydantic-settings when available; falls back to a stdlib reader so the
skeleton imports and runs before dependencies are installed.
"""
from __future__ import annotations

import os
from pathlib import Path

# Default index location: anchored to the repo root (not the process CWD), so the
# server finds the same index regardless of where it was launched from.
_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_CHROMA = str(_REPO_ROOT / "chroma_db")

try:  # preferred: validated settings
    from pydantic_settings import BaseSettings, SettingsConfigDict

    class Settings(BaseSettings):
        model_config = SettingsConfigDict(env_file=".env", extra="ignore")

        llm_provider: str = "openai"
        openai_api_key: str = ""
        openai_model: str = "gpt-4o-mini"
        aws_region: str = "us-east-1"
        bedrock_model_id: str = "anthropic.claude-3-5-sonnet-20240620-v1:0"

        embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"
        chroma_dir: str = _DEFAULT_CHROMA
        top_k: int = 6

    settings = Settings()

except ModuleNotFoundError:  # fallback: plain env reader
    from dataclasses import dataclass

    @dataclass
    class Settings:  # type: ignore[no-redef]
        llm_provider: str = os.getenv("LLM_PROVIDER", "openai")
        openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
        openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        aws_region: str = os.getenv("AWS_REGION", "us-east-1")
        bedrock_model_id: str = os.getenv(
            "BEDROCK_MODEL_ID", "anthropic.claude-3-5-sonnet-20240620-v1:0"
        )
        embed_model: str = os.getenv(
            "EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        )
        chroma_dir: str = os.getenv("CHROMA_DIR", _DEFAULT_CHROMA)
        top_k: int = int(os.getenv("TOP_K", "6"))

    settings = Settings()
