"""
api/config.py
=============
Pydantic Settings — all configuration loaded from environment variables / .env file.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import List, Optional

try:
    from pydantic_settings import BaseSettings, SettingsConfigDict
except ImportError:
    from pydantic import BaseSettings  # type: ignore
    SettingsConfigDict = None  # type: ignore


class Settings(BaseSettings):
    # ── App ────────────────────────────────────────────────────────────────
    app_name:    str = "VoiceGuard API"
    app_version: str = "1.0.0"
    debug:       bool = False

    # ── Model ──────────────────────────────────────────────────────────────
    cm_checkpoint:  str = "models/cm_detect2b_v4.pt"
    hf_repo_id:     str = ""         # Pull checkpoint from HF Hub on startup
    hf_token:       str = ""

    # ── Server ─────────────────────────────────────────────────────────────
    host: str = "0.0.0.0"
    port: int = 8000

    # ── Auth ───────────────────────────────────────────────────────────────
    api_key_required: bool = False
    api_keys:         str  = ""      # Comma-separated list of valid API keys

    # ── Rate limiting ──────────────────────────────────────────────────────
    rate_limit_per_minute: int = 60

    # ── CORS ───────────────────────────────────────────────────────────────
    cors_origins: str = "*"          # Comma-separated origins

    # ── Results ────────────────────────────────────────────────────────────
    benchmark_json: str = "results/benchmark_asvspoof19.json"

    if SettingsConfigDict is not None:
        model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")
    else:
        class Config:
            env_file = ".env"

    @property
    def valid_api_keys(self) -> List[str]:
        return [k.strip() for k in self.api_keys.split(",") if k.strip()]

    @property
    def cors_origins_list(self) -> List[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache()
def get_settings() -> Settings:
    return Settings()
