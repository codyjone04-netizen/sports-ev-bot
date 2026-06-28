"""
config/settings.py — Central configuration via pydantic-settings.
All values are read from environment variables or .env file.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # ── Database ─────────────────────────────────────────────────────────────
    database_url: str = Field(default="postgresql://postgres:password@localhost:5432/sportsev")
    redis_url: str = Field(default="redis://localhost:6379/0")

    # ── Discord ───────────────────────────────────────────────────────────────
    discord_token: Optional[str] = None
    discord_channel_id: Optional[int] = None

    # ── Kalshi ────────────────────────────────────────────────────────────────
    kalshi_api_key: Optional[str] = None
    kalshi_base_url: str = "https://trading-api.kalshi.com/trade-api/v2"

    # ── Polymarket ────────────────────────────────────────────────────────────
    polymarket_api_key: Optional[str] = None
    polymarket_base_url: str = "https://clob.polymarket.com"

    # ── Sports Data ───────────────────────────────────────────────────────────
    odds_api_key: Optional[str] = None
    odds_api_base_url: str = "https://api.the-odds-api.com/v4"
    api_football_key: Optional[str] = None
    api_football_base_url: str = "https://v3.football.api-sports.io"

    # ── Weather ───────────────────────────────────────────────────────────────
    openweather_key: Optional[str] = None

    # ── EV / Betting ──────────────────────────────────────────────────────────
    confidence_threshold: float = 0.85
    min_ev_threshold: float = 0.05
    alert_stake: float = 100.0
    max_kelly_fraction: float = 0.25
    scan_interval_seconds: int = 300

    # ── App ───────────────────────────────────────────────────────────────────
    app_env: str = "development"
    log_level: str = "INFO"
    secret_key: str = "change_me_in_production"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
