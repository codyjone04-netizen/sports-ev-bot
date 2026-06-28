"""
app/alerts/discord_bot.py — Discord integration for EV alerts.

Sends rich embeds when:
- Individual pick exceeds confidence + EV thresholds
- A new optimal parlay is constructed
- Lineup / injury news materially changes an existing pick
"""
from __future__ import annotations

import asyncio
from datetime import timezone
from typing import Any

from app.core.logging_config import get_logger
from app.core.models import Market, Prediction
from app.ml.features import MarketContext
from app.ml.parlay_builder import ParlayResult
from config.settings import get_settings

logger = get_logger(__name__)
settings = get_settings()

try:
    import discord  # type: ignore
    _HAS_DISCORD = True
except ImportError:
    _HAS_DISCORD = False
    logger.warning("discord_py_not_installed", detail="pip install discord.py")


def _american_str(dec: float) -> str:
    """e.g. 2.15 → +115"""
    if dec >= 2.0:
        n = int(round((dec - 1) * 100))
        return f"+{n}"
    else:
        n = int(round(-100 / (dec - 1)))
        return str(n)


def _risk_emoji(risk: str) -> str:
    return {"low": "🟢", "medium": "🟡", "high": "🔴"}.get(risk, "⚪")


def _sport_emoji(sport: str) -> str:
    return {
        "soccer": "⚽",
        "basketball": "🏀",
        "american_football": "🏈",
        "baseball": "⚾",
        "hockey": "🏒",
        "tennis": "🎾",
        "prediction": "🔮",
    }.get(sport, "🎯")


class MockChannel:
    """Fallback when discord.py is not installed — logs to console."""

    async def send(self, content: str = "", embed: Any = None) -> None:
        if embed:
            logger.info("discord_mock_embed", title=getattr(embed, "title", ""), description=getattr(embed, "description", ""))
        else:
            logger.info("discord_mock_message", content=content)


class DiscordAlerter:
    """
    Wraps a discord.py Client (or mock) to send formatted alerts.
    Can be used standalone or injected into MarketScanner.
    """

    def __init__(self) -> None:
        self._client: Any = None
        self._channel: Any = None
        self._ready = asyncio.Event()

    async def start(self) -> None:
        """Connect to Discord. Falls back to console logger if no token/library."""
        if not _HAS_DISCORD or not settings.discord_token:
            logger.warning("discord_disabled", reason="no token or discord.py not installed")
            self._channel = MockChannel()
            self._ready.set()
            return

        intents = discord.Intents.default()
        self._client = discord.Client(intents=intents)

        @self._client.event
        async def on_ready() -> None:
            logger.info("discord_connected", user=str(self._client.user))
            if settings.discord_channel_id:
                self._channel = self._client.get_channel(settings.discord_channel_id)
            self._ready.set()

        asyncio.create_task(self._client.start(settings.discord_token))
        try:
            await asyncio.wait_for(self._ready.wait(), timeout=15.0)
        except asyncio.TimeoutError:
            logger.error("discord_connect_timeout")
            self._channel = MockChannel()
            self._ready.set()

    async def _send_embed(self, **kwargs: Any) -> None:
        if self._channel is None:
            return
        if _HAS_DISCORD:
            embed = discord.Embed(**kwargs)
            try:
                await self._channel.send(embed=embed)
            except Exception as exc:
                logger.error("discord_send_error", error=str(exc))
        else:
            await self._channel.send(embed=kwargs)

    # ── Pick alert ────────────────────────────────────────────────────────────

    async def send_pick_alert(
        self,
        market: Market,
        prediction: Prediction,
        ctx: MarketContext,
    ) -> None:
        sport_emoji = _sport_emoji(market.sport.value if market.sport else "other")
        risk_emoji = _risk_emoji(prediction.risk_level.value if prediction.risk_level else "medium")
        ev_pct = f"{prediction.expected_value:+.1%}"
        conf_pct = f"{prediction.confidence:.0%}"
        odds_str = _american_str(market.decimal_odds)
        kelly_pct = f"{(prediction.kelly_fraction or 0):.1%}"
        stake = settings.alert_stake * (prediction.kelly_fraction or 0)

        title = f"{sport_emoji} HIGH EV PICK — {conf_pct} Confidence"
        description = (
            f"**{market.event_name}**\n"
            f"📌 **Pick:** {market.selection}\n"
            f"📊 **EV:** {ev_pct} | **Conf:** {conf_pct}\n"
            f"💰 **Odds:** {odds_str} | **Model Prob:** {prediction.model_probability:.1%}\n"
            f"🔥 **Kelly:** {kelly_pct} (≈ ${stake:.0f} on ${settings.alert_stake:.0f} bankroll)\n"
            f"{risk_emoji} **Risk:** {(prediction.risk_level.value or 'medium').title()}\n\n"
            f"📈 **Why:**\n{prediction.explanation or 'See model output.'}"
        )

        await self._send_embed(
            title=title,
            description=description,
            color=0x00FF88 if prediction.expected_value >= 0.10 else 0xFFAA00,
        )
        logger.info("discord_pick_alert_sent", event=market.event_name, ev=ev_pct)

    # ── Parlay alert ──────────────────────────────────────────────────────────

    async def send_parlay_alert(self, parlay: ParlayResult) -> None:
        n = len(parlay.legs)
        ev_pct = f"{parlay.combined_ev:+.1%}"
        conf_pct = f"{parlay.combined_confidence:.0%}"
        odds_str = f"{parlay.combined_odds:.2f}x"

        legs_text = "\n".join(
            f"  **Leg {i+1}:** {leg.selection} ({leg.event_name}) "
            f"@ {leg.decimal_odds:.2f} | EV {leg.expected_value:+.1%} | Conf {leg.confidence:.0%}"
            for i, leg in enumerate(parlay.legs)
        )

        title = f"🎰 {n}-LEG PARLAY — {conf_pct} Confidence"
        description = (
            f"**Combined Odds:** {odds_str}\n"
            f"**Combined EV:** {ev_pct}\n"
            f"**Confidence:** {conf_pct}\n\n"
            f"{legs_text}"
        )

        await self._send_embed(
            title=title,
            description=description,
            color=0x9B59B6,
        )
        logger.info("discord_parlay_alert_sent", legs=n, ev=ev_pct)

    # ── News / injury alert ───────────────────────────────────────────────────

    async def send_news_alert(self, headline: str, impact: str, prediction_ids: list[str]) -> None:
        description = (
            f"📰 **{headline}**\n\n"
            f"⚠️ Impact: {impact}\n"
            f"Affected predictions: {len(prediction_ids)}"
        )
        await self._send_embed(title="🚨 LINEUP/INJURY UPDATE", description=description, color=0xFF4444)
