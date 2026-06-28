"""
app/scrapers/kalshi.py — Fetch active prediction markets from Kalshi.

Kalshi REST API v2 documentation:
https://trading-api.kalshi.com/trade-api/v2/swagger-ui
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any

import httpx

from app.core.logging_config import get_logger
from app.ml.features import MarketContext

logger = get_logger(__name__)


class KalshiScraper:
    """
    Pull active markets from Kalshi and convert them to MarketContext objects.
    """

    BASE_URL = "https://trading-api.kalshi.com/trade-api/v2"

    def __init__(self, api_key: str | None = None) -> None:
        self._api_key = api_key
        self._headers: dict[str, str] = {"Content-Type": "application/json"}
        if api_key:
            self._headers["Authorization"] = f"Bearer {api_key}"

    async def fetch_markets(
        self,
        limit: int = 200,
        status: str = "open",
    ) -> list[dict[str, Any]]:
        """Return raw market dicts from Kalshi."""
        url = f"{self.BASE_URL}/markets"
        params: dict[str, Any] = {"limit": limit, "status": status}
        raw: list[dict] = []

        async with httpx.AsyncClient(timeout=30.0) as client:
            cursor: str | None = None
            while True:
                if cursor:
                    params["cursor"] = cursor
                try:
                    resp = await client.get(url, headers=self._headers, params=params)
                    resp.raise_for_status()
                    data = resp.json()
                except httpx.HTTPError as exc:
                    logger.error("kalshi_http_error", error=str(exc))
                    break

                markets = data.get("markets", [])
                raw.extend(markets)
                cursor = data.get("cursor")
                if not cursor or len(markets) < limit:
                    break
                await asyncio.sleep(0.2)  # rate-limit courtesy

        logger.info("kalshi_markets_fetched", count=len(raw))
        return raw

    def _implied_prob(self, yes_ask: float | None, yes_bid: float | None) -> float:
        """Mid-price as implied probability (Kalshi prices are cents = probability)."""
        if yes_ask is not None and yes_bid is not None:
            return ((yes_ask + yes_bid) / 2) / 100
        if yes_ask is not None:
            return yes_ask / 100
        if yes_bid is not None:
            return yes_bid / 100
        return 0.5

    def to_market_contexts(self, raw_markets: list[dict]) -> list[tuple[dict, MarketContext]]:
        """
        Convert Kalshi raw dicts to (raw_dict, MarketContext) pairs.
        Returns pairs so the caller can persist both.
        """
        results: list[tuple[dict, MarketContext]] = []
        for m in raw_markets:
            try:
                yes_ask = m.get("yes_ask")  # cents
                yes_bid = m.get("yes_bid")
                ip = self._implied_prob(yes_ask, yes_bid)
                if ip <= 0 or ip >= 1:
                    continue

                # Kalshi odds: 1/ip for YES, 1/(1-ip) for NO
                decimal_yes = 1.0 / ip if ip > 0 else 2.0
                volume = m.get("volume", 0) or 0

                close_time = m.get("close_time")
                event_date: datetime | None = None
                if close_time:
                    try:
                        event_date = datetime.fromisoformat(close_time.rstrip("Z")).replace(tzinfo=timezone.utc)
                    except ValueError:
                        pass

                ctx = MarketContext(
                    market_type="prediction_market",
                    sport="prediction",
                    decimal_odds=decimal_yes,
                    implied_probability=ip,
                    volume=float(volume),
                    kalshi_yes_price=ip,
                    extra={
                        "ticker": m.get("ticker", ""),
                        "title": m.get("title", ""),
                        "category": m.get("category", ""),
                    },
                )
                results.append((m, ctx))
            except Exception as exc:
                logger.warning("kalshi_parse_error", error=str(exc), market_id=m.get("ticker"))
        return results


async def demo() -> None:
    scraper = KalshiScraper()
    markets = await scraper.fetch_markets(limit=10)
    for m in markets[:3]:
        print(m.get("ticker"), m.get("title"))


if __name__ == "__main__":
    asyncio.run(demo())
