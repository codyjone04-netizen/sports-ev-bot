"""
app/scrapers/polymarket.py — Fetch markets from Polymarket CLOB API.

Docs: https://docs.polymarket.com
"""
from __future__ import annotations

import asyncio
from typing import Any

import httpx

from app.core.logging_config import get_logger
from app.ml.features import MarketContext

logger = get_logger(__name__)


class PolymarketScraper:
    BASE_URL = "https://clob.polymarket.com"
    GAMMA_URL = "https://gamma-api.polymarket.com"

    def __init__(self, api_key: str | None = None) -> None:
        self._api_key = api_key
        self._headers: dict[str, str] = {"Content-Type": "application/json"}
        if api_key:
            self._headers["POLY_API_KEY"] = api_key

    async def fetch_markets(self, limit: int = 100, active: bool = True) -> list[dict[str, Any]]:
        """Fetch markets from Polymarket Gamma API (public, no auth needed)."""
        url = f"{self.GAMMA_URL}/markets"
        params: dict[str, Any] = {
            "active": str(active).lower(),
            "limit": min(limit, 100),
            "order": "volume24hr",
            "ascending": "false",
        }
        raw: list[dict] = []

        async with httpx.AsyncClient(timeout=30.0) as client:
            offset = 0
            while len(raw) < limit:
                params["offset"] = offset
                try:
                    resp = await client.get(url, params=params)
                    resp.raise_for_status()
                    data = resp.json()
                except httpx.HTTPError as exc:
                    logger.error("polymarket_http_error", error=str(exc))
                    break

                batch = data if isinstance(data, list) else data.get("data", [])
                if not batch:
                    break
                raw.extend(batch)
                if len(batch) < 100:
                    break
                offset += len(batch)

        logger.info("polymarket_markets_fetched", count=len(raw))
        return raw[:limit]

    def to_market_contexts(self, raw_markets: list[dict]) -> list[tuple[dict, MarketContext]]:
        results: list[tuple[dict, MarketContext]] = []
        for m in raw_markets:
            try:
                # Polymarket: outcomes have prices
                outcomes = m.get("outcomes", [])
                prices = m.get("outcomePrices", [])

                if not outcomes or not prices:
                    continue

                # We'll create a context for each YES outcome
                for i, outcome in enumerate(outcomes):
                    if outcome.lower() not in ("yes", "true", "1"):
                        continue
                    try:
                        ip = float(prices[i]) if i < len(prices) else 0.5
                    except (ValueError, TypeError):
                        ip = 0.5

                    if ip <= 0 or ip >= 1:
                        continue

                    decimal_odds = 1.0 / ip
                    volume = float(m.get("volume24hr", 0) or 0)

                    ctx = MarketContext(
                        market_type="prediction_market",
                        sport="prediction",
                        decimal_odds=decimal_odds,
                        implied_probability=ip,
                        volume=volume,
                        poly_yes_price=ip,
                        extra={
                            "slug": m.get("slug", ""),
                            "question": m.get("question", ""),
                            "category": m.get("category", ""),
                            "url": f"https://polymarket.com/event/{m.get('slug', '')}",
                        },
                    )
                    results.append((m, ctx))
                    break  # one context per market (YES outcome)

            except Exception as exc:
                logger.warning("polymarket_parse_error", error=str(exc), market=m.get("slug"))

        return results


async def demo() -> None:
    scraper = PolymarketScraper()
    markets = await scraper.fetch_markets(limit=5)
    for m in markets:
        print(m.get("question"), m.get("outcomePrices"))


if __name__ == "__main__":
    asyncio.run(demo())
