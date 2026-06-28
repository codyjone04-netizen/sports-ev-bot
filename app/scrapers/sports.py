"""
app/scrapers/sports.py — Fetch sports odds from The Odds API.
Uses only active sports to avoid 422 errors.
"""
from __future__ import annotations

import asyncio
from typing import Any

import httpx

from app.core.logging_config import get_logger
from app.ml.features import MarketContext

logger = get_logger(__name__)

ODDS_API_MARKET_MAP = {
    "h2h": "moneyline",
    "spreads": "spread",
    "totals": "total",
    "btts": "btts",
    "player_shots_on_target": "player_shots_on_target",
    "player_shots": "player_shots",
    "player_passes": "player_passes",
    "player_tackles": "player_tackles",
    "player_saves": "player_saves",
    "player_anytime_scorer": "anytime_scorer",
    "team_totals": "team_goals",
    "corners": "corners",
    "cards": "cards",
    "player_assists": "assists",
}

SPORT_MAP = {
    "soccer": "soccer",
    "basketball_nba": "basketball",
    "americanfootball_nfl": "american_football",
    "baseball_mlb": "baseball",
    "icehockey_nhl": "hockey",
    "tennis_atp": "tennis",
    "soccer_fifa_world_cup": "soccer",
    "soccer_epl": "soccer",
    "soccer_usa_mls": "soccer",
}


class OddsAPIScraper:
    def __init__(self, api_key: str) -> None:
        self._api_key = api_key
        self._base = "https://api.the-odds-api.com/v4"

    async def fetch_active_sports(self) -> list[str]:
        """Fetch only sports that currently have active events."""
        async with httpx.AsyncClient(timeout=20.0) as client:
            try:
                resp = await client.get(
                    f"{self._base}/sports",
                    params={"apiKey": self._api_key, "all": "false"},
                )
                resp.raise_for_status()
                sports = resp.json()
                # Return keys of active sports only
                return [s["key"] for s in sports if not s.get("has_outrights", False)]
            except httpx.HTTPError as exc:
                logger.error("odds_api_sports_error", error=str(exc))
                return []

    async def fetch_odds(
        self,
        sport: str,
        regions: str = "us,uk,eu",
        markets: str = "h2h,totals",
        odds_format: str = "decimal",
    ) -> list[dict]:
        params: dict[str, Any] = {
            "apiKey": self._api_key,
            "regions": regions,
            "markets": markets,
            "oddsFormat": odds_format,
        }
        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                resp = await client.get(
                    f"{self._base}/sports/{sport}/odds",
                    params=params,
                )
                resp.raise_for_status()
                return resp.json()
            except httpx.HTTPError as exc:
                logger.error("odds_api_http_error", sport=sport, error=str(exc))
                return []

    def parse_events_to_contexts(
        self, events: list[dict], sport_key: str = "soccer"
    ) -> list[tuple[dict, str, MarketContext]]:
        sport = SPORT_MAP.get(sport_key, "other")
        results: list[tuple[dict, str, MarketContext]] = []

        for event in events:
            home = event.get("home_team", "")
            away = event.get("away_team", "")
            event_name = f"{home} vs {away}"
            event_id = event.get("id", "")
            event_date = event.get("commence_time")
            opening_odds_map: dict[str, float] = {}

            for bk_idx, bk in enumerate(event.get("bookmakers", [])):
                for market in bk.get("markets", []):
                    market_key = market.get("key", "")
                    market_type = ODDS_API_MARKET_MAP.get(market_key, "other")
                    if market_type == "other":
                        continue

                    for outcome in market.get("outcomes", []):
                        sel = outcome.get("name", "")
                        dec_odds = float(outcome.get("price", 2.0))
                        line = outcome.get("point")
                        ip = 1.0 / dec_odds if dec_odds > 0 else 0.5

                        combo_key = f"{event_id}|{market_key}|{sel}"
                        if bk_idx == 0:
                            opening_odds_map[combo_key] = dec_odds
                        opening = opening_odds_map.get(combo_key, dec_odds)
                        line_movement = dec_odds - opening

                        is_player_prop = market_key.startswith("player_")
                        player_name = sel if is_player_prop else ""
                        selection = sel if not is_player_prop else f"Over {line}"

                        ctx = MarketContext(
                            market_type=market_type,
                            sport=sport,
                            decimal_odds=dec_odds,
                            implied_probability=ip,
                            opening_decimal_odds=opening,
                            line_movement=line_movement,
                            line=float(line) if line is not None else None,
                            home_team=home,
                            away_team=away,
                            player_name=player_name,
                        )

                        raw = {
                            "event_id": event_id,
                            "event_name": event_name,
                            "event_date": event_date,
                            "home_team": home,
                            "away_team": away,
                            "bookmaker": bk.get("key"),
                            "market_key": market_key,
                            "market_type": market_type,
                            "selection": selection,
                            "player_name": player_name,
                            "decimal_odds": dec_odds,
                            "line": line,
                        }
                        results.append((raw, event_id, ctx))

        logger.info("odds_api_parsed", records=len(results))
        return results


class APIFootballScraper:
    BASE_URL = "https://v3.football.api-sports.io"

    def __init__(self, api_key: str) -> None:
        self._headers = {"x-apisports-key": api_key}

    async def _get(self, endpoint: str, params: dict) -> dict:
        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                resp = await client.get(
                    f"{self.BASE_URL}/{endpoint}",
                    headers=self._headers,
                    params=params,
                )
                resp.raise_for_status()
                return resp.json()
            except httpx.HTTPError as exc:
                logger.error("api_football_error", endpoint=endpoint, error=str(exc))
                return {}

    async def get_fixtures(self, league: int = 1, season: int = 2026, next_n: int = 20) -> list[dict]:
        data = await self._get("fixtures", {"league": league, "season": season, "next": next_n})
        return data.get("response", [])
