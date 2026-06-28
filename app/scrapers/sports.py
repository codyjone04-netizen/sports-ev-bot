"""
app/scrapers/sports.py — Fetch sports odds, player props, and match stats.

Primary sources
---------------
* The Odds API (https://the-odds-api.com) — moneyline, spreads, totals, props
* API-Football (https://api-sports.io)   — team/player stats, lineups, injuries
"""
from __future__ import annotations

import asyncio
from typing import Any

import httpx

from app.core.logging_config import get_logger
from app.ml.features import MarketContext

logger = get_logger(__name__)

# Market type mappings from Odds API key → our MarketType enum value
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
    "player_to_score": "anytime_scorer",
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
}


class OddsAPIScraper:
    """Pull odds from The Odds API — covers 40+ sports and 80+ books."""

    def __init__(self, api_key: str) -> None:
        self._api_key = api_key
        self._base = "https://api.the-odds-api.com/v4"

    async def fetch_sports(self) -> list[dict]:
        """List of active sports."""
        async with httpx.AsyncClient(timeout=20.0) as client:
            resp = await client.get(
                f"{self._base}/sports",
                params={"apiKey": self._api_key, "all": "false"},
            )
            resp.raise_for_status()
            return resp.json()

    async def fetch_odds(
        self,
        sport: str = "soccer_fifa_world_cup",
        regions: str = "us,uk,eu",
        markets: str = "h2h,totals,btts",
        odds_format: str = "decimal",
        bookmakers: str | None = None,
    ) -> list[dict]:
        """Fetch event odds for one sport."""
        params: dict[str, Any] = {
            "apiKey": self._api_key,
            "regions": regions,
            "markets": markets,
            "oddsFormat": odds_format,
        }
        if bookmakers:
            params["bookmakers"] = bookmakers

        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                resp = await client.get(f"{self._base}/sports/{sport}/odds", params=params)
                resp.raise_for_status()
                return resp.json()
            except httpx.HTTPError as exc:
                logger.error("odds_api_http_error", sport=sport, error=str(exc))
                return []

    async def fetch_player_props(
        self,
        sport: str = "soccer_fifa_world_cup",
        event_id: str = "",
        prop_markets: str = "player_shots,player_passes",
    ) -> list[dict]:
        """Fetch player prop markets for a specific event."""
        params: dict[str, Any] = {
            "apiKey": self._api_key,
            "regions": "us,uk,eu",
            "markets": prop_markets,
            "oddsFormat": "decimal",
        }
        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                resp = await client.get(
                    f"{self._base}/sports/{sport}/events/{event_id}/odds",
                    params=params,
                )
                resp.raise_for_status()
                return resp.json()
            except httpx.HTTPError as exc:
                logger.error("odds_api_props_error", event=event_id, error=str(exc))
                return []

    def parse_events_to_contexts(
        self, events: list[dict], sport_key: str = "soccer"
    ) -> list[tuple[dict, str, MarketContext]]:
        """
        Parse Odds API event list into (raw_market_dict, external_id, MarketContext) triples.
        One triple per bookmaker × market × outcome.
        """
        sport = SPORT_MAP.get(sport_key, "other")
        results: list[tuple[dict, str, MarketContext]] = []

        for event in events:
            home = event.get("home_team", "")
            away = event.get("away_team", "")
            event_name = f"{home} vs {away}"
            event_id = event.get("id", "")
            event_date = event.get("commence_time")

            bookmakers = event.get("bookmakers", [])
            # Track opening odds: use first bookmaker as "opening"
            opening_odds_map: dict[str, float] = {}

            for bk_idx, bk in enumerate(bookmakers):
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

                        # Opening odds: first bookmaker's price
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
    """Fetch team stats, player stats, lineups and injuries from API-Football."""

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

    async def get_team_stats(self, team: int, league: int, season: int) -> dict:
        data = await self._get("teams/statistics", {"team": team, "league": league, "season": season})
        return data.get("response", {})

    async def get_player_stats(self, player: int, season: int) -> dict:
        data = await self._get("players", {"id": player, "season": season})
        resp = data.get("response", [])
        return resp[0] if resp else {}

    async def get_injuries(self, league: int, season: int, fixture: int | None = None) -> list[dict]:
        params: dict[str, Any] = {"league": league, "season": season}
        if fixture:
            params["fixture"] = fixture
        data = await self._get("injuries", params)
        return data.get("response", [])

    async def get_lineups(self, fixture: int) -> list[dict]:
        data = await self._get("fixtures/lineups", {"fixture": fixture})
        return data.get("response", [])

    async def get_h2h(self, team1: int, team2: int, last: int = 10) -> list[dict]:
        data = await self._get("fixtures/headtohead", {"h2h": f"{team1}-{team2}", "last": last})
        return data.get("response", [])

    def enrich_context(self, ctx: MarketContext, team_stats: dict, h2h: list[dict]) -> MarketContext:
        """
        Populate MarketContext fields from team stats + H2H data.
        Returns a modified copy.
        """
        from dataclasses import replace

        # Team offense/defense from statistics API
        fixtures_data = team_stats.get("fixtures", {})
        goals_data = team_stats.get("goals", {})
        for_goals = goals_data.get("for", {}).get("average", {})
        against_goals = goals_data.get("against", {}).get("average", {})

        goals_scored_avg = float(for_goals.get("total", 0) or 0)
        goals_conceded_avg = float(against_goals.get("total", 0) or 0)

        # H2H stats
        h2h_home = sum(1 for f in h2h if f.get("teams", {}).get("home", {}).get("winner"))
        h2h_away = sum(1 for f in h2h if f.get("teams", {}).get("away", {}).get("winner"))
        h2h_draws = len(h2h) - h2h_home - h2h_away
        total_goals = []
        for f in h2h:
            goals = f.get("goals", {})
            home_g = goals.get("home", 0) or 0
            away_g = goals.get("away", 0) or 0
            total_goals.append(home_g + away_g)

        btts_count = sum(1 for f in h2h if (
            (f.get("goals", {}).get("home", 0) or 0) > 0 and
            (f.get("goals", {}).get("away", 0) or 0) > 0
        ))

        ctx = replace(
            ctx,
            home_goals_scored_avg=goals_scored_avg,
            home_goals_conceded_avg=goals_conceded_avg,
            h2h_home_wins=h2h_home,
            h2h_away_wins=h2h_away,
            h2h_draws=h2h_draws,
            h2h_matches=len(h2h),
            h2h_avg_total_goals=sum(total_goals) / max(len(total_goals), 1),
            h2h_btts_rate=btts_count / max(len(h2h), 1),
        )
        return ctx
