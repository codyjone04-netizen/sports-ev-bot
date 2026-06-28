"""
app/ml/features.py — Feature engineering for every market type.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class MarketContext:
    market_type: str = ""
    sport: str = ""
    decimal_odds: float = 2.0
    implied_probability: float = 0.5
    opening_decimal_odds: float | None = None
    volume: float | None = None
    line: float | None = None
    home_team: str = ""
    away_team: str = ""
    player_name: str = ""
    home_goals_scored_avg: float = 0.0
    home_goals_conceded_avg: float = 0.0
    home_shots_avg: float = 0.0
    home_possession_avg: float = 50.0
    home_form_points: float = 0.0
    home_btts_rate: float = 0.0
    away_goals_scored_avg: float = 0.0
    away_goals_conceded_avg: float = 0.0
    away_shots_avg: float = 0.0
    away_possession_avg: float = 50.0
    away_form_points: float = 0.0
    away_btts_rate: float = 0.0
    h2h_home_wins: int = 0
    h2h_draws: int = 0
    h2h_away_wins: int = 0
    h2h_btts_rate: float = 0.0
    h2h_avg_total_goals: float = 0.0
    h2h_matches: int = 0
    player_stat_avg: float = 0.0
    player_stat_std: float = 0.0
    player_minutes_avg: float = 0.0
    player_form_trend: float = 0.0
    is_home: bool = True
    days_since_last_match_home: int = 7
    days_since_last_match_away: int = 7
    weather_wind_speed: float = 0.0
    weather_precipitation: float = 0.0
    is_rivalry: bool = False
    venue_altitude: float = 0.0
    line_movement: float = 0.0
    sharp_money_indicator: float = 0.0
    market_age_hours: float = 0.0
    poly_yes_price: float | None = None
    kalshi_yes_price: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)


def _safe(val: float | None, default: float = 0.0) -> float:
    if val is None or math.isnan(val) or math.isinf(val):
        return default
    return float(val)


def engineer_features(ctx: MarketContext) -> dict[str, float]:
    f: dict[str, float] = {}

    f["implied_prob"] = _safe(ctx.implied_probability, 0.5)
    f["decimal_odds"] = _safe(ctx.decimal_odds, 2.0)
    f["log_odds"] = math.log(max(ctx.decimal_odds, 1.001))
    f["overround"] = _safe(1 / ctx.decimal_odds)
    f["opening_odds"] = _safe(ctx.opening_decimal_odds, ctx.decimal_odds)
    f["line_movement"] = _safe(ctx.line_movement)
    f["sharp_money"] = _safe(ctx.sharp_money_indicator)
    f["volume_log"] = math.log1p(_safe(ctx.volume, 0.0))
    f["market_age_h"] = _safe(ctx.market_age_hours)

    f["home_goals_scored"] = _safe(ctx.home_goals_scored_avg)
    f["home_goals_conceded"] = _safe(ctx.home_goals_conceded_avg)
    f["home_shots"] = _safe(ctx.home_shots_avg)
    f["home_possession"] = _safe(ctx.home_possession_avg, 50.0)
    f["home_form"] = _safe(ctx.home_form_points)
    f["home_btts_rate"] = _safe(ctx.home_btts_rate)

    f["away_goals_scored"] = _safe(ctx.away_goals_scored_avg)
    f["away_goals_conceded"] = _safe(ctx.away_goals_conceded_avg)
    f["away_shots"] = _safe(ctx.away_shots_avg)
    f["away_possession"] = _safe(ctx.away_possession_avg, 50.0)
    f["away_form"] = _safe(ctx.away_form_points)
    f["away_btts_rate"] = _safe(ctx.away_btts_rate)

    f["total_goals_expectation"] = f["home_goals_scored"] + f["away_goals_scored"]
    f["goal_diff_expectation"] = f["home_goals_scored"] - f["away_goals_scored"]
    f["btts_combined_rate"] = f["home_btts_rate"] * f["away_btts_rate"]
    f["defense_weakness"] = f["home_goals_conceded"] + f["away_goals_conceded"]

    total_h2h = max(ctx.h2h_matches, 1)
    f["h2h_home_win_rate"] = ctx.h2h_home_wins / total_h2h
    f["h2h_draw_rate"] = ctx.h2h_draws / total_h2h
    f["h2h_away_win_rate"] = ctx.h2h_away_wins / total_h2h
    f["h2h_btts_rate"] = _safe(ctx.h2h_btts_rate)
    f["h2h_avg_goals"] = _safe(ctx.h2h_avg_total_goals)
    f["h2h_matches"] = float(ctx.h2h_matches)

    f["player_stat_avg"] = _safe(ctx.player_stat_avg)
    f["player_stat_std"] = _safe(ctx.player_stat_std)
    f["player_minutes_avg"] = _safe(ctx.player_minutes_avg)
    f["player_form_trend"] = _safe(ctx.player_form_trend)
    if ctx.player_stat_std and ctx.player_stat_std > 0 and ctx.line is not None:
        f["player_line_z"] = (ctx.line - ctx.player_stat_avg) / ctx.player_stat_std
    else:
        f["player_line_z"] = 0.0

    f["is_home"] = float(ctx.is_home)
    f["home_rest_days"] = float(min(ctx.days_since_last_match_home, 30))
    f["away_rest_days"] = float(min(ctx.days_since_last_match_away, 30))
    f["rest_advantage"] = f["home_rest_days"] - f["away_rest_days"]
    f["wind_speed"] = _safe(ctx.weather_wind_speed)
    f["precipitation"] = _safe(ctx.weather_precipitation)
    f["is_rivalry"] = float(ctx.is_rivalry)
    f["altitude"] = _safe(ctx.venue_altitude)

    market_types = [
        "moneyline", "spread", "total", "btts", "team_goals",
        "player_shots", "player_passes", "player_tackles", "player_saves",
        "corners", "cards", "assists", "anytime_scorer", "possession",
        "prediction_market",
    ]
    for mt in market_types:
        f[f"mt_{mt}"] = float(ctx.market_type == mt)

    for sp in ["soccer", "basketball", "american_football", "baseball", "hockey", "tennis"]:
        f[f"sp_{sp}"] = float(ctx.sport == sp)

    f["poly_yes_price"] = _safe(ctx.poly_yes_price, 0.5)
    f["kalshi_yes_price"] = _safe(ctx.kalshi_yes_price, 0.5)
    f["pred_market_consensus"] = (f["poly_yes_price"] + f["kalshi_yes_price"]) / 2

    return f


def features_to_array(features: dict[str, float]) -> np.ndarray:
    return np.array([features[k] for k in sorted(features.keys())], dtype=np.float32)


FEATURE_NAMES: list[str] = sorted(engineer_features(MarketContext()).keys())
