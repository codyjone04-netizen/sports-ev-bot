"""
tests/test_core.py — Unit tests for EV calculation, features, and parlay builder.
Run with: pytest tests/
"""
import math
import pytest
import numpy as np

from app.ml.features import MarketContext, engineer_features, features_to_array, FEATURE_NAMES
from app.ml.parlay_builder import ScoredPick, build_parlays
from app.ml.pipeline import _kelly


# ── Feature engineering ───────────────────────────────────────────────────────

def test_feature_keys_consistent():
    ctx = MarketContext()
    f = engineer_features(ctx)
    assert set(f.keys()) == set(FEATURE_NAMES)


def test_all_features_finite():
    ctx = MarketContext(
        decimal_odds=2.5,
        implied_probability=0.4,
        home_goals_scored_avg=1.8,
        away_goals_scored_avg=1.2,
        h2h_btts_rate=0.6,
        player_stat_avg=3.5,
        player_stat_std=1.2,
        line=3.5,
    )
    f = engineer_features(ctx)
    for k, v in f.items():
        assert math.isfinite(v), f"Feature {k} = {v} is not finite"


def test_feature_array_shape():
    ctx = MarketContext()
    arr = features_to_array(engineer_features(ctx))
    assert arr.shape == (len(FEATURE_NAMES),)
    assert arr.dtype == np.float32


def test_btts_rate_combined():
    ctx = MarketContext(home_btts_rate=0.8, away_btts_rate=0.7)
    f = engineer_features(ctx)
    assert abs(f["btts_combined_rate"] - 0.56) < 1e-6


def test_player_line_z():
    ctx = MarketContext(player_stat_avg=4.0, player_stat_std=1.0, line=5.0)
    f = engineer_features(ctx)
    assert abs(f["player_line_z"] - 1.0) < 1e-6


# ── Kelly criterion ───────────────────────────────────────────────────────────

def test_kelly_positive_ev():
    # 60% chance at 2.0 odds → f = (1*0.6 - 0.4)/1 = 0.2
    k = _kelly(prob=0.6, decimal_odds=2.0)
    assert abs(k - 0.2) < 1e-6


def test_kelly_negative_ev_returns_zero():
    # 40% chance at 2.0 odds → negative edge → 0
    k = _kelly(prob=0.4, decimal_odds=2.0)
    assert k == 0.0


def test_kelly_capped():
    # 95% at 10x → raw Kelly = 84.4% → capped at 25%
    k = _kelly(prob=0.95, decimal_odds=10.0, max_fraction=0.25)
    assert k == 0.25


# ── Parlay builder ────────────────────────────────────────────────────────────

def _make_pick(eid: str, sel: str, odds: float, prob: float, ev: float, conf: float) -> ScoredPick:
    return ScoredPick(
        prediction_id=eid,
        event_name=f"Event {eid}",
        selection=sel,
        market_type="moneyline",
        sport="soccer",
        decimal_odds=odds,
        model_probability=prob,
        expected_value=ev,
        confidence=conf,
        kelly_fraction=0.05,
        home_team=f"Home{eid}",
        away_team=f"Away{eid}",
    )


def test_parlay_builder_returns_combos():
    picks = [
        _make_pick("A", "HomeA", 1.8, 0.65, 0.17, 0.80),
        _make_pick("B", "HomeB", 2.1, 0.70, 0.47, 0.85),
        _make_pick("C", "OverC", 1.9, 0.62, 0.18, 0.78),
    ]
    results = build_parlays(picks, min_confidence=0.60, min_ev=0.05)
    assert 2 in results
    assert 3 in results


def test_parlay_combined_odds():
    picks = [
        _make_pick("X", "SelX", 2.0, 0.60, 0.20, 0.80),
        _make_pick("Y", "SelY", 3.0, 0.65, 0.95, 0.85),
    ]
    results = build_parlays(picks, min_confidence=0.50, min_ev=0.05)
    assert 2 in results
    combo = results[2][0]
    assert abs(combo.combined_odds - 6.0) < 1e-6


def test_correlated_moneyline_rejected():
    # Two moneylines from the same game should be rejected
    p1 = ScoredPick(
        prediction_id="1", event_name="E", selection="HomeA",
        market_type="moneyline", sport="soccer",
        decimal_odds=2.0, model_probability=0.6, expected_value=0.2,
        confidence=0.8, kelly_fraction=0.05,
        home_team="HomeA", away_team="AwayA",
    )
    p2 = ScoredPick(
        prediction_id="2", event_name="E", selection="AwayA",
        market_type="moneyline", sport="soccer",
        decimal_odds=2.5, model_probability=0.55, expected_value=0.375,
        confidence=0.82, kelly_fraction=0.06,
        home_team="HomeA", away_team="AwayA",
    )
    results = build_parlays([p1, p2], min_confidence=0.50, min_ev=0.0, max_correlation=0.25)
    # Should have no 2-leg combos since the only pair is correlated
    assert 2 not in results or len(results.get(2, [])) == 0


def test_empty_picks_returns_empty():
    assert build_parlays([]) == {}


def test_insufficient_picks():
    picks = [_make_pick("A", "X", 2.0, 0.6, 0.2, 0.8)]
    assert build_parlays(picks) == {}
