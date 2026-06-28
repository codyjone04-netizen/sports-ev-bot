"""
app/ml/parlay_builder.py — Build optimal 2–5 leg parlays from scored predictions.

Algorithm
---------
1. Filter predictions above confidence and EV thresholds.
2. Try all combinations of size 2, 3, 4, 5.
3. Score each combo by combined EV (not just multiplied odds).
4. Reject correlated combos (same event, opposing outcomes).
5. Return the best combo per leg count.
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ScoredPick:
    """A single prediction that passed quality thresholds."""
    prediction_id: str
    event_name: str
    selection: str
    market_type: str
    sport: str
    decimal_odds: float
    model_probability: float
    expected_value: float
    confidence: float
    kelly_fraction: float
    home_team: str = ""
    away_team: str = ""
    player_name: str = ""
    event_date: Any = None
    source: str = ""
    extra: dict = field(default_factory=dict)

    @property
    def event_key(self) -> str:
        """Unique key for the sporting event (to detect correlated legs)."""
        if self.home_team and self.away_team:
            return f"{self.home_team}|{self.away_team}"
        return self.event_name


@dataclass
class ParlayResult:
    legs: list[ScoredPick]
    combined_odds: float        # product of decimal odds
    combined_ev: float          # true combined EV (geometric)
    combined_confidence: float  # geometric mean of leg confidences
    correlation_penalty: float  # 0 = uncorrelated, 1 = fully correlated
    adjusted_score: float       # final ranking score


def _legs_are_correlated(a: ScoredPick, b: ScoredPick) -> bool:
    """
    Return True if two legs share enough risk that combining them is dangerous.
    Rules:
    - Same event AND same market type AND opposing selections → reject
    - Both legs are moneylines from the same game → reject
    """
    same_event = a.event_key == b.event_key
    if not same_event:
        return False
    # Same event — allow BTTS + Over, but block moneyline A + moneyline B from same game
    if a.market_type == "moneyline" and b.market_type == "moneyline":
        return True
    # Block opposing result selections from same event (e.g. home ML + away ML)
    if a.market_type == b.market_type and a.selection != b.selection:
        return True
    return False


def _correlation_penalty(legs: list[ScoredPick]) -> float:
    """
    Simple correlation penalty in [0, 1].
    Each correlated pair contributes 0.5; capped at 1.
    """
    pairs = list(itertools.combinations(legs, 2))
    if not pairs:
        return 0.0
    correlated = sum(1 for a, b in pairs if _legs_are_correlated(a, b))
    return min(correlated / max(len(pairs), 1), 1.0)


def _combined_ev(legs: list[ScoredPick]) -> float:
    """
    True combined EV = P(all legs win) × combined_odds - 1
    P(all win) ≈ product of individual probabilities (independence assumption).
    """
    p_all = math.prod(leg.model_probability for leg in legs)
    combined_odds = math.prod(leg.decimal_odds for leg in legs)
    return (p_all * combined_odds) - 1.0


def _combined_confidence(legs: list[ScoredPick]) -> float:
    """Geometric mean of confidences — penalises low-confidence legs."""
    if not legs:
        return 0.0
    log_sum = sum(math.log(max(leg.confidence, 1e-9)) for leg in legs)
    return math.exp(log_sum / len(legs))


def build_parlays(
    picks: list[ScoredPick],
    min_confidence: float = 0.65,
    min_ev: float = 0.0,
    max_legs: int = 5,
    min_legs: int = 2,
    max_correlation: float = 0.25,
    top_n_per_size: int = 3,
) -> dict[int, list[ParlayResult]]:
    """
    Build and rank parlay combinations.

    Parameters
    ----------
    picks          : list of scored predictions (already filtered by caller)
    min_confidence : minimum per-leg confidence to include
    min_ev         : minimum per-leg EV to include
    max_legs       : maximum combo size (max 5)
    top_n_per_size : return top N combos per leg count

    Returns
    -------
    dict mapping leg_count → list[ParlayResult] sorted descending by adjusted_score
    """
    # Filter to qualifying picks
    eligible = [
        p for p in picks
        if p.confidence >= min_confidence and p.expected_value >= min_ev
    ]

    if len(eligible) < min_legs:
        return {}

    results: dict[int, list[ParlayResult]] = {}

    for n in range(min_legs, min(max_legs, len(eligible)) + 1):
        combos: list[ParlayResult] = []

        for combo in itertools.combinations(eligible, n):
            legs = list(combo)
            penalty = _correlation_penalty(legs)

            if penalty >= max_correlation:
                continue  # too correlated — skip

            c_odds = math.prod(leg.decimal_odds for leg in legs)
            c_ev = _combined_ev(legs)
            c_conf = _combined_confidence(legs)

            # Score: EV adjusted downward by correlation and low confidence
            adjusted = c_ev * (1 - penalty) * c_conf

            combos.append(ParlayResult(
                legs=legs,
                combined_odds=c_odds,
                combined_ev=c_ev,
                combined_confidence=c_conf,
                correlation_penalty=penalty,
                adjusted_score=adjusted,
            ))

        if combos:
            combos.sort(key=lambda r: r.adjusted_score, reverse=True)
            results[n] = combos[:top_n_per_size]

    return results


def format_parlay_summary(parlay: ParlayResult) -> str:
    """Return a human-readable summary string for a parlay."""
    lines = [f"🎰 {len(parlay.legs)}-LEG PARLAY"]
    lines.append(f"   Combined Odds : {parlay.combined_odds:.2f}x")
    lines.append(f"   Combined EV   : {parlay.combined_ev:+.1%}")
    lines.append(f"   Confidence    : {parlay.combined_confidence:.1%}")
    lines.append("")
    for i, leg in enumerate(parlay.legs, 1):
        lines.append(f"   Leg {i}: {leg.selection} ({leg.event_name})")
        lines.append(f"         Odds {leg.decimal_odds:.2f} | EV {leg.expected_value:+.1%} | Conf {leg.confidence:.0%}")
    return "\n".join(lines)
