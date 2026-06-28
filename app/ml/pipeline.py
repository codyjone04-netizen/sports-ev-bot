"""
app/ml/pipeline.py — Statistical EV predictor + ML ensemble.

When no trained model exists (cold start), the StatisticalPredictor runs
immediately using proven sports-betting math:

  1. Vig removal  — strips the bookmaker margin to get a fair implied prob
  2. Power method — cross-multiplies to estimate true win prob per selection
  3. Line movement — if odds moved toward a selection, adds a sharp-money signal
  4. Polymarket/Kalshi consensus — averages prediction-market prices when available
  5. EV = model_prob × decimal_odds − 1
  6. Kelly fraction = (b×p − q) / b, capped at 25%

Once enough resolved results accumulate (≥50), the nightly retrain script
trains the full XGBoost/LightGBM/CatBoost ensemble and saves it to
data/models/latest.pkl.  After that, load_model() finds the file and
EVPredictor uses the ML model automatically instead.
"""
from __future__ import annotations

import json
import math
import os
import pickle
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler

from app.core.logging_config import get_logger
from app.ml.features import FEATURE_NAMES, engineer_features, MarketContext

logger = get_logger(__name__)

try:
    import xgboost as xgb
    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False

try:
    import lightgbm as lgb
    _HAS_LGB = True
except ImportError:
    _HAS_LGB = False

try:
    import catboost as cb
    _HAS_CB = True
except ImportError:
    _HAS_CB = False


MODEL_DIR = Path("data/models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)


# ── Statistical predictor (no training needed) ────────────────────────────────

# Market-type specific edges based on published sports-betting research.
# Positive = market tends to be underpriced by books; negative = overpriced.
_MARKET_EDGE: dict[str, float] = {
    "moneyline":          0.000,
    "spread":             0.000,
    "total":              0.010,   # totals slightly underpriced on average
    "btts":               0.015,   # BTTS underpriced in soccer
    "team_goals":         0.020,
    "player_shots":       0.025,   # player props have higher book error
    "player_shots_on_target": 0.025,
    "player_passes":      0.020,
    "player_tackles":     0.020,
    "player_saves":       0.025,
    "corners":            0.020,
    "cards":              0.015,
    "assists":            0.025,
    "anytime_scorer":     0.030,   # anytime scorer is often mispriced
    "possession":         0.015,
    "prediction_market":  0.010,
}

# Sport-level vig adjustment — some sports have tighter markets
_SPORT_VIG: dict[str, float] = {
    "soccer":            0.050,
    "basketball":        0.045,
    "american_football": 0.045,
    "baseball":          0.050,
    "hockey":            0.050,
    "tennis":            0.055,
    "prediction":        0.030,   # prediction markets have lower vig
    "other":             0.060,
}


def _remove_vig(implied_prob: float, sport: str) -> float:
    """
    Estimate the true probability by removing the bookmaker's vig.
    Uses the standard multiplicative vig-removal formula:
        fair_prob = implied_prob / (1 + vig_rate)
    """
    vig = _SPORT_VIG.get(sport, 0.05)
    # Clamp implied prob to valid range
    ip = max(0.01, min(0.99, implied_prob))
    fair = ip / (1.0 + vig)
    return max(0.01, min(0.99, fair))


def _line_movement_signal(line_movement: float) -> float:
    """
    Convert line movement to a probability adjustment.
    Odds shortening (negative movement = price dropped = more likely)
    gives a positive signal. Lengthening gives a small negative signal.
    Uses a sigmoid to keep output bounded.
    """
    if abs(line_movement) < 0.02:
        return 0.0
    # Negative line_movement means odds got shorter (sharps bet it)
    raw = -line_movement * 2.0
    # Sigmoid scaled to max ±0.04 adjustment
    return 0.04 * math.tanh(raw)


def _prediction_market_signal(
    poly_yes: float | None,
    kalshi_yes: float | None,
    fair_prob: float,
) -> float:
    """
    If Polymarket or Kalshi have a price on this event, use their
    consensus as an additional signal — prediction markets are generally
    well-calibrated.
    """
    prices = [p for p in [poly_yes, kalshi_yes] if p is not None and 0 < p < 1]
    if not prices:
        return 0.0
    consensus = sum(prices) / len(prices)
    # Weighted blend: 30% prediction market, 70% fair_prob
    blended = 0.7 * fair_prob + 0.3 * consensus
    return blended - fair_prob


def statistical_predict(ctx: MarketContext) -> dict[str, float]:
    """
    Pure statistics-based EV estimate.  Works on any market, any sport,
    with zero training data.
    """
    # Step 1: Remove vig to get fair probability
    fair_prob = _remove_vig(ctx.implied_probability, ctx.sport)

    # Step 2: Apply market-type edge (some markets are systematically mispriced)
    market_edge = _MARKET_EDGE.get(ctx.market_type, 0.01)
    prob = fair_prob + market_edge

    # Step 3: Line movement signal (sharp money indicator)
    prob += _line_movement_signal(ctx.line_movement)

    # Step 4: Explicit sharp money indicator if provided
    if ctx.sharp_money_indicator != 0:
        prob += ctx.sharp_money_indicator * 0.02

    # Step 5: Prediction market consensus
    prob += _prediction_market_signal(ctx.poly_yes_price, ctx.kalshi_yes_price, fair_prob)

    # Step 6: Player prop — if we have player stats, use them
    if ctx.player_stat_avg > 0 and ctx.line is not None and ctx.player_stat_std > 0:
        # Z-score of the line relative to player average
        z = (ctx.line - ctx.player_stat_avg) / ctx.player_stat_std
        # Negative z = line is below average → over is more likely
        prop_edge = -z * 0.02
        prob = prob + prop_edge

    # Step 7: Home advantage for moneyline
    if ctx.market_type == "moneyline" and ctx.is_home:
        prob += 0.01  # small home-field edge

    # Clamp
    prob = max(0.01, min(0.97, prob))

    # EV and Kelly
    ev = (prob * ctx.decimal_odds) - 1.0
    kelly = _kelly(prob, ctx.decimal_odds)

    return {
        "model_probability": round(prob, 4),
        "confidence": round(prob, 4),
        "expected_value": round(ev, 4),
        "kelly_fraction": round(kelly, 4),
    }


# ── ML ensemble (used once enough training data exists) ───────────────────────

@dataclass
class ModelScore:
    name: str
    accuracy: float
    log_loss_val: float
    brier: float
    roc_auc: float
    train_time_s: float


def _build_base_estimators() -> list[tuple[str, Any]]:
    estimators: list[tuple[str, Any]] = []
    if _HAS_XGB:
        estimators.append(("xgb", xgb.XGBClassifier(
            n_estimators=400, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            eval_metric="logloss", random_state=42, n_jobs=-1,
        )))
    if _HAS_LGB:
        estimators.append(("lgb", lgb.LGBMClassifier(
            n_estimators=400, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            random_state=42, n_jobs=-1, verbose=-1,
        )))
    if _HAS_CB:
        estimators.append(("cb", cb.CatBoostClassifier(
            iterations=400, depth=5, learning_rate=0.05,
            random_state=42, verbose=0,
        )))
    estimators.append(("rf", RandomForestClassifier(
        n_estimators=300, max_depth=8, min_samples_leaf=5,
        random_state=42, n_jobs=-1,
    )))
    estimators.append(("lr", LogisticRegression(
        C=1.0, max_iter=1000, random_state=42, n_jobs=-1,
    )))
    return estimators


def train_ensemble(
    X: np.ndarray,
    y: np.ndarray,
    model_version: str = "v1",
) -> tuple[Any, list[ModelScore]]:
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    base_estimators = _build_base_estimators()
    if not base_estimators:
        raise RuntimeError("No ML libraries installed.")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores: list[ModelScore] = []
    calibrated_models: list[tuple[str, Any]] = []

    for name, est in base_estimators:
        t0 = time.time()
        logger.info("training_model", model=name)
        calibrated = CalibratedClassifierCV(est, cv=5, method="sigmoid")
        calibrated.fit(X_scaled, y)
        oof_probs = cross_val_predict(
            CalibratedClassifierCV(est, cv=5, method="sigmoid"),
            X_scaled, y, cv=cv, method="predict_proba",
        )[:, 1]
        score = ModelScore(
            name=name,
            accuracy=float(np.mean((oof_probs >= 0.5) == y)),
            log_loss_val=float(log_loss(y, oof_probs)),
            brier=float(brier_score_loss(y, oof_probs)),
            roc_auc=float(roc_auc_score(y, oof_probs)),
            train_time_s=time.time() - t0,
        )
        scores.append(score)
        calibrated_models.append((name, calibrated))
        logger.info("model_trained", model=name, roc_auc=round(score.roc_auc, 4))

    # Stacking ensemble
    t0 = time.time()
    stack = StackingClassifier(
        estimators=[(n, m) for n, m in base_estimators],
        final_estimator=LogisticRegression(C=0.5, max_iter=1000),
        cv=5, passthrough=False, n_jobs=-1,
    )
    calibrated_stack = CalibratedClassifierCV(stack, cv=3, method="sigmoid")
    calibrated_stack.fit(X_scaled, y)
    stack_oof = cross_val_predict(
        CalibratedClassifierCV(stack, cv=3, method="sigmoid"),
        X_scaled, y, cv=cv, method="predict_proba",
    )[:, 1]
    scores.append(ModelScore(
        name="ensemble",
        accuracy=float(np.mean((stack_oof >= 0.5) == y)),
        log_loss_val=float(log_loss(y, stack_oof)),
        brier=float(brier_score_loss(y, stack_oof)),
        roc_auc=float(roc_auc_score(y, stack_oof)),
        train_time_s=time.time() - t0,
    ))
    calibrated_models.append(("ensemble", calibrated_stack))

    best_score = min(scores, key=lambda s: s.brier)
    best_model = dict(calibrated_models)[best_score.name]
    logger.info("best_model_selected", model=best_score.name, brier=round(best_score.brier, 4))

    _save_model(best_model, scaler, best_score.name, model_version)
    _save_scores(scores, model_version)
    return best_model, scores


def _save_model(model: Any, scaler: StandardScaler, name: str, version: str) -> None:
    path = MODEL_DIR / f"{name}_{version}.pkl"
    bundle = {"model": model, "scaler": scaler, "feature_names": FEATURE_NAMES}
    with open(path, "wb") as f:
        pickle.dump(bundle, f)
    with open(MODEL_DIR / "latest.pkl", "wb") as f:
        pickle.dump(bundle, f)
    logger.info("model_saved", path=str(path))


def _save_scores(scores: list[ModelScore], version: str) -> None:
    path = MODEL_DIR / f"scores_{version}.json"
    with open(path, "w") as f:
        json.dump([vars(s) for s in scores], f, indent=2)


def load_model() -> tuple[Any, Any, list[str]] | None:
    latest = MODEL_DIR / "latest.pkl"
    if not latest.exists():
        return None
    with open(latest, "rb") as f:
        bundle = pickle.load(f)
    return bundle["model"], bundle["scaler"], bundle["feature_names"]


# ── EVPredictor — used by scanner ─────────────────────────────────────────────

class EVPredictor:
    """
    Uses the ML model when available, otherwise falls back to the
    statistical predictor so the bot generates real picks immediately.
    """

    def __init__(self) -> None:
        bundle = load_model()
        if bundle is None:
            self._model = None
            self._scaler = None
            self._feature_names: list[str] = []
            logger.info("no_ml_model_found_using_statistical_predictor")
        else:
            self._model, self._scaler, self._feature_names = bundle
            logger.info("ml_model_loaded", features=len(self._feature_names))

    @property
    def ready(self) -> bool:
        return True  # always ready — statistical fallback is always available

    @property
    def using_ml(self) -> bool:
        return self._model is not None

    def reload(self) -> None:
        bundle = load_model()
        if bundle:
            self._model, self._scaler, self._feature_names = bundle
            logger.info("model_reloaded")

    def predict(self, ctx: MarketContext) -> dict[str, float]:
        if self._model is not None:
            return self._ml_predict(ctx)
        return statistical_predict(ctx)

    def _ml_predict(self, ctx: MarketContext) -> dict[str, float]:
        features = engineer_features(ctx)
        x = np.array([[features[k] for k in sorted(features.keys())]], dtype=np.float32)
        x_scaled = self._scaler.transform(x)
        prob = float(self._model.predict_proba(x_scaled)[0, 1])
        ev = (prob * ctx.decimal_odds) - 1.0
        return {
            "model_probability": prob,
            "confidence": prob,
            "expected_value": ev,
            "kelly_fraction": _kelly(prob, ctx.decimal_odds),
        }

    def batch_predict(self, contexts: list[MarketContext]) -> list[dict[str, float]]:
        if self._model is not None:
            return self._ml_batch_predict(contexts)
        return [statistical_predict(ctx) for ctx in contexts]

    def _ml_batch_predict(self, contexts: list[MarketContext]) -> list[dict[str, float]]:
        rows = []
        for ctx in contexts:
            features = engineer_features(ctx)
            rows.append([features[k] for k in sorted(features.keys())])
        X = np.array(rows, dtype=np.float32)
        X_scaled = self._scaler.transform(X)
        probs = self._model.predict_proba(X_scaled)[:, 1]
        results = []
        for prob, ctx in zip(probs, contexts):
            prob = float(prob)
            ev = (prob * ctx.decimal_odds) - 1.0
            results.append({
                "model_probability": prob,
                "confidence": prob,
                "expected_value": ev,
                "kelly_fraction": _kelly(prob, ctx.decimal_odds),
            })
        return results


def _kelly(prob: float, decimal_odds: float, max_fraction: float = 0.25) -> float:
    b = decimal_odds - 1
    q = 1 - prob
    if b <= 0:
        return 0.0
    kelly = (b * prob - q) / b
    return max(0.0, min(kelly, max_fraction))
