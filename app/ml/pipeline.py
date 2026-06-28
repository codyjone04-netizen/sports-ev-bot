"""
app/ml/pipeline.py — Train, evaluate, calibrate, and predict with an ensemble.

Models trained
--------------
* XGBoost
* LightGBM
* CatBoost
* RandomForest
* LogisticRegression (calibration baseline)
* Stacking ensemble (LR meta-learner over all base models)

Each model is:
1. Trained on historical match/market data
2. Platt-calibrated so P(win) = actual win rate
3. Scored on held-out val set for auto-selection

The best single model AND the ensemble are both persisted.
"""
from __future__ import annotations

import json
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
from app.ml.features import FEATURE_NAMES, engineer_features, features_to_array, MarketContext

logger = get_logger(__name__)

# Optional imports — degrade gracefully if package not installed
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
            n_estimators=400,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
        )))

    if _HAS_LGB:
        estimators.append(("lgb", lgb.LGBMClassifier(
            n_estimators=400,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )))

    if _HAS_CB:
        estimators.append(("cb", cb.CatBoostClassifier(
            iterations=400,
            depth=5,
            learning_rate=0.05,
            random_state=42,
            verbose=0,
        )))

    estimators.append(("rf", RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1,
    )))

    estimators.append(("lr", LogisticRegression(
        C=1.0,
        max_iter=1000,
        random_state=42,
        n_jobs=-1,
    )))

    return estimators


def train_ensemble(
    X: np.ndarray,
    y: np.ndarray,
    model_version: str = "v1",
) -> tuple[Any, list[ModelScore]]:
    """
    Train all base models + stacking ensemble.
    Returns (best_model, scores_list).
    """
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    base_estimators = _build_base_estimators()
    if not base_estimators:
        raise RuntimeError("No ML libraries installed. Run: pip install xgboost lightgbm catboost scikit-learn")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores: list[ModelScore] = []
    calibrated_models: list[tuple[str, Any]] = []

    for name, est in base_estimators:
        t0 = time.time()
        logger.info("training_model", model=name)

        # Calibrate with Platt scaling via cross-val
        calibrated = CalibratedClassifierCV(est, cv=5, method="sigmoid")
        calibrated.fit(X_scaled, y)

        # OOF predictions for scoring
        oof_probs = cross_val_predict(
            CalibratedClassifierCV(est, cv=5, method="sigmoid"),
            X_scaled, y, cv=cv, method="predict_proba",
        )[:, 1]

        acc = float(np.mean((oof_probs >= 0.5) == y))
        ll = float(log_loss(y, oof_probs))
        bs = float(brier_score_loss(y, oof_probs))
        auc = float(roc_auc_score(y, oof_probs))
        elapsed = time.time() - t0

        score = ModelScore(name=name, accuracy=acc, log_loss_val=ll, brier=bs, roc_auc=auc, train_time_s=elapsed)
        scores.append(score)
        calibrated_models.append((name, calibrated))

        logger.info(
            "model_trained",
            model=name,
            accuracy=round(acc, 4),
            log_loss=round(ll, 4),
            roc_auc=round(auc, 4),
            seconds=round(elapsed, 1),
        )

    # ── Stacking ensemble ─────────────────────────────────────────────────────
    logger.info("training_stacking_ensemble")
    t0 = time.time()

    stack = StackingClassifier(
        estimators=[(n, m) for n, m in base_estimators],
        final_estimator=LogisticRegression(C=0.5, max_iter=1000),
        cv=5,
        passthrough=False,
        n_jobs=-1,
    )
    calibrated_stack = CalibratedClassifierCV(stack, cv=3, method="sigmoid")
    calibrated_stack.fit(X_scaled, y)

    stack_oof = cross_val_predict(
        CalibratedClassifierCV(stack, cv=3, method="sigmoid"),
        X_scaled, y, cv=cv, method="predict_proba",
    )[:, 1]

    stack_score = ModelScore(
        name="ensemble",
        accuracy=float(np.mean((stack_oof >= 0.5) == y)),
        log_loss_val=float(log_loss(y, stack_oof)),
        brier=float(brier_score_loss(y, stack_oof)),
        roc_auc=float(roc_auc_score(y, stack_oof)),
        train_time_s=time.time() - t0,
    )
    scores.append(stack_score)
    calibrated_models.append(("ensemble", calibrated_stack))

    # ── Select best model by Brier score (lower = better) ────────────────────
    best_score = min(scores, key=lambda s: s.brier)
    best_model = dict(calibrated_models)[best_score.name]

    logger.info("best_model_selected", model=best_score.name, brier=round(best_score.brier, 4))

    # ── Persist everything ────────────────────────────────────────────────────
    _save_model(best_model, scaler, best_score.name, model_version)
    _save_scores(scores, model_version)

    return best_model, scores


def _save_model(model: Any, scaler: StandardScaler, name: str, version: str) -> None:
    path = MODEL_DIR / f"{name}_{version}.pkl"
    with open(path, "wb") as f:
        pickle.dump({"model": model, "scaler": scaler, "feature_names": FEATURE_NAMES}, f)
    # Also write as "latest"
    latest = MODEL_DIR / "latest.pkl"
    with open(latest, "wb") as f:
        pickle.dump({"model": model, "scaler": scaler, "feature_names": FEATURE_NAMES}, f)
    logger.info("model_saved", path=str(path))


def _save_scores(scores: list[ModelScore], version: str) -> None:
    data = [vars(s) for s in scores]
    path = MODEL_DIR / f"scores_{version}.json"
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_model() -> tuple[Any, Any, list[str]] | None:
    """Load latest persisted model. Returns (model, scaler, feature_names) or None."""
    latest = MODEL_DIR / "latest.pkl"
    if not latest.exists():
        return None
    with open(latest, "rb") as f:
        bundle = pickle.load(f)
    return bundle["model"], bundle["scaler"], bundle["feature_names"]


class EVPredictor:
    """
    Thin wrapper around the trained ensemble for online prediction.
    Loads the model once and reuses it across many calls.
    """

    def __init__(self) -> None:
        bundle = load_model()
        if bundle is None:
            self._model = None
            self._scaler = None
            self._feature_names: list[str] = []
            logger.warning("no_model_loaded", detail="Run training first")
        else:
            self._model, self._scaler, self._feature_names = bundle
            logger.info("model_loaded", features=len(self._feature_names))

    @property
    def ready(self) -> bool:
        return self._model is not None

    def reload(self) -> None:
        bundle = load_model()
        if bundle:
            self._model, self._scaler, self._feature_names = bundle
            logger.info("model_reloaded")

    def predict(self, ctx: MarketContext) -> dict[str, float]:
        """
        Predict probability, EV, Kelly fraction for a market context.

        Returns
        -------
        dict with keys: model_probability, confidence, expected_value, kelly_fraction
        """
        if not self.ready:
            # Fallback: return implied probability
            ip = ctx.implied_probability
            return {
                "model_probability": ip,
                "confidence": ip,
                "expected_value": 0.0,
                "kelly_fraction": 0.0,
            }

        features = engineer_features(ctx)
        x = np.array([[features[k] for k in sorted(features.keys())]], dtype=np.float32)
        x_scaled = self._scaler.transform(x)

        prob = float(self._model.predict_proba(x_scaled)[0, 1])
        ev = (prob * ctx.decimal_odds) - 1.0
        kelly = _kelly(prob, ctx.decimal_odds)

        return {
            "model_probability": prob,
            "confidence": prob,  # model is calibrated, so prob ≈ confidence
            "expected_value": ev,
            "kelly_fraction": kelly,
        }

    def batch_predict(self, contexts: list[MarketContext]) -> list[dict[str, float]]:
        if not self.ready or not contexts:
            return [self.predict(c) for c in contexts]

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
    """
    Full Kelly criterion, capped at max_fraction.
    f = (b*p - q) / b  where b = decimal_odds - 1, q = 1 - p
    """
    b = decimal_odds - 1
    q = 1 - prob
    if b <= 0:
        return 0.0
    kelly = (b * prob - q) / b
    return max(0.0, min(kelly, max_fraction))
