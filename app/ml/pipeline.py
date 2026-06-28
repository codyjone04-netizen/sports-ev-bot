"""
app/ml/pipeline.py — Statistical EV predictor + ML ensemble.
"""
from __future__ import annotations
import json, math, pickle, time
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
    import xgboost as xgb; _HAS_XGB = True
except ImportError:
    _HAS_XGB = False
try:
    import lightgbm as lgb; _HAS_LGB = True
except ImportError:
    _HAS_LGB = False
try:
    import catboost as cb; _HAS_CB = True
except ImportError:
    _HAS_CB = False

MODEL_DIR = Path("data/models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

_MARKET_EDGE = {
    "moneyline": 0.000, "spread": 0.000, "total": 0.010,
    "btts": 0.015, "team_goals": 0.020, "player_shots": 0.025,
    "player_shots_on_target": 0.025, "player_passes": 0.020,
    "player_tackles": 0.020, "player_saves": 0.025, "corners": 0.020,
    "cards": 0.015, "assists": 0.025, "anytime_scorer": 0.030,
    "possession": 0.015, "prediction_market": 0.010,
}
_SPORT_VIG = {
    "soccer": 0.050, "basketball": 0.045, "american_football": 0.045,
    "baseball": 0.050, "hockey": 0.050, "tennis": 0.055,
    "prediction": 0.030, "other": 0.060,
}

def _remove_vig(implied_prob: float, sport: str) -> float:
    vig = _SPORT_VIG.get(sport, 0.05)
    ip = max(0.01, min(0.99, implied_prob))
    return max(0.01, min(0.99, ip / (1.0 + vig)))

def _line_movement_signal(line_movement: float) -> float:
    if abs(line_movement) < 0.02:
        return 0.0
    return 0.04 * math.tanh(-line_movement * 2.0)

def _prediction_market_signal(poly_yes, kalshi_yes, fair_prob):
    prices = [p for p in [poly_yes, kalshi_yes] if p is not None and 0 < p < 1]
    if not prices:
        return 0.0
    consensus = sum(prices) / len(prices)
    return (0.7 * fair_prob + 0.3 * consensus) - fair_prob

def statistical_predict(ctx: MarketContext) -> dict[str, float]:
    fair_prob = _remove_vig(ctx.implied_probability, ctx.sport)
    prob = fair_prob + _MARKET_EDGE.get(ctx.market_type, 0.01)
    prob += _line_movement_signal(ctx.line_movement)
    if ctx.sharp_money_indicator != 0:
        prob += ctx.sharp_money_indicator * 0.02
    prob += _prediction_market_signal(ctx.poly_yes_price, ctx.kalshi_yes_price, fair_prob)
    if ctx.player_stat_avg > 0 and ctx.line is not None and ctx.player_stat_std > 0:
        z = (ctx.line - ctx.player_stat_avg) / ctx.player_stat_std
        prob += -z * 0.02
    if ctx.market_type == "moneyline" and ctx.is_home:
        prob += 0.01
    prob = max(0.01, min(0.97, prob))
    ev = (prob * ctx.decimal_odds) - 1.0
    return {
        "model_probability": round(prob, 4),
        "confidence": round(prob, 4),
        "expected_value": round(ev, 4),
        "kelly_fraction": round(_kelly(prob, ctx.decimal_odds), 4),
    }

@dataclass
class ModelScore:
    name: str; accuracy: float; log_loss_val: float
    brier: float; roc_auc: float; train_time_s: float

def _build_base_estimators():
    est = []
    if _HAS_XGB:
        est.append(("xgb", xgb.XGBClassifier(n_estimators=400, max_depth=5, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, eval_metric="logloss", random_state=42, n_jobs=-1)))
    if _HAS_LGB:
        est.append(("lgb", lgb.LGBMClassifier(n_estimators=400, max_depth=5, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1, verbose=-1)))
    if _HAS_CB:
        est.append(("cb", cb.CatBoostClassifier(iterations=400, depth=5, learning_rate=0.05, random_state=42, verbose=0)))
    est.append(("rf", RandomForestClassifier(n_estimators=300, max_depth=8, min_samples_leaf=5, random_state=42, n_jobs=-1)))
    est.append(("lr", LogisticRegression(C=1.0, max_iter=1000, random_state=42, n_jobs=-1)))
    return est

def train_ensemble(X, y, model_version="v1"):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    base_estimators = _build_base_estimators()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores, calibrated_models = [], []
    for name, est in base_estimators:
        t0 = time.time()
        cal = CalibratedClassifierCV(est, cv=5, method="sigmoid")
        cal.fit(X_scaled, y)
        oof = cross_val_predict(CalibratedClassifierCV(est, cv=5, method="sigmoid"), X_scaled, y, cv=cv, method="predict_proba")[:, 1]
        scores.append(ModelScore(name=name, accuracy=float(np.mean((oof >= 0.5) == y)), log_loss_val=float(log_loss(y, oof)), brier=float(brier_score_loss(y, oof)), roc_auc=float(roc_auc_score(y, oof)), train_time_s=time.time()-t0))
        calibrated_models.append((name, cal))
    stack = StackingClassifier(estimators=[(n, m) for n, m in base_estimators], final_estimator=LogisticRegression(C=0.5, max_iter=1000), cv=5, n_jobs=-1)
    cal_stack = CalibratedClassifierCV(stack, cv=3, method="sigmoid")
    cal_stack.fit(X_scaled, y)
    stack_oof = cross_val_predict(CalibratedClassifierCV(stack, cv=3, method="sigmoid"), X_scaled, y, cv=cv, method="predict_proba")[:, 1]
    scores.append(ModelScore(name="ensemble", accuracy=float(np.mean((stack_oof >= 0.5) == y)), log_loss_val=float(log_loss(y, stack_oof)), brier=float(brier_score_loss(y, stack_oof)), roc_auc=float(roc_auc_score(y, stack_oof)), train_time_s=0))
    calibrated_models.append(("ensemble", cal_stack))
    best = min(scores, key=lambda s: s.brier)
    best_model = dict(calibrated_models)[best.name]
    _save_model(best_model, scaler, best.name, model_version)
    _save_scores(scores, model_version)
    return best_model, scores

def _save_model(model, scaler, name, version):
    bundle = {"model": model, "scaler": scaler, "feature_names": FEATURE_NAMES}
    with open(MODEL_DIR / f"{name}_{version}.pkl", "wb") as f: pickle.dump(bundle, f)
    with open(MODEL_DIR / "latest.pkl", "wb") as f: pickle.dump(bundle, f)

def _save_scores(scores, version):
    with open(MODEL_DIR / f"scores_{version}.json", "w") as f: json.dump([vars(s) for s in scores], f, indent=2)

def load_model():
    latest = MODEL_DIR / "latest.pkl"
    if not latest.exists(): return None
    with open(latest, "rb") as f: bundle = pickle.load(f)
    return bundle["model"], bundle["scaler"], bundle["feature_names"]

class EVPredictor:
    def __init__(self):
        bundle = load_model()
        if bundle is None:
            self._model = self._scaler = None
            self._feature_names = []
            logger.info("no_ml_model_found_using_statistical_predictor")
        else:
            self._model, self._scaler, self._feature_names = bundle
            logger.info("ml_model_loaded", features=len(self._feature_names))

    @property
    def ready(self): return True

    @property
    def using_ml(self): return self._model is not None

    def reload(self):
        bundle = load_model()
        if bundle: self._model, self._scaler, self._feature_names = bundle

    def predict(self, ctx):
        if self._model is not None: return self._ml_predict(ctx)
        return statistical_predict(ctx)

    def _ml_predict(self, ctx):
        features = engineer_features(ctx)
        x = np.array([[features[k] for k in sorted(features.keys())]], dtype=np.float32)
        prob = float(self._model.predict_proba(self._scaler.transform(x))[0, 1])
        return {"model_probability": prob, "confidence": prob, "expected_value": (prob * ctx.decimal_odds) - 1.0, "kelly_fraction": _kelly(prob, ctx.decimal_odds)}

    def batch_predict(self, contexts):
        if self._model is not None: return self._ml_batch_predict(contexts)
        return [statistical_predict(ctx) for ctx in contexts]

    def _ml_batch_predict(self, contexts):
        rows = [[engineer_features(ctx)[k] for k in sorted(engineer_features(ctx).keys())] for ctx in contexts]
        probs = self._model.predict_proba(self._scaler.transform(np.array(rows, dtype=np.float32)))[:, 1]
        return [{"model_probability": float(p), "confidence": float(p), "expected_value": (float(p) * ctx.decimal_odds) - 1.0, "kelly_fraction": _kelly(float(p), ctx.decimal_odds)} for p, ctx in zip(probs, contexts)]

def _kelly(prob, decimal_odds, max_fraction=0.25):
    b = decimal_odds - 1
    if b <= 0: return 0.0
    return max(0.0, min((b * prob - (1 - prob)) / b, max_fraction))
