"""
scripts/retrain.py — Nightly retraining pipeline.

Pulls resolved predictions from the DB, builds a training dataset,
trains the ensemble, and writes a ModelRun audit record.

Run via cron or APScheduler:
    python -m scripts.retrain
"""
from __future__ import annotations

import asyncio
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# Make sure project root is on PYTHONPATH when run directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.database import AsyncSessionLocal, init_db
from app.core.logging_config import configure_logging, get_logger
from app.core.models import Market, ModelRun, Prediction, PredictionResult, ResultOutcome
from app.ml.features import MarketContext, engineer_features
from app.ml.pipeline import FEATURE_NAMES, train_ensemble
from sqlalchemy import select

configure_logging()
logger = get_logger("retrain")


async def build_training_data() -> tuple[np.ndarray, np.ndarray]:
    """
    Pull predictions + outcomes from DB, build (X, y) arrays.
    Each row = feature vector for one prediction.
    y = 1 if WIN, 0 if LOSS.
    """
    async with AsyncSessionLocal() as session:
        stmt = (
            select(Prediction, Market, PredictionResult)
            .join(Market, Prediction.market_id == Market.id)
            .join(PredictionResult, PredictionResult.prediction_id == Prediction.id)
            .where(PredictionResult.outcome.in_([ResultOutcome.WIN, ResultOutcome.LOSS]))
        )
        rows = (await session.execute(stmt)).all()

    if not rows:
        logger.warning("no_training_data", rows=0)
        return np.empty((0, len(FEATURE_NAMES))), np.empty(0)

    X_rows, y_list = [], []
    for pred, market, result in rows:
        # Re-construct a minimal MarketContext from stored data
        ctx = MarketContext(
            market_type=market.market_type.value if market.market_type else "other",
            sport=market.sport.value if market.sport else "other",
            decimal_odds=market.decimal_odds,
            implied_probability=market.implied_probability,
            opening_decimal_odds=market.opening_odds,
            line_movement=market.line_movement or 0.0,
            line=market.line,
            home_team=market.home_team or "",
            away_team=market.away_team or "",
            player_name=market.player_name or "",
            volume=market.volume,
            # Stored features snapshot would go here in production
        )
        features = engineer_features(ctx)
        X_rows.append([features[k] for k in sorted(features.keys())])
        y_list.append(1 if result.outcome == ResultOutcome.WIN else 0)

    X = np.array(X_rows, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)
    logger.info("training_data_built", samples=len(y), positive_rate=float(y.mean()))
    return X, y


async def record_model_run(scores: list, version: str, n_train: int) -> None:
    """Write training metrics to ModelRun table."""
    async with AsyncSessionLocal() as session:
        # Mark all previous runs inactive
        stmt = select(ModelRun).where(ModelRun.is_active == True)
        prev = (await session.execute(stmt)).scalars().all()
        for r in prev:
            r.is_active = False

        best = min(scores, key=lambda s: s.brier)
        run = ModelRun(
            id=uuid.uuid4(),
            model_name=best.name,
            version=version,
            train_samples=n_train,
            val_samples=int(n_train * 0.2),
            accuracy=best.accuracy,
            log_loss=best.log_loss_val,
            brier_score=best.brier,
            roc_auc=best.roc_auc,
            is_active=True,
            trained_at=datetime.now(timezone.utc),
        )
        session.add(run)
        await session.commit()
        logger.info("model_run_recorded", model=best.name, brier=round(best.brier, 4))


async def main() -> None:
    await init_db()
    logger.info("retrain_started")

    X, y = await build_training_data()
    if len(y) < 50:
        logger.warning("insufficient_data", samples=len(y), required=50)
        return

    version = datetime.now(timezone.utc).strftime("v%Y%m%d_%H%M")
    logger.info("training_with_samples", n=len(y), version=version)

    _, scores = train_ensemble(X, y, model_version=version)
    await record_model_run(scores, version, n_train=len(y))

    logger.info("retrain_complete", version=version)


if __name__ == "__main__":
    asyncio.run(main())
