"""
app/api/main.py — FastAPI application.

Endpoints
---------
GET  /health                        — liveness probe
GET  /api/picks/today               — top EV picks sorted by EV descending
GET  /api/parlays/today             — best parlays sorted by combined EV
GET  /api/predictions               — paginated prediction history
GET  /api/stats/roi                 — historical ROI / win-rate summary
GET  /api/stats/model               — model info (ML or statistical)
GET  /api/markets                   — raw market snapshots
POST /api/predictions/{id}/result   — record actual outcome
POST /api/scan                      — trigger an immediate scan (admin)
GET  /dashboard                     — web dashboard (HTML)
"""
from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from sqlalchemy import func, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db, init_db
from app.core.logging_config import configure_logging, get_logger
from app.core.models import (
    Market,
    Parlay,
    Prediction,
    PredictionResult,
    ResultOutcome,
)
from config.settings import get_settings

configure_logging()
logger = get_logger(__name__)
settings = get_settings()

from app.core.scanner import MarketScanner  # noqa: E402
_scanner: MarketScanner | None = None


def get_scanner() -> MarketScanner:
    global _scanner
    if _scanner is None:
        _scanner = MarketScanner()
    return _scanner


app = FastAPI(
    title="Sports EV Bot API",
    description="AI-powered prediction market and sports betting EV analyzer",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup() -> None:
    await init_db()
    scanner = get_scanner()
    asyncio.create_task(scanner.run_forever())
    logger.info("app_started")


# ── Schemas ───────────────────────────────────────────────────────────────────

class RecordResultRequest(BaseModel):
    outcome: str  # win | loss | push | void
    actual_value: Optional[float] = None
    profit_loss: Optional[float] = None
    notes: Optional[str] = None


class ScanResponse(BaseModel):
    markets: int
    predictions: int
    high_ev_picks: int
    alerts: int
    elapsed_s: float


# ── Health ────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "time": datetime.now(timezone.utc).isoformat()}


# ── Picks ─────────────────────────────────────────────────────────────────────

@app.get("/api/picks/today")
async def picks_today(
    limit: int = Query(30, ge=1, le=100),
    min_ev: float = Query(0.0),
    min_conf: float = Query(0.0),
    sport: Optional[str] = None,
    market_type: Optional[str] = None,
):
    """
    Top predictions sorted by EV descending.
    Returns picks from the last 2 hours (most recent scan window).
    Falls back to last 24h if recent window is empty.
    """
    async with get_db() as session:
        def build_stmt(since: datetime):
            s = (
                select(Prediction, Market)
                .join(Market, Prediction.market_id == Market.id)
                .where(Prediction.expected_value >= min_ev)
                .where(Prediction.confidence >= min_conf)
                .where(Prediction.created_at >= since)
                .order_by(Prediction.expected_value.desc())
                .limit(limit)
            )
            if sport:
                s = s.where(Market.sport == sport)
            if market_type:
                s = s.where(Market.market_type == market_type)
            return s

        now = datetime.now(timezone.utc)
        # Try last 2 hours first (current scan)
        rows = (await session.execute(build_stmt(now - timedelta(hours=2)))).all()
        window = "2h"
        # Fall back to last 24h
        if not rows:
            rows = (await session.execute(build_stmt(now - timedelta(hours=24)))).all()
            window = "24h"

        return {
            "window": window,
            "count": len(rows),
            "picks": [
                {
                    "prediction_id": str(pred.id),
                    "event": market.event_name,
                    "selection": market.selection,
                    "market_type": market.market_type.value if market.market_type else "",
                    "sport": market.sport.value if market.sport else "",
                    "source": market.source,
                    "decimal_odds": market.decimal_odds,
                    "american_odds": market.american_odds,
                    "implied_prob": round(market.implied_probability, 4),
                    "model_prob": round(pred.model_probability, 4),
                    "confidence": round(pred.confidence, 4),
                    "expected_value": round(pred.expected_value, 4),
                    "ev_pct": f"{pred.expected_value * 100:+.1f}%",
                    "kelly_fraction": round(pred.kelly_fraction or 0, 4),
                    "kelly_pct": f"{(pred.kelly_fraction or 0) * 100:.1f}%",
                    "risk_level": pred.risk_level.value if pred.risk_level else "",
                    "explanation": pred.explanation,
                    "event_date": market.event_date.isoformat() if market.event_date else None,
                    "created_at": pred.created_at.isoformat() if pred.created_at else None,
                }
                for pred, market in rows
            ],
        }


# ── Parlays ───────────────────────────────────────────────────────────────────

@app.get("/api/parlays/today")
async def parlays_today(
    limit: int = Query(12, ge=1, le=50),
    min_legs: int = Query(2, ge=2, le=5),
    max_legs: int = Query(5, ge=2, le=5),
):
    """Best parlays sorted by combined EV, most recent scan first."""
    async with get_db() as session:
        now = datetime.now(timezone.utc)

        # Try last 2h, fall back to 24h
        for hours in [2, 24]:
            since = now - timedelta(hours=hours)
            stmt = (
                select(Parlay)
                .where(Parlay.created_at >= since)
                .where(Parlay.num_legs >= min_legs)
                .where(Parlay.num_legs <= max_legs)
                .order_by(Parlay.combined_ev.desc())
                .limit(limit)
            )
            parlays = (await session.execute(stmt)).scalars().all()
            if parlays:
                break

        results = []
        for p in parlays:
            legs_stmt = (
                select(Prediction, Market)
                .join(Market, Prediction.market_id == Market.id)
                .join(Prediction.parlay_legs)
                .where(text(f"parlay_legs.parlay_id = '{p.id}'"))
                .order_by(text("parlay_legs.leg_order"))
            )
            leg_rows = (await session.execute(legs_stmt)).all()
            results.append({
                "parlay_id": str(p.id),
                "num_legs": p.num_legs,
                "combined_odds": round(p.combined_odds, 2),
                "combined_ev": round(p.combined_ev, 4),
                "combined_ev_pct": f"{p.combined_ev * 100:+.1f}%",
                "combined_confidence": round(p.combined_confidence, 4),
                "risk_level": p.risk_level.value if p.risk_level else "",
                "correlation_score": round(p.correlation_score or 0, 3),
                "created_at": p.created_at.isoformat() if p.created_at else None,
                "legs": [
                    {
                        "event": market.event_name,
                        "selection": market.selection,
                        "market_type": market.market_type.value if market.market_type else "",
                        "sport": market.sport.value if market.sport else "",
                        "decimal_odds": round(market.decimal_odds, 2),
                        "american_odds": market.american_odds,
                        "ev": round(pred.expected_value, 4),
                        "ev_pct": f"{pred.expected_value * 100:+.1f}%",
                        "confidence": round(pred.confidence, 4),
                        "event_date": market.event_date.isoformat() if market.event_date else None,
                    }
                    for pred, market in leg_rows
                ],
            })
        return {"count": len(results), "parlays": results}


# ── Predictions (paginated history) ──────────────────────────────────────────

@app.get("/api/predictions")
async def predictions_list(
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=200),
    sport: Optional[str] = None,
    market_type: Optional[str] = None,
):
    offset = (page - 1) * page_size
    async with get_db() as session:
        stmt = (
            select(Prediction, Market)
            .join(Market, Prediction.market_id == Market.id)
            .order_by(Prediction.created_at.desc())
            .offset(offset)
            .limit(page_size)
        )
        if sport:
            stmt = stmt.where(Market.sport == sport)
        if market_type:
            stmt = stmt.where(Market.market_type == market_type)
        rows = (await session.execute(stmt)).all()
        return [
            {
                "prediction_id": str(pred.id),
                "market_id": str(market.id),
                "event": market.event_name,
                "selection": market.selection,
                "sport": market.sport.value if market.sport else "",
                "market_type": market.market_type.value if market.market_type else "",
                "decimal_odds": market.decimal_odds,
                "confidence": round(pred.confidence, 4),
                "expected_value": round(pred.expected_value, 4),
                "alert_sent": pred.alert_sent,
                "created_at": pred.created_at.isoformat() if pred.created_at else None,
            }
            for pred, market in rows
        ]


# ── Result recording ──────────────────────────────────────────────────────────

@app.post("/api/predictions/{prediction_id}/result")
async def record_result(prediction_id: str, body: RecordResultRequest):
    """Record the actual outcome of a prediction for model retraining."""
    try:
        pid = uuid.UUID(prediction_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid prediction_id")
    try:
        outcome = ResultOutcome(body.outcome)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid outcome: {body.outcome}")

    async with get_db() as session:
        pred = await session.get(Prediction, pid)
        if not pred:
            raise HTTPException(status_code=404, detail="Prediction not found")
        result = PredictionResult(
            id=uuid.uuid4(),
            prediction_id=pid,
            outcome=outcome,
            actual_value=body.actual_value,
            profit_loss=body.profit_loss,
            resolved_at=datetime.now(timezone.utc),
            notes=body.notes,
        )
        session.add(result)
        await session.commit()
        return {"status": "recorded", "outcome": outcome.value}


# ── Stats ─────────────────────────────────────────────────────────────────────

@app.get("/api/stats/roi")
async def stats_roi():
    async with get_db() as session:
        stmt = select(PredictionResult)
        results = (await session.execute(stmt)).scalars().all()
        if not results:
            return {"total": 0, "wins": 0, "losses": 0, "win_rate": 0, "total_pl": 0, "roi": 0}
        resolved = [r for r in results if r.outcome in (ResultOutcome.WIN, ResultOutcome.LOSS)]
        wins = sum(1 for r in resolved if r.outcome == ResultOutcome.WIN)
        total_pl = sum(r.profit_loss or 0 for r in resolved)
        total_stake = len(resolved) * settings.alert_stake
        return {
            "total": len(resolved),
            "wins": wins,
            "losses": len(resolved) - wins,
            "win_rate": round(wins / max(len(resolved), 1), 4),
            "total_pl": round(total_pl, 2),
            "roi": round(total_pl / max(total_stake, 1), 4),
        }


@app.get("/api/stats/model")
async def stats_model():
    """Model info — shows whether using ML or statistical predictor."""
    from app.core.models import ModelRun
    from app.ml.pipeline import load_model
    using_ml = load_model() is not None
    async with get_db() as session:
        stmt = (
            select(ModelRun)
            .where(ModelRun.is_active == True)
            .order_by(ModelRun.trained_at.desc())
            .limit(1)
        )
        run = (await session.execute(stmt)).scalar_one_or_none()

        # Count total predictions in DB
        total_preds = (await session.execute(select(func.count()).select_from(Prediction))).scalar()
        total_markets = (await session.execute(select(func.count()).select_from(Market))).scalar()

    base = {
        "predictor_mode": "ml_ensemble" if using_ml else "statistical",
        "total_predictions": total_preds,
        "total_markets_scanned": total_markets,
        "thresholds": {
            "min_ev": settings.min_ev_threshold,
            "min_confidence": settings.confidence_threshold,
        },
    }
    if run:
        base.update({
            "model": run.model_name,
            "version": run.version,
            "train_samples": run.train_samples,
            "accuracy": run.accuracy,
            "log_loss": run.log_loss,
            "brier_score": run.brier_score,
            "roc_auc": run.roc_auc,
            "trained_at": run.trained_at.isoformat() if run.trained_at else None,
        })
    return base


# ── Markets ───────────────────────────────────────────────────────────────────

@app.get("/api/markets")
async def markets_list(
    limit: int = Query(50, ge=1, le=500),
    source: Optional[str] = None,
):
    async with get_db() as session:
        stmt = select(Market).order_by(Market.scraped_at.desc()).limit(limit)
        if source:
            stmt = stmt.where(Market.source == source)
        markets = (await session.execute(stmt)).scalars().all()
        return [
            {
                "id": str(m.id),
                "source": m.source,
                "event": m.event_name,
                "selection": m.selection,
                "sport": m.sport.value if m.sport else "",
                "market_type": m.market_type.value if m.market_type else "",
                "decimal_odds": m.decimal_odds,
                "implied_probability": m.implied_probability,
                "scraped_at": m.scraped_at.isoformat() if m.scraped_at else None,
            }
            for m in markets
        ]


# ── Admin ─────────────────────────────────────────────────────────────────────

@app.post("/api/scan")
async def trigger_scan():
    """Manually trigger an immediate market scan and return results."""
    scanner = get_scanner()
    result = await scanner.scan_once()
    return {
        "markets": result.get("markets", 0),
        "predictions": result.get("predictions", 0),
        "high_ev_picks": result.get("high_ev_picks", 0),
        "alerts": result.get("alerts", 0),
        "elapsed_s": result.get("elapsed_s", 0.0),
    }


# ── Dashboard ─────────────────────────────────────────────────────────────────

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard():
    from app.dashboard.html import render_dashboard
    return HTMLResponse(content=render_dashboard())
