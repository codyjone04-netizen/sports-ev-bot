"""
app/core/models.py — SQLAlchemy ORM models.

Tables
------
markets          : snapshot of every scraped market opportunity
predictions      : model output for each market
parlays          : constructed multi-leg combinations
parlay_legs      : join table for parlay → predictions
results          : outcome recorded after event resolves
model_runs       : audit log for ML training runs
"""
from __future__ import annotations

import enum
import uuid
from datetime import datetime

from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import DeclarativeBase, relationship
from sqlalchemy.sql import func


class Base(DeclarativeBase):
    pass


# ── Enums ─────────────────────────────────────────────────────────────────────

class MarketType(str, enum.Enum):
    MONEYLINE = "moneyline"
    SPREAD = "spread"
    TOTAL = "total"
    BTTS = "btts"
    TEAM_GOALS = "team_goals"
    PLAYER_SHOTS = "player_shots"
    PLAYER_SHOTS_ON_TARGET = "player_shots_on_target"
    PLAYER_PASSES = "player_passes"
    PLAYER_TACKLES = "player_tackles"
    PLAYER_SAVES = "player_saves"
    CORNERS = "corners"
    CARDS = "cards"
    ASSISTS = "assists"
    ANYTIME_SCORER = "anytime_scorer"
    POSSESSION = "possession"
    PREDICTION_MARKET = "prediction_market"
    OTHER = "other"


class Sport(str, enum.Enum):
    SOCCER = "soccer"
    BASKETBALL = "basketball"
    AMERICAN_FOOTBALL = "american_football"
    BASEBALL = "baseball"
    HOCKEY = "hockey"
    TENNIS = "tennis"
    PREDICTION = "prediction"
    OTHER = "other"


class ResultOutcome(str, enum.Enum):
    WIN = "win"
    LOSS = "loss"
    PUSH = "push"
    PENDING = "pending"
    VOID = "void"


class RiskLevel(str, enum.Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


# ── Models ────────────────────────────────────────────────────────────────────

class Market(Base):
    """Raw market snapshot captured from any data source."""
    __tablename__ = "markets"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    external_id = Column(String(256), nullable=True, index=True)  # source's own ID
    source = Column(String(64), nullable=False)                   # e.g. "kalshi", "draftkings"
    sport = Column(Enum(Sport), nullable=False, default=Sport.OTHER)
    market_type = Column(Enum(MarketType), nullable=False)
    event_name = Column(String(512), nullable=False)
    event_date = Column(DateTime(timezone=True), nullable=True)
    home_team = Column(String(256), nullable=True)
    away_team = Column(String(256), nullable=True)
    player_name = Column(String(256), nullable=True)
    selection = Column(String(512), nullable=False)               # e.g. "Brazil", "Over 2.5"
    line = Column(Float, nullable=True)                           # e.g. 2.5 for totals
    decimal_odds = Column(Float, nullable=False)
    american_odds = Column(Integer, nullable=True)
    implied_probability = Column(Float, nullable=False)
    volume = Column(Float, nullable=True)                         # market liquidity
    opening_odds = Column(Float, nullable=True)
    line_movement = Column(Float, nullable=True)                  # decimal_odds - opening_odds
    extra_data = Column(JSON, nullable=True)                      # source-specific extras
    scraped_at = Column(DateTime(timezone=True), server_default=func.now())
    predictions = relationship("Prediction", back_populates="market", cascade="all, delete-orphan")

    __table_args__ = (
        UniqueConstraint("source", "external_id", "scraped_at", name="uq_market_snapshot"),
    )


class Prediction(Base):
    """ML model output for a single market selection."""
    __tablename__ = "predictions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    market_id = Column(UUID(as_uuid=True), ForeignKey("markets.id", ondelete="CASCADE"), nullable=False)
    model_name = Column(String(128), nullable=False)              # e.g. "ensemble_v3"
    model_version = Column(String(64), nullable=True)
    model_probability = Column(Float, nullable=False)             # P(outcome) per model
    confidence = Column(Float, nullable=False)                    # calibrated confidence [0,1]
    expected_value = Column(Float, nullable=False)                # EV as fraction, e.g. 0.12 = +12%
    kelly_fraction = Column(Float, nullable=True)                 # suggested stake fraction
    risk_level = Column(Enum(RiskLevel), nullable=False, default=RiskLevel.MEDIUM)
    features_snapshot = Column(JSON, nullable=True)               # feature values used
    feature_importances = Column(JSON, nullable=True)
    explanation = Column(Text, nullable=True)                     # human-readable rationale
    alert_sent = Column(Boolean, default=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    market = relationship("Market", back_populates="predictions")
    result = relationship("PredictionResult", back_populates="prediction", uselist=False)
    parlay_legs = relationship("ParlayLeg", back_populates="prediction")


class Parlay(Base):
    """Multi-leg bet combination."""
    __tablename__ = "parlays"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    num_legs = Column(Integer, nullable=False)
    combined_odds = Column(Float, nullable=False)                 # product of decimal odds
    combined_ev = Column(Float, nullable=False)
    combined_confidence = Column(Float, nullable=False)           # product of confidences
    risk_level = Column(Enum(RiskLevel), nullable=False)
    correlation_score = Column(Float, default=0.0)                # lower = less correlated
    alert_sent = Column(Boolean, default=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    legs = relationship("ParlayLeg", back_populates="parlay", cascade="all, delete-orphan")


class ParlayLeg(Base):
    """Join table linking a Parlay to its constituent Predictions."""
    __tablename__ = "parlay_legs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    parlay_id = Column(UUID(as_uuid=True), ForeignKey("parlays.id", ondelete="CASCADE"), nullable=False)
    prediction_id = Column(UUID(as_uuid=True), ForeignKey("predictions.id", ondelete="CASCADE"), nullable=False)
    leg_order = Column(Integer, nullable=False)
    parlay = relationship("Parlay", back_populates="legs")
    prediction = relationship("Prediction", back_populates="parlay_legs")


class PredictionResult(Base):
    """Actual outcome recorded after event resolves — used to retrain and track ROI."""
    __tablename__ = "prediction_results"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    prediction_id = Column(UUID(as_uuid=True), ForeignKey("predictions.id", ondelete="CASCADE"), unique=True, nullable=False)
    outcome = Column(Enum(ResultOutcome), nullable=False, default=ResultOutcome.PENDING)
    actual_value = Column(Float, nullable=True)                   # e.g. goals scored
    profit_loss = Column(Float, nullable=True)                    # in units
    resolved_at = Column(DateTime(timezone=True), nullable=True)
    notes = Column(Text, nullable=True)
    prediction = relationship("Prediction", back_populates="result")


class ModelRun(Base):
    """Audit log for each ML training run."""
    __tablename__ = "model_runs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    model_name = Column(String(128), nullable=False)
    version = Column(String(64), nullable=False)
    train_samples = Column(Integer, nullable=True)
    val_samples = Column(Integer, nullable=True)
    accuracy = Column(Float, nullable=True)
    log_loss = Column(Float, nullable=True)
    brier_score = Column(Float, nullable=True)
    roc_auc = Column(Float, nullable=True)
    feature_importances = Column(JSON, nullable=True)
    hyperparams = Column(JSON, nullable=True)
    is_active = Column(Boolean, default=False)
    trained_at = Column(DateTime(timezone=True), server_default=func.now())
