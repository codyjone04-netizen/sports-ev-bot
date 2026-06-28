"""
app/core/scanner.py — Market scanning orchestration.
"""
from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timezone

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import AsyncSessionLocal
from app.core.logging_config import get_logger
from app.core.models import Market, MarketType, Parlay, ParlayLeg, Prediction, RiskLevel, Sport
from app.ml.features import MarketContext
from app.ml.parlay_builder import ParlayResult, ScoredPick, build_parlays
from app.ml.pipeline import EVPredictor
from config.settings import get_settings

logger = get_logger(__name__)
settings = get_settings()


def _risk_level(ev: float, confidence: float) -> RiskLevel:
    if ev >= 0.15 and confidence >= 0.80:
        return RiskLevel.LOW
    elif ev >= 0.08 and confidence >= 0.70:
        return RiskLevel.MEDIUM
    return RiskLevel.HIGH


def _market_type_enum(mt: str) -> MarketType:
    try:
        return MarketType(mt)
    except ValueError:
        return MarketType.OTHER


def _sport_enum(sp: str) -> Sport:
    try:
        return Sport(sp)
    except ValueError:
        return Sport.OTHER


def _decimal_to_american(dec: float) -> int:
    if dec >= 2.0:
        return int(round((dec - 1) * 100))
    else:
        return int(round(-100 / (dec - 1)))


class MarketScanner:
    def __init__(self) -> None:
        self._predictor = EVPredictor()
        self._discord_alert = None
        self._running = False

    def set_discord_alerter(self, alerter: object) -> None:
        self._discord_alert = alerter

    async def _collect_kalshi(self) -> list[tuple[dict, MarketContext]]:
        try:
            from app.scrapers.kalshi import KalshiScraper
            scraper = KalshiScraper(api_key=settings.kalshi_api_key)
            raw = await scraper.fetch_markets(limit=200)
            return scraper.to_market_contexts(raw)
        except Exception as exc:
            logger.error("kalshi_collect_failed", error=str(exc))
            return []

    async def _collect_polymarket(self) -> list[tuple[dict, MarketContext]]:
        try:
            from app.scrapers.polymarket import PolymarketScraper
            scraper = PolymarketScraper(api_key=settings.polymarket_api_key)
            raw = await scraper.fetch_markets(limit=100)
            return scraper.to_market_contexts(raw)
        except Exception as exc:
            logger.error("polymarket_collect_failed", error=str(exc))
            return []

    async def _collect_sports_odds(self) -> list[tuple[dict, str, MarketContext]]:
        if not settings.odds_api_key:
            return []
        try:
            from app.scrapers.sports import OddsAPIScraper
            scraper = OddsAPIScraper(settings.odds_api_key)
            all_results: list[tuple[dict, str, MarketContext]] = []
            sports = [
                ("soccer_fifa_world_cup", "h2h,totals,btts"),
                ("soccer_epl", "h2h,totals,btts"),
                ("basketball_nba", "h2h,totals"),
                ("americanfootball_nfl", "h2h,spreads,totals"),
            ]
            for sport_key, markets in sports:
                events = await scraper.fetch_odds(sport=sport_key, markets=markets)
                parsed = scraper.parse_events_to_contexts(events, sport_key=sport_key)
                all_results.extend(parsed)
                await asyncio.sleep(0.5)
            return all_results
        except Exception as exc:
            logger.error("sports_odds_collect_failed", error=str(exc))
            return []

    async def _persist_market(self, session: AsyncSession, raw: dict, ctx: MarketContext, source: str) -> Market:
        market = Market(
            id=uuid.uuid4(),
            external_id=str(raw.get("ticker") or raw.get("event_id") or raw.get("slug") or "")[:256],
            source=source,
            sport=_sport_enum(ctx.sport),
            market_type=_market_type_enum(ctx.market_type),
            event_name=str(raw.get("event_name") or raw.get("title") or raw.get("question") or "")[:512],
            event_date=raw.get("event_date") or raw.get("close_time"),
            home_team=str(raw.get("home_team") or ctx.home_team or "")[:256],
            away_team=str(raw.get("away_team") or ctx.away_team or "")[:256],
            player_name=str(raw.get("player_name") or ctx.player_name or "")[:256],
            selection=str(raw.get("selection") or raw.get("outcome") or "YES")[:512],
            line=ctx.line,
            decimal_odds=ctx.decimal_odds,
            american_odds=_decimal_to_american(ctx.decimal_odds),
            implied_probability=ctx.implied_probability,
            volume=ctx.volume,
            opening_odds=ctx.opening_decimal_odds,
            line_movement=ctx.line_movement,
            extra_data=ctx.extra,
            scraped_at=datetime.now(timezone.utc),
        )
        session.add(market)
        return market

    async def _persist_prediction(self, session: AsyncSession, market: Market, pred: dict, explanation: str) -> Prediction:
        ev = pred["expected_value"]
        conf = pred["confidence"]
        risk = _risk_level(ev, conf)
        prediction = Prediction(
            id=uuid.uuid4(),
            market_id=market.id,
            model_name="ensemble",
            model_version="latest",
            model_probability=pred["model_probability"],
            confidence=conf,
            expected_value=ev,
            kelly_fraction=pred["kelly_fraction"],
            risk_level=risk,
            explanation=explanation,
            alert_sent=False,
        )
        session.add(prediction)
        return prediction

    async def _persist_parlay(self, session: AsyncSession, parlay: ParlayResult, prediction_map: dict) -> None:
        db_parlay = Parlay(
            id=uuid.uuid4(),
            num_legs=len(parlay.legs),
            combined_odds=parlay.combined_odds,
            combined_ev=parlay.combined_ev,
            combined_confidence=parlay.combined_confidence,
            risk_level=_risk_level(parlay.combined_ev, parlay.combined_confidence),
            correlation_score=parlay.correlation_penalty,
            alert_sent=False,
        )
        session.add(db_parlay)
        await session.flush()
        for i, leg in enumerate(parlay.legs):
            pred = prediction_map.get(leg.prediction_id)
            if not pred:
                continue
            session.add(ParlayLeg(
                id=uuid.uuid4(),
                parlay_id=db_parlay.id,
                prediction_id=pred.id,
                leg_order=i,
            ))

    def _make_explanation(self, ctx: MarketContext, pred: dict) -> str:
        ev = pred["expected_value"]
        prob = pred["model_probability"]
        ip = ctx.implied_probability
        edge = prob - ip
        parts = [
            f"Model prob: {prob:.1%} vs implied {ip:.1%} (edge: {edge:+.1%})",
            f"EV: {ev:+.1%} | Kelly: {pred['kelly_fraction']:.1%}",
        ]
        if ctx.market_type == "btts":
            parts.append(f"H2H BTTS rate: {ctx.h2h_btts_rate:.0%}")
        elif ctx.market_type in ("player_shots", "player_passes"):
            parts.append(f"Player avg: {ctx.player_stat_avg:.1f}")
        if ctx.line_movement and abs(ctx.line_movement) > 0.1:
            parts.append(f"Line moved: {ctx.line_movement:+.2f}")
        return " | ".join(parts)

    async def scan_once(self) -> dict:
        logger.info("scan_started")
        t0 = asyncio.get_event_loop().time()

        # Collect from all sources
        results = await asyncio.gather(
            self._collect_kalshi(),
            self._collect_polymarket(),
            self._collect_sports_odds(),
            return_exceptions=True,
        )

        kalshi_data = results[0] if not isinstance(results[0], Exception) else []
        poly_data = results[1] if not isinstance(results[1], Exception) else []
        sports_data = results[2] if not isinstance(results[2], Exception) else []

        contexts: list[tuple[str, dict, MarketContext]] = []
        for raw, ctx in kalshi_data:
            contexts.append(("kalshi", raw, ctx))
        for raw, ctx in poly_data:
            contexts.append(("polymarket", raw, ctx))
        for raw, _eid, ctx in sports_data:
            contexts.append(("odds_api", raw, ctx))

        logger.info("markets_collected", total=len(contexts))

        if not contexts:
            return {"markets": 0, "predictions": 0, "high_ev_picks": 0, "alerts": 0, "elapsed_s": 0.0}

        # Batch predict
        ctx_list = [c for _, _, c in contexts]
        preds = self._predictor.batch_predict(ctx_list)

        scored_picks: list[ScoredPick] = []
        n_alerted = 0
        n_saved = 0

        async with AsyncSessionLocal() as session:
            prediction_map: dict[str, Prediction] = {}

            for (source, raw, ctx), pred in zip(contexts, preds):
                try:
                    market = await self._persist_market(session, raw, ctx, source)
                    await session.flush()
                    explanation = self._make_explanation(ctx, pred)
                    db_pred = await self._persist_prediction(session, market, pred, explanation)
                    await session.flush()
                    n_saved += 1

                    if pred["expected_value"] >= settings.min_ev_threshold:
                        pick_id = str(db_pred.id)
                        prediction_map[pick_id] = db_pred
                        scored_picks.append(ScoredPick(
                            prediction_id=pick_id,
                            event_name=market.event_name,
                            selection=market.selection,
                            market_type=ctx.market_type,
                            sport=ctx.sport,
                            decimal_odds=ctx.decimal_odds,
                            model_probability=pred["model_probability"],
                            expected_value=pred["expected_value"],
                            confidence=pred["confidence"],
                            kelly_fraction=pred["kelly_fraction"],
                            home_team=ctx.home_team,
                            away_team=ctx.away_team,
                            player_name=ctx.player_name,
                            source=source,
                        ))

                    if (
                        pred["confidence"] >= settings.confidence_threshold
                        and pred["expected_value"] >= settings.min_ev_threshold
                        and self._discord_alert
                    ):
                        await self._discord_alert.send_pick_alert(market, db_pred, ctx)
                        db_pred.alert_sent = True
                        n_alerted += 1

                except Exception as exc:
                    logger.error("persist_error", error=str(exc))
                    try:
                        await session.rollback()
                    except Exception:
                        pass
                    continue

            if len(scored_picks) >= 2:
                parlays = build_parlays(
                    scored_picks,
                    min_confidence=settings.confidence_threshold - 0.15,
                    min_ev=settings.min_ev_threshold,
                )
                for _n, top_combos in parlays.items():
                    for combo in top_combos[:1]:
                        try:
                            await self._persist_parlay(session, combo, prediction_map)
                        except Exception as exc:
                            logger.error("parlay_persist_error", error=str(exc))

            try:
                await session.commit()
            except Exception as exc:
                logger.error("commit_error", error=str(exc))
                await session.rollback()

        elapsed = asyncio.get_event_loop().time() - t0
        summary = {
            "markets": len(contexts),
            "predictions": n_saved,
            "high_ev_picks": len(scored_picks),
            "alerts": n_alerted,
            "elapsed_s": round(elapsed, 1),
        }
        logger.info("scan_complete", **summary)
        return summary

    async def run_forever(self) -> None:
        self._running = True
        logger.info("scanner_started", interval_s=settings.scan_interval_seconds)
        while self._running:
            try:
                await self.scan_once()
            except Exception as exc:
                logger.error("scan_cycle_error", error=str(exc))
            await asyncio.sleep(settings.scan_interval_seconds)

    def stop(self) -> None:
        self._running = False
