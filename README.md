# Sports EV Bot — AI Prediction & Betting Market Analyzer

A production-ready AI application that scans every available betting and prediction market, identifies the highest Expected Value (EV) opportunities, and constructs optimal 2–5 leg parlays using ML ensembles.

---

## Architecture

```
sports-ev-bot/
├── app/
│   ├── api/            # FastAPI routes
│   ├── core/           # DB models, config, scheduler
│   ├── ml/             # Feature engineering + ML pipeline
│   ├── scrapers/       # Data collectors (Kalshi, Polymarket, sports APIs)
│   ├── alerts/         # Discord bot
│   └── dashboard/      # Web dashboard (React via single HTML artifact)
├── tests/              # Unit + integration tests
├── config/             # YAML configuration
├── docker/             # Dockerfile + compose
└── scripts/            # DB init, seed, retrain
```

---

## Quick Start

### 1. Configure environment

```bash
cp .env.example .env
# Fill in your API keys and DB credentials
```

### 2. Start with Docker Compose

```bash
docker-compose up --build
```

### 3. Access the dashboard

```
http://localhost:8000/dashboard
```

### 4. API docs

```
http://localhost:8000/docs
```

---

## Key Features

| Feature | Description |
|---|---|
| Market scanning | Scans every market type every 5 minutes |
| EV calculation | Kelly criterion + model probability vs implied probability |
| ML ensemble | XGBoost + LightGBM + CatBoost + Logistic Regression |
| Auto-retrain | Nightly retraining on completed predictions |
| Discord alerts | Fires when confidence ≥ threshold (default 85%) |
| Parlay builder | Best 2–5 leg combos with correlation detection |
| Dashboard | Live ROI, win rate, calibration curves, model accuracy |

---

## Environment Variables

| Variable | Description |
|---|---|
| `DATABASE_URL` | PostgreSQL connection string |
| `REDIS_URL` | Redis connection string |
| `DISCORD_TOKEN` | Discord bot token |
| `DISCORD_CHANNEL_ID` | Channel to post alerts |
| `KALSHI_API_KEY` | Kalshi API key |
| `POLYMARKET_API_KEY` | Polymarket API key |
| `ODDS_API_KEY` | The Odds API key |
| `API_FOOTBALL_KEY` | API-Football key |
| `OPENWEATHER_KEY` | OpenWeatherMap key |
| `CONFIDENCE_THRESHOLD` | Min confidence to alert (default 0.85) |
| `MIN_EV_THRESHOLD` | Min EV to alert (default 0.05) |
| `ALERT_STAKE` | Default stake for Kelly sizing |

---

## ML Pipeline

1. **Feature engineering** — 80+ features per market (team form, H2H, odds movement, weather, lineup changes, etc.)
2. **Model training** — XGBoost, LightGBM, CatBoost, RandomForest, LogisticRegression trained in parallel
3. **Ensemble** — Stacking with Logistic Regression meta-learner
4. **EV calculation** — `EV = (model_prob × decimal_odds) - 1`
5. **Kelly sizing** — `f = (bp - q) / b` for position sizing
6. **Calibration** — Platt scaling to ensure confidence = actual win rate

---

## Parlay Construction

1. Score each market by EV
2. Select top N markets above confidence threshold
3. Try all combinations of 2–5 legs
4. Reject combinations with correlated risk (same game, same team outcome)
5. Rank by combined EV and parlay payout
6. Surface top combo per leg count

---

## Discord Alert Format

```
🎯 HIGH EV PICK — 92% Confidence

📌 Pick: Brazil BTTS vs Argentina
📊 EV: +18.4%
💰 Odds: +115 | Model Prob: 67.3%
🔥 Kelly: 4.2% stake
⚠️  Risk: Medium

📈 Supporting Stats:
• Both teams scored in 8/10 recent H2H
• Argentina avg 2.1 goals/game last 5
• Brazil avg 1.8 goals/game last 5
• Odds moved from +130 → +115 (sharp money)

🔗 https://...
```
