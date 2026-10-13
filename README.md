# StreamGuard

Real-time fraud-detection MLOps service: streaming transaction scoring via
FastAPI, online learning with drift detection and auto-retrain, and a live
latency/drift dashboard.

> Day 1 of 7 — scaffold, scoring API, and baseline model. Full README
> (architecture, setup, evaluation, roadmap) lands on Day 7.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python scripts/train_baseline.py   # reproducible synthetic data, seed=42
uvicorn app.main:app --reload
```

- `GET http://localhost:8000/health` — service + model status
- `POST http://localhost:8000/score` — score one transaction (see example below)

```bash
curl -X POST http://localhost:8000/score -H "Content-Type: application/json" -d '{
  "amount": 499.99,
  "merchant_category": "online_retail",
  "hour_of_day": 3,
  "card_present": false
}'
```

## Run tests

```bash
pytest -q
```

## Project layout

```
app/
  main.py        # FastAPI app: /health, /score
  schemas.py     # Pydantic transaction + response models
  model.py       # feature engineering + model loading/scoring
  models/        # trained artifacts (baseline_model.joblib)
scripts/
  train_baseline.py  # synthetic data generation + training
tests/
  test_api.py    # schema, endpoint, and edge-case tests
```

## Roadmap

Day 2: Redpanda streaming ingest · Day 3: River online learning + drift
detection · Day 4: MLflow + XGBoost challenger · Day 5: Streamlit dashboard ·
Day 6: evaluation + latency benchmarks · Day 7: full README + polish.
