# StreamGuard

Real-time fraud-detection MLOps service: streaming transaction scoring via
FastAPI, online learning with drift detection and auto-retrain, and a live
latency/drift dashboard.

> Day 4 of 7 — MLflow tracking + XGBoost challenger with a conservative
> promotion rule. Full README (architecture, setup, evaluation, roadmap)
> lands on Day 7.

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

## Streaming (Day 2)

```powershell
docker compose up -d        # Redpanda + Console + API (stop standalone redpanda-1 first: docker stop redpanda-1)
python scripts/produce_transactions.py --count 50
python scripts/consume_and_score.py --max-messages 50
```

- Producer publishes synthetic transactions to the `transactions` topic.
- Consumer scores each message through the same scoring path as `POST /score`.
- Redpanda Console (topic browser) at http://localhost:8080.

## Online learning + drift detection (Day 3)

```bash
# Feed a labeled transaction to the River shadow learner
curl -X POST http://localhost:8000/feedback -H "Content-Type: application/json" -d '{
  "transaction": {"amount": 499.99, "merchant_category": "online_retail", "hour_of_day": 3, "card_present": false},
  "label": true
}'
# Check drift + registry status
curl http://localhost:8000/drift
```

- River `LogisticRegression` learns from every labeled transaction, shadowing the batch model.
- ADWIN watches the prediction-error stream; confirmed drift raises a retrain request and versions a new candidate in the model registry.
- The batch retrain itself lands on Day 4 with MLflow.

## MLflow + XGBoost challenger (Day 4)

```bash
docker compose up -d mlflow   # tracking server + UI at http://localhost:5000
python scripts/train_challenger.py [--data labeled.csv] [--tracking-uri URI]
curl http://localhost:8000/models
```

- The challenger trains XGBoost on accumulated labeled data (or reproducible
  synthetic data, seed=42, when no CSV is given) and logs params, metrics
  (AUC, precision, recall, per-score latency), and the model artifact to
  MLflow when the server is reachable. Without a server it falls back to
  local logging: `evals/challenger_metrics.json`. It never crashes for lack
  of MLflow.
- Promotion rule: the challenger is promoted only if its AUC beats the
  champion by at least 0.005 AND its p50 latency is at most 1.2x the
  champion's. Promotion bumps the registry version.
- `POST /score` now sends the serving model version as an
  `X-Model-Version` response header (also in the JSON body).
- `GET /models` lists the champion, the latest challenger, and their status.

## Run tests

```bash
pytest -q
```

## Project layout

```
app/
  main.py        # FastAPI app: /health, /score, /feedback, /drift, /models
  schemas.py     # Pydantic transaction + response models
  model.py       # feature engineering + model loading/scoring
  online.py      # Day 3: River online learner, ADWIN drift, registry
  challenger.py  # Day 4: champion/challenger compare + promotion rule
  streaming.py   # Redpanda topic, message codec, Kafka client factories
  models/        # trained artifacts (baseline_model.joblib)
scripts/
  train_baseline.py  # synthetic data generation + training
  train_challenger.py  # Day 4: XGBoost challenger + MLflow logging
  produce_transactions.py  # Day 2: publish synthetic txns to Redpanda
  consume_and_score.py     # Day 2: consume + score off the topic
evals/
  challenger_metrics.json  # Day 4: latest challenger run (local fallback)
tests/
  test_api.py    # schema, endpoint, and edge-case tests
  test_online.py # Day 3: learner improvement, ADWIN drift, retrain trigger
  test_challenger.py  # Day 4: promotion rule, challenger run, /models
  test_streaming.py  # Day 2: message codec + produce/consume/score round trip
```

## Roadmap

Day 2: Redpanda streaming ingest · Day 3: River online learning + drift
detection · Day 4: MLflow + XGBoost challenger · Day 5: Streamlit dashboard ·
Day 6: evaluation + latency benchmarks · Day 7: full README + polish.
