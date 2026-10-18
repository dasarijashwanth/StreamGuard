# StreamGuard

**A real-time fraud-detection MLOps service.** Transactions stream through
Redpanda into a FastAPI scoring API; a River online learner shadows the
batch model with ADWIN drift detection and an auto-retrain trigger; an
MLflow-tracked XGBoost challenger competes against the champion under a
strict promotion rule; and a Streamlit dashboard watches fraud rate,
latency, drift, and model versions live.

> Complete 7-day build. Test suite: 50 passed, 1 skipped (broker
> integration, needs Docker).

## Features

- **Real-time scoring API** — FastAPI `POST /score` scores a transaction
  in ~1ms (p50) with the serving model version in an `X-Model-Version`
  header.
- **Streaming ingest** — synthetic producer publishes to a Redpanda
  `transactions` topic; the consumer scores each message through the same
  path as `POST /score`.
- **Online learning + drift detection** — River `LogisticRegression`
  learns from labeled feedback via `POST /feedback`; ADWIN watches the
  error stream and versions a retrain candidate in the model registry when
  drift is confirmed.
- **Champion/challenger with MLflow** — an XGBoost challenger trains on
  accumulated labels (or seeded synthetic data), logs to MLflow, and is
  promoted only if AUC beats the champion by 0.005+ with p50 latency at
  most 1.2x. The challenger was honestly rejected on synthetic data
  (0.9998 vs 0.9999), which is the correct behavior.
- **Live dashboard** — Streamlit panels for total scored, fraud rate,
  latency p50/p99, drift status, champion/challenger versions, score
  distribution, and flagged transactions, backed by an in-memory
  `GET /metrics` collector.
- **Evaluation + regression gate** — holdout eval (seed 123, unseen by
  training) reports AUC 0.9990, precision 0.9927 / recall 0.9067 at
  threshold 0.5, ECE 0.0037; the test suite fails if a future model swap
  drops below the committed baseline.
- **Fully reproducible** — every dataset is seeded synthetic data;
  `scripts/train_baseline.py` regenerates everything from scratch.

## Architecture

```
Synthetic producer -> Redpanda `transactions` topic -> consumer
  -> FastAPI POST /score (LogisticRegression champion)
  -> River shadow learner + ADWIN drift detector
  -> MLflow / XGBoost challenger + model registry
  -> Streamlit dashboard reading /health, /metrics, /drift, /models
```

## Prerequisites

- Python 3.12 (local `.venv` works; see Quickstart)
- Docker + `docker compose` for Redpanda, MLflow, and the dashboard
- Windows PowerShell: stop any standalone `redpanda-1` container first
  (`docker stop redpanda-1`) to avoid a port 9092 conflict.

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

## Live dashboard (Day 5)

```powershell
docker compose up -d        # API + Redpanda + dashboard in one command
python scripts/produce_transactions.py --count 50
python scripts/consume_and_score.py --max-messages 50
```

Open http://localhost:8501. Panels refresh every 5 seconds (adjustable in
the sidebar) and show: total scored, fraud rate, latency p50/p99, drift
status, champion/challenger versions, a score-distribution histogram, and
the recently flagged transactions table.

Local run without Docker (API must be up first):

```powershell
streamlit run dashboard/app.py   # dashboard at http://localhost:8501
```

- The dashboard reads `GET /health`, `/metrics`, `/drift`, and `/models`.
  If the API is down, panels render as empty instead of crashing.
- `GET /metrics` is new on Day 5: the API records every `/score` call in an
  in-memory collector (bounded to the most recent 500 scores) and exposes
  totals, fraud rate, latency p50/p99, and recent scores. Metrics reset
  when the API restarts; nothing is persisted.

```bash
pytest -q
```

## Evaluation + latency benchmarks (Day 6)

```bash
python scripts/evaluate.py            # holdout metrics + latency -> evals/day6_results.json, evals/evaluation.md
python scripts/benchmark_latency.py   # latency only, add --save to merge into evals/day6_results.json
```

- The holdout uses seed 123 (training used seed 42), so it is data the
  champion never saw. Measured on this run: AUC **0.9990**,
  precision 0.9927 / recall 0.9067 at threshold 0.5, calibration error
  (ECE) 0.0037.
- Latency on this run: in-process single-score p50 0.103ms / p99 0.143ms
  (9,286 req/s); HTTP `POST /score` p50 1.128ms / p99 2.772ms (777 req/s)
  over 1,000 requests each.
- Regression gate: `tests/test_regression_gate.py` fails the suite if a
  future model swap scores AUC below the committed result minus 0.01, or
  precision at 0.5 below the committed result minus 0.05 (floored at
  0.90). Re-run `scripts/evaluate.py` after any model change to refresh
  the committed baseline.

## Limitations

- All data is synthetic (seed 42 for training, seed 123 for the holdout),
  so the near-perfect AUC (holdout 0.9990) reflects a cleanly separable
  toy problem, not real fraud data.
- The challenger was NOT promoted: champion AUC 0.9999 vs challenger
  0.9998 missed the +0.005 promotion margin. Real promotion needs
  messier labeled data.
- `GET /metrics` is in-memory, bounded to the last 500 scores, and resets
  when the API restarts; nothing is persisted.
- The challenger falls back to local `evals/challenger_metrics.json` when
  no MLflow server is reachable.

## Project layout

```
app/
  main.py        # FastAPI app: /health, /score, /feedback, /drift, /models, /metrics
  schemas.py     # Pydantic transaction + response models
  model.py       # feature engineering + model loading/scoring
  online.py      # Day 3: River online learner, ADWIN drift, registry
  challenger.py  # Day 4: champion/challenger compare + promotion rule
  metrics.py     # Day 5: in-memory scoring metrics for GET /metrics
  streaming.py   # Redpanda topic, message codec, Kafka client factories
  evaluate.py    # Day 6: holdout metrics, calibration, latency benchmarks
  synthetic.py   # Day 6: reproducible synthetic transaction generator
  models/        # trained artifacts (baseline_model.joblib)
dashboard/
  app.py         # Day 5: Streamlit live dashboard UI
  data.py        # Day 5: API fetch + binning/filtering helpers (no Streamlit)
scripts/
  train_baseline.py  # synthetic data generation + training
  train_challenger.py  # Day 4: XGBoost challenger + MLflow logging
  evaluate.py        # Day 6: holdout eval + latency -> evals/
  benchmark_latency.py  # Day 6: latency-only re-run
  produce_transactions.py  # Day 2: publish synthetic txns to Redpanda
  consume_and_score.py     # Day 2: consume + score off the topic
evals/
  challenger_metrics.json  # Day 4: latest challenger run (local fallback)
  day6_results.json        # Day 6: committed holdout + latency results
  evaluation.md            # Day 6: human-readable results table
tests/
  test_api.py    # schema, endpoint, and edge-case tests
  test_online.py # Day 3: learner improvement, ADWIN drift, retrain trigger
  test_challenger.py  # Day 4: promotion rule, challenger run, /models
  test_metrics.py     # Day 5: /metrics endpoint + metrics collector
  test_dashboard.py   # Day 5: dashboard data helpers (no Streamlit server)
  test_evaluation.py  # Day 6: metric math on hand-computed datasets
  test_benchmark.py   # Day 6: latency benchmark smoke tests
  test_regression_gate.py  # Day 6: champion must clear committed baseline
  test_streaming.py  # Day 2: message codec + produce/consume/score round trip
```

## Roadmap

Day 2: Redpanda streaming ingest ✅ · Day 3: River online learning + drift
detection ✅ · Day 4: MLflow + XGBoost challenger ✅ · Day 5: Streamlit
dashboard ✅ · Day 6: evaluation + latency benchmarks ✅ · Day 7: full
README + polish ✅.
