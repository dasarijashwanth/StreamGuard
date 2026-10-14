# StreamGuard — 7-Day Build Plan

**What:** Real-time fraud-detection MLOps service. Streaming transaction scoring
via FastAPI, online learning with drift detection and auto-retrain, and a live
latency/drift dashboard.

**Stack:** Python, scikit-learn / XGBoost, FastAPI, Redpanda, River, MLflow, Streamlit.

> Rule: every day's slice is shippable on its own. Nothing lands on `main`
> without Jashwanth's explicit approval.

---

## Day 1 — Tue Oct 13: Scaffold + scoring API + baseline model ✅
- FastAPI app with `GET /health` and `POST /score`
- Pydantic transaction schemas with validation
- Baseline `LogisticRegression` trained on reproducible synthetic data (`seed=42`),
  artifact saved under `app/models/`
- `/score` returns fraud probability, binary decision, and model version
- `requirements.txt`, `.gitignore`, `Dockerfile`, README skeleton
- pytest suite: schema validation, `/health`, `/score` happy path + edge cases

## Day 2 — Wed Oct 14: Streaming ingest (Redpanda) ✅
- `docker-compose.yml`: Redpanda broker + console, API service
- Producer script emitting synthetic transactions to a `transactions` topic
- Consumer wiring transactions into the scoring path
- Integration test: produce → consume → score round trip

## Day 3 — Thu Oct 15: Online learning + drift detection
- River `LogisticRegression` online learner shadowing the batch model
- ADWIN drift detector on prediction error stream
- Auto-retrain trigger when drift is confirmed; model registry versioning
- Unit tests for drift detector behavior on synthetic concept drift

## Day 4 — Fri Oct 16: MLflow + batch XGBoost challenger
- MLflow tracking server (docker-compose service)
- Nightly batch job: train XGBoost on accumulated labeled data, log params/metrics
- Champion/challenger comparison (AUC, latency); promotion rule
- API gains model-selection header (`X-Model-Version`)

## Day 5 — Sat Oct 17: Streamlit live dashboard
- Live fraud-rate, latency p50/p99, and drift-status panels
- Score distribution + recent flagged transactions table
- Reads from Redpanda topic / API metrics endpoint
- One-command `docker-compose up` demo

## Day 6 — Sun Oct 18: Evaluation + latency benchmarks
- Holdout evaluation harness: AUC, precision/recall @ thresholds, calibration
- Latency benchmark: single-score p50/p99, 1k-rps throughput test
- Results table committed to `evals/`; regression gate for future model swaps

## Day 7 — Mon Oct 19: Full README, Dockerfile polish, final review
- Complete README: overview, features, architecture diagram, setup, usage,
  evaluation results, roadmap
- Multi-stage Dockerfile, `.dockerignore`, compose profiles
- Final test pass, lint, approval gate, then push
