"""Day 4 tests: promotion rule, challenger training, /score header, /models."""

import json
from pathlib import Path

from fastapi.testclient import TestClient

from app.challenger import (
    ChallengerRegistry,
    reset_challenger_registry,
    should_promote,
)
from app.main import app
from scripts.train_challenger import train_challenger

client = TestClient(app)

LEGIT_TXN = {
    "transaction_id": "txn-legit-001",
    "amount": 42.50,
    "merchant_category": "grocery",
    "hour_of_day": 14,
    "card_present": True,
}


def _metrics(auc: float, p50: float) -> dict:
    return {"auc": auc, "precision": 0.9, "recall": 0.8, "p50_latency_ms": p50}


def test_promotion_rule_promotes_strictly_better_challenger():
    promoted, _ = should_promote(_metrics(0.90, 1.0), _metrics(0.92, 1.1))
    assert promoted is True


def test_promotion_rule_rejects_worse_auc():
    promoted, reason = should_promote(_metrics(0.90, 1.0), _metrics(0.89, 0.5))
    assert promoted is False
    assert "AUC" in reason


def test_promotion_rule_rejects_too_slow_challenger():
    promoted, reason = should_promote(_metrics(0.90, 1.0), _metrics(0.95, 1.3))
    assert promoted is False
    assert "latency" in reason


def test_registry_promotion_bumps_version():
    registry = ChallengerRegistry()
    registry.register_champion(_metrics(0.90, 1.0), "0.1.0")
    registry.register_challenger(_metrics(0.92, 1.1), "0.1.0")
    decision = registry.maybe_promote()
    assert decision["promoted"] is True
    assert registry.champion["status"] == "champion"
    assert registry.registry.current_version == "0.1.1"


def test_train_challenger_runs_without_mlflow_and_writes_metrics(tmp_path):
    """End-to-end: synthetic data, no MLflow server, metrics JSON written."""
    out_dir = tmp_path / "evals"
    result = train_challenger(n_samples=2000, tracking_uri=None, output_dir=out_dir)
    assert result["challenger"]["auc"] > 0.9
    assert result["champion"] is not None
    assert result["mlflow_logged"] is False
    metrics_path = out_dir / "challenger_metrics.json"
    assert metrics_path.exists()
    saved = json.loads(metrics_path.read_text())
    assert saved["challenger"]["auc"] == result["challenger"]["auc"]
    assert "promoted" in saved
    assert (out_dir / "challenger_model.joblib").exists()


def test_score_has_model_version_header():
    resp = client.post("/score", json=LEGIT_TXN)
    assert resp.status_code == 200
    assert resp.headers["X-Model-Version"] == "0.1.0"
    assert resp.headers["X-Model-Version"] == resp.json()["model_version"]


def test_models_endpoint_returns_registry():
    reset_challenger_registry()
    resp = client.get("/models")
    assert resp.status_code == 200
    body = resp.json()
    assert body["champion"] is not None
    assert body["champion"]["version"] == "0.1.0"
    assert body["champion"]["status"] == "champion"
    assert body["champion"]["auc"] > 0.9  # seeded from the baseline bundle
    assert body["challenger"] is None
    assert body["registry_version"] == "0.1.0"
