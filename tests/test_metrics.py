"""Day 5 tests: /metrics endpoint + in-memory metrics collector."""

from fastapi.testclient import TestClient

from app.main import app
from app.metrics import MetricsCollector, reset_metrics

client = TestClient(app)

LEGIT_TXN = {
    "transaction_id": "txn-metrics-legit",
    "amount": 42.50,
    "merchant_category": "grocery",
    "hour_of_day": 14,
    "card_present": True,
}

FRAUD_TXN = {
    "transaction_id": "txn-metrics-fraud",
    "amount": 4500.00,
    "merchant_category": "online_retail",
    "hour_of_day": 3,
    "card_present": False,
}


def setup_function():
    reset_metrics()


def test_metrics_empty_before_any_scores():
    resp = client.get("/metrics")
    assert resp.status_code == 200
    body = resp.json()
    assert body["total_scored"] == 0
    assert body["fraud_count"] == 0
    assert body["fraud_rate"] == 0.0
    assert body["p50_latency_ms"] is None
    assert body["p99_latency_ms"] is None
    assert body["recent"] == []


def test_score_records_metrics():
    client.post("/score", json=LEGIT_TXN)
    client.post("/score", json=FRAUD_TXN)
    body = client.get("/metrics").json()
    assert body["total_scored"] == 2
    assert body["fraud_count"] == 1
    assert body["fraud_rate"] == 0.5
    assert body["p50_latency_ms"] is not None
    assert body["p99_latency_ms"] is not None
    assert body["p99_latency_ms"] >= body["p50_latency_ms"]
    ids = [r["transaction_id"] for r in body["recent"]]
    assert ids == ["txn-metrics-legit", "txn-metrics-fraud"]
    flagged = [r for r in body["recent"] if r["is_fraud"]]
    assert len(flagged) == 1
    assert flagged[0]["fraud_probability"] >= flagged[0]["threshold"]


def test_collector_percentiles_nearest_rank():
    collector = MetricsCollector()
    for latency in (10.0, 20.0, 30.0, 40.0):
        collector.record(
            transaction_id=None,
            fraud_probability=0.1,
            is_fraud=False,
            latency_ms=latency,
            threshold=0.5,
        )
    snap = collector.snapshot()
    assert snap["total_scored"] == 4
    assert snap["p50_latency_ms"] == 20.0
    assert snap["p99_latency_ms"] == 40.0


def test_collector_recent_window_is_bounded():
    collector = MetricsCollector(max_recent=3)
    for i in range(5):
        collector.record(
            transaction_id=f"txn-{i}",
            fraud_probability=0.1,
            is_fraud=False,
            latency_ms=1.0,
            threshold=0.5,
        )
    snap = collector.snapshot()
    assert snap["total_scored"] == 5
    assert [r["transaction_id"] for r in snap["recent"]] == ["txn-2", "txn-3", "txn-4"]


def test_collector_reset_clears_state():
    collector = MetricsCollector()
    collector.record(
        transaction_id="txn-1",
        fraud_probability=0.9,
        is_fraud=True,
        latency_ms=5.0,
        threshold=0.5,
    )
    collector.reset()
    snap = collector.snapshot()
    assert snap["total_scored"] == 0
    assert snap["fraud_count"] == 0
    assert snap["recent"] == []
