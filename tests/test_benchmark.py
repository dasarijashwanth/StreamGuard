"""Day 6 tests: latency benchmark smoke tests (small N, fast)."""

from fastapi.testclient import TestClient

from app.evaluate import benchmark_api_latency, benchmark_score_latency
from app.main import app
from app.schemas import Transaction

TXNS = [
    Transaction(amount=42.50, merchant_category="grocery", hour_of_day=14, card_present=True),
    Transaction(amount=4500.00, merchant_category="online_retail", hour_of_day=3, card_present=False),
]


def test_benchmark_score_latency_smoke():
    stats = benchmark_score_latency(TXNS, n_requests=10, warmup=2)
    assert stats["n_requests"] == 10
    assert stats["p50_latency_ms"] > 0
    assert stats["p99_latency_ms"] >= stats["p50_latency_ms"]
    assert stats["mean_latency_ms"] > 0
    assert stats["requests_per_sec"] > 0


def test_benchmark_api_latency_smoke():
    with TestClient(app) as client:
        stats = benchmark_api_latency(client, [t.model_dump() for t in TXNS], n_requests=10, warmup=2)
    assert stats["n_requests"] == 10
    assert stats["p50_latency_ms"] > 0
    assert stats["p99_latency_ms"] >= stats["p50_latency_ms"]
    assert stats["requests_per_sec"] > 0
