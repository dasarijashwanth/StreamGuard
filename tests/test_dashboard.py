"""Day 5 tests: dashboard data helpers (no Streamlit server needed)."""

import ast
from pathlib import Path

import httpx
import pytest

from dashboard.data import bin_scores, drift_summary, fetch_all, fetch_json, flagged_scores


def _response(status: int, **kwargs) -> httpx.Response:
    """Build an httpx.Response with a request attached (raise_for_status needs it)."""
    return httpx.Response(status, request=httpx.Request("GET", "http://api:8000/x"), **kwargs)


def test_fetch_json_success(monkeypatch):
    def fake_get(url, timeout):
        assert url == "http://api:8000/health"
        return _response(200, json={"status": "ok"})

    monkeypatch.setattr(httpx, "get", fake_get)
    assert fetch_json("http://api:8000", "health") == {"status": "ok"}


def test_fetch_json_failure_returns_none(monkeypatch):
    def fake_get(url, timeout):
        raise httpx.ConnectError("refused", request=httpx.Request("GET", url))

    monkeypatch.setattr(httpx, "get", fake_get)
    assert fetch_json("http://api:8000", "health") is None


def test_fetch_json_bad_json_returns_none(monkeypatch):
    def fake_get(url, timeout):
        return _response(200, text="not json")

    monkeypatch.setattr(httpx, "get", fake_get)
    assert fetch_json("http://api:8000", "health") is None


def test_fetch_all_returns_all_endpoints(monkeypatch):
    def fake_get(url, timeout):
        return _response(200, json={"endpoint": url.rsplit("/", 1)[-1]})

    monkeypatch.setattr(httpx, "get", fake_get)
    result = fetch_all("http://api:8000")
    assert set(result) == {"health", "metrics", "drift", "models"}
    assert result["metrics"] == {"endpoint": "metrics"}


def test_bin_scores_buckets_and_edge():
    scores = [
        {"fraud_probability": 0.05},
        {"fraud_probability": 0.15},
        {"fraud_probability": 0.95},
        {"fraud_probability": 1.0},  # exactly 1.0 lands in the last bin
    ]
    bins = bin_scores(scores)
    assert len(bins) == 10
    assert bins[0]["count"] == 1
    assert bins[1]["count"] == 1
    assert bins[9]["count"] == 2
    assert sum(b["count"] for b in bins) == 4
    assert bins[0]["bin"] == "0.0-0.1"


def test_bin_scores_empty():
    assert sum(b["count"] for b in bin_scores([])) == 0


def test_flagged_scores_newest_first():
    scores = [
        {"transaction_id": "a", "is_fraud": True},
        {"transaction_id": "b", "is_fraud": False},
        {"transaction_id": "c", "is_fraud": True},
    ]
    flagged = flagged_scores(scores)
    assert [s["transaction_id"] for s in flagged] == ["c", "a"]


def test_drift_summary_stable_and_detected():
    stable = drift_summary(
        {"samples_seen": 10, "error_rate": 0.1, "drift_events": 0,
         "drift_detected": False, "retrain_requested": False}
    )
    assert stable["status"] == "stable"
    detected = drift_summary(
        {"samples_seen": 10, "error_rate": 0.4, "drift_events": 1,
         "drift_detected": True, "retrain_requested": True}
    )
    assert detected["status"] == "drift detected"


def test_drift_summary_none_is_unknown():
    assert drift_summary(None)["status"] == "unknown"


def test_dashboard_app_parses_without_streamlit():
    """app.py must be valid Python; it is only executed by `streamlit run`."""
    app_path = Path(__file__).resolve().parent.parent / "dashboard" / "app.py"
    tree = ast.parse(app_path.read_text())
    assert isinstance(tree, ast.Module)
    assert "streamlit" in app_path.read_text()
    assert "fetch_all" in app_path.read_text()


def test_fetch_json_http_error_returns_none(monkeypatch):
    def fake_get(url, timeout):
        return _response(500, json={"detail": "boom"})

    monkeypatch.setattr(httpx, "get", fake_get)
    assert fetch_json("http://api:8000", "health") is None
