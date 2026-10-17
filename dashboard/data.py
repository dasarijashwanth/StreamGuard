"""Data helpers for the StreamGuard dashboard (Day 5).

Kept free of Streamlit so tests can exercise the fetch, binning, and
filtering logic without a running Streamlit server. `dashboard/app.py`
owns all UI; this module owns talking to the API and shaping data.
"""

from typing import Any, Dict, List, Optional

import httpx

DEFAULT_API_BASE_URL = "http://localhost:8000"
ENDPOINTS = ("health", "metrics", "drift", "models")
BIN_COUNT = 10


def fetch_json(base_url: str, path: str, timeout: float = 3.0) -> Optional[Dict[str, Any]]:
    """GET one API endpoint; returns None on any failure (API down, timeout).

    The dashboard must keep rendering with stale/empty panels instead of
    crashing when the API is unreachable.
    """
    try:
        resp = httpx.get(f"{base_url.rstrip('/')}/{path}", timeout=timeout)
        resp.raise_for_status()
        return resp.json()
    except (httpx.HTTPError, ValueError):
        return None


def fetch_all(base_url: str) -> Dict[str, Optional[Dict[str, Any]]]:
    """Fetch every panel's endpoint; each value is a dict or None."""
    return {name: fetch_json(base_url, name) for name in ENDPOINTS}


def bin_scores(scores: List[Dict[str, Any]], bin_count: int = BIN_COUNT) -> List[Dict[str, Any]]:
    """Bucket fraud probabilities into equal-width bins for the histogram.

    Returns [{"bin": "0.0-0.1", "count": n}, ...] with exactly `bin_count`
    bins. A score of exactly 1.0 lands in the last bin.
    """
    counts = [0] * bin_count
    for score in scores:
        prob = float(score.get("fraud_probability", 0.0))
        idx = min(int(prob * bin_count), bin_count - 1)
        counts[idx] += 1
    width = 1.0 / bin_count
    return [
        {"bin": f"{i * width:.1f}-{(i + 1) * width:.1f}", "count": counts[i]}
        for i in range(bin_count)
    ]


def flagged_scores(scores: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Recent scores the model flagged as fraud, newest first."""
    return [s for s in reversed(scores) if s.get("is_fraud")]


def drift_summary(drift: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Normalize the /drift payload into display-ready values."""
    if not drift:
        return {
            "status": "unknown",
            "samples_seen": 0,
            "error_rate": 0.0,
            "drift_events": 0,
            "retrain_requested": False,
        }
    detected = bool(drift.get("drift_detected"))
    retrain = bool(drift.get("retrain_requested"))
    status = "drift detected" if detected else ("retrain requested" if retrain else "stable")
    return {
        "status": status,
        "samples_seen": int(drift.get("samples_seen", 0)),
        "error_rate": float(drift.get("error_rate", 0.0)),
        "drift_events": int(drift.get("drift_events", 0)),
        "retrain_requested": retrain,
    }
