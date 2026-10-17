"""Holdout evaluation + latency benchmarks (StreamGuard Day 6).

Pure functions live here so tests can check metric math on tiny known
datasets without a model, a server, or disk. The CLI scripts
(scripts/evaluate.py, scripts/benchmark_latency.py) are thin wrappers.

Metrics:
  - AUC (ROC) on the full holdout.
  - Precision / recall at thresholds 0.3 / 0.5 / 0.7 / 0.9.
  - Calibration: 10 equal-width bins of predicted probability, each with
    count, mean predicted probability, and observed fraud rate, plus the
    expected calibration error (ECE).
Latency (nearest-rank percentiles, same definition as Day 5 metrics):
  - In-process single-score timing via score_transaction (model only).
  - HTTP-level timing via TestClient POST /score (full request path).
"""

import math
import time
from typing import Any, Dict, List, Sequence

import numpy as np
from sklearn.metrics import precision_score, recall_score, roc_auc_score

from app.model import score_transaction
from app.schemas import Transaction

DEFAULT_THRESHOLDS = (0.3, 0.5, 0.7, 0.9)
N_CALIBRATION_BINS = 10


def _percentile(sorted_values: List[float], pct: float) -> float:
    """Nearest-rank percentile (same definition as app/metrics.py)."""
    rank = math.ceil(pct / 100.0 * len(sorted_values))
    return sorted_values[max(0, min(rank - 1, len(sorted_values) - 1))]


def precision_recall_at_thresholds(
    y_true: Sequence[int],
    y_proba: Sequence[float],
    thresholds: Sequence[float] = DEFAULT_THRESHOLDS,
) -> Dict[str, Dict[str, float]]:
    """Precision and recall at each threshold. Keys are str(threshold)."""
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    out: Dict[str, Dict[str, float]] = {}
    for t in thresholds:
        preds = (y_proba >= t).astype(int)
        out[str(t)] = {
            "precision": float(precision_score(y_true, preds, zero_division=0)),
            "recall": float(recall_score(y_true, preds, zero_division=0)),
        }
    return out


def calibration_table(
    y_true: Sequence[int],
    y_proba: Sequence[float],
    n_bins: int = N_CALIBRATION_BINS,
) -> Dict[str, Any]:
    """Binned predicted-vs-observed table + expected calibration error.

    Bins are equal-width slices of [0, 1]. Empty bins report count 0 with
    None for the rates so callers can tell "no data" from "0% fraud".
    """
    y_true = np.asarray(y_true, dtype=float)
    y_proba = np.asarray(y_proba, dtype=float)
    n = len(y_true)
    bins: List[Dict[str, Any]] = []
    ece = 0.0
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    for i in range(n_bins):
        lo, hi = float(edges[i]), float(edges[i + 1])
        # Last bin is closed on the right so proba == 1.0 lands somewhere.
        mask = (y_proba >= lo) & (y_proba < hi) if i < n_bins - 1 else (y_proba >= lo) & (y_proba <= hi)
        count = int(mask.sum())
        if count == 0:
            bins.append({"bin": [lo, hi], "count": 0, "mean_predicted": None, "observed_fraud_rate": None})
            continue
        mean_pred = float(y_proba[mask].mean())
        observed = float(y_true[mask].mean())
        bins.append({"bin": [lo, hi], "count": count, "mean_predicted": mean_pred, "observed_fraud_rate": observed})
        ece += abs(mean_pred - observed) * (count / n)
    return {"bins": bins, "ece": float(ece)}


def evaluate_holdout(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    thresholds: Sequence[float] = DEFAULT_THRESHOLDS,
) -> Dict[str, Any]:
    """Full holdout report for one model: AUC, P/R per threshold, calibration."""
    proba = model.predict_proba(X)[:, 1]
    return {
        "n": int(len(y)),
        "fraud_rate": float(np.mean(y)) if len(y) else 0.0,
        "auc": float(roc_auc_score(y, proba)),
        "precision_recall": precision_recall_at_thresholds(y, proba, thresholds),
        "calibration": calibration_table(y, proba),
    }


def benchmark_score_latency(
    transactions: Sequence[Transaction],
    n_requests: int,
    warmup: int = 20,
) -> Dict[str, float]:
    """In-process latency: score_transaction called n_requests times.

    Cycles through `transactions` so the benchmark never needs a live
    server. Returns p50/p99/mean latency in ms and requests/sec.
    """
    txns = list(transactions)
    for i in range(warmup):
        score_transaction(txns[i % len(txns)])
    latencies: List[float] = []
    start = time.perf_counter()
    for i in range(n_requests):
        _, _, _, _, latency_ms = score_transaction(txns[i % len(txns)])
        latencies.append(latency_ms)
    elapsed_s = time.perf_counter() - start
    ordered = sorted(latencies)
    return {
        "n_requests": int(n_requests),
        "p50_latency_ms": float(_percentile(ordered, 50)),
        "p99_latency_ms": float(_percentile(ordered, 99)),
        "mean_latency_ms": float(np.mean(latencies)),
        "requests_per_sec": float(n_requests / elapsed_s) if elapsed_s > 0 else 0.0,
    }


def benchmark_api_latency(
    client: Any,
    transactions: Sequence[Dict[str, Any]],
    n_requests: int,
    warmup: int = 20,
) -> Dict[str, float]:
    """HTTP-level latency: POST /score called n_requests times via TestClient.

    Measures the full request path (JSON, validation, scoring, response).
    Raises on any non-200 response so a broken API fails loudly instead
    of producing a meaningless latency number.
    """
    txns = list(transactions)
    for i in range(warmup):
        resp = client.post("/score", json=txns[i % len(txns)])
        resp.raise_for_status()
    latencies: List[float] = []
    start = time.perf_counter()
    for i in range(n_requests):
        t0 = time.perf_counter()
        resp = client.post("/score", json=txns[i % len(txns)])
        resp.raise_for_status()
        latencies.append((time.perf_counter() - t0) * 1000.0)
    elapsed_s = time.perf_counter() - start
    ordered = sorted(latencies)
    return {
        "n_requests": int(n_requests),
        "p50_latency_ms": float(_percentile(ordered, 50)),
        "p99_latency_ms": float(_percentile(ordered, 99)),
        "mean_latency_ms": float(np.mean(latencies)),
        "requests_per_sec": float(n_requests / elapsed_s) if elapsed_s > 0 else 0.0,
    }
