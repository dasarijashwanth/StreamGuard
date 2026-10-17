"""In-memory scoring metrics for StreamGuard (Day 5).

The API records one entry per scored transaction: latency, fraud
probability, and fraud flag. The dashboard reads the aggregate snapshot
from GET /metrics. Everything is process-local and bounded (a deque of
the most recent 500 scores), so there is no new infrastructure to run.

Thread-safe: /score can be called concurrently, and the dashboard polls
/metrics while scoring continues.
"""

import math
import threading
from collections import deque
from typing import Any, Dict, List, Optional

MAX_RECENT = 500


def _percentile(sorted_values: List[float], pct: float) -> Optional[float]:
    """Nearest-rank percentile; None when there is no data yet."""
    if not sorted_values:
        return None
    rank = math.ceil(pct / 100.0 * len(sorted_values))
    return sorted_values[max(0, min(rank - 1, len(sorted_values) - 1))]


class MetricsCollector:
    """Thread-safe rolling store of scoring outcomes."""

    def __init__(self, max_recent: int = MAX_RECENT) -> None:
        self._lock = threading.Lock()
        self._latencies: deque = deque(maxlen=max_recent)
        self._recent: deque = deque(maxlen=max_recent)
        self._total = 0
        self._fraud = 0

    def record(
        self,
        *,
        transaction_id: Optional[str],
        fraud_probability: float,
        is_fraud: bool,
        latency_ms: float,
        threshold: float,
    ) -> None:
        """Record one scored transaction."""
        with self._lock:
            self._total += 1
            if is_fraud:
                self._fraud += 1
            self._latencies.append(float(latency_ms))
            self._recent.append(
                {
                    "transaction_id": transaction_id,
                    "fraud_probability": round(float(fraud_probability), 4),
                    "is_fraud": bool(is_fraud),
                    "latency_ms": round(float(latency_ms), 3),
                    "threshold": float(threshold),
                }
            )

    def snapshot(self) -> Dict[str, Any]:
        """Aggregate + recent-scores snapshot for GET /metrics."""
        with self._lock:
            latencies = sorted(self._latencies)
            recent = list(self._recent)
            total = self._total
            fraud = self._fraud
        return {
            "total_scored": total,
            "fraud_count": fraud,
            "fraud_rate": (fraud / total) if total else 0.0,
            "p50_latency_ms": _percentile(latencies, 50),
            "p99_latency_ms": _percentile(latencies, 99),
            "recent": recent,
        }

    def reset(self) -> None:
        """Clear all state (tests only)."""
        with self._lock:
            self._latencies.clear()
            self._recent.clear()
            self._total = 0
            self._fraud = 0


_metrics: MetricsCollector | None = None


def get_metrics() -> MetricsCollector:
    """Process-wide singleton; tests reset it via `reset_metrics`."""
    global _metrics
    if _metrics is None:
        _metrics = MetricsCollector()
    return _metrics


def reset_metrics() -> None:
    """Drop the singleton (tests only)."""
    global _metrics
    _metrics = None
