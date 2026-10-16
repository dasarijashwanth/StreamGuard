"""Champion/challenger comparison + promotion rule (StreamGuard Day 4).

The batch challenger (XGBoost, scripts/train_challenger.py) is compared
against the incumbent champion (Day 1 logistic-regression baseline) on the
same held-out test set. Promotion is deliberately conservative: the
challenger must beat the champion on AUC by a margin AND stay within a
latency budget. `should_promote` is pure Python (no MLflow, no disk) so the
rule itself is trivially unit-testable.

The registry here mirrors the Day 3 `ModelRegistry` pattern from
app/online.py: versions are bumped on promotion and every decision is
recorded with a reason.
"""

import time
from typing import Any, Dict, Optional, Tuple

import numpy as np
from sklearn.metrics import precision_score, recall_score, roc_auc_score

from app.online import ModelRegistry, _next_patch

AUC_MARGIN = 0.005
LATENCY_FACTOR = 1.2


def evaluate_model(model: Any, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    """Score one model on a held-out set: AUC, precision, recall, p50 latency.

    Latency is measured per single-row predict_proba call (the serving
    shape), median over up to 200 rows so the benchmark stays fast.
    """
    proba = model.predict_proba(X)[:, 1]
    preds = (proba >= 0.5).astype(int)
    n_latency = min(200, len(X))
    latencies = []
    for i in range(n_latency):
        start = time.perf_counter()
        model.predict_proba(X[i : i + 1])
        latencies.append((time.perf_counter() - start) * 1000.0)
    return {
        "auc": float(roc_auc_score(y, proba)),
        "precision": float(precision_score(y, preds, zero_division=0)),
        "recall": float(recall_score(y, preds, zero_division=0)),
        "p50_latency_ms": float(np.median(latencies)),
    }


def should_promote(champion: Dict[str, float], challenger: Dict[str, float]) -> Tuple[bool, str]:
    """Promotion rule: challenger must clear BOTH bars.

    - AUC: challenger >= champion + 0.005 (a real gain, not noise).
    - Latency: challenger p50 <= champion p50 * 1.2 (no more than 20% slower).
    """
    if challenger["auc"] < champion["auc"] + AUC_MARGIN:
        return False, (
            f"challenger AUC {challenger['auc']:.4f} does not beat "
            f"champion {champion['auc']:.4f} + margin {AUC_MARGIN}"
        )
    if challenger["p50_latency_ms"] > champion["p50_latency_ms"] * LATENCY_FACTOR:
        return False, (
            f"challenger p50 latency {challenger['p50_latency_ms']:.3f}ms exceeds "
            f"champion {champion['p50_latency_ms']:.3f}ms x {LATENCY_FACTOR}"
        )
    return True, (
        f"challenger AUC {challenger['auc']:.4f} beats champion "
        f"{champion['auc']:.4f} within latency budget"
    )


class ChallengerRegistry:
    """Tracks the champion, the latest challenger, and promotion history."""

    def __init__(self, registry: Optional[ModelRegistry] = None) -> None:
        self.registry = registry or ModelRegistry()
        self.champion: Optional[Dict[str, Any]] = None
        self.challenger: Optional[Dict[str, Any]] = None
        self.history: list = []

    def register_champion(self, metrics: Dict[str, float], version: str) -> None:
        """Seed the incumbent champion (e.g. the Day 1 baseline)."""
        self.champion = {**metrics, "version": version, "status": "champion"}

    def register_challenger(self, metrics: Dict[str, float], version: str) -> None:
        """Record a freshly trained challenger awaiting comparison."""
        self.challenger = {**metrics, "version": version, "status": "challenger"}

    def maybe_promote(self) -> Dict[str, Any]:
        """Apply the promotion rule; bump the registry version on promotion.

        Returns the decision dict: {promoted, reason, champion, challenger}.
        With no incumbent champion, the challenger wins by default.
        """
        if self.challenger is None:
            return {"promoted": False, "reason": "no challenger registered"}
        if self.champion is None:
            promoted, reason = True, "no incumbent champion; challenger wins by default"
        else:
            promoted, reason = should_promote(self.champion, self.challenger)
        if promoted:
            new_version = _next_patch(self.registry.current_version)
            self.registry.promote(new_version, f"Challenger promoted: {reason}")
            self.champion = {**self.challenger, "version": new_version, "status": "champion"}
            self.challenger = {**self.challenger, "status": "promoted"}
        else:
            self.challenger = {**self.challenger, "status": "rejected"}
        decision = {
            "promoted": promoted,
            "reason": reason,
            "champion": dict(self.champion) if self.champion else None,
            "challenger": dict(self.challenger) if self.challenger else None,
        }
        self.history.append(decision)
        return decision

    def snapshot(self) -> Dict[str, Any]:
        """Registry state for GET /models."""
        return {
            "champion": dict(self.champion) if self.champion else None,
            "challenger": dict(self.challenger) if self.challenger else None,
            "registry_version": self.registry.current_version,
            "registry_history": list(self.registry.history),
        }


_challenger_registry: Optional[ChallengerRegistry] = None


def get_challenger_registry() -> ChallengerRegistry:
    """Process-wide singleton; tests reset it via `reset_challenger_registry`."""
    global _challenger_registry
    if _challenger_registry is None:
        _challenger_registry = ChallengerRegistry()
    return _challenger_registry


def reset_challenger_registry() -> None:
    """Drop the singleton (tests only)."""
    global _challenger_registry
    _challenger_registry = None
