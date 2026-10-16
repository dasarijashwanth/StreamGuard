"""River online learner + ADWIN drift detection (StreamGuard Day 3).

The River model *shadows* the batch model from Day 1: every scored
transaction that later gets a true label is fed to the online learner, and
the prediction error stream is watched by an ADWIN drift detector. When
ADWIN confirms drift, we raise an auto-retrain trigger and version a new
candidate in the model registry. The actual batch retrain lands on Day 4
with MLflow; Day 3 owns detection, triggering, and versioning.
"""

import math
from typing import Dict, List, Optional

from river import compose, drift, linear_model, optim, preprocessing

from app.schemas import MERCHANT_CATEGORIES, Transaction


def transaction_to_feature_dict(txn: Transaction) -> Dict[str, float]:
    """Same features as `transaction_to_features`, as a River feature dict.

    Keys: amount_log, hour_sin, hour_cos, card_present, cat_<category>.
    """
    features = {
        "amount_log": math.log1p(txn.amount),
        "hour_sin": math.sin(2 * math.pi * txn.hour_of_day / 24),
        "hour_cos": math.cos(2 * math.pi * txn.hour_of_day / 24),
        "card_present": 1.0 if txn.card_present else 0.0,
    }
    for category in MERCHANT_CATEGORIES:
        features[f"cat_{category}"] = 1.0 if txn.merchant_category == category else 0.0
    return features


class ModelRegistry:
    """In-memory model version registry with promotion history."""

    def __init__(self, initial_version: str = "0.1.0") -> None:
        self.current_version = initial_version
        self.history: List[Dict[str, str]] = [
            {"version": initial_version, "reason": "Day 1 batch baseline"}
        ]

    def promote(self, version: str, reason: str) -> None:
        """Record a new candidate version with the reason it was created."""
        self.current_version = version
        self.history.append({"version": version, "reason": reason})


def _next_patch(version: str) -> str:
    """Bump the patch segment of a semver-ish version string."""
    parts = version.split(".")
    try:
        parts[-1] = str(int(parts[-1]) + 1)
    except ValueError:
        parts.append("1")
    return ".".join(parts)


class DriftMonitor:
    """ADWIN detector over the 0/1 prediction-error stream."""

    def __init__(self, delta: float = 0.002) -> None:
        self.detector = drift.ADWIN(delta=delta)
        self.samples_seen = 0
        self.errors = 0
        self.drift_events = 0
        self.drift_detected = False

    def update(self, error: int) -> bool:
        """Feed one error (0 = correct, 1 = wrong). Returns True on new drift."""
        self.samples_seen += 1
        self.errors += int(error)
        self.detector.update(int(error))
        if self.detector.drift_detected:
            self.drift_events += 1
            self.drift_detected = True
            return True
        return False

    @property
    def error_rate(self) -> float:
        if self.samples_seen == 0:
            return 0.0
        return self.errors / self.samples_seen


class OnlineLearner:
    """River logistic-regression learner shadowing the batch model."""

    def __init__(self, registry: Optional[ModelRegistry] = None) -> None:
        self.model = compose.Pipeline(
            preprocessing.StandardScaler(),
            linear_model.LogisticRegression(optimizer=optim.SGD(0.01), l2=1e-4),
        )
        self.monitor = DriftMonitor()
        self.registry = registry or ModelRegistry()
        self.retrain_requested = False

    def predict_proba(self, txn: Transaction) -> float:
        """Shadow-model fraud probability for one transaction."""
        features = transaction_to_feature_dict(txn)
        return float(self.model.predict_proba_one(features).get(True, 0.0))

    def learn(self, txn: Transaction, label: bool) -> Dict[str, object]:
        """Learn from one labeled transaction; check drift; maybe trigger retrain.

        Returns a status dict with the drift outcome and registry state.
        """
        features = transaction_to_feature_dict(txn)
        predicted = self.model.predict_proba_one(features).get(True, 0.0)
        error = int((predicted >= 0.5) != bool(label))
        self.model.learn_one(features, bool(label))
        drifted = self.monitor.update(error)
        if drifted and not self.retrain_requested:
            self.retrain_requested = True
            new_version = _next_patch(self.registry.current_version)
            self.registry.promote(new_version, "ADWIN drift confirmed, retrain requested")
        return {
            "drift_detected": drifted,
            "retrain_requested": self.retrain_requested,
            "samples_seen": self.monitor.samples_seen,
            "error_rate": round(self.monitor.error_rate, 4),
            "drift_events": self.monitor.drift_events,
            "model_version": self.registry.current_version,
        }

    def status(self) -> Dict[str, object]:
        """Current drift/registry snapshot for GET /drift."""
        return {
            "samples_seen": self.monitor.samples_seen,
            "errors": self.monitor.errors,
            "error_rate": round(self.monitor.error_rate, 4),
            "drift_events": self.monitor.drift_events,
            "drift_detected": self.monitor.drift_detected,
            "retrain_requested": self.retrain_requested,
            "model_version": self.registry.current_version,
            "registry_history": list(self.registry.history),
        }


_learner: Optional[OnlineLearner] = None


def get_online_learner() -> OnlineLearner:
    """Process-wide singleton; tests reset it via `reset_online_learner`."""
    global _learner
    if _learner is None:
        _learner = OnlineLearner()
    return _learner


def reset_online_learner() -> None:
    """Drop the singleton (tests only)."""
    global _learner
    _learner = None
