"""Day 6 tests: evaluation metric math on tiny hand-computed datasets."""

import numpy as np
import pytest

from app.evaluate import calibration_table, evaluate_holdout, precision_recall_at_thresholds


class _StubModel:
    """Predicts fixed probabilities regardless of input."""

    def __init__(self, proba):
        self._proba = np.asarray(proba, dtype=float)

    def predict_proba(self, X):
        n = len(X)
        p1 = self._proba[:n]
        return np.column_stack([1 - p1, p1])


def test_precision_recall_at_thresholds():
    y = [0, 0, 1, 1]
    proba = [0.1, 0.4, 0.6, 0.9]
    out = precision_recall_at_thresholds(y, proba)
    assert out["0.5"] == {"precision": 1.0, "recall": 1.0}
    assert out["0.7"] == {"precision": 1.0, "recall": 0.5}
    assert out["0.3"]["precision"] == 2 / 3
    assert out["0.3"]["recall"] == 1.0
    assert out["0.9"] == {"precision": 1.0, "recall": 0.5}


def test_precision_recall_all_negative_predictions():
    # No prediction clears the threshold: precision is 0 by zero_division rule.
    out = precision_recall_at_thresholds([0, 1], [0.1, 0.2], thresholds=[0.9])
    assert out["0.9"] == {"precision": 0.0, "recall": 0.0}


def test_calibration_table_two_bins():
    y = [0, 0, 1, 1]
    proba = [0.1, 0.2, 0.8, 0.9]
    table = calibration_table(y, proba, n_bins=2)
    assert table["ece"] == pytest.approx(0.15)
    low, high = table["bins"]
    assert low["count"] == 2
    assert low["mean_predicted"] == pytest.approx(0.15)
    assert low["observed_fraud_rate"] == 0.0
    assert high["count"] == 2
    assert high["mean_predicted"] == pytest.approx(0.85)
    assert high["observed_fraud_rate"] == 1.0


def test_calibration_table_empty_bins_report_none():
    table = calibration_table([0, 1], [0.05, 0.06], n_bins=2)
    low, high = table["bins"]
    assert low["count"] == 2
    assert high["count"] == 0
    assert high["mean_predicted"] is None
    assert high["observed_fraud_rate"] is None


def test_evaluate_holdout_stub_model():
    X = np.zeros((2, 3))
    y = np.array([0, 1])
    report = evaluate_holdout(_StubModel([0.2, 0.8]), X, y)
    assert report["n"] == 2
    assert report["fraud_rate"] == 0.5
    assert report["auc"] == 1.0
    assert report["precision_recall"]["0.5"] == {"precision": 1.0, "recall": 1.0}
    assert len(report["calibration"]["bins"]) == 10
