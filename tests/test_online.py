"""Day 3 tests: River online learner, ADWIN drift, retrain trigger, registry."""

import numpy as np
from fastapi.testclient import TestClient

from app.main import app
from app.model import transaction_to_features
from app.online import DriftMonitor, OnlineLearner, reset_online_learner, transaction_to_feature_dict
from app.schemas import MERCHANT_CATEGORIES, Transaction

client = TestClient(app)


def _txn(amount: float, card_present: bool, hour: int = 12) -> Transaction:
    return Transaction(
        amount=amount,
        merchant_category="grocery",
        hour_of_day=hour,
        card_present=card_present,
    )


def test_feature_dict_matches_batch_vector():
    txn = _txn(250.0, True, hour=9)
    as_dict = transaction_to_feature_dict(txn)
    expected_order = ["amount_log", "hour_sin", "hour_cos", "card_present"] + [
        f"cat_{c}" for c in MERCHANT_CATEGORIES
    ]
    assert list(as_dict.keys()) == expected_order
    assert np.allclose(list(as_dict.values()), transaction_to_features(txn))


def test_online_learner_improves_on_stationary_stream():
    """Error rate must fall as the learner sees a fixed, learnable rule."""
    rng = np.random.default_rng(7)
    learner = OnlineLearner()
    errors = []
    for _ in range(400):
        amount = float(rng.uniform(10, 1000))
        card_present = bool(rng.integers(0, 2))
        label = amount > 300 and not card_present
        txn = _txn(amount, card_present)
        predicted = learner.predict_proba(txn) >= 0.5
        errors.append(int(predicted != label))
        learner.learn(txn, label)
    first_half = sum(errors[:100]) / 100
    last_quarter = sum(errors[300:]) / 100
    assert last_quarter < first_half
    assert last_quarter < 0.25


def test_adwin_fires_on_concept_drift():
    """A 0-error stream flipping to all errors must trip ADWIN."""
    monitor = DriftMonitor()
    for _ in range(300):
        assert monitor.update(0) is False
    fired = any(monitor.update(1) for _ in range(300))
    assert fired
    assert monitor.drift_events >= 1


def test_drift_triggers_retrain_and_version_bump():
    """Flipped labels after a stable rule must request a retrain + new version."""
    reset_online_learner()
    rng = np.random.default_rng(11)
    learner = OnlineLearner()
    for _ in range(200):
        amount = float(rng.uniform(10, 1000))
        card_present = bool(rng.integers(0, 2))
        label = amount > 300 and not card_present  # rule A
        learner.learn(_txn(amount, card_present), label)
    for _ in range(400):
        amount = float(rng.uniform(10, 1000))
        card_present = bool(rng.integers(0, 2))
        label = not (amount > 300 and not card_present)  # rule A flipped
        learner.learn(_txn(amount, card_present), label)
    assert learner.retrain_requested is True
    assert learner.registry.current_version != "0.1.0"
    assert len(learner.registry.history) >= 2


def test_feedback_and_drift_endpoints():
    reset_online_learner()
    payload = {
        "transaction": {
            "amount": 500.0,
            "merchant_category": "online_retail",
            "hour_of_day": 2,
            "card_present": False,
        },
        "label": True,
    }
    resp = client.post("/feedback", json=payload)
    assert resp.status_code == 200
    assert resp.json()["samples_seen"] == 1
    status = client.get("/drift")
    assert status.status_code == 200
    assert status.json()["samples_seen"] == 1
    assert status.json()["model_version"] == "0.1.0"
