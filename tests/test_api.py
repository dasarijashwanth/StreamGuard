"""Day 1 tests: schemas, /health, /score happy path and edge cases."""

from fastapi.testclient import TestClient

from app.main import app
from app.schemas import Transaction

client = TestClient(app)

LEGIT_TXN = {
    "transaction_id": "txn-legit-001",
    "amount": 42.50,
    "merchant_category": "grocery",
    "hour_of_day": 14,
    "card_present": True,
}

FRAUD_TXN = {
    "transaction_id": "txn-fraud-001",
    "amount": 4500.00,
    "merchant_category": "online_retail",
    "hour_of_day": 3,
    "card_present": False,
}


def test_health_ok():
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["model_version"] == "0.1.0"


def test_score_legit_low_risk():
    resp = client.post("/score", json=LEGIT_TXN)
    assert resp.status_code == 200
    body = resp.json()
    assert body["transaction_id"] == "txn-legit-001"
    assert 0.0 <= body["fraud_probability"] <= 1.0
    assert body["is_fraud"] is False
    assert body["model_version"] == "0.1.0"
    assert body["latency_ms"] >= 0


def test_score_fraud_high_risk():
    resp = client.post("/score", json=FRAUD_TXN)
    assert resp.status_code == 200
    body = resp.json()
    assert body["is_fraud"] is True
    assert body["fraud_probability"] >= 0.5


def test_score_separates_risk():
    legit = client.post("/score", json=LEGIT_TXN).json()["fraud_probability"]
    fraud = client.post("/score", json=FRAUD_TXN).json()["fraud_probability"]
    assert fraud > legit


def test_score_without_transaction_id():
    payload = {k: v for k, v in LEGIT_TXN.items() if k != "transaction_id"}
    resp = client.post("/score", json=payload)
    assert resp.status_code == 200
    assert resp.json()["transaction_id"] is None


def test_reject_negative_amount():
    resp = client.post("/score", json={**LEGIT_TXN, "amount": -5})
    assert resp.status_code == 422


def test_reject_zero_amount():
    resp = client.post("/score", json={**LEGIT_TXN, "amount": 0})
    assert resp.status_code == 422


def test_reject_hour_out_of_range():
    resp = client.post("/score", json={**LEGIT_TXN, "hour_of_day": 25})
    assert resp.status_code == 422
    resp = client.post("/score", json={**LEGIT_TXN, "hour_of_day": -1})
    assert resp.status_code == 422


def test_reject_missing_amount():
    payload = {k: v for k, v in LEGIT_TXN.items() if k != "amount"}
    resp = client.post("/score", json=payload)
    assert resp.status_code == 422


def test_unknown_category_coerced_to_other():
    resp = client.post("/score", json={**LEGIT_TXN, "merchant_category": "Space Tourism"})
    assert resp.status_code == 200
    txn = Transaction(**{**LEGIT_TXN, "merchant_category": "Space Tourism"})
    assert txn.merchant_category == "other"


def test_category_normalization():
    txn = Transaction(**{**LEGIT_TXN, "merchant_category": "  Online_Retail "})
    assert txn.merchant_category == "online_retail"
