"""Day 2 tests: message codec + produce -> consume -> score round trip."""

import json
import socket
import uuid

import pytest

from app.model import score_transaction
from app.schemas import Transaction
from app.streaming import (
    DEFAULT_BOOTSTRAP,
    TOPIC,
    decode_transaction,
    encode_transaction,
    make_consumer,
    make_producer,
)

TXN = Transaction(
    transaction_id="txn-test-001",
    amount=4500.00,
    merchant_category="online_retail",
    hour_of_day=3,
    card_present=False,
)


def test_encode_decode_round_trip():
    assert decode_transaction(encode_transaction(TXN)) == TXN


def test_decode_rejects_invalid_bytes():
    with pytest.raises(Exception):
        decode_transaction(b"not json")


def test_encoded_message_is_json_with_expected_fields():
    payload = json.loads(encode_transaction(TXN))
    assert payload["transaction_id"] == "txn-test-001"
    assert payload["amount"] == 4500.00
    assert payload["merchant_category"] == "online_retail"


def _broker_reachable() -> bool:
    host, port = DEFAULT_BOOTSTRAP.split(":")
    try:
        with socket.create_connection((host, int(port)), timeout=2):
            return True
    except OSError:
        return False


@pytest.mark.skipif(not _broker_reachable(), reason="No Redpanda broker on localhost:9092")
def test_produce_consume_score_round_trip():
    """Integration: publish one txn, consume it back, score it end to end.

    Needs a live broker (docker compose up, or the Day 2 on-ramp script).
    Uses a throwaway consumer group so offsets never collide with other runs.
    """
    txn = Transaction(
        transaction_id=f"txn-it-{uuid.uuid4().hex[:8]}",
        amount=42.50,
        merchant_category="grocery",
        hour_of_day=14,
        card_present=True,
    )
    producer = make_producer()
    producer.send(TOPIC, key=txn.transaction_id.encode(), value=encode_transaction(txn))
    producer.flush()
    producer.close()

    consumer = make_consumer(group_id=f"streamguard-test-{uuid.uuid4().hex[:8]}", timeout_ms=15000)
    found = None
    for message in consumer:
        candidate = decode_transaction(message.value)
        if candidate.transaction_id == txn.transaction_id:
            found = candidate
            break
    consumer.close()

    assert found is not None, "Published transaction never came back off the topic"
    probability, is_fraud, version, threshold, latency_ms = score_transaction(found)
    assert 0.0 <= probability <= 1.0
    assert version == "0.1.0"
    assert latency_ms >= 0
