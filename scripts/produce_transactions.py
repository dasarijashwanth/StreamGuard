"""Day 2 producer: publish synthetic transactions to the Redpanda topic.

Run:  python scripts/produce_transactions.py [--count N] [--bootstrap-servers HOST:PORT]

Reuses the Day 1 synthetic generator (seed=42) so produced traffic matches
the distribution the baseline model was trained on. Each message gets a
unique transaction_id and is JSON-encoded via app.streaming.
"""

import argparse
import sys
import uuid
from pathlib import Path

# Allow `python scripts/produce_transactions.py` from the project root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.schemas import Transaction
from app.streaming import DEFAULT_BOOTSTRAP, TOPIC, encode_transaction, make_producer
from scripts.train_baseline import generate_transactions


def main() -> None:
    parser = argparse.ArgumentParser(description="Publish synthetic transactions to Redpanda.")
    parser.add_argument("--count", type=int, default=50, help="Transactions to publish.")
    parser.add_argument("--bootstrap-servers", default=DEFAULT_BOOTSTRAP)
    args = parser.parse_args()

    rows, _labels = generate_transactions(args.count, fraud_rate=0.03, seed=42)
    producer = make_producer(args.bootstrap_servers)
    for row in rows:
        txn = Transaction(transaction_id=f"txn-{uuid.uuid4().hex[:8]}", **row)
        producer.send(TOPIC, key=txn.transaction_id.encode(), value=encode_transaction(txn))
    producer.flush()
    producer.close()
    print(f"Published {len(rows)} transactions to topic '{TOPIC}'.")


if __name__ == "__main__":
    main()
