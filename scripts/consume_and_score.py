"""Day 2 consumer: score transactions off the Redpanda topic in real time.

Run:  python scripts/consume_and_score.py [--max-messages N] [--bootstrap-servers HOST:PORT]

Consumes from the `transactions` topic, validates each message, scores it
through the same scoring path the API uses (`app.model.score_transaction`),
and prints one JSON line per message. Stops after --max-messages (0 = run
until Ctrl+C). Day 3 will feed these scores into the online learner.
"""

import argparse
import json
import sys
from pathlib import Path

# Allow `python scripts/consume_and_score.py` from the project root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.model import score_transaction
from app.streaming import DEFAULT_BOOTSTRAP, decode_transaction, make_consumer


def main() -> None:
    parser = argparse.ArgumentParser(description="Score transactions from the Redpanda topic.")
    parser.add_argument("--max-messages", type=int, default=0, help="Stop after N messages (0 = until Ctrl+C).")
    parser.add_argument("--bootstrap-servers", default=DEFAULT_BOOTSTRAP)
    args = parser.parse_args()

    consumer = make_consumer(args.bootstrap_servers)
    scored = 0
    try:
        for message in consumer:
            txn = decode_transaction(message.value)
            probability, is_fraud, version, threshold, latency_ms = score_transaction(txn)
            print(json.dumps({
                "transaction_id": txn.transaction_id,
                "fraud_probability": round(probability, 4),
                "is_fraud": is_fraud,
                "model_version": version,
                "threshold": threshold,
                "latency_ms": round(latency_ms, 3),
            }), flush=True)
            scored += 1
            if args.max_messages and scored >= args.max_messages:
                break
    except KeyboardInterrupt:
        pass
    finally:
        consumer.close()
    print(f"Scored {scored} transactions.", file=sys.stderr)


if __name__ == "__main__":
    main()
