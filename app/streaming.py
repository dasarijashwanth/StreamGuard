"""Kafka/Redpanda plumbing for StreamGuard Day 2.

One shared home for the topic name, message encoding, and client factories,
so the producer script, consumer script, and tests can never disagree on the
wire format. Day 3's online learner consumes from this same topic.
"""

from kafka import KafkaConsumer, KafkaProducer

from app.schemas import Transaction

TOPIC = "transactions"
DEFAULT_BOOTSTRAP = "localhost:9092"
CONSUMER_GROUP = "streamguard-scorer"


def encode_transaction(txn: Transaction) -> bytes:
    """Serialize a Transaction to JSON bytes for the topic."""
    return txn.model_dump_json().encode("utf-8")


def decode_transaction(data: bytes) -> Transaction:
    """Deserialize topic bytes back into a validated Transaction."""
    return Transaction.model_validate_json(data)


def make_producer(bootstrap_servers: str = DEFAULT_BOOTSTRAP) -> KafkaProducer:
    """Create a producer that sends raw bytes (we encode with encode_transaction)."""
    return KafkaProducer(bootstrap_servers=bootstrap_servers)


def make_consumer(
    bootstrap_servers: str = DEFAULT_BOOTSTRAP,
    group_id: str = CONSUMER_GROUP,
    timeout_ms: int = 1000,
) -> KafkaConsumer:
    """Create a consumer on the transactions topic, earliest offset first."""
    return KafkaConsumer(
        TOPIC,
        bootstrap_servers=bootstrap_servers,
        group_id=group_id,
        auto_offset_reset="earliest",
        enable_auto_commit=True,
        consumer_timeout_ms=timeout_ms,
    )
