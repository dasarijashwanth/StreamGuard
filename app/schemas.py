"""Pydantic schemas for StreamGuard API contracts.

Day 1 covers the core transaction + scoring contracts. Streaming envelopes
(Redpanda message wrappers) and online-learning feedback schemas plug in here
on Days 2-3 without breaking existing fields.
"""

from typing import Optional

from pydantic import BaseModel, Field, field_validator

MERCHANT_CATEGORIES = [
    "grocery",
    "gas",
    "restaurant",
    "online_retail",
    "travel",
    "atm",
    "other",
]

MODEL_VERSION = "0.1.0"


class Transaction(BaseModel):
    """A single payment transaction to be scored."""

    transaction_id: Optional[str] = Field(
        default=None, description="Client-supplied idempotency key, if any."
    )
    amount: float = Field(gt=0, le=1_000_000, description="Transaction amount in USD.")
    merchant_category: str = Field(description="Merchant category; unknown values map to 'other'.")
    hour_of_day: int = Field(ge=0, le=23, description="Local hour the transaction occurred (0-23).")
    card_present: bool = Field(description="True for in-person chip/tap, False for card-not-present.")

    @field_validator("merchant_category")
    @classmethod
    def normalize_category(cls, value: str) -> str:
        """Lowercase/strip the category; bucket anything unknown as 'other'.

        Real-world merchant feeds are messy, so we coerce rather than reject.
        """
        normalized = value.strip().lower().replace(" ", "_")
        return normalized if normalized in MERCHANT_CATEGORIES else "other"


class ScoreResponse(BaseModel):
    """Fraud score for one transaction."""

    transaction_id: Optional[str] = None
    fraud_probability: float = Field(ge=0.0, le=1.0)
    is_fraud: bool = Field(description="True when fraud_probability >= threshold.")
    model_version: str
    threshold: float
    latency_ms: float = Field(description="Server-side scoring time in milliseconds.")
