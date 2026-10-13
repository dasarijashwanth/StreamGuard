"""Model loading, feature engineering, and scoring for StreamGuard.

The feature pipeline lives in exactly one place (`transaction_to_features`)
so training (scripts/train_baseline.py) and serving (app/main.py) can never
drift apart. Day 3's River online learner and Day 4's XGBoost challenger reuse
this same function.
"""

import math
import time
from pathlib import Path
from typing import Any, Dict, Tuple

import joblib
import numpy as np

from app.schemas import MERCHANT_CATEGORIES, Transaction

ARTIFACT_PATH = Path(__file__).parent / "models" / "baseline_model.joblib"
DEFAULT_THRESHOLD = 0.5

_model_bundle: Dict[str, Any] | None = None


def transaction_to_features(txn: Transaction) -> np.ndarray:
    """Convert a Transaction into the model feature vector.

    Features: log1p(amount), sin/cos hour encoding, card_present flag,
    one-hot merchant category. Order is fixed and shared with training.
    """
    amount_log = math.log1p(txn.amount)
    hour_sin = math.sin(2 * math.pi * txn.hour_of_day / 24)
    hour_cos = math.cos(2 * math.pi * txn.hour_of_day / 24)
    card_present = 1.0 if txn.card_present else 0.0
    category_onehot = [1.0 if txn.merchant_category == c else 0.0 for c in MERCHANT_CATEGORIES]
    return np.array(
        [amount_log, hour_sin, hour_cos, card_present, *category_onehot],
        dtype=np.float64,
    )


def load_bundle() -> Dict[str, Any]:
    """Load (once) the trained artifact bundle from disk."""
    global _model_bundle
    if _model_bundle is None:
        if not ARTIFACT_PATH.exists():
            raise FileNotFoundError(
                f"Model artifact not found at {ARTIFACT_PATH}. "
                "Run `python scripts/train_baseline.py` first."
            )
        _model_bundle = joblib.load(ARTIFACT_PATH)
    return _model_bundle


def score_transaction(txn: Transaction) -> Tuple[float, bool, str, float, float]:
    """Score one transaction.

    Returns (fraud_probability, is_fraud, model_version, threshold, latency_ms).
    """
    bundle = load_bundle()
    start = time.perf_counter()
    features = transaction_to_features(txn).reshape(1, -1)
    probability = float(bundle["model"].predict_proba(features)[0][1])
    latency_ms = (time.perf_counter() - start) * 1000.0
    threshold = float(bundle.get("threshold", DEFAULT_THRESHOLD))
    return probability, probability >= threshold, bundle["version"], threshold, latency_ms
