"""Train the Day 1 baseline fraud model on reproducible synthetic data.

Run:  python scripts/train_baseline.py
Seed is fixed (42) so the artifact is byte-reproducible for a given
scikit-learn version. Writes app/models/baseline_model.joblib.

Synthetic world design (deliberately separable, fraud rate ~3%):
  - Legit: moderate lognormal amounts, daytime hours, card usually present,
    everyday merchant categories.
  - Fraud: large amounts, night hours (0-5), card rarely present,
    online_retail / travel / atm heavy.
"""

import sys
from datetime import datetime, timezone
from pathlib import Path

# Allow `python scripts/train_baseline.py` from the project root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from app.model import transaction_to_features
from app.schemas import MERCHANT_CATEGORIES, MODEL_VERSION, Transaction

SEED = 42
N_SAMPLES = 20_000
FRAUD_RATE = 0.03
ARTIFACT_PATH = Path(__file__).parent.parent / "app" / "models" / "baseline_model.joblib"

LEGIT_CATS = ["grocery", "gas", "restaurant", "grocery", "gas", "other"]
FRAUD_CATS = ["online_retail", "travel", "atm", "online_retail", "other"]


def _sample_hours(rng: np.random.Generator, n: int, night_bias: bool) -> np.ndarray:
    if night_bias:
        # Fraud clusters between midnight and 5am
        values = [0, 1, 2, 3, 4, 5, 12, 14, 20]
        probs = np.array([0.2, 0.2, 0.2, 0.15, 0.1, 0.05, 0.04, 0.03, 0.03])
    else:
        # Legit clusters in daytime/evening
        values = list(range(24))
        probs = np.array([0.01, 0.01, 0.01, 0.005, 0.005, 0.01, 0.02, 0.04, 0.06,
                          0.07, 0.07, 0.08, 0.08, 0.07, 0.06, 0.06, 0.06, 0.07,
                          0.08, 0.07, 0.05, 0.03, 0.02, 0.015])
    probs = probs / probs.sum()  # guard against rounding drift
    return rng.choice(values, size=n, p=probs)


def generate_transactions(n: int, fraud_rate: float, seed: int):
    """Generate raw transaction dicts + labels."""
    rng = np.random.default_rng(seed)
    n_fraud = int(n * fraud_rate)
    n_legit = n - n_fraud

    rows, labels = [], []

    for _ in range(n_legit):
        rows.append({
            "amount": float(np.clip(rng.lognormal(3.2, 0.9), 1, 50_000)),
            "merchant_category": str(rng.choice(LEGIT_CATS)),
            "hour_of_day": int(_sample_hours(rng, 1, night_bias=False)[0]),
            "card_present": bool(rng.random() < 0.85),
        })
        labels.append(0)

    for _ in range(n_fraud):
        rows.append({
            "amount": float(np.clip(rng.lognormal(5.0, 1.1), 1, 50_000)),
            "merchant_category": str(rng.choice(FRAUD_CATS)),
            "hour_of_day": int(_sample_hours(rng, 1, night_bias=True)[0]),
            "card_present": bool(rng.random() < 0.25),
        })
        labels.append(1)

    # Shuffle so order carries no signal
    idx = rng.permutation(n)
    rows = [rows[i] for i in idx]
    return rows, np.array([labels[i] for i in idx])


def main() -> None:
    rows, y = generate_transactions(N_SAMPLES, FRAUD_RATE, SEED)
    X = np.vstack([transaction_to_features(Transaction(**r)) for r in rows])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )

    model = LogisticRegression(max_iter=1000, random_state=SEED)
    model.fit(X_train, y_train)

    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    print(f"train_size={len(X_train)} test_size={len(X_test)} "
          f"fraud_rate={y.mean():.3f} test_auc={auc:.4f}")

    ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "categories": MERCHANT_CATEGORIES,
            "version": MODEL_VERSION,
            "threshold": 0.5,
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "seed": SEED,
            "test_auc": float(auc),
        },
        ARTIFACT_PATH,
    )
    print(f"Saved artifact -> {ARTIFACT_PATH}")


if __name__ == "__main__":
    main()
