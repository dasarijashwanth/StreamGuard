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
from app.synthetic import generate_transactions

SEED = 42
N_SAMPLES = 20_000
FRAUD_RATE = 0.03
ARTIFACT_PATH = Path(__file__).parent.parent / "app" / "models" / "baseline_model.joblib"


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
