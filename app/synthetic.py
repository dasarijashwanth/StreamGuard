"""Reproducible synthetic transaction generator (moved from scripts/train_baseline.py on Day 6).

The generator lives here so training, evaluation, benchmarks, and tests all
draw from the same synthetic world and can never drift apart.

Synthetic world design (deliberately separable, fraud rate ~3%):
  - Legit: moderate lognormal amounts, daytime hours, card usually present,
    everyday merchant categories.
  - Fraud: large amounts, night hours (0-5), card rarely present,
    online_retail / travel / atm heavy.
"""

import numpy as np

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
