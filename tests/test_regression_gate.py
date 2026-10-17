"""Day 6 tests: regression gate.

Fails the suite if the champion regresses below the committed Day 6
results (evals/day6_results.json) minus tolerance, so a future model swap
cannot silently ship a worse model. Uses a small fresh holdout (seed 123,
same seed as the committed run) to keep the test fast.
"""

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from app.model import load_bundle, transaction_to_features
from app.schemas import Transaction
from app.synthetic import generate_transactions

RESULTS_PATH = Path(__file__).parent.parent / "evals" / "day6_results.json"
GATE_HOLDOUT_N = 2000
GATE_SEED = 123


def test_champion_passes_regression_gate():
    if not RESULTS_PATH.exists():
        raise AssertionError(
            "evals/day6_results.json is missing; run `python scripts/evaluate.py` first."
        )
    baseline = json.loads(RESULTS_PATH.read_text())
    gate = baseline["regression_gate"]

    rows, y = generate_transactions(GATE_HOLDOUT_N, 0.03, GATE_SEED)
    X = np.vstack([transaction_to_features(Transaction(**r)) for r in rows])
    proba = load_bundle()["model"].predict_proba(X)[:, 1]

    auc = float(roc_auc_score(y, proba))
    assert auc >= gate["min_auc"], (
        f"champion AUC {auc:.4f} below regression gate {gate['min_auc']:.4f} "
        f"(committed {baseline['evaluation']['auc']:.4f} minus tolerance {gate['auc_tolerance']})"
    )
    preds = (proba >= 0.5).astype(int)
    tp = int(((preds == 1) & (y == 1)).sum())
    fp = int(((preds == 1) & (y == 0)).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    assert precision >= gate["min_precision_at_0.5"], (
        f"precision@0.5 {precision:.4f} below gate {gate['min_precision_at_0.5']:.4f}"
    )
