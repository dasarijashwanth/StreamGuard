"""Train the Day 4 batch challenger (XGBoost) and compare vs the champion.

Run:  python scripts/train_challenger.py [--data labeled.csv] [--tracking-uri URI]

Data: by default, reproducible synthetic labeled data from the same
generator/seed as scripts/train_baseline.py (seed=42). Pass --data with a
CSV of labeled transactions (columns: amount, merchant_category,
hour_of_day, card_present, label) to train on real accumulated labels.

MLflow: when --tracking-uri (or MLFLOW_TRACKING_URI) points at a
reachable server, params/metrics/artifact are logged there. When no
server is reachable (the normal local case), the run falls back to
local logging: metrics JSON under evals/ + a printed summary. It never
crashes for lack of MLflow.
"""

import argparse
import json
import os
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

# Allow `python scripts/train_challenger.py` from the project root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import numpy as np
from sklearn.model_selection import train_test_split

from app.challenger import evaluate_model, should_promote
from app.model import load_bundle, transaction_to_features
from app.schemas import MODEL_VERSION, Transaction
from scripts.train_baseline import SEED, generate_transactions

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHALLENGER_ARTIFACT_PATH = PROJECT_ROOT / "app" / "models" / "challenger_model.joblib"
DEFAULT_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")

XGB_PARAMS = {
    "n_estimators": 200,
    "max_depth": 4,
    "learning_rate": 0.05,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "eval_metric": "logloss",
    "n_jobs": -1,
    "random_state": SEED,
}


def _tracking_server_reachable(uri: str, timeout: float = 2.0) -> bool:
    """Cheap reachability probe: HTTP GET the tracking server base URL."""
    if not uri.startswith(("http://", "https://")):
        return True  # local file store: always usable
    try:
        with urllib.request.urlopen(uri.rstrip("/") + "/", timeout=timeout) as resp:
            return resp.status < 500
    except Exception:
        return False


def _load_labeled_rows(data_path: str | None, n_samples: int):
    """Return (feature matrix, labels) from a CSV or the synthetic generator."""
    if data_path:
        import pandas as pd

        df = pd.read_csv(data_path)
        required = {"amount", "merchant_category", "hour_of_day", "card_present", "label"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"CSV {data_path} missing columns: {sorted(missing)}")
        rows = df.to_dict("records")
        y = np.array([int(r.pop("label")) for r in rows])
        X = np.vstack([transaction_to_features(Transaction(**r)) for r in rows])
        return X, y
    rows, y = generate_transactions(n_samples, 0.03, SEED)
    X = np.vstack([transaction_to_features(Transaction(**r)) for r in rows])
    return X, y


def _log_to_mlflow(tracking_uri: str, params: dict, champion: dict, challenger: dict,
                   decision: dict, artifact_path: Path) -> bool:
    """Log the run to MLflow. Returns False (never raises) on any failure."""
    try:
        import mlflow
    except ImportError:
        print("mlflow not installed; skipping MLflow logging")
        return False
    try:
        mlflow.set_tracking_uri(tracking_uri)
        with mlflow.start_run(run_name=f"challenger-{datetime.now(timezone.utc):%Y%m%d-%H%M%S}"):
            mlflow.log_params({f"xgb_{k}": v for k, v in params.items()})
            mlflow.log_params({"seed": SEED, "model_type": "XGBClassifier"})
            mlflow.log_metrics({
                "challenger_auc": challenger["auc"],
                "challenger_precision": challenger["precision"],
                "challenger_recall": challenger["recall"],
                "challenger_p50_latency_ms": challenger["p50_latency_ms"],
                "champion_auc": champion["auc"] if champion else 0.0,
                "auc_delta": challenger["auc"] - (champion["auc"] if champion else 0.0),
            })
            mlflow.log_artifact(str(artifact_path))
        print(f"Logged run to MLflow at {tracking_uri}")
        return True
    except Exception as exc:  # tracking server died mid-run, bad URI, etc.
        print(f"MLflow logging failed ({exc}); local metrics JSON is the record")
        return False


def train_challenger(data_path: str | None = None, n_samples: int = 20_000,
                     tracking_uri: str | None = None,
                     output_dir: Path | None = None) -> dict:
    """Train the XGBoost challenger, compare vs champion, log results.

    Returns the full result dict (champion + challenger metrics, decision).
    `tracking_uri=None` skips MLflow entirely (fast local path used by tests).
    """
    from xgboost import XGBClassifier

    X, y = _load_labeled_rows(data_path, n_samples)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )

    model = XGBClassifier(**XGB_PARAMS)
    model.fit(X_train, y_train)
    challenger_metrics = evaluate_model(model, X_test, y_test)

    try:
        bundle = load_bundle()
        champion_metrics = evaluate_model(bundle["model"], X_test, y_test)
        champion_version = bundle["version"]
    except FileNotFoundError:
        champion_metrics, champion_version = None, None
        print("No champion artifact found; challenger wins by default")

    if champion_metrics is None:
        promoted, reason = True, "no incumbent champion; challenger wins by default"
    else:
        promoted, reason = should_promote(champion_metrics, challenger_metrics)

    result = {
        "model_type": "XGBClassifier",
        "params": XGB_PARAMS,
        "seed": SEED,
        "train_size": len(X_train),
        "test_size": len(X_test),
        "champion": ({**champion_metrics, "version": champion_version}
                     if champion_metrics else None),
        "challenger": {**challenger_metrics, "version": MODEL_VERSION},
        "promoted": promoted,
        "reason": reason,
        "mlflow_logged": False,
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }

    out_dir = output_dir or (PROJECT_ROOT / "evals")
    out_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = out_dir / "challenger_model.joblib" if output_dir else CHALLENGER_ARTIFACT_PATH
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "version": MODEL_VERSION, "threshold": 0.5,
                 "trained_at": result["trained_at"], "seed": SEED,
                 "test_metrics": challenger_metrics}, artifact_path)

    if tracking_uri and _tracking_server_reachable(tracking_uri):
        result["mlflow_logged"] = _log_to_mlflow(
            tracking_uri, XGB_PARAMS, champion_metrics, challenger_metrics,
            {"promoted": promoted, "reason": reason}, artifact_path)
    elif tracking_uri:
        print(f"MLflow at {tracking_uri} not reachable; using local logging only")

    metrics_path = out_dir / "challenger_metrics.json"
    serializable = json.loads(json.dumps(result, default=str))
    metrics_path.write_text(json.dumps(serializable, indent=2))
    print(f"challenger_auc={challenger_metrics['auc']:.4f} "
          f"champion_auc={champion_metrics['auc'] if champion_metrics else float('nan'):.4f} "
          f"promoted={promoted} ({reason})")
    print(f"Saved metrics -> {metrics_path}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the XGBoost challenger.")
    parser.add_argument("--data", default=None, help="CSV of labeled transactions")
    parser.add_argument("--n-samples", type=int, default=20_000,
                        help="Synthetic samples when --data is not given")
    parser.add_argument("--tracking-uri", default=DEFAULT_TRACKING_URI,
                        help="MLflow tracking URI (empty string skips MLflow)")
    args = parser.parse_args()
    started = time.time()
    train_challenger(
        data_path=args.data,
        n_samples=args.n_samples,
        tracking_uri=args.tracking_uri or None,
    )
    print(f"Done in {time.time() - started:.1f}s")


if __name__ == "__main__":
    main()
