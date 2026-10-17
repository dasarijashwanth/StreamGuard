"""Day 6 evaluation CLI: holdout metrics + latency, results to evals/.

Run:  python scripts/evaluate.py [--n 5000] [--seed 123] [--skip-benchmark]

The holdout uses seed 123, deliberately different from the training seed
(42), so it is data the champion never saw during training. Writes:
  - evals/day6_results.json (machine-readable results)
  - evals/evaluation.md (human-readable results table)
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

# Allow `python scripts/evaluate.py` from the project root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from fastapi.testclient import TestClient

from app.evaluate import benchmark_api_latency, benchmark_score_latency, evaluate_holdout
from app.main import app
from app.model import load_bundle, transaction_to_features
from app.schemas import Transaction
from app.synthetic import generate_transactions

EVALS_DIR = Path(__file__).parent.parent / "evals"
HOLDOUT_SEED = 123
HOLDOUT_N = 5_000
HOLDOUT_FRAUD_RATE = 0.03


def _results_markdown(results: dict) -> str:
    """Human-readable results table for evals/evaluation.md."""
    pr = results["evaluation"]["precision_recall"]
    cal = results["evaluation"]["calibration"]
    lat = results["latency"]
    lines = [
        "# StreamGuard evaluation (Day 6)",
        "",
        f"Champion model version: {results['model_version']} · "
        f"holdout n={results['evaluation']['n']} (seed {results['holdout_seed']}, "
        f"fraud rate {results['evaluation']['fraud_rate']:.3f}) · "
        f"evaluated {results['evaluated_at']}",
        "",
        "## Holdout metrics",
        "",
        f"- AUC: **{results['evaluation']['auc']:.4f}**",
        "",
        "| Threshold | Precision | Recall |",
        "|---:|---:|---:|",
    ]
    for t in sorted(pr, key=float):
        lines.append(f"| {t} | {pr[t]['precision']:.4f} | {pr[t]['recall']:.4f} |")
    lines += [
        "",
        "## Calibration",
        "",
        f"Expected calibration error (ECE): **{cal['ece']:.4f}**",
        "",
        "| Predicted bin | Count | Mean predicted | Observed fraud rate |",
        "|---|---:|---:|---:|",
    ]
    for b in cal["bins"]:
        lo, hi = b["bin"]
        mp = f"{b['mean_predicted']:.4f}" if b["mean_predicted"] is not None else "-"
        obs = f"{b['observed_fraud_rate']:.4f}" if b["observed_fraud_rate"] is not None else "-"
        lines.append(f"| [{lo:.1f}, {hi:.1f}] | {b['count']} | {mp} | {obs} |")
    latency_lines = ["", "## Latency", ""]
    if lat is None:
        latency_lines.append("Latency benchmarks were skipped for this run (`--skip-benchmark`).")
    else:
        latency_lines += [
            "| Benchmark | Requests | p50 (ms) | p99 (ms) | Mean (ms) | Req/sec |",
            "|---|---:|---:|---:|---:|---:|",
            f"| In-process `score_transaction` | {lat['in_process']['n_requests']} | "
            f"{lat['in_process']['p50_latency_ms']:.3f} | {lat['in_process']['p99_latency_ms']:.3f} | "
            f"{lat['in_process']['mean_latency_ms']:.3f} | {lat['in_process']['requests_per_sec']:.1f} |",
            f"| HTTP `POST /score` (TestClient) | {lat['api']['n_requests']} | "
            f"{lat['api']['p50_latency_ms']:.3f} | {lat['api']['p99_latency_ms']:.3f} | "
            f"{lat['api']['mean_latency_ms']:.3f} | {lat['api']['requests_per_sec']:.1f} |",
        ]
    lines += latency_lines + [
        "",
        "## Regression gate",
        "",
        f"`tests/test_regression_gate.py` fails the suite if a future model swap "
        f"scores AUC below {results['regression_gate']['min_auc']:.4f} on a fresh "
        f"holdout (committed AUC {results['evaluation']['auc']:.4f} minus "
        f"tolerance {results['regression_gate']['auc_tolerance']}), or precision at "
        f"threshold 0.5 below {results['regression_gate']['min_precision_at_0.5']:.4f}.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the champion on a fresh holdout.")
    parser.add_argument("--n", type=int, default=HOLDOUT_N, help="holdout size (default 5000)")
    parser.add_argument("--seed", type=int, default=HOLDOUT_SEED, help="holdout seed (default 123)")
    parser.add_argument("--skip-benchmark", action="store_true", help="skip latency benchmarks")
    args = parser.parse_args()

    bundle = load_bundle()
    rows, y = generate_transactions(args.n, HOLDOUT_FRAUD_RATE, args.seed)
    X = np.vstack([transaction_to_features(Transaction(**r)) for r in rows])
    evaluation = evaluate_holdout(bundle["model"], X, y)

    latency = None
    if not args.skip_benchmark:
        txns = [Transaction(**r) for r in rows[:50]]
        in_process = benchmark_score_latency(txns, n_requests=1000)
        with TestClient(app) as client:
            api = benchmark_api_latency(client, [t.model_dump() for t in txns], n_requests=1000)
        latency = {"in_process": in_process, "api": api}

    min_auc = round(min(0.99, evaluation["auc"] - 0.01), 4)
    min_precision = round(max(0.90, evaluation["precision_recall"]["0.5"]["precision"] - 0.05), 4)
    results = {
        "model_version": bundle["version"],
        "holdout_seed": args.seed,
        "holdout_n": args.n,
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "evaluation": evaluation,
        "latency": latency,
        "regression_gate": {
            "min_auc": min_auc,
            "auc_tolerance": 0.01,
            "min_precision_at_0.5": min_precision,
            "precision_tolerance": 0.05,
        },
    }

    EVALS_DIR.mkdir(parents=True, exist_ok=True)
    (EVALS_DIR / "day6_results.json").write_text(json.dumps(results, indent=2))
    (EVALS_DIR / "evaluation.md").write_text(_results_markdown(results))
    print(f"AUC={evaluation['auc']:.4f} "
          f"P@0.5={evaluation['precision_recall']['0.5']['precision']:.4f} "
          f"R@0.5={evaluation['precision_recall']['0.5']['recall']:.4f} "
          f"ECE={evaluation['calibration']['ece']:.4f}")
    if latency:
        print(f"in-process p50={latency['in_process']['p50_latency_ms']:.3f}ms "
              f"p99={latency['in_process']['p99_latency_ms']:.3f}ms "
              f"{latency['in_process']['requests_per_sec']:.1f} req/s")
        print(f"API p50={latency['api']['p50_latency_ms']:.3f}ms "
              f"p99={latency['api']['p99_latency_ms']:.3f}ms "
              f"{latency['api']['requests_per_sec']:.1f} req/s")
    print("Wrote evals/day6_results.json and evals/evaluation.md")


if __name__ == "__main__":
    main()
