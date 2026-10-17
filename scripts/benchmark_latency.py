"""Day 6 latency benchmark CLI (standalone re-run, no eval needed).

Run:  python scripts/benchmark_latency.py [--requests 1000] [--warmup 20] [--save]

Times single-score latency two ways: in-process (model only) and over HTTP
via TestClient POST /score (full request path). Prints a summary table.
With --save, merges the numbers into evals/day6_results.json under
"latency" (the file scripts/evaluate.py writes).
"""

import argparse
import json
import sys
from pathlib import Path

# Allow `python scripts/benchmark_latency.py` from the project root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient

from app.evaluate import benchmark_api_latency, benchmark_score_latency
from app.main import app
from app.schemas import Transaction
from app.synthetic import generate_transactions

EVALS_DIR = Path(__file__).parent.parent / "evals"
RESULTS_PATH = EVALS_DIR / "day6_results.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark /score latency.")
    parser.add_argument("--requests", type=int, default=1000, help="timed requests per benchmark (default 1000)")
    parser.add_argument("--warmup", type=int, default=20, help="untimed warmup requests (default 20)")
    parser.add_argument("--save", action="store_true", help="merge results into evals/day6_results.json")
    args = parser.parse_args()

    rows, _ = generate_transactions(50, 0.03, 123)
    txns = [Transaction(**r) for r in rows]

    in_process = benchmark_score_latency(txns, n_requests=args.requests, warmup=args.warmup)
    with TestClient(app) as client:
        api = benchmark_api_latency(client, [t.model_dump() for t in txns], n_requests=args.requests, warmup=args.warmup)

    print(f"in-process: n={in_process['n_requests']} p50={in_process['p50_latency_ms']:.3f}ms "
          f"p99={in_process['p99_latency_ms']:.3f}ms mean={in_process['mean_latency_ms']:.3f}ms "
          f"{in_process['requests_per_sec']:.1f} req/s")
    print(f"API:        n={api['n_requests']} p50={api['p50_latency_ms']:.3f}ms "
          f"p99={api['p99_latency_ms']:.3f}ms mean={api['mean_latency_ms']:.3f}ms "
          f"{api['requests_per_sec']:.1f} req/s")

    if args.save:
        results = json.loads(RESULTS_PATH.read_text()) if RESULTS_PATH.exists() else {}
        results["latency"] = {"in_process": in_process, "api": api}
        EVALS_DIR.mkdir(parents=True, exist_ok=True)
        RESULTS_PATH.write_text(json.dumps(results, indent=2))
        print(f"Saved latency into {RESULTS_PATH}")


if __name__ == "__main__":
    main()
