# StreamGuard evaluation (Day 6)

Champion model version: 0.1.0 · holdout n=5000 (seed 123, fraud rate 0.030) · evaluated 2026-10-17T22:12:07.398199+00:00

## Holdout metrics

- AUC: **0.9990**

| Threshold | Precision | Recall |
|---:|---:|---:|
| 0.3 | 0.9662 | 0.9533 |
| 0.5 | 0.9927 | 0.9067 |
| 0.7 | 1.0000 | 0.8600 |
| 0.9 | 1.0000 | 0.7800 |

## Calibration

Expected calibration error (ECE): **0.0037**

| Predicted bin | Count | Mean predicted | Observed fraud rate |
|---|---:|---:|---:|
| [0.0, 0.1] | 4824 | 0.0022 | 0.0006 |
| [0.1, 0.2] | 19 | 0.1480 | 0.0526 |
| [0.2, 0.3] | 9 | 0.2441 | 0.3333 |
| [0.3, 0.4] | 5 | 0.3457 | 0.6000 |
| [0.4, 0.5] | 6 | 0.4391 | 0.6667 |
| [0.5, 0.6] | 3 | 0.5516 | 1.0000 |
| [0.6, 0.7] | 5 | 0.6397 | 0.8000 |
| [0.7, 0.8] | 4 | 0.7264 | 1.0000 |
| [0.8, 0.9] | 8 | 0.8654 | 1.0000 |
| [0.9, 1.0] | 117 | 0.9878 | 1.0000 |

## Latency

| Benchmark | Requests | p50 (ms) | p99 (ms) | Mean (ms) | Req/sec |
|---|---:|---:|---:|---:|---:|
| In-process `score_transaction` | 1000 | 0.103 | 0.143 | 0.107 | 9286.4 |
| HTTP `POST /score` (TestClient) | 1000 | 1.128 | 2.772 | 1.287 | 776.8 |

## Regression gate

`tests/test_regression_gate.py` fails the suite if a future model swap scores AUC below 0.9890 on a fresh holdout (committed AUC 0.9990 minus tolerance 0.01), or precision at threshold 0.5 below 0.9427.
