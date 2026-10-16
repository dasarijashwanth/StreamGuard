"""StreamGuard FastAPI service: real-time fraud scoring."""

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from app.model import load_bundle, score_transaction
from app.online import get_online_learner
from app.schemas import DriftStatus, FeedbackRequest, ScoreResponse, Transaction


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model artifact once at startup so /score stays fast."""
    load_bundle()
    yield


app = FastAPI(
    title="StreamGuard",
    description="Real-time fraud-detection scoring API.",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict:
    """Liveness probe: service status plus loaded model version."""
    try:
        bundle = load_bundle()
        return {
            "status": "ok",
            "model_loaded": True,
            "model_version": bundle["version"],
        }
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc))


@app.post("/score", response_model=ScoreResponse)
def score(txn: Transaction) -> ScoreResponse:
    """Score a single transaction for fraud.

    Returns fraud probability, binary decision at the model threshold,
    model version, and server-side scoring latency.
    """
    probability, is_fraud, version, threshold, latency_ms = score_transaction(txn)
    return ScoreResponse(
        transaction_id=txn.transaction_id,
        fraud_probability=round(probability, 4),
        is_fraud=is_fraud,
        model_version=version,
        threshold=threshold,
        latency_ms=round(latency_ms, 3),
    )


@app.post("/feedback")
def feedback(req: FeedbackRequest) -> dict:
    """Feed one labeled transaction to the online learner (Day 3).

    Updates the River shadow model, checks ADWIN drift, and triggers a
    retrain request + registry version bump when drift is confirmed.
    """
    return get_online_learner().learn(req.transaction, req.label)


@app.get("/drift", response_model=DriftStatus)
def drift_status() -> DriftStatus:
    """Current drift detection + model registry snapshot."""
    status = get_online_learner().status()
    return DriftStatus(
        samples_seen=status["samples_seen"],
        errors=status["errors"],
        error_rate=status["error_rate"],
        drift_events=status["drift_events"],
        drift_detected=status["drift_detected"],
        retrain_requested=status["retrain_requested"],
        model_version=str(status["model_version"]),
    )
