"""StreamGuard live dashboard (Day 5).

Polls the FastAPI service (GET /health, /metrics, /drift, /models) and
shows fraud rate, latency p50/p99, drift status, the score distribution,
and a table of recently flagged transactions. Panels refresh every few
seconds; if the API is down, panels render as empty instead of crashing.

Run locally (API must be up first):
    streamlit run dashboard/app.py
Or with Docker (one command for the whole stack):
    docker compose up -d
Then open http://localhost:8501.
"""

import os
import sys
from pathlib import Path

import streamlit as st

# Make `dashboard.data` importable whether Streamlit is launched from the
# project root or from inside the dashboard/ directory.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dashboard.data import (  # noqa: E402
    DEFAULT_API_BASE_URL,
    bin_scores,
    drift_summary,
    fetch_all,
    flagged_scores,
)

st.set_page_config(page_title="StreamGuard", layout="wide")
st.title("StreamGuard live dashboard")

with st.sidebar:
    st.header("Settings")
    api_base_url = st.text_input("API base URL", value=os.getenv("API_BASE_URL", DEFAULT_API_BASE_URL))
    refresh_seconds = st.slider("Refresh every (seconds)", min_value=2, max_value=30, value=5)
    st.caption("The API must be running. Start it with `docker compose up -d` or `uvicorn app.main:app`.")


@st.fragment(run_every=refresh_seconds)
def render_panels() -> None:
    """Fetch all endpoints and render every panel. Re-runs on the timer."""
    data = fetch_all(api_base_url)
    health = data.get("health") or {}
    metrics = data.get("metrics") or {}
    drift = drift_summary(data.get("drift"))
    models = data.get("models") or {}

    if not health and not metrics:
        st.warning("API unreachable. Start it, then wait for the next refresh.")
        return

    model_version = health.get("model_version", "unknown")
    st.subheader(f"Serving model: {model_version}")

    total = int(metrics.get("total_scored", 0))
    fraud_rate = float(metrics.get("fraud_rate", 0.0))
    p50 = metrics.get("p50_latency_ms")
    p99 = metrics.get("p99_latency_ms")

    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
    kpi1.metric("Scored", f"{total}")
    kpi2.metric("Fraud rate", f"{fraud_rate:.1%}")
    kpi3.metric("Latency p50", f"{p50:.2f} ms" if p50 is not None else "no data")
    kpi4.metric("Latency p99", f"{p99:.2f} ms" if p99 is not None else "no data")
    kpi5.metric("Drift", drift["status"])

    col_left, col_right = st.columns(2)
    with col_left:
        st.subheader("Drift status")
        st.write(f"Samples seen: {drift['samples_seen']}")
        st.write(f"Error rate: {drift['error_rate']:.1%}")
        st.write(f"Drift events: {drift['drift_events']}")
        st.write(f"Retrain requested: {'yes' if drift['retrain_requested'] else 'no'}")
    with col_right:
        st.subheader("Models")
        champion = models.get("champion") or {}
        challenger = models.get("challenger") or {}
        st.write(f"Champion: {champion.get('version', 'none')} ({champion.get('status', '-')})")
        st.write(f"Challenger: {challenger.get('version', 'none')} ({challenger.get('status', '-')})")
        st.write(f"Registry version: {models.get('registry_version', 'unknown')}")

    recent = metrics.get("recent", [])
    st.subheader("Score distribution (recent scores)")
    st.bar_chart(bin_scores(recent), x="bin", y="count")

    st.subheader("Recently flagged transactions")
    flagged = flagged_scores(recent)
    if flagged:
        st.table(
            [
                {
                    "transaction_id": s.get("transaction_id") or "-",
                    "fraud_probability": s.get("fraud_probability"),
                    "latency_ms": s.get("latency_ms"),
                    "threshold": s.get("threshold"),
                }
                for s in flagged
            ]
        )
    else:
        st.write("No flagged transactions in the recent window.")


render_panels()
