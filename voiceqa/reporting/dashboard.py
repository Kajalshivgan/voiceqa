"""Streamlit dashboard: results by category, latency distribution, and
failure clustering. Run with:

    streamlit run voiceqa/reporting/dashboard.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from voiceqa import config


def load_results(results_path: Path) -> pd.DataFrame:
    data = json.loads(results_path.read_text(encoding="utf-8"))
    return pd.DataFrame(data)


def load_summary(summary_path: Path) -> dict:
    if summary_path.exists():
        return json.loads(summary_path.read_text(encoding="utf-8"))
    return {}


def render_dashboard(df: pd.DataFrame, summary: dict) -> None:
    st.title("VoiceQA Dashboard")
    st.caption("LLM-driven voice assistant testing — results by category, latency, and failures.")

    if df.empty:
        st.warning("No results found. Run `python run_pipeline.py` first.")
        return

    col1, col2, col3 = st.columns(3)
    col1.metric("Overall pass rate", f"{df['pass'].mean():.0%}")
    col2.metric("Total scenarios", len(df))
    if "latency_ms" in df.columns and df["latency_ms"].notna().any():
        col3.metric("Median latency", f"{df['latency_ms'].median():.0f} ms")

    if summary.get("nl_summary"):
        st.info(summary["nl_summary"])

    st.subheader("Pass rate by category")
    st.bar_chart(df.groupby("category")["pass"].mean())

    st.subheader("Latency distribution")
    if "latency_ms" in df.columns and df["latency_ms"].notna().any():
        st.bar_chart(df["latency_ms"])
    else:
        st.caption("No latency data available.")

    st.subheader("Score breakdown by category")
    score_cols = [c for c in ("intent_correctness", "ui_state_correctness", "response_naturalness") if c in df.columns]
    if score_cols:
        st.dataframe(df.groupby("category")[score_cols].mean().round(2))

    st.subheader("Failures")
    failures = df[~df["pass"]]
    if failures.empty:
        st.success("No failures.")
    else:
        display_cols = [c for c in ("command", "category", "expected_intent", "transcript", "reasoning") if c in df.columns]
        st.dataframe(failures[display_cols])

    with st.expander("All results"):
        st.dataframe(df)


def main() -> None:
    st.set_page_config(page_title="VoiceQA Dashboard", layout="wide")

    default_results = config.RESULTS_DIR / "results.json"
    default_summary = config.RESULTS_DIR / "summary.json"

    st.sidebar.header("Results source")
    results_path_str = st.sidebar.text_input("Results JSON path", str(default_results))
    results_path = Path(results_path_str)
    summary_path = results_path.parent / "summary.json"

    if not results_path.exists():
        st.title("VoiceQA Dashboard")
        st.warning(
            f"No results file found at `{results_path}`. "
            "Run the pipeline first, e.g.:\n\n"
            "`python run_pipeline.py --scenarios voiceqa/scenarios/scenarios.json --mock`"
        )
        return

    df = load_results(results_path)
    summary = load_summary(summary_path)
    render_dashboard(df, summary)


if __name__ == "__main__":
    main()
