"""
GTAA Interactive Dashboard — Streamlit
Run: streamlit run dashboard/app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

st.set_page_config(
    page_title="GTAA Multi-Agent Dashboard",
    page_icon="📊",
    layout="wide",
)

RESULTS_DIR = Path("results")


@st.cache_data
def load_results():
    """Load backtest results from disk."""
    data = {}

    eq_path = RESULTS_DIR / "equity_curve.csv"
    if eq_path.exists():
        eq = pd.read_csv(eq_path, index_col=0, parse_dates=True)
        data["equity"] = eq.squeeze()

    bench_path = RESULTS_DIR / "benchmark_curve.csv"
    if bench_path.exists():
        bench = pd.read_csv(bench_path, index_col=0, parse_dates=True)
        data["benchmark"] = bench.squeeze()

    stats_path = RESULTS_DIR / "stats.json"
    if stats_path.exists():
        with open(stats_path) as f:
            data["stats"] = json.load(f)

    weights_path = RESULTS_DIR / "weights_history.json"
    if weights_path.exists():
        with open(weights_path) as f:
            data["weights_history"] = json.load(f)

    config_path = RESULTS_DIR / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            data["config"] = json.load(f)

    return data


def render_header():
    st.title("📊 GTAA Multi-Agent System")
    st.caption("Global Tactical Asset Allocation — Momentum Rotation Engine")


def render_kpis(stats: dict):
    gtaa = stats.get("gtaa", {})
    bench = stats.get("benchmark", {})

    cols = st.columns(6)
    metrics = [
        ("CAGR", "cagr", True),
        ("Sharpe", "sharpe", False),
        ("Max DD", "max_drawdown", True),
        ("Sortino", "sortino", False),
        ("Calmar", "calmar", False),
        ("Win Rate", "monthly_win_rate", True),
    ]

    for col, (label, key, is_pct) in zip(cols, metrics):
        g = gtaa.get(key, 0)
        b = bench.get(key, 0)
        delta = g - b

        if is_pct:
            col.metric(label, f"{g:.1%}", f"{delta:+.1%} vs SPY")
        else:
            col.metric(label, f"{g:.2f}", f"{delta:+.2f} vs SPY")


def render_equity_chart(equity: pd.Series, benchmark: pd.Series):
    st.subheader("Equity Curve")

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=equity.index, y=equity.values,
        name="GTAA System",
        line=dict(color="#2962FF", width=2),
    ))

    fig.add_trace(go.Scatter(
        x=benchmark.index, y=benchmark.values,
        name="SPY Buy & Hold",
        line=dict(color="#FF6D00", width=1.5, dash="dot"),
    ))

    fig.update_layout(
        height=450,
        yaxis_title="Portfolio Value ($)",
        xaxis_title="",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(l=0, r=0, t=30, b=0),
        yaxis_type="log",
    )

    st.plotly_chart(fig, use_container_width=True)


def render_drawdown_chart(equity: pd.Series, benchmark: pd.Series):
    st.subheader("Drawdown")

    def dd(series):
        cummax = series.cummax()
        return (series - cummax) / cummax

    dd_gtaa = dd(equity)
    dd_bench = dd(benchmark)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=dd_gtaa.index, y=dd_gtaa.values,
        name="GTAA", fill="tozeroy",
        line=dict(color="#2962FF", width=1),
        fillcolor="rgba(41, 98, 255, 0.2)",
    ))

    fig.add_trace(go.Scatter(
        x=dd_bench.index, y=dd_bench.values,
        name="SPY", fill="tozeroy",
        line=dict(color="#FF6D00", width=1),
        fillcolor="rgba(255, 109, 0, 0.1)",
    ))

    fig.update_layout(
        height=300,
        yaxis_title="Drawdown",
        yaxis_tickformat=".0%",
        margin=dict(l=0, r=0, t=10, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )

    st.plotly_chart(fig, use_container_width=True)


def render_weights_over_time(weights_history: list):
    st.subheader("Portfolio Allocation Over Time")

    if not weights_history:
        st.info("No weight history available.")
        return

    # Build a DataFrame from weights history
    records = []
    for entry in weights_history:
        date = entry["date"]
        for ticker, weight in entry.get("weights", {}).items():
            records.append({"date": date, "ticker": ticker, "weight": weight})

    df = pd.DataFrame(records)
    if df.empty:
        return

    df["date"] = pd.to_datetime(df["date"])

    # Pivot and fill
    pivot = df.pivot_table(index="date", columns="ticker", values="weight", aggfunc="first").fillna(0)

    fig = go.Figure()
    colors = px.colors.qualitative.Set3 + px.colors.qualitative.Pastel

    for i, col in enumerate(pivot.columns):
        fig.add_trace(go.Scatter(
            x=pivot.index, y=pivot[col],
            name=col,
            stackgroup="one",
            line=dict(width=0),
            fillcolor=colors[i % len(colors)],
        ))

    fig.update_layout(
        height=400,
        yaxis_title="Weight",
        yaxis_tickformat=".0%",
        margin=dict(l=0, r=0, t=10, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, font_size=10),
    )

    st.plotly_chart(fig, use_container_width=True)


def render_regime_timeline(weights_history: list):
    st.subheader("Regime History")

    if not weights_history:
        return

    regime_data = pd.DataFrame([
        {"date": pd.to_datetime(w["date"]), "regime": w.get("regime", "UNKNOWN")}
        for w in weights_history
    ])

    # V4 5-regime color map
    color_map = {
        "PANIC": "#F44336", "HIGH_FEAR": "#FF9800", "ELEVATED": "#FFC107",
        "NORMAL": "#4CAF50", "COMPLACENT": "#2196F3",
        # V3 compat
        "RISK_ON": "#4CAF50", "RISK_OFF": "#FF9800", "CRISIS": "#F44336",
        "UNKNOWN": "#9E9E9E",
    }

    fig = go.Figure()
    for regime in regime_data["regime"].unique():
        mask = regime_data["regime"] == regime
        subset = regime_data[mask]
        if len(subset) > 0:
            fig.add_trace(go.Scatter(
                x=subset["date"], y=[regime] * len(subset),
                mode="markers",
                name=regime,
                marker=dict(color=color_map.get(regime, "#9E9E9E"), size=8),
            ))

    fig.update_layout(
        height=200,
        margin=dict(l=0, r=0, t=10, b=0),
        legend=dict(orientation="h"),
    )

    st.plotly_chart(fig, use_container_width=True)


def render_monthly_returns(equity: pd.Series):
    st.subheader("Monthly Returns Heatmap")

    returns = equity.pct_change().dropna()
    monthly = returns.resample("ME").sum()
    monthly.index = monthly.index.to_period("M")

    # Build year x month matrix
    df = pd.DataFrame({
        "year": monthly.index.year,
        "month": monthly.index.month,
        "return": monthly.values,
    })

    pivot = df.pivot_table(index="year", columns="month", values="return", aggfunc="first")
    pivot.columns = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                      "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    fig = go.Figure(data=go.Heatmap(
        z=pivot.values * 100,
        x=pivot.columns,
        y=pivot.index,
        colorscale="RdYlGn",
        zmid=0,
        text=[[f"{v:.1f}%" if not np.isnan(v) else "" for v in row] for row in pivot.values * 100],
        texttemplate="%{text}",
        textfont_size=10,
    ))

    fig.update_layout(
        height=max(200, len(pivot) * 30),
        margin=dict(l=0, r=0, t=10, b=0),
    )

    st.plotly_chart(fig, use_container_width=True)


def render_current_portfolio(weights_history: list):
    st.subheader("Current Portfolio")

    if not weights_history:
        st.info("No portfolio data.")
        return

    latest = weights_history[-1]
    weights = latest.get("weights", {})

    col1, col2 = st.columns(2)

    with col1:
        st.caption(f"As of {latest['date']} | Regime: {latest.get('regime', 'N/A')}")
        st.caption(f"Conviction: {latest.get('conviction', 0):.2f} | Turnover: {latest.get('turnover', 0):.1%}")

        # Table
        df = pd.DataFrame([
            {"Asset": t, "Weight": f"{w:.1%}"}
            for t, w in sorted(weights.items(), key=lambda x: -x[1])
            if w > 0.001
        ])
        st.dataframe(df, hide_index=True, use_container_width=True)

    with col2:
        # Pie chart
        labels = list(weights.keys())
        values = list(weights.values())

        fig = go.Figure(data=[go.Pie(
            labels=labels, values=values,
            hole=0.4,
            textinfo="label+percent",
            textfont_size=11,
        )])
        fig.update_layout(height=350, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig, use_container_width=True)


def render_config(config: dict):
    with st.expander("System Configuration"):
        st.json(config)


def main():
    render_header()

    data = load_results()

    if not data:
        st.error(
            "No backtest results found. Run the backtest first:\n\n"
            "```bash\npython backtest.py\n```"
        )
        return

    # KPIs
    if "stats" in data:
        render_kpis(data["stats"])

    st.divider()

    # Equity curve
    if "equity" in data and "benchmark" in data:
        render_equity_chart(data["equity"], data["benchmark"])
        render_drawdown_chart(data["equity"], data["benchmark"])

    # Allocation over time
    if "weights_history" in data:
        col1, col2 = st.columns(2)
        with col1:
            render_weights_over_time(data["weights_history"])
        with col2:
            render_current_portfolio(data["weights_history"])

    # Regime timeline
    if "weights_history" in data:
        render_regime_timeline(data["weights_history"])

    # Monthly returns
    if "equity" in data:
        render_monthly_returns(data["equity"])

    # Config
    if "config" in data:
        render_config(data["config"])


if __name__ == "__main__":
    main()
