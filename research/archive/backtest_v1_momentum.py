#!/usr/bin/env python3
"""
GTAA Backtest Runner
Runs the full multi-agent backtest and outputs results.
"""

import json
import logging
import sys
import time
from pathlib import Path

import pandas as pd
import numpy as np

from config.settings import GTAAConfig
from engine.backtester import BacktestEngine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("gtaa")

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


def print_stats(stats: dict, benchmark_stats: dict):
    """Print comparison table."""
    print("\n" + "=" * 65)
    print("  GTAA MULTI-AGENT BACKTEST RESULTS")
    print("=" * 65)
    print(f"  {'Metric':<30} {'GTAA':>12} {'SPY (B&H)':>12}")
    print("-" * 65)

    rows = [
        ("CAGR", "cagr", True),
        ("Annual Volatility", "annual_vol", True),
        ("Sharpe Ratio", "sharpe", False),
        ("Sortino Ratio", "sortino", False),
        ("Max Drawdown", "max_drawdown", True),
        ("Calmar Ratio", "calmar", False),
        ("Monthly Win Rate", "monthly_win_rate", True),
        ("Total Return", "total_return", True),
    ]

    for label, key, is_pct in rows:
        g = stats.get(key, 0)
        b = benchmark_stats.get(key, 0)
        if is_pct:
            print(f"  {label:<30} {g:>11.2%} {b:>11.2%}")
        else:
            print(f"  {label:<30} {g:>11.2f} {b:>11.2f}")

    print(f"  {'Total Trades':<30} {stats.get('total_trades', 0):>11} {'1':>12}")
    print(f"  {'Years':<30} {stats.get('years', 0):>11.1f} {stats.get('years', 0):>11.1f}")
    print("=" * 65)


def compute_benchmark_stats(benchmark_curve: pd.Series) -> dict:
    """Compute stats for the benchmark buy-and-hold."""
    returns = benchmark_curve.pct_change().dropna()
    years = (benchmark_curve.index[-1] - benchmark_curve.index[0]).days / 365.25

    cagr = (benchmark_curve.iloc[-1] / benchmark_curve.iloc[0]) ** (1 / years) - 1 if years > 0 else 0
    vol = returns.std() * np.sqrt(252)
    sharpe = (returns.mean() * 252) / vol if vol > 0 else 0
    downside = returns[returns < 0]
    downside_vol = downside.std() * np.sqrt(252) if len(downside) > 0 else 0.001
    sortino = (returns.mean() * 252) / downside_vol

    cummax = benchmark_curve.cummax()
    dd = (benchmark_curve - cummax) / cummax
    max_dd = dd.min()
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0

    monthly = returns.resample("ME").sum()
    win_rate = (monthly > 0).sum() / max(len(monthly), 1)

    return {
        "cagr": cagr,
        "annual_vol": vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_dd,
        "calmar": calmar,
        "monthly_win_rate": win_rate,
        "total_return": benchmark_curve.iloc[-1] / benchmark_curve.iloc[0] - 1,
    }


def main():
    print("\n🏗️  GTAA Multi-Agent System — Backtest Starting\n")

    # Use tuned production config
    from config.production import get_production_config
    config = get_production_config()

    # Allow CLI overrides
    if "--start" in sys.argv:
        idx = sys.argv.index("--start")
        config.backtest.start_date = sys.argv[idx + 1]
    if "--end" in sys.argv:
        idx = sys.argv.index("--end")
        config.backtest.end_date = sys.argv[idx + 1]
    if "--weekly" in sys.argv:
        config.allocation.rebalance_frequency = "weekly"

    print(f"  Period: {config.backtest.start_date} → {config.backtest.end_date}")
    print(f"  Capital: ${config.backtest.initial_capital:,.0f}")
    print(f"  Rebalance: {config.allocation.rebalance_frequency}")
    print(f"  Top N assets: {config.allocation.top_n_assets}")
    print(f"  Vol target: {config.risk.vol_target:.0%}")
    print(f"  Slippage: {config.costs.slippage_bps:.0f} bps")
    print()

    # Run backtest
    engine = BacktestEngine(config)
    start_time = time.time()

    def progress(pct):
        bar = "█" * int(pct * 30) + "░" * (30 - int(pct * 30))
        print(f"\r  [{bar}] {pct:.0%}", end="", flush=True)

    result = engine.run(progress_callback=progress)
    elapsed = time.time() - start_time
    print(f"\r  Backtest complete in {elapsed:.1f}s" + " " * 40)

    # Benchmark stats
    benchmark_stats = compute_benchmark_stats(result.benchmark_curve)

    # Print results
    print_stats(result.stats, benchmark_stats)

    # Save results
    result.equity_curve.to_csv(RESULTS_DIR / "equity_curve.csv")
    result.benchmark_curve.to_csv(RESULTS_DIR / "benchmark_curve.csv")

    with open(RESULTS_DIR / "stats.json", "w") as f:
        json.dump({"gtaa": result.stats, "benchmark": benchmark_stats}, f, indent=2)

    with open(RESULTS_DIR / "weights_history.json", "w") as f:
        json.dump(result.weights_history, f, indent=2, default=str)

    with open(RESULTS_DIR / "trade_log.json", "w") as f:
        json.dump(result.trade_log, f, indent=2, default=str)

    with open(RESULTS_DIR / "config.json", "w") as f:
        json.dump(result.config, f, indent=2)

    print(f"\n  Results saved to {RESULTS_DIR}/")
    print()

    return result


if __name__ == "__main__":
    main()
