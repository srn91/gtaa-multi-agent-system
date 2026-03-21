#!/usr/bin/env python3
"""
GTAA V2 Backtest — Tuned parameters.
Fixes: cash drag, bond overweight, vol target too low.
"""

import json
import logging
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

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


def compute_benchmark_stats(benchmark_curve: pd.Series) -> dict:
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
        "cagr": cagr, "annual_vol": vol, "sharpe": sharpe, "sortino": sortino,
        "max_drawdown": max_dd, "calmar": calmar, "monthly_win_rate": win_rate,
        "total_return": benchmark_curve.iloc[-1] / benchmark_curve.iloc[0] - 1,
    }


def run_variant(name: str, config: GTAAConfig) -> dict:
    """Run a single backtest variant and return stats."""
    engine = BacktestEngine(config)
    result = engine.run()
    bench_stats = compute_benchmark_stats(result.benchmark_curve)

    # Save this variant
    variant_dir = RESULTS_DIR / name
    variant_dir.mkdir(exist_ok=True)
    result.equity_curve.to_csv(variant_dir / "equity_curve.csv")
    result.benchmark_curve.to_csv(variant_dir / "benchmark_curve.csv")
    with open(variant_dir / "stats.json", "w") as f:
        json.dump({"gtaa": result.stats, "benchmark": bench_stats}, f, indent=2)
    with open(variant_dir / "weights_history.json", "w") as f:
        json.dump(result.weights_history, f, indent=2, default=str)

    return {
        "name": name,
        "gtaa": result.stats,
        "benchmark": bench_stats,
        "equity_curve": result.equity_curve,
        "benchmark_curve": result.benchmark_curve,
        "weights_history": result.weights_history,
    }


def main():
    print("\n🔬 GTAA V2 — Parameter Optimization\n")
    print("Running multiple configurations to find the optimal setup.\n")

    variants = {}

    # ─── V2a: Raise vol target, reduce risk parity ───
    print("  [1/4] V2a: Higher vol target (18%), less risk parity (25%)...")
    config_a = GTAAConfig()
    config_a.risk.vol_target = 0.18
    config_a.allocation.risk_parity_blend = 0.25
    config_a.allocation.regime_tilt_strength = 0.40
    config_a.risk.min_cash_pct = 0.01
    variants["v2a_higher_vol"] = run_variant("v2a_higher_vol", config_a)

    # ─── V2b: Pure momentum (no risk parity) ───
    print("  [2/4] V2b: Pure momentum, no risk parity...")
    config_b = GTAAConfig()
    config_b.risk.vol_target = 0.18
    config_b.allocation.risk_parity_blend = 0.0  # pure momentum
    config_b.allocation.regime_tilt_strength = 0.35
    config_b.allocation.top_n_assets = 6  # more concentrated
    config_b.risk.min_cash_pct = 0.01
    variants["v2b_pure_momentum"] = run_variant("v2b_pure_momentum", config_b)

    # ─── V2c: Aggressive — higher vol, concentrated ───
    print("  [3/4] V2c: Aggressive — 22% vol target, top 5, pure momentum...")
    config_c = GTAAConfig()
    config_c.risk.vol_target = 0.22
    config_c.allocation.risk_parity_blend = 0.0
    config_c.allocation.top_n_assets = 5
    config_c.allocation.regime_tilt_strength = 0.45
    config_c.risk.min_cash_pct = 0.01
    config_c.risk.max_position_pct = 0.35
    config_c.momentum.skip_recent_days = 2
    variants["v2c_aggressive"] = run_variant("v2c_aggressive", config_c)

    # ─── V2d: Weekly rebalance, aggressive ───
    print("  [4/4] V2d: Weekly rebalance, aggressive...")
    config_d = GTAAConfig()
    config_d.risk.vol_target = 0.20
    config_d.allocation.risk_parity_blend = 0.10
    config_d.allocation.top_n_assets = 6
    config_d.allocation.rebalance_frequency = "weekly"
    config_d.allocation.regime_tilt_strength = 0.40
    config_d.risk.min_cash_pct = 0.01
    config_d.risk.max_position_pct = 0.30
    config_d.momentum.skip_recent_days = 3
    variants["v2d_weekly"] = run_variant("v2d_weekly", config_d)

    # ─── Print comparison table ───
    print("\n" + "=" * 95)
    print("  VARIANT COMPARISON")
    print("=" * 95)
    print(f"  {'Variant':<25} {'CAGR':>8} {'Sharpe':>8} {'Sortino':>8} {'MaxDD':>8} {'Calmar':>8} {'WinRate':>8}")
    print("-" * 95)

    for name, v in variants.items():
        s = v["gtaa"]
        print(
            f"  {name:<25} "
            f"{s.get('cagr', 0):>7.1%} "
            f"{s.get('sharpe', 0):>7.2f} "
            f"{s.get('sortino', 0):>7.2f} "
            f"{s.get('max_drawdown', 0):>7.1%} "
            f"{s.get('calmar', 0):>7.2f} "
            f"{s.get('monthly_win_rate', 0):>7.0%}"
        )

    # Benchmark row
    bs = list(variants.values())[0]["benchmark"]
    print(
        f"  {'SPY (benchmark)':<25} "
        f"{bs.get('cagr', 0):>7.1%} "
        f"{bs.get('sharpe', 0):>7.2f} "
        f"{bs.get('sortino', 0):>7.2f} "
        f"{bs.get('max_drawdown', 0):>7.1%} "
        f"{bs.get('calmar', 0):>7.2f} "
        f"{bs.get('monthly_win_rate', 0):>7.0%}"
    )
    print("=" * 95)

    # Find best by Sharpe
    best_sharpe = max(variants.items(), key=lambda x: x[1]["gtaa"].get("sharpe", 0))
    best_cagr = max(variants.items(), key=lambda x: x[1]["gtaa"].get("cagr", 0))

    print(f"\n  Best Sharpe: {best_sharpe[0]} ({best_sharpe[1]['gtaa']['sharpe']:.2f})")
    print(f"  Best CAGR:   {best_cagr[0]} ({best_cagr[1]['gtaa']['cagr']:.1%})")

    # Save best variant as the main results
    best = best_sharpe[1]
    best["equity_curve"].to_csv(RESULTS_DIR / "equity_curve.csv")
    best["benchmark_curve"].to_csv(RESULTS_DIR / "benchmark_curve.csv")
    with open(RESULTS_DIR / "stats.json", "w") as f:
        json.dump({"gtaa": best["gtaa"], "benchmark": best["benchmark"]}, f, indent=2)
    with open(RESULTS_DIR / "weights_history.json", "w") as f:
        json.dump(best["weights_history"], f, indent=2, default=str)

    # Save comparison
    comparison = {
        name: {"gtaa": v["gtaa"], "benchmark": v["benchmark"]}
        for name, v in variants.items()
    }
    with open(RESULTS_DIR / "variant_comparison.json", "w") as f:
        json.dump(comparison, f, indent=2)

    print(f"\n  Best variant saved to {RESULTS_DIR}/")
    print()


if __name__ == "__main__":
    main()
