#!/usr/bin/env python3
"""
GTAA V3 — Structural fix: regime-aware momentum selection.
RISK_ON: raw momentum (favors equities), RISK_OFF: vol-adjusted (favors bonds).
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
    engine = BacktestEngine(config)
    result = engine.run()
    bench_stats = compute_benchmark_stats(result.benchmark_curve)

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
    print("\n🔬 GTAA V3 — Regime-Aware Momentum (Structural Fix)\n")

    variants = {}

    # ─── V3a: Raw momentum in RISK_ON, no risk parity, higher vol target ───
    print("  [1/4] V3a: Raw momentum + 20% vol target...")
    c = GTAAConfig()
    c.risk.vol_target = 0.20
    c.allocation.risk_parity_blend = 0.0
    c.allocation.top_n_assets = 6
    c.allocation.regime_tilt_strength = 0.40
    c.risk.min_cash_pct = 0.01
    c.risk.max_position_pct = 0.30
    c.momentum.skip_recent_days = 3
    variants["v3a_raw_mom"] = run_variant("v3a_raw_mom", c)

    # ─── V3b: Same but more concentrated (top 4) ───
    print("  [2/4] V3b: Top 4, concentrated...")
    c = GTAAConfig()
    c.risk.vol_target = 0.22
    c.allocation.risk_parity_blend = 0.0
    c.allocation.top_n_assets = 4
    c.allocation.regime_tilt_strength = 0.45
    c.risk.min_cash_pct = 0.01
    c.risk.max_position_pct = 0.40
    c.momentum.skip_recent_days = 3
    variants["v3b_concentrated"] = run_variant("v3b_concentrated", c)

    # ─── V3c: Top 6, slight risk parity for diversification ───
    print("  [3/4] V3c: Top 6, 15% risk parity blend...")
    c = GTAAConfig()
    c.risk.vol_target = 0.20
    c.allocation.risk_parity_blend = 0.15
    c.allocation.top_n_assets = 6
    c.allocation.regime_tilt_strength = 0.40
    c.risk.min_cash_pct = 0.01
    c.risk.max_position_pct = 0.30
    c.momentum.skip_recent_days = 3
    variants["v3c_blend"] = run_variant("v3c_blend", c)

    # ─── V3d: Aggressive with crypto, top 5 ───
    print("  [4/4] V3d: Aggressive with higher max position (45%)...")
    c = GTAAConfig()
    c.risk.vol_target = 0.25
    c.allocation.risk_parity_blend = 0.0
    c.allocation.top_n_assets = 5
    c.allocation.regime_tilt_strength = 0.50
    c.risk.min_cash_pct = 0.01
    c.risk.max_position_pct = 0.45
    c.risk.max_asset_class_pct = 0.70
    c.momentum.skip_recent_days = 2
    variants["v3d_aggressive"] = run_variant("v3d_aggressive", c)

    # ─── Print comparison ───
    print("\n" + "=" * 95)
    print("  V3 VARIANT COMPARISON (Regime-Aware Momentum)")
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

    # Find best
    best_sharpe = max(variants.items(), key=lambda x: x[1]["gtaa"].get("sharpe", 0))
    best_cagr = max(variants.items(), key=lambda x: x[1]["gtaa"].get("cagr", 0))
    print(f"\n  Best Sharpe: {best_sharpe[0]} ({best_sharpe[1]['gtaa']['sharpe']:.2f})")
    print(f"  Best CAGR:   {best_cagr[0]} ({best_cagr[1]['gtaa']['cagr']:.1%})")

    # Save best CAGR variant as main
    best = best_cagr[1]
    best["equity_curve"].to_csv(RESULTS_DIR / "equity_curve.csv")
    best["benchmark_curve"].to_csv(RESULTS_DIR / "benchmark_curve.csv")
    with open(RESULTS_DIR / "stats.json", "w") as f:
        json.dump({"gtaa": best["gtaa"], "benchmark": best["benchmark"]}, f, indent=2)
    with open(RESULTS_DIR / "weights_history.json", "w") as f:
        json.dump(best["weights_history"], f, indent=2, default=str)
    with open(RESULTS_DIR / "config.json", "w") as f:
        json.dump({"note": f"Best CAGR variant: {best_cagr[0]}"}, f, indent=2)

    # Diagnose equity exposure
    print("\n  === EQUITY EXPOSURE DIAGNOSIS (Best CAGR variant) ===")
    class_map = {
        'SPY': 'equity', 'QQQ': 'equity', 'IWM': 'equity', 'MDY': 'equity',
        'EFA': 'equity', 'EEM': 'equity', 'VGK': 'equity', 'EWJ': 'equity',
        'BTC-USD': 'crypto', 'ETH-USD': 'crypto',
    }
    wh = best["weights_history"]
    eq_exp = [sum(w for t, w in e['weights'].items() if class_map.get(t) == 'equity') for e in wh]
    crypto_exp = [sum(w for t, w in e['weights'].items() if class_map.get(t) == 'crypto') for e in wh]
    risk_on = [e for e in wh if e.get('regime') == 'RISK_ON']
    eq_risk_on = [sum(w for t, w in e['weights'].items() if class_map.get(t) == 'equity') for e in risk_on]

    import numpy as np
    print(f"  Overall avg equity:     {np.mean(eq_exp):.1%}")
    print(f"  RISK_ON avg equity:     {np.mean(eq_risk_on):.1%}" if eq_risk_on else "  No RISK_ON periods")
    print(f"  Avg crypto exposure:    {np.mean(crypto_exp):.1%}")
    print()


if __name__ == "__main__":
    main()
