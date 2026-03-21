#!/usr/bin/env python3
"""
Monte Carlo Simulation Engine for GTAA V5

Instead of trusting ONE backtest path, we run 10,000 simulated paths
by bootstrapping historical daily returns to answer:
  - What's the realistic range of CAGR outcomes?
  - What's the probability of beating SPY?
  - What's the worst-case drawdown at 95% confidence?
  - What's the probability of losing money?

Two methods:
  1. BLOCK BOOTSTRAP: Resample blocks of 21 trading days (preserves autocorrelation)
  2. PATH SHUFFLE: Randomly reorder monthly return sequences

Block bootstrap is more realistic because markets have momentum clustering.
"""

import json
import time
import logging
from pathlib import Path

import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")

RESULTS_DIR = Path("results")


def load_backtest_returns():
    """Load daily returns from the V5 backtest equity curve."""
    eq_path = RESULTS_DIR / "equity_curve.csv"
    bench_path = RESULTS_DIR / "benchmark_curve.csv"

    if not eq_path.exists():
        raise FileNotFoundError("Run backtest_v5.py first to generate equity_curve.csv")

    eq = pd.read_csv(eq_path, index_col=0, parse_dates=True).squeeze()
    bench = pd.read_csv(bench_path, index_col=0, parse_dates=True).squeeze()

    strat_returns = eq.pct_change().dropna()
    bench_returns = bench.pct_change().dropna()

    return strat_returns, bench_returns, eq, bench


def block_bootstrap(returns: np.ndarray, n_days: int, block_size: int = 21) -> np.ndarray:
    """
    Block bootstrap: sample blocks of `block_size` consecutive days.
    Preserves short-term autocorrelation (momentum/mean-reversion structure).
    """
    n = len(returns)
    path = []
    while len(path) < n_days:
        start = np.random.randint(0, n - block_size)
        block = returns[start:start + block_size]
        path.extend(block.tolist())
    return np.array(path[:n_days])


def run_monte_carlo(
    strat_returns: pd.Series,
    bench_returns: pd.Series,
    n_simulations: int = 10000,
    n_days: int = None,
    block_size: int = 21,
    initial_capital: float = 100000.0,
) -> dict:
    """
    Run Monte Carlo simulation.

    For each simulation:
      1. Block-bootstrap strategy daily returns
      2. Block-bootstrap benchmark daily returns (same randomization)
      3. Compute equity curve, CAGR, Sharpe, max DD for each
      4. Collect distributions
    """
    if n_days is None:
        n_days = len(strat_returns)

    strat_arr = strat_returns.values
    bench_arr = bench_returns.values

    # Storage
    cagrs = []
    sharpes = []
    max_dds = []
    sortinos = []
    final_values = []
    beat_spy_count = 0
    loss_count = 0

    bench_cagrs = []

    years = n_days / 252.0

    print(f"\n  Running {n_simulations:,} Monte Carlo simulations...")
    print(f"  Block size: {block_size} days | Simulation length: {n_days} days ({years:.1f} years)")
    print()

    t0 = time.time()

    for i in range(n_simulations):
        # Block bootstrap strategy returns
        sim_strat = block_bootstrap(strat_arr, n_days, block_size)
        sim_bench = block_bootstrap(bench_arr, n_days, block_size)

        # Strategy equity curve
        strat_equity = initial_capital * np.cumprod(1 + sim_strat)
        bench_equity = initial_capital * np.cumprod(1 + sim_bench)

        # CAGR
        strat_cagr = (strat_equity[-1] / initial_capital) ** (1 / years) - 1
        bench_cagr = (bench_equity[-1] / initial_capital) ** (1 / years) - 1

        # Sharpe
        ann_ret = np.mean(sim_strat) * 252
        ann_vol = np.std(sim_strat) * np.sqrt(252)
        sharpe = ann_ret / ann_vol if ann_vol > 0 else 0

        # Sortino
        downside = sim_strat[sim_strat < 0]
        ds_vol = np.std(downside) * np.sqrt(252) if len(downside) > 0 else 0.001
        sortino = ann_ret / ds_vol

        # Max drawdown
        cummax = np.maximum.accumulate(strat_equity)
        dd = (strat_equity - cummax) / cummax
        max_dd = np.min(dd)

        # Collect
        cagrs.append(strat_cagr)
        sharpes.append(sharpe)
        max_dds.append(max_dd)
        sortinos.append(sortino)
        final_values.append(strat_equity[-1])
        bench_cagrs.append(bench_cagr)

        if strat_cagr > bench_cagr:
            beat_spy_count += 1
        if strat_equity[-1] < initial_capital:
            loss_count += 1

        # Progress
        if (i + 1) % 2000 == 0:
            pct = (i + 1) / n_simulations
            bar = "█" * int(pct * 30) + "░" * (30 - int(pct * 30))
            elapsed = time.time() - t0
            eta = elapsed / pct * (1 - pct) if pct > 0 else 0
            print(f"\r  [{bar}] {pct:.0%} ({i+1:,}/{n_simulations:,}) ETA: {eta:.0f}s", end="", flush=True)

    elapsed = time.time() - t0
    print(f"\r  Complete: {n_simulations:,} simulations in {elapsed:.1f}s" + " " * 30)

    # Convert to arrays
    cagrs = np.array(cagrs)
    sharpes = np.array(sharpes)
    max_dds = np.array(max_dds)
    sortinos = np.array(sortinos)
    final_values = np.array(final_values)
    bench_cagrs = np.array(bench_cagrs)

    # Compute percentiles
    percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]

    results = {
        "n_simulations": n_simulations,
        "n_days": n_days,
        "years": round(years, 1),
        "block_size": block_size,

        "cagr": {
            "mean": round(float(np.mean(cagrs)), 4),
            "median": round(float(np.median(cagrs)), 4),
            "std": round(float(np.std(cagrs)), 4),
            "percentiles": {str(p): round(float(np.percentile(cagrs, p)), 4) for p in percentiles},
        },
        "sharpe": {
            "mean": round(float(np.mean(sharpes)), 4),
            "median": round(float(np.median(sharpes)), 4),
            "percentiles": {str(p): round(float(np.percentile(sharpes, p)), 4) for p in percentiles},
        },
        "max_drawdown": {
            "mean": round(float(np.mean(max_dds)), 4),
            "median": round(float(np.median(max_dds)), 4),
            "percentile_5": round(float(np.percentile(max_dds, 5)), 4),  # worst 5%
            "percentile_1": round(float(np.percentile(max_dds, 1)), 4),  # worst 1%
        },
        "sortino": {
            "mean": round(float(np.mean(sortinos)), 4),
            "median": round(float(np.median(sortinos)), 4),
        },
        "final_value": {
            "mean": round(float(np.mean(final_values)), 0),
            "median": round(float(np.median(final_values)), 0),
            "percentiles": {str(p): round(float(np.percentile(final_values, p)), 0) for p in percentiles},
        },
        "probability_beat_spy": round(beat_spy_count / n_simulations, 4),
        "probability_loss": round(loss_count / n_simulations, 4),

        "benchmark_cagr": {
            "mean": round(float(np.mean(bench_cagrs)), 4),
            "median": round(float(np.median(bench_cagrs)), 4),
        },

        # Raw arrays for histogram plotting
        "_cagrs": cagrs.tolist(),
        "_sharpes": sharpes.tolist(),
        "_max_dds": max_dds.tolist(),
        "_final_values": final_values.tolist(),
    }

    return results


def print_results(results: dict):
    """Print Monte Carlo results in a clean format."""
    print("\n" + "=" * 75)
    print("  MONTE CARLO SIMULATION RESULTS")
    print(f"  {results['n_simulations']:,} simulations | {results['years']} years | Block size: {results['block_size']}d")
    print("=" * 75)

    c = results["cagr"]
    print(f"\n  📊 CAGR Distribution:")
    print(f"     Mean:   {c['mean']:.1%}    Median: {c['median']:.1%}    Std: {c['std']:.1%}")
    print(f"     1st:    {c['percentiles']['1']:.1%}")
    print(f"     5th:    {c['percentiles']['5']:.1%}    ← Realistic worst case")
    print(f"     25th:   {c['percentiles']['25']:.1%}")
    print(f"     50th:   {c['percentiles']['50']:.1%}    ← Expected outcome")
    print(f"     75th:   {c['percentiles']['75']:.1%}")
    print(f"     95th:   {c['percentiles']['95']:.1%}    ← Realistic best case")
    print(f"     99th:   {c['percentiles']['99']:.1%}")

    s = results["sharpe"]
    print(f"\n  📊 Sharpe Distribution:")
    print(f"     Mean: {s['mean']:.2f}    Median: {s['median']:.2f}")
    print(f"     5th:  {s['percentiles']['5']:.2f}    95th: {s['percentiles']['95']:.2f}")

    d = results["max_drawdown"]
    print(f"\n  📊 Max Drawdown Distribution:")
    print(f"     Mean:   {d['mean']:.1%}")
    print(f"     Median: {d['median']:.1%}")
    print(f"     5th percentile (worst 5%):  {d['percentile_5']:.1%}")
    print(f"     1st percentile (worst 1%):  {d['percentile_1']:.1%}")

    fv = results["final_value"]
    print(f"\n  💰 Final Portfolio Value ($100k start):")
    print(f"     Mean:   ${fv['mean']:,.0f}")
    print(f"     Median: ${fv['median']:,.0f}")
    print(f"     5th:    ${fv['percentiles']['5']:,.0f}    ← Worst realistic outcome")
    print(f"     95th:   ${fv['percentiles']['95']:,.0f}    ← Best realistic outcome")

    print(f"\n  🎯 Key Probabilities:")
    print(f"     Beat SPY:        {results['probability_beat_spy']:.1%}")
    print(f"     Lose money:      {results['probability_loss']:.1%}")
    print(f"     CAGR > 15%:      {sum(1 for x in results['_cagrs'] if x > 0.15) / len(results['_cagrs']):.1%}")
    print(f"     CAGR > 20%:      {sum(1 for x in results['_cagrs'] if x > 0.20) / len(results['_cagrs']):.1%}")
    print(f"     Max DD > -30%:   {sum(1 for x in results['_max_dds'] if x < -0.30) / len(results['_max_dds']):.1%}")
    print(f"     Sharpe > 1.0:    {sum(1 for x in results['_sharpes'] if x > 1.0) / len(results['_sharpes']):.1%}")

    print(f"\n  📈 Benchmark (SPY) Monte Carlo:")
    print(f"     Mean CAGR:   {results['benchmark_cagr']['mean']:.1%}")
    print(f"     Median CAGR: {results['benchmark_cagr']['median']:.1%}")

    print("\n" + "=" * 75)


def main():
    print("\n🎲 GTAA Monte Carlo Simulation Engine")
    print("   10,000 bootstrapped paths from V5 backtest returns\n")

    strat_returns, bench_returns, eq, bench = load_backtest_returns()

    print(f"  Loaded {len(strat_returns)} daily returns")
    print(f"  Strategy: {eq.iloc[-1]/eq.iloc[0]-1:.1%} total return")
    print(f"  Benchmark: {bench.iloc[-1]/bench.iloc[0]-1:.1%} total return")

    results = run_monte_carlo(
        strat_returns=strat_returns,
        bench_returns=bench_returns,
        n_simulations=10000,
        block_size=21,  # 1-month blocks preserve momentum structure
    )

    print_results(results)

    # Save results (without raw arrays for smaller file)
    save_results = {k: v for k, v in results.items() if not k.startswith("_")}
    with open(RESULTS_DIR / "monte_carlo.json", "w") as f:
        json.dump(save_results, f, indent=2)

    # Save histogram data separately
    np.savez(
        RESULTS_DIR / "monte_carlo_distributions.npz",
        cagrs=np.array(results["_cagrs"]),
        sharpes=np.array(results["_sharpes"]),
        max_dds=np.array(results["_max_dds"]),
        final_values=np.array(results["_final_values"]),
    )

    print(f"\n  Results saved to {RESULTS_DIR}/monte_carlo.json")
    print(f"  Distributions saved to {RESULTS_DIR}/monte_carlo_distributions.npz\n")


if __name__ == "__main__":
    main()
