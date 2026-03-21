#!/usr/bin/env python3
"""
GTAA V4 — ML-Enhanced Multi-Agent Backtest
Incorporates ideas from ALL class projects:
  - XGBoost 5-regime classifier (Regime-Aware Options Engine)
  - RF+KNN direction ensemble (Metal Futures project)
  - Bull ratio + ATR filter (LSTM project)
  - VaR-based position sizing (Options Engine)
  - Volatility targeting (LSTM project)
  - Walk-forward validation (Options Framework)
"""

import json
import logging
import time
from pathlib import Path

import pandas as pd
import numpy as np
import yfinance as yf

from config.settings import GTAAConfig, ASSET_UNIVERSE
from data.data_loader import DataLoader
from agents.research_agent import ResearchAgent
from agents.ml_regime_agent import MLRegimeAgent, REGIME_META, Regime5
from agents.ml_direction_agent import MLDirectionAgent
from agents.risk_agent import RiskAgent
from agents.allocation_agent import AllocationAgent
from agents.pm_agent import PMAgent
from agents.review_agent import ReviewAgent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("gtaa_v4")

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


def compute_stats(equity_curve: pd.Series) -> dict:
    returns = equity_curve.pct_change().dropna()
    years = (equity_curve.index[-1] - equity_curve.index[0]).days / 365.25
    if years <= 0 or equity_curve.iloc[0] <= 0:
        return {}
    cagr = (equity_curve.iloc[-1] / equity_curve.iloc[0]) ** (1 / years) - 1
    vol = returns.std() * np.sqrt(252)
    sharpe = (returns.mean() * 252) / vol if vol > 0 else 0
    downside = returns[returns < 0]
    downside_vol = downside.std() * np.sqrt(252) if len(downside) > 0 else 0.001
    sortino = (returns.mean() * 252) / downside_vol
    cummax = equity_curve.cummax()
    dd = (equity_curve - cummax) / cummax
    max_dd = dd.min()
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0
    monthly = returns.resample("ME").sum()
    win_rate = (monthly > 0).sum() / max(len(monthly), 1)
    return {
        "cagr": round(cagr, 4), "annual_vol": round(vol, 4),
        "sharpe": round(sharpe, 4), "sortino": round(sortino, 4),
        "max_drawdown": round(max_dd, 4), "calmar": round(calmar, 4),
        "monthly_win_rate": round(win_rate, 4),
        "total_return": round(equity_curve.iloc[-1] / equity_curve.iloc[0] - 1, 4),
        "years": round(years, 2),
    }


def compute_var_budget(equity: float, regime_int: int) -> float:
    """VaR-based risk budget per regime (from Options Engine)."""
    var_pcts = {
        Regime5.PANIC: 0.028,
        Regime5.HIGH_FEAR: 0.021,
        Regime5.ELEVATED: 0.012,
        Regime5.NORMAL: 0.015,
        Regime5.COMPLACENT: 0.018,
    }
    var_pct = var_pcts.get(regime_int, 0.015)
    var_mult = REGIME_META.get(regime_int, {}).get("var_mult", 1.0)
    return equity * var_pct * var_mult


def apply_vol_targeting(weights: dict, returns: pd.DataFrame,
                         date: pd.Timestamp, target_vol: float = 0.15) -> dict:
    """
    Volatility targeting (from LSTM project).
    Scale portfolio leverage to keep realized vol near target.
    """
    idx = returns.index.get_loc(date) if date in returns.index else len(returns) - 1
    lookback = 63
    start = max(0, idx - lookback)
    subset = returns.iloc[start:idx + 1]

    tickers = [t for t in weights if t in subset.columns and weights[t] > 0.001]
    if len(tickers) < 2:
        return weights

    w_vec = np.array([weights.get(t, 0) for t in tickers])
    cov = subset[tickers].cov() * 252
    port_var = w_vec @ cov.values @ w_vec
    port_vol = np.sqrt(max(port_var, 0))

    if port_vol > 0:
        leverage = target_vol / port_vol
        leverage = np.clip(leverage, 0.3, 2.0)  # from LSTM: cap at 2x
    else:
        leverage = 1.0

    scaled = {t: w * leverage if t in tickers else w for t, w in weights.items()}

    # Re-normalize if total > 1 (can't be more than 100% invested without leverage)
    total = sum(scaled.values())
    if total > 1.0:
        scaled = {t: w / total for t, w in scaled.items()}

    return scaled


def blend_momentum_and_ml(
    momentum_rankings: list,
    ml_predictions: dict,
    trend_flags: dict,
    regime_action: str,
    top_n: int = 6,
    ml_weight: float = 0.4,
) -> dict:
    """
    Combine momentum rankings with ML direction predictions.
    momentum_weight = (1 - ml_weight), ml_weight for ML scores.
    Only include assets where ML agrees with momentum direction.
    """
    # Build composite score per asset
    scores = {}
    for r in momentum_rankings:
        ticker = r["ticker"]
        raw_mom = r.get("raw_momentum", 0)

        # Momentum component (normalized to 0-1 range)
        mom_score = np.clip(raw_mom * 5 + 0.5, 0, 1)  # center at 0.5

        # ML direction component
        ml_pred = ml_predictions.get(ticker, {})
        ml_direction = ml_pred.get("direction", 0)
        ml_confidence = ml_pred.get("confidence", 0)

        # ML score: direction * confidence, mapped to 0-1
        ml_score = np.clip(ml_direction * ml_confidence * 0.5 + 0.5, 0, 1)

        # Composite
        composite = mom_score * (1 - ml_weight) + ml_score * ml_weight

        # Trend filter: penalize assets below trend (unless defensive regime)
        if regime_action == "AGGRESSIVE" or regime_action == "MODERATE":
            if not trend_flags.get(ticker, True):
                composite *= 0.3  # heavy penalty for below-trend assets

        # ML veto: if ML strongly disagrees with momentum, reduce score
        if raw_mom > 0 and ml_direction < 0 and ml_confidence > 0.6:
            composite *= 0.5  # ML says bearish, momentum says bullish — halve it
        elif raw_mom < 0 and ml_direction > 0 and ml_confidence > 0.6:
            composite *= 0.5  # reverse disagreement

        scores[ticker] = composite

    # Sort by composite score, take top N
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    # For NO_TRADE regimes, go to cash
    if regime_action == "NO_TRADE":
        return {"BIL": 1.0}

    # For DEFENSIVE, only take assets above certain threshold
    threshold = 0.4 if regime_action == "DEFENSIVE" else 0.3
    eligible = [(t, s) for t, s in ranked if s > threshold]

    if len(eligible) == 0:
        return {"BIL": 1.0}

    top = eligible[:top_n]

    # Rank-based weighting
    n = len(top)
    raw_weights = [(n - i) * top[i][1] for i in range(n)]  # score-weighted ranks
    total = sum(raw_weights)

    if total <= 0:
        return {"BIL": 1.0}

    return {t: w / total for (t, _), w in zip(top, raw_weights)}


def main():
    print("\n🧬 GTAA V4 — ML-Enhanced Multi-Agent Backtest")
    print("   Fusing ideas from ALL class projects\n")

    config = GTAAConfig()
    config.backtest.start_date = '2020-01-01'  # shorter period for faster ML backtest
    config.risk.vol_target = 0.20
    config.allocation.risk_parity_blend = 0.0
    config.allocation.top_n_assets = 6
    config.risk.min_cash_pct = 0.01
    config.risk.max_position_pct = 0.35
    config.risk.max_asset_class_pct = 0.65

    # Load data
    loader = DataLoader(universe=config.universe, start=config.backtest.start_date,
                         end=config.backtest.end_date)
    prices = loader.load()
    returns = loader.returns

    # Load VIX data for ML regime
    print("  Loading VIX data for ML regime classifier...")
    try:
        vix_raw = yf.download("^VIX", start=config.backtest.start_date,
                               end=config.backtest.end_date, progress=False)
        if isinstance(vix_raw.columns, pd.MultiIndex):
            vix_series = vix_raw["Close"]["^VIX"].reindex(prices.index).ffill()
        else:
            vix_series = vix_raw["Close"].reindex(prices.index).ffill()
    except Exception as e:
        logger.warning(f"VIX download failed: {e}, using synthetic proxy")
        spy_vol = prices["SPY"].pct_change().rolling(20).std() * np.sqrt(252) * 100
        vix_series = spy_vol.clip(10, 80)

    # Initialize agents
    print("  Initializing 7-agent team...")
    research = ResearchAgent(
        lookback_windows=config.momentum.lookback_windows,
        lookback_weights=config.momentum.lookback_weights,
        skip_recent=config.momentum.skip_recent_days,
    )
    ml_regime = MLRegimeAgent(retrain_every=252)
    ml_direction = MLDirectionAgent(retrain_every=252)
    risk_agent = RiskAgent(config=config.risk)
    pm = PMAgent(rebalance_threshold=config.risk.rebalance_threshold)
    review = ReviewAgent()

    # Backtest loop
    warmup = 300  # need more warmup for ML training
    valid_dates = prices.index[warmup:]
    rebalance_dates = set()
    groups = valid_dates.to_series().groupby(valid_dates.to_period("M"))
    for _, g in groups:
        if len(g) > 0:
            rebalance_dates.add(g.iloc[-1])

    equity = config.backtest.initial_capital
    current_weights: dict = {"BIL": 1.0}
    pm.set_current_weights(current_weights)

    equity_curve = []
    dates_out = []
    weights_history = []
    bench_equity = config.backtest.initial_capital

    total_days = len(valid_dates)
    print(f"  Period: {valid_dates[0].date()} to {valid_dates[-1].date()} ({total_days} days)")
    print(f"  ML retrain every: 63 days (quarterly)")
    print()

    start_time = time.time()

    for i, date in enumerate(valid_dates):
        # Benchmark
        if "SPY" in returns.columns and date in returns.index:
            br = returns.loc[date, "SPY"]
            if pd.notna(br):
                bench_equity *= (1 + br)

        # Daily portfolio return
        port_ret = 0.0
        new_values = {}
        for ticker, weight in current_weights.items():
            if ticker in returns.columns and date in returns.index:
                ret = returns.loc[date, ticker]
                if pd.notna(ret):
                    new_values[ticker] = weight * (1 + ret)
                    port_ret += weight * ret
                else:
                    new_values[ticker] = weight
            else:
                new_values[ticker] = weight

        equity *= (1 + port_ret)
        total = sum(new_values.values())
        if total > 0:
            current_weights = {t: v / total for t, v in new_values.items()}

        equity_curve.append(equity)
        dates_out.append(date)
        review.update_equity(date, equity)

        # Rebalance
        if date in rebalance_dates:
            try:
                # 1. Research Agent (momentum)
                r_sig = research.analyze(date, prices, returns, {"universe": config.universe})

                # 2. ML Regime Agent (XGBoost 5-regime)
                rg_sig = ml_regime.analyze(date, prices, returns, {"vix_series": vix_series})

                # 3. ML Direction Agent (RF+KNN per asset)
                dir_sig = ml_direction.analyze(date, prices, returns, {})

                # 4. Blend momentum + ML direction into target weights
                regime_action = rg_sig.data.get("action", "MODERATE")
                regime_int = rg_sig.data.get("regime_int", Regime5.NORMAL)

                blended_weights = blend_momentum_and_ml(
                    momentum_rankings=r_sig.data.get("rankings", []),
                    ml_predictions=dir_sig.data.get("predictions", {}),
                    trend_flags=r_sig.data.get("trend_flags", {}),
                    regime_action=regime_action,
                    top_n=config.allocation.top_n_assets,
                    ml_weight=0.35,  # 35% ML, 65% momentum
                )

                # 5. Volatility targeting (from LSTM project)
                vol_targeted = apply_vol_targeting(
                    blended_weights, returns, date,
                    target_vol=0.15,  # from LSTM: 15% target
                )

                # 6. Risk Agent review
                risk_sig = risk_agent.analyze(date, prices, returns, {
                    "proposed_weights": vol_targeted,
                    "universe": config.universe,
                    "equity_value": equity,
                    "regime": rg_sig.data.get("regime_3way", "RISK_ON"),
                })

                # 7. PM Agent final decision
                # Create a compatible allocation signal
                from agents.base_agent import Signal as _Sig
                alloc_sig = _Sig(
                    agent_name="Allocation",
                    timestamp=datetime.now(),
                    signal_type="allocation",
                    data={"target_weights": vol_targeted},
                    confidence=0.7,
                    reasoning="Blended momentum + ML direction with vol targeting",
                )

                pm_sig = pm.analyze(date, prices, returns, {
                    "research_signal_obj": r_sig,
                    "regime_signal_obj": rg_sig,
                    "allocation_signal_obj": alloc_sig,
                    "risk_signal_obj": risk_sig,
                    "equity_value": equity,
                    "universe": config.universe,
                })

                if pm_sig.data.get("execute_rebalance"):
                    new_weights = pm_sig.data["final_weights"]

                    # Transaction costs
                    all_t = set(current_weights.keys()) | set(new_weights.keys())
                    cost = 0
                    for t in all_t:
                        trade_val = abs(current_weights.get(t, 0) - new_weights.get(t, 0)) * equity
                        if trade_val > 100:
                            slippage = 15 if config.universe.get(t, {}).get("class") == "crypto" else 5
                            cost += trade_val * (slippage / 10000)
                    equity -= cost

                    turnover = pm_sig.data.get("turnover", 0)
                    review.log_trade(date, current_weights, new_weights, equity, turnover)

                    weights_history.append({
                        "date": str(date.date()),
                        "weights": new_weights,
                        "regime": rg_sig.data.get("regime", "NORMAL"),
                        "regime_action": regime_action,
                        "turnover": turnover,
                        "cost": round(cost, 2),
                        "equity": round(equity, 2),
                        "conviction": pm_sig.data.get("conviction", 0),
                        "ml_trained": rg_sig.data.get("ml_trained", False),
                        "n_ml_models": dir_sig.data.get("n_models_trained", 0),
                    })

                    current_weights = new_weights
                    pm.set_current_weights(current_weights)

            except Exception as e:
                logger.error(f"Rebalance error {date.date()}: {e}")
                import traceback
                traceback.print_exc()
                continue

        if i % 100 == 0:
            pct = i / total_days
            bar = "█" * int(pct * 30) + "░" * (30 - int(pct * 30))
            print(f"\r  [{bar}] {pct:.0%}", end="", flush=True)

    elapsed = time.time() - start_time
    print(f"\r  Complete in {elapsed:.0f}s" + " " * 40)

    # Compute results
    eq_series = pd.Series(equity_curve, index=dates_out)
    bench_series = pd.Series(
        [config.backtest.initial_capital] + [0] * (len(dates_out) - 1),
        index=dates_out,
    )
    # Rebuild benchmark properly
    bench_vals = []
    be = config.backtest.initial_capital
    for d in dates_out:
        if "SPY" in returns.columns and d in returns.index:
            br = returns.loc[d, "SPY"]
            if pd.notna(br):
                be *= (1 + br)
        bench_vals.append(be)
    bench_series = pd.Series(bench_vals, index=dates_out)

    gtaa_stats = compute_stats(eq_series)
    bench_stats = compute_stats(bench_series)

    # Print results
    print("\n" + "=" * 75)
    print("  GTAA V4 — ML-ENHANCED RESULTS")
    print("=" * 75)
    print(f"  {'Metric':<30} {'GTAA V4':>12} {'SPY':>12} {'Delta':>12}")
    print("-" * 75)

    for label, key, is_pct in [
        ("CAGR", "cagr", True),
        ("Annual Vol", "annual_vol", True),
        ("Sharpe", "sharpe", False),
        ("Sortino", "sortino", False),
        ("Max Drawdown", "max_drawdown", True),
        ("Calmar", "calmar", False),
        ("Win Rate", "monthly_win_rate", True),
        ("Total Return", "total_return", True),
    ]:
        g = gtaa_stats.get(key, 0)
        b = bench_stats.get(key, 0)
        d = g - b
        if is_pct:
            print(f"  {label:<30} {g:>11.1%} {b:>11.1%} {d:>+11.1%}")
        else:
            print(f"  {label:<30} {g:>11.2f} {b:>11.2f} {d:>+11.2f}")

    print(f"  {'Trades':<30} {len(weights_history):>11}")
    print("=" * 75)

    # Diagnose
    if weights_history:
        class_map = {
            'SPY': 'eq', 'QQQ': 'eq', 'IWM': 'eq', 'MDY': 'eq',
            'EFA': 'eq', 'EEM': 'eq', 'VGK': 'eq', 'EWJ': 'eq',
            'BTC-USD': 'crypto', 'ETH-USD': 'crypto',
        }
        eq_exp = [sum(w for t, w in e['weights'].items() if class_map.get(t) == 'eq') for e in weights_history]
        crypto_exp = [sum(w for t, w in e['weights'].items() if class_map.get(t) == 'crypto') for e in weights_history]
        regimes = [e.get('regime', 'NORMAL') for e in weights_history]

        from collections import Counter
        regime_counts = Counter(regimes)

        print(f"\n  Avg equity exposure:    {np.mean(eq_exp):.1%}")
        print(f"  Avg crypto exposure:    {np.mean(crypto_exp):.1%}")
        print(f"  Regime distribution:    {dict(regime_counts)}")
        print(f"  ML models trained:      {weights_history[-1].get('n_ml_models', 0)}")

    # Save results
    eq_series.to_csv(RESULTS_DIR / "equity_curve.csv")
    bench_series.to_csv(RESULTS_DIR / "benchmark_curve.csv")
    with open(RESULTS_DIR / "stats.json", "w") as f:
        json.dump({"gtaa": gtaa_stats, "benchmark": bench_stats}, f, indent=2)
    with open(RESULTS_DIR / "weights_history.json", "w") as f:
        json.dump(weights_history, f, indent=2, default=str)
    with open(RESULTS_DIR / "config.json", "w") as f:
        json.dump({"version": "V4_ML_Enhanced"}, f, indent=2)

    print(f"\n  Results saved to {RESULTS_DIR}/\n")


if __name__ == "__main__":
    from datetime import datetime
    main()
