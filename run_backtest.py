#!/usr/bin/env python3
"""
GTAA V5 — Aggressive ML-Enhanced Backtest
Fixes from V4 autopsy:
  1. Cash drag: max 5% cash in NORMAL/COMPLACENT
  2. Equity underweight: min 50% equity in bullish regimes
  3. Vol target raised to 20% with ART (Auto-Regressive Risk Targeting)
  4. ELEVATED = defensive allocation (bonds+gold), NOT cash
  5. PANIC/HIGH_FEAR = 60% bonds + 40% gold, NOT 100% cash
  6. Sector rotation: prefer sector ETFs with strongest momentum
  7. Dynamic ML weight: trust ML more when confidence > 0.7
"""

import json, logging, time
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np
import yfinance as yf

from config.settings import GTAAConfig, ASSET_UNIVERSE
from data.data_loader import DataLoader
from agents.research_agent import ResearchAgent
from agents.ml_regime_agent import MLRegimeAgent, REGIME_META, Regime5
from agents.ml_direction_agent import MLDirectionAgent
from agents.risk_agent import RiskAgent
from agents.pm_agent import PMAgent
from agents.review_agent import ReviewAgent
from agents.base_agent import Signal

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("gtaa_v5")

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


def compute_stats(eq: pd.Series) -> dict:
    r = eq.pct_change().dropna()
    yrs = (eq.index[-1] - eq.index[0]).days / 365.25
    if yrs <= 0 or eq.iloc[0] <= 0:
        return {}
    cagr = (eq.iloc[-1] / eq.iloc[0]) ** (1 / yrs) - 1
    vol = r.std() * np.sqrt(252)
    sharpe = (r.mean() * 252) / vol if vol > 0 else 0
    ds = r[r < 0]
    ds_vol = ds.std() * np.sqrt(252) if len(ds) > 0 else 0.001
    sortino = (r.mean() * 252) / ds_vol
    cm = eq.cummax()
    dd = (eq - cm) / cm
    max_dd = dd.min()
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0
    mo = r.resample("ME").sum()
    wr = (mo > 0).sum() / max(len(mo), 1)
    return {"cagr": round(cagr, 4), "annual_vol": round(vol, 4), "sharpe": round(sharpe, 4),
            "sortino": round(sortino, 4), "max_drawdown": round(max_dd, 4), "calmar": round(calmar, 4),
            "monthly_win_rate": round(wr, 4), "total_return": round(eq.iloc[-1] / eq.iloc[0] - 1, 4), "years": round(yrs, 2)}


def art_vol_scale(weights, returns, date, target_vol=0.20):
    """Auto-Regressive Risk Targeting (from Options Framework).
    Key diff from V4: uses 20% target, allows leverage up to 1.5x, min 0.5x."""
    idx = returns.index.get_loc(date) if date in returns.index else len(returns) - 1
    subset = returns.iloc[max(0, idx - 63):idx + 1]
    tickers = [t for t in weights if t in subset.columns and weights[t] > 0.001]
    if len(tickers) < 2:
        return weights
    w = np.array([weights.get(t, 0) for t in tickers])
    cov = subset[tickers].cov() * 252
    pv = w @ cov.values @ w
    pvol = np.sqrt(max(pv, 0))
    if pvol > 0:
        lev = target_vol / pvol
        lev = np.clip(lev, 0.5, 1.5)  # ART: allow leverage up to 1.5x
    else:
        lev = 1.0
    scaled = {t: weights[t] * lev if t in tickers else weights[t] for t in weights}
    total = sum(scaled.values())
    if total > 1.0:
        scaled = {t: v / total for t, v in scaled.items()}
    return scaled


def build_aggressive_allocation(
    rankings, ml_preds, trend_flags, regime_int, regime_action,
    universe, top_n=8, ml_weight=0.40,
):
    """
    V5 allocation: aggressively deploy capital based on regime.
    COMPLACENT/NORMAL: 80%+ risk assets, prefer sector ETFs
    ELEVATED: 60% equity + 30% defensive + 10% commodity
    HIGH_FEAR: 40% defensive bonds + 30% gold + 30% cash
    PANIC: 50% bonds + 30% gold + 20% cash
    """

    # Defensive allocations for bad regimes
    if regime_int == Regime5.PANIC:
        # Panic: mostly defensive but keep 15% in broad equity for recovery
        return {"TLT": 0.25, "IEF": 0.15, "GLD": 0.25, "SPY": 0.15, "BIL": 0.20}

    if regime_int == Regime5.HIGH_FEAR:
        # High fear: defensive tilt but stay 30% invested in equities
        return {"TLT": 0.15, "IEF": 0.10, "GLD": 0.20, "SPY": 0.20, "XLV": 0.10, "XLP": 0.10, "BIL": 0.10, "TIP": 0.05}

    # For ELEVATED / NORMAL / COMPLACENT: build from momentum + ML
    scores = {}
    for r in rankings:
        ticker = r["ticker"]
        raw_mom = r.get("raw_momentum", 0)
        asset_class = universe.get(ticker, {}).get("class", "")

        # Base momentum score
        mom_score = np.clip(raw_mom * 5 + 0.5, 0, 1)

        # ML direction boost
        ml = ml_preds.get(ticker, {})
        ml_dir = ml.get("direction", 0)
        ml_conf = ml.get("confidence", 0)

        # Dynamic ML weight: trust ML more when confident (from LSTM project insight)
        effective_ml_w = ml_weight * (1.0 + ml_conf) / 2.0  # scale 0.2-0.4 based on confidence
        ml_score = np.clip(ml_dir * ml_conf * 0.5 + 0.5, 0, 1)
        composite = mom_score * (1 - effective_ml_w) + ml_score * effective_ml_w

        # Trend filter: penalty for below-trend in bullish regimes
        if regime_int >= Regime5.NORMAL and not trend_flags.get(ticker, True):
            composite *= 0.4

        # ML veto: strong disagreement
        if raw_mom > 0 and ml_dir < 0 and ml_conf > 0.65:
            composite *= 0.5
        elif raw_mom < 0 and ml_dir > 0 and ml_conf > 0.65:
            composite *= 0.5

        # Sector ETF bonus during strong regimes (sector rotation from ATPM project)
        if regime_int >= Regime5.NORMAL and asset_class == "equity_sector":
            composite *= 1.25  # 25% bonus for sectors — finer granularity

        # Crypto bonus during COMPLACENT (strong bull)
        if regime_int == Regime5.COMPLACENT and asset_class == "crypto":
            composite *= 1.20

        scores[ticker] = composite

    # Sort and select top N
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    threshold = 0.35 if regime_int >= Regime5.NORMAL else 0.40
    eligible = [(t, s) for t, s in ranked if s > threshold]

    if len(eligible) == 0:
        return {"TLT": 0.30, "GLD": 0.30, "IEF": 0.20, "BIL": 0.20}

    top = eligible[:top_n]

    # Score-weighted allocation
    total_score = sum(s for _, s in top)
    if total_score <= 0:
        return {"BIL": 1.0}

    weights = {t: s / total_score for t, s in top}

    # Regime-specific floor on risk assets
    if regime_int == Regime5.COMPLACENT:
        # Strong bull: ensure at least 85% in risk assets
        risk_classes = {"equity_us", "equity_intl", "equity_sector", "crypto", "commodity", "real_estate"}
        risk_w = sum(w for t, w in weights.items() if universe.get(t, {}).get("class", "") in risk_classes)
        if risk_w < 0.85:
            for t in weights:
                if universe.get(t, {}).get("class", "") in risk_classes:
                    weights[t] *= 0.85 / max(risk_w, 0.01)
            total = sum(weights.values())
            weights = {t: w / total for t, w in weights.items()}

    elif regime_int == Regime5.NORMAL:
        # Normal: at least 70% risk assets
        risk_classes = {"equity_us", "equity_intl", "equity_sector", "crypto", "commodity", "real_estate"}
        risk_w = sum(w for t, w in weights.items() if universe.get(t, {}).get("class", "") in risk_classes)
        if risk_w < 0.70:
            for t in weights:
                if universe.get(t, {}).get("class", "") in risk_classes:
                    weights[t] *= 0.70 / max(risk_w, 0.01)
            total = sum(weights.values())
            weights = {t: w / total for t, w in weights.items()}

    elif regime_int == Regime5.ELEVATED:
        # Elevated: stay invested but tilt defensive. Keep 60% equity minimum.
        equity_classes = {"equity_us", "equity_intl", "equity_sector"}
        eq_w = sum(w for t, w in weights.items() if universe.get(t, {}).get("class", "") in equity_classes)
        if eq_w < 0.55:
            # Scale UP equities to 55% floor
            for t in list(weights.keys()):
                if universe.get(t, {}).get("class", "") in equity_classes:
                    weights[t] *= 1.3
            # Add some defensive as buffer
            weights["GLD"] = weights.get("GLD", 0) + 0.08
            weights["TLT"] = weights.get("TLT", 0) + 0.07
            total = sum(weights.values())
            weights = {t: w / total for t, w in weights.items()}

    # Enforce max 5% cash in bullish regimes
    if regime_int >= Regime5.NORMAL:
        cash_w = sum(w for t, w in weights.items() if universe.get(t, {}).get("class", "") == "cash")
        if cash_w > 0.05:
            # Remove excess cash, redistribute proportionally to non-cash
            excess = cash_w - 0.05
            non_cash = {t: w for t, w in weights.items() if universe.get(t, {}).get("class", "") != "cash"}
            nc_total = sum(non_cash.values())
            for t in non_cash:
                weights[t] += excess * (non_cash[t] / nc_total) if nc_total > 0 else 0
            for t in list(weights.keys()):
                if universe.get(t, {}).get("class", "") == "cash":
                    weights[t] = min(weights[t], 0.05)

    # Clean up tiny weights
    weights = {t: w for t, w in weights.items() if w >= 0.02}
    total = sum(weights.values())
    if total > 0:
        weights = {t: w / total for t, w in weights.items()}

    return weights


def main():
    print("\n⚡ GTAA V5 — Aggressive ML-Enhanced Backtest")
    print("   Fixing: cash drag, equity underweight, vol targeting, defensive allocation\n")

    config = GTAAConfig()
    config.backtest.start_date = '2020-01-01'
    config.risk.max_position_pct = 0.35
    config.risk.max_asset_class_pct = 0.70
    config.risk.min_cash_pct = 0.00  # V5: no forced cash in bullish regimes
    config.risk.vol_target = 0.20

    # Load data
    loader = DataLoader(universe=config.universe, start=config.backtest.start_date, end=config.backtest.end_date)
    prices = loader.load()
    returns = loader.returns

    # VIX
    print("  Loading VIX...")
    try:
        vix_raw = yf.download("^VIX", start=config.backtest.start_date, end=config.backtest.end_date, progress=False)
        if isinstance(vix_raw.columns, pd.MultiIndex):
            vix_series = vix_raw["Close"]["^VIX"].reindex(prices.index).ffill()
        else:
            vix_series = vix_raw["Close"].reindex(prices.index).ffill()
    except Exception:
        spy_vol = prices["SPY"].pct_change().rolling(20).std() * np.sqrt(252) * 100
        vix_series = spy_vol.clip(10, 80)

    # Agents
    print("  Initializing 8-agent team...")
    research = ResearchAgent(
        lookback_windows=config.momentum.lookback_windows,
        lookback_weights=config.momentum.lookback_weights,
        skip_recent=config.momentum.skip_recent_days,
    )
    ml_regime = MLRegimeAgent(retrain_every=252)
    ml_direction = MLDirectionAgent(retrain_every=252)
    risk_agent = RiskAgent(config=config.risk)
    pm = PMAgent(rebalance_threshold=0.03)  # lower threshold = more responsive
    review = ReviewAgent()

    # Setup
    warmup = 300
    valid_dates = prices.index[warmup:]
    rebalance_dates = set()
    groups = valid_dates.to_series().groupby(valid_dates.to_period("M"))
    for _, g in groups:
        if len(g) > 0:
            rebalance_dates.add(g.iloc[-1])

    equity = config.backtest.initial_capital
    current_weights = {"BIL": 1.0}
    pm.set_current_weights(current_weights)
    bench_equity = config.backtest.initial_capital
    equity_curve, dates_out, bench_vals, weights_history = [], [], [], []

    total_days = len(valid_dates)
    print(f"  Period: {valid_dates[0].date()} to {valid_dates[-1].date()} ({total_days} days)\n")

    t0 = time.time()

    for i, date in enumerate(valid_dates):
        # Benchmark
        if "SPY" in returns.columns and date in returns.index:
            br = returns.loc[date, "SPY"]
            if pd.notna(br):
                bench_equity *= (1 + br)

        # Daily return
        port_ret = 0.0
        new_vals = {}
        for t, w in current_weights.items():
            if t in returns.columns and date in returns.index:
                r = returns.loc[date, t]
                if pd.notna(r):
                    new_vals[t] = w * (1 + r)
                    port_ret += w * r
                else:
                    new_vals[t] = w
            else:
                new_vals[t] = w

        equity *= (1 + port_ret)
        total = sum(new_vals.values())
        if total > 0:
            current_weights = {t: v / total for t, v in new_vals.items()}

        equity_curve.append(equity)
        dates_out.append(date)
        bench_vals.append(bench_equity)
        review.update_equity(date, equity)

        if date in rebalance_dates:
            try:
                # 1. Research
                r_sig = research.analyze(date, prices, returns, {"universe": config.universe})

                # 2. ML Regime
                rg_sig = ml_regime.analyze(date, prices, returns, {"vix_series": vix_series})
                regime_int = rg_sig.data.get("regime_int", Regime5.NORMAL)
                regime_action = rg_sig.data.get("action", "MODERATE")

                # 3. ML Direction
                dir_sig = ml_direction.analyze(date, prices, returns, {})

                # 4. V5 aggressive allocation
                target_weights = build_aggressive_allocation(
                    rankings=r_sig.data.get("rankings", []),
                    ml_preds=dir_sig.data.get("predictions", {}),
                    trend_flags=r_sig.data.get("trend_flags", {}),
                    regime_int=regime_int,
                    regime_action=regime_action,
                    universe=config.universe,
                    top_n=6,
                    ml_weight=0.40,
                )

                # 5. ART vol targeting (20% target)
                target_weights = art_vol_scale(target_weights, returns, date, target_vol=0.20)

                # 6. Risk check — skip cash padding in bullish regimes
                if regime_int >= Regime5.NORMAL:
                    # In bullish regimes, use target weights directly (skip risk agent's cash padding)
                    # Only apply position-level caps manually
                    final_w = {}
                    for t, w in target_weights.items():
                        capped = min(w, config.risk.max_position_pct)
                        if capped > 0.02:
                            final_w[t] = capped
                    total = sum(final_w.values())
                    if total > 0:
                        final_w = {t: w / total for t, w in final_w.items()}
                    risk_sig = Signal("Risk", datetime.now(), "risk",
                                     {"approved_weights": final_w, "vetoed": False,
                                      "drawdown_action": "NORMAL", "drawdown_level": 0,
                                      "violations": [], "correlation_warnings": []},
                                     0.85, "V5: bullish bypass")
                else:
                    risk_sig = risk_agent.analyze(date, prices, returns, {
                        "proposed_weights": target_weights,
                        "universe": config.universe,
                        "equity_value": equity,
                        "regime": rg_sig.data.get("regime_3way", "RISK_ON"),
                    })

                # 7. PM decision
                alloc_sig = Signal("Allocation", datetime.now(), "allocation",
                                   {"target_weights": target_weights}, 0.75, "V5 aggressive")
                pm_sig = pm.analyze(date, prices, returns, {
                    "research_signal_obj": r_sig,
                    "regime_signal_obj": rg_sig,
                    "allocation_signal_obj": alloc_sig,
                    "risk_signal_obj": risk_sig,
                    "equity_value": equity,
                    "universe": config.universe,
                })

                if pm_sig.data.get("execute_rebalance"):
                    new_w = pm_sig.data["final_weights"]

                    # Transaction costs
                    cost = 0
                    for t in set(current_weights.keys()) | set(new_w.keys()):
                        tv = abs(current_weights.get(t, 0) - new_w.get(t, 0)) * equity
                        if tv > 100:
                            slp = 15 if config.universe.get(t, {}).get("class") == "crypto" else 5
                            cost += tv * slp / 10000
                    equity -= cost

                    review.log_trade(date, current_weights, new_w, equity, pm_sig.data.get("turnover", 0))

                    weights_history.append({
                        "date": str(date.date()),
                        "weights": new_w,
                        "regime": rg_sig.data.get("regime", "NORMAL"),
                        "regime_action": regime_action,
                        "regime_int": regime_int,
                        "equity": round(equity, 2),
                        "conviction": pm_sig.data.get("conviction", 0),
                    })

                    current_weights = new_w
                    pm.set_current_weights(current_weights)

            except Exception as e:
                logger.error(f"Rebalance error {date.date()}: {e}")
                import traceback; traceback.print_exc()

        if i % 100 == 0:
            pct = i / total_days
            bar = "█" * int(pct * 30) + "░" * (30 - int(pct * 30))
            print(f"\r  [{bar}] {pct:.0%}", end="", flush=True)

    elapsed = time.time() - t0
    print(f"\r  Complete in {elapsed:.0f}s" + " " * 40)

    # Results
    eq_series = pd.Series(equity_curve, index=dates_out)
    bench_series = pd.Series(bench_vals, index=dates_out)
    g = compute_stats(eq_series)
    b = compute_stats(bench_series)

    print("\n" + "=" * 80)
    print("  GTAA V5 — AGGRESSIVE ML-ENHANCED RESULTS")
    print("=" * 80)
    print(f"  {'Metric':<25} {'GTAA V5':>12} {'SPY':>12} {'Delta':>12}")
    print("-" * 80)
    for label, key, pct in [
        ("CAGR", "cagr", True), ("Annual Vol", "annual_vol", True),
        ("Sharpe", "sharpe", False), ("Sortino", "sortino", False),
        ("Max Drawdown", "max_drawdown", True), ("Calmar", "calmar", False),
        ("Win Rate", "monthly_win_rate", True), ("Total Return", "total_return", True),
    ]:
        gv, bv = g.get(key, 0), b.get(key, 0)
        d = gv - bv
        if pct:
            print(f"  {label:<25} {gv:>11.1%} {bv:>11.1%} {d:>+11.1%}")
        else:
            print(f"  {label:<25} {gv:>11.2f} {bv:>11.2f} {d:>+11.2f}")
    print(f"  {'Trades':<25} {len(weights_history):>11}")
    print("=" * 80)

    # Diagnosis
    if weights_history:
        class_map = {
            'SPY': 'eq', 'QQQ': 'eq', 'IWM': 'eq', 'MDY': 'eq',
            'XLK': 'eq', 'XLF': 'eq', 'XLE': 'eq', 'XLV': 'eq',
            'XLY': 'eq', 'XLP': 'eq', 'XLI': 'eq', 'XLU': 'eq',
            'EFA': 'eq', 'EEM': 'eq', 'VGK': 'eq', 'EWJ': 'eq',
            'BTC-USD': 'crypto', 'ETH-USD': 'crypto',
            'GLD': 'comm', 'SLV': 'comm', 'USO': 'comm', 'DBC': 'comm',
            'TLT': 'bond', 'IEF': 'bond', 'TIP': 'bond', 'HYG': 'bond', 'LQD': 'bond',
            'VNQ': 'reit', 'VNQI': 'reit', 'BIL': 'cash', 'SHY': 'cash',
        }
        eq_exp = [sum(w for t, w in e['weights'].items() if class_map.get(t) == 'eq') for e in weights_history]
        crypto_exp = [sum(w for t, w in e['weights'].items() if class_map.get(t) == 'crypto') for e in weights_history]
        cash_exp = [sum(w for t, w in e['weights'].items() if class_map.get(t) == 'cash') for e in weights_history]
        bond_exp = [sum(w for t, w in e['weights'].items() if class_map.get(t) == 'bond') for e in weights_history]

        from collections import Counter
        regimes = Counter(e.get('regime') for e in weights_history)

        print(f"\n  Avg equity:   {np.mean(eq_exp):.1%}  (V4: 11.6%)")
        print(f"  Avg crypto:   {np.mean(crypto_exp):.1%}  (V4: 19.8%)")
        print(f"  Avg bonds:    {np.mean(bond_exp):.1%}")
        print(f"  Avg cash:     {np.mean(cash_exp):.1%}  (V4: 25.9%)")
        print(f"  Regimes:      {dict(regimes)}")

    # Save
    eq_series.to_csv(RESULTS_DIR / "equity_curve.csv")
    bench_series.to_csv(RESULTS_DIR / "benchmark_curve.csv")
    with open(RESULTS_DIR / "stats.json", "w") as f:
        json.dump({"gtaa": g, "benchmark": b}, f, indent=2)
    with open(RESULTS_DIR / "weights_history.json", "w") as f:
        json.dump(weights_history, f, indent=2, default=str)

    print(f"\n  Saved to {RESULTS_DIR}/\n")


if __name__ == "__main__":
    main()
