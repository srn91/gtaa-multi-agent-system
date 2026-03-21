#!/usr/bin/env python3
"""
GTAA V6 — Concentrated Conviction Engine

Philosophy change from V5:
  V5: Diversify across 6-8 assets → dilutes winners
  V6: Concentrate into 3-4 HIGHEST CONVICTION bets → amplifies winners

Core logic:
  1. CONVICTION CONCENTRATION: Top 3-4 assets get 70-90% of capital
  2. SECTOR ROTATION > BROAD INDEX: XLK with momentum > SPY always
  3. MOMENTUM PERSISTENCE: Don't rotate out of winners until momentum dies
  4. REGIME-ADAPTIVE AGGRESSION: COMPLACENT = max risk, PANIC = defensive not cash
  5. BI-WEEKLY REBALANCE: Catch momentum shifts 2x faster
  6. MEAN REVERSION ENTRY: Detect oversold bounces (from LSTM project)
  7. DRAWDOWN RE-ENTRY: When regime flips back to NORMAL, re-enter aggressively
"""

import json, logging, time
from pathlib import Path
from datetime import datetime
from collections import Counter

import pandas as pd
import numpy as np
import yfinance as yf

from config.settings import GTAAConfig, ASSET_UNIVERSE
from data.data_loader import DataLoader
from agents.research_agent import ResearchAgent
from agents.ml_regime_agent import MLRegimeAgent, REGIME_META, Regime5
from agents.ml_direction_agent import MLDirectionAgent
from agents.review_agent import ReviewAgent
from agents.base_agent import Signal

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("gtaa_v6")

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


def compute_momentum_score(prices, ticker, idx, windows=[21, 63, 126, 252], weights=[0.35, 0.30, 0.20, 0.15]):
    """Raw momentum score — no vol adjustment, pure return strength."""
    if ticker not in prices.columns:
        return 0.0
    series = prices[ticker].iloc[:idx + 1]
    if len(series) < max(windows):
        return 0.0
    total = 0.0
    for w, wt in zip(windows, weights):
        if len(series) >= w + 5:
            ret = series.iloc[-5] / series.iloc[-w - 5] - 1  # skip last 5 days
            total += ret * wt
    return total


def detect_oversold_bounce(prices, ticker, idx):
    """Mean reversion signal: detect oversold conditions primed for bounce.
    From LSTM project: price below 20d SMA + RSI proxy < 0.35 + positive 5d momentum."""
    if ticker not in prices.columns:
        return False, 0.0
    s = prices[ticker].iloc[:idx + 1]
    if len(s) < 30:
        return False, 0.0

    sma20 = s.iloc[-20:].mean()
    price = s.iloc[-1]
    rets = s.pct_change().dropna()

    # RSI proxy
    recent = rets.iloc[-14:]
    rsi = (recent > 0).sum() / len(recent) if len(recent) > 0 else 0.5

    # Recent 5d momentum turning positive
    mom5 = s.iloc[-1] / s.iloc[-6] - 1 if len(s) >= 6 else 0

    # Oversold: below SMA20 + RSI < 0.35 + 5d momentum turning positive
    is_oversold = price < sma20 and rsi < 0.35 and mom5 > 0
    strength = max(0, (sma20 - price) / sma20) if is_oversold else 0

    return is_oversold, strength


def build_conviction_portfolio(
    prices, returns, idx, date,
    rankings, ml_preds, trend_flags,
    regime_int, universe, prev_weights,
):
    """
    V6 CONVICTION ENGINE — The core innovation.

    Instead of equal-weighting 6-8 assets:
    1. Score every asset on momentum + ML direction + trend + mean-reversion
    2. Take the TOP 3-4 by conviction score
    3. Weight by conviction (not equal weight)
    4. Apply regime multipliers
    5. Maintain momentum persistence (don't sell winners just to rebalance)
    """

    # ── STEP 1: Score every asset ──
    scores = {}
    for r in rankings:
        ticker = r["ticker"]
        ac = universe.get(ticker, {}).get("class", "")

        # A) Raw momentum (most important signal)
        raw_mom = r.get("raw_momentum", 0)
        mom_score = raw_mom  # keep raw — don't normalize away the signal

        # B) ML direction agreement
        ml = ml_preds.get(ticker, {})
        ml_dir = ml.get("direction", 0)
        ml_conf = ml.get("confidence", 0)
        ml_boost = ml_dir * ml_conf * 0.3  # up to ±0.3 boost

        # C) Trend confirmation
        above_trend = trend_flags.get(ticker, False)
        trend_boost = 0.1 if above_trend else -0.15

        # D) Mean reversion bounce (from LSTM project)
        is_bounce, bounce_str = detect_oversold_bounce(prices, ticker, idx)
        bounce_boost = bounce_str * 0.5 if is_bounce else 0

        # E) Momentum persistence bonus: if already held + still positive momentum
        persistence_bonus = 0
        if ticker in prev_weights and prev_weights[ticker] > 0.05 and raw_mom > 0:
            persistence_bonus = 0.08  # reward holding winners

        # F) Sector rotation bonus (from ATPM project)
        sector_bonus = 0
        if ac == "equity_sector" and regime_int >= Regime5.NORMAL:
            sector_bonus = 0.05

        # Composite conviction score
        conviction = mom_score + ml_boost + trend_boost + bounce_boost + persistence_bonus + sector_bonus

        # Asset class multipliers per regime
        if regime_int == Regime5.COMPLACENT:
            if ac in ("equity_us", "equity_sector", "equity_intl"):
                conviction *= 1.3
            elif ac == "crypto":
                conviction *= 1.4  # crypto gets biggest boost in low-vol bull
        elif regime_int == Regime5.NORMAL:
            if ac in ("equity_us", "equity_sector"):
                conviction *= 1.2
            elif ac == "crypto":
                conviction *= 1.1
        elif regime_int == Regime5.ELEVATED:
            if ac in ("equity_sector", "equity_us"):
                conviction *= 1.1  # Still favor equities in ELEVATED
            elif ac in ("commodity",):
                conviction *= 1.15  # Slight gold boost
            elif ac == "crypto":
                conviction *= 0.8  # Reduce crypto

        scores[ticker] = {
            "conviction": conviction,
            "raw_mom": raw_mom,
            "ml_dir": ml_dir,
            "ml_conf": ml_conf,
            "above_trend": above_trend,
            "bounce": is_bounce,
        }

    # ── STEP 2: Defensive regimes ──
    if regime_int == Regime5.PANIC:
        # Panic: defensive mix but NOT 100% cash
        return {"GLD": 0.30, "TLT": 0.20, "XLP": 0.15, "XLV": 0.15, "IEF": 0.10, "BIL": 0.10}

    if regime_int == Regime5.HIGH_FEAR:
        # High fear: 50% equity (defensive sectors) + 50% defensive
        return {"XLP": 0.15, "XLV": 0.15, "SPY": 0.10, "XLU": 0.10, "GLD": 0.20, "TLT": 0.15, "IEF": 0.10, "BIL": 0.05}

    # ── STEP 3: Select top assets by conviction ──
    ranked = sorted(scores.items(), key=lambda x: x[1]["conviction"], reverse=True)

    # Only take assets with positive conviction
    positive = [(t, s) for t, s in ranked if s["conviction"] > 0]
    if len(positive) == 0:
        return {"SPY": 0.40, "GLD": 0.30, "TLT": 0.20, "BIL": 0.10}

    # Top N based on regime — MORE CONCENTRATED
    if regime_int == Regime5.COMPLACENT:
        top_n = 3  # VERY concentrated — 3 highest conviction bets
    elif regime_int == Regime5.NORMAL:
        top_n = 4
    else:  # ELEVATED
        top_n = 5

    top = positive[:top_n]

    # ── STEP 4: Conviction-weighted allocation ──
    # CUBE the convictions to massively amplify the top pick
    convictions = np.array([s["conviction"] for _, s in top])
    convictions = np.clip(convictions, 0.01, None)

    exp_conv = convictions ** 2.5  # between square and cube for heavy concentration
    weights_arr = exp_conv / exp_conv.sum()

    weights = {t: float(w) for (t, _), w in zip(top, weights_arr)}

    # ── STEP 5: Position caps ──
    max_pos = 0.40 if regime_int == Regime5.COMPLACENT else (0.35 if regime_int >= Regime5.NORMAL else 0.25)
    capped = False
    for t in weights:
        if weights[t] > max_pos:
            weights[t] = max_pos
            capped = True
    if capped:
        total = sum(weights.values())
        if total > 0:
            weights = {t: w / total for t, w in weights.items()}

    # ── STEP 6: Ensure minimum risk asset exposure in bullish regimes ──
    risk_classes = {"equity_us", "equity_intl", "equity_sector", "crypto", "commodity", "real_estate"}
    risk_w = sum(w for t, w in weights.items() if universe.get(t, {}).get("class", "") in risk_classes)

    target_risk = {Regime5.COMPLACENT: 0.95, Regime5.NORMAL: 0.85, Regime5.ELEVATED: 0.70}
    min_risk = target_risk.get(regime_int, 0.50)

    if risk_w < min_risk:
        # Scale up risk assets
        scale = min_risk / max(risk_w, 0.01)
        for t in list(weights.keys()):
            if universe.get(t, {}).get("class", "") in risk_classes:
                weights[t] *= scale
        total = sum(weights.values())
        weights = {t: w / total for t, w in weights.items()}

    # Clean tiny weights
    weights = {t: w for t, w in weights.items() if w >= 0.03}
    total = sum(weights.values())
    if total > 0:
        weights = {t: w / total for t, w in weights.items()}

    return weights


def art_vol_scale(weights, returns, date, target_vol=0.20):
    """Auto-Regressive Risk Targeting."""
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
        lev = np.clip(lev, 0.6, 1.5)
    else:
        lev = 1.0
    scaled = {t: weights[t] * lev if t in tickers else weights[t] for t in weights}
    total = sum(scaled.values())
    if total > 1.0:
        scaled = {t: v / total for t, v in scaled.items()}
    return scaled


def main():
    print("\n🔥 GTAA V6 — Concentrated Conviction Engine")
    print("   Philosophy: Concentrate into highest-conviction bets\n")

    config = GTAAConfig()
    config.backtest.start_date = '2020-01-01'

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
    print("  Initializing agents...")
    research = ResearchAgent(
        lookback_windows=[21, 63, 126, 252],
        lookback_weights=[0.35, 0.30, 0.20, 0.15],
        skip_recent=3,
    )
    ml_regime = MLRegimeAgent(retrain_every=252)
    ml_direction = MLDirectionAgent(retrain_every=252)
    review = ReviewAgent()

    # Setup — MONTHLY rebalance (bi-weekly was too noisy)
    warmup = 300
    valid_dates = prices.index[warmup:]
    rebalance_dates = set()
    groups = valid_dates.to_series().groupby(valid_dates.to_period("M"))
    for _, g in groups:
        if len(g) > 0:
            rebalance_dates.add(g.iloc[-1])

    equity = config.backtest.initial_capital
    current_weights = {"BIL": 1.0}
    bench_equity = config.backtest.initial_capital
    equity_curve, dates_out, bench_vals, weights_history = [], [], [], []
    prev_regime = Regime5.NORMAL

    total_days = len(valid_dates)
    print(f"  Period: {valid_dates[0].date()} to {valid_dates[-1].date()}")
    print(f"  Rebalance: bi-weekly ({len(rebalance_dates)} dates)")
    print()

    t0 = time.time()

    for i, date in enumerate(valid_dates):
        if "SPY" in returns.columns and date in returns.index:
            br = returns.loc[date, "SPY"]
            if pd.notna(br):
                bench_equity *= (1 + br)

        # Daily portfolio return
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

        # Drawdown circuit breaker: if DD > 15%, cap at ELEVATED. If > 25%, force HIGH_FEAR
        peak = max(equity_curve)
        dd = (equity - peak) / peak if peak > 0 else 0

        if date in rebalance_dates:
            try:
                idx = prices.index.get_loc(date)

                # 1. Research
                r_sig = research.analyze(date, prices, returns, {"universe": config.universe})

                # 2. ML Regime
                rg_sig = ml_regime.analyze(date, prices, returns, {"vix_series": vix_series})
                regime_int = rg_sig.data.get("regime_int", Regime5.NORMAL)

                # 3. ML Direction
                dir_sig = ml_direction.analyze(date, prices, returns, {})

                # 4. Drawdown override — gentler
                if dd < -0.25:
                    regime_int = min(regime_int, Regime5.HIGH_FEAR)
                elif dd < -0.15:
                    regime_int = min(regime_int, Regime5.ELEVATED)

                # 5. REGIME TRANSITION BOOST: when switching from fear → normal, go aggressive
                regime_transition_boost = False
                if prev_regime <= Regime5.ELEVATED and regime_int >= Regime5.NORMAL:
                    regime_transition_boost = True

                # 6. Build conviction portfolio
                target_w = build_conviction_portfolio(
                    prices, returns, idx, date,
                    rankings=r_sig.data.get("rankings", []),
                    ml_preds=dir_sig.data.get("predictions", {}),
                    trend_flags=r_sig.data.get("trend_flags", {}),
                    regime_int=regime_int,
                    universe=config.universe,
                    prev_weights=current_weights,
                )

                # 7. ART vol targeting
                target_w = art_vol_scale(target_w, returns, date, target_vol=0.22)

                # 8. Execute: no PM gating — conviction engine decides directly
                # Transaction costs
                cost = 0
                for t in set(current_weights.keys()) | set(target_w.keys()):
                    tv = abs(current_weights.get(t, 0) - target_w.get(t, 0)) * equity
                    if tv > 100:
                        slp = 15 if config.universe.get(t, {}).get("class") == "crypto" else 5
                        cost += tv * slp / 10000
                equity -= cost

                weights_history.append({
                    "date": str(date.date()),
                    "weights": {t: round(w, 4) for t, w in target_w.items() if w > 0.02},
                    "regime": rg_sig.data.get("regime", "NORMAL"),
                    "regime_int": regime_int,
                    "equity": round(equity, 2),
                    "cost": round(cost, 2),
                    "dd": round(dd, 4),
                    "transition_boost": regime_transition_boost,
                })

                current_weights = target_w
                prev_regime = regime_int

            except Exception as e:
                logger.error(f"Error {date.date()}: {e}")
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
    print("  GTAA V6 — CONCENTRATED CONVICTION ENGINE")
    print("=" * 80)
    print(f"  {'Metric':<25} {'GTAA V6':>12} {'SPY':>12} {'Delta':>12}")
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
        n_pos = [len([w for w in e['weights'].values() if w > 0.03]) for e in weights_history]

        regimes = Counter(e.get('regime') for e in weights_history)

        # Most held tickers
        ticker_freq = Counter()
        for e in weights_history:
            for t, w in e['weights'].items():
                if w > 0.05:
                    ticker_freq[t] += 1

        print(f"\n  Avg equity:     {np.mean(eq_exp):.1%}")
        print(f"  Avg crypto:     {np.mean(crypto_exp):.1%}")
        print(f"  Avg cash:       {np.mean(cash_exp):.1%}")
        print(f"  Avg positions:  {np.mean(n_pos):.1f}")
        print(f"  Regimes:        {dict(regimes)}")
        print(f"  Top holdings:   {ticker_freq.most_common(8)}")

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
