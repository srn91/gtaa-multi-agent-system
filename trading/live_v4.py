#!/usr/bin/env python3
"""
V4 Live Signal Pipeline — Runs ML-enhanced agent team on current data.
Entry point for GitHub Actions paper trading.
"""

import os
import json
import logging
from datetime import datetime

import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
logger = logging.getLogger("gtaa_live")


def run_live_signal():
    """Run the full V4 agent pipeline on current data and optionally execute via Alpaca."""
    from config.settings import GTAAConfig, ASSET_UNIVERSE
    from data.data_loader import DataLoader
    from agents.research_agent import ResearchAgent
    from agents.ml_regime_agent import MLRegimeAgent
    from agents.ml_direction_agent import MLDirectionAgent
    from agents.risk_agent import RiskAgent
    from agents.pm_agent import PMAgent
    from agents.base_agent import Signal
    from backtest_v4 import blend_momentum_and_ml, apply_vol_targeting
    import yfinance as yf

    config = GTAAConfig()
    config.risk.max_position_pct = 0.35
    config.risk.max_asset_class_pct = 0.65
    config.risk.min_cash_pct = 0.01

    # Load recent data (3 years for ML training)
    logger.info("Loading market data...")
    loader = DataLoader(universe=config.universe, start="2022-01-01")
    prices = loader.load(use_cache=False)
    returns = loader.returns
    today = prices.index[-1]

    # Load VIX
    try:
        vix_raw = yf.download("^VIX", start="2022-01-01", progress=False)
        if isinstance(vix_raw.columns, pd.MultiIndex):
            vix_series = vix_raw["Close"]["^VIX"].reindex(prices.index).ffill()
        else:
            vix_series = vix_raw["Close"].reindex(prices.index).ffill()
    except Exception:
        spy_vol = prices["SPY"].pct_change().rolling(20).std() * np.sqrt(252) * 100
        vix_series = spy_vol.clip(10, 80)

    logger.info(f"Data through {today.date()}, {len(prices.columns)} assets")

    # === Run Agent Pipeline ===

    # 1. Research (momentum)
    research = ResearchAgent()
    r_sig = research.analyze(today, prices, returns, {"universe": config.universe})
    logger.info(f"Research: top 5 = {r_sig.data['top_5']}")

    # 2. ML Regime (XGBoost 5-regime)
    ml_regime = MLRegimeAgent(retrain_every=9999)  # force train on first call
    rg_sig = ml_regime.analyze(today, prices, returns, {"vix_series": vix_series})
    logger.info(f"Regime: {rg_sig.data['regime']} (action={rg_sig.data['action']}, conf={rg_sig.confidence:.2f})")

    # 3. ML Direction (RF+KNN)
    ml_dir = MLDirectionAgent(retrain_every=9999)
    dir_sig = ml_dir.analyze(today, prices, returns, {})
    logger.info(f"Direction: {len(dir_sig.data['bullish_tickers'])} bullish, "
                f"{len(dir_sig.data['bearish_tickers'])} bearish")

    # 4. Blend into target weights
    regime_action = rg_sig.data.get("action", "MODERATE")
    blended = blend_momentum_and_ml(
        r_sig.data.get("rankings", []),
        dir_sig.data.get("predictions", {}),
        r_sig.data.get("trend_flags", {}),
        regime_action,
        top_n=6,
        ml_weight=0.35,
    )

    # 5. Vol targeting
    vol_targeted = apply_vol_targeting(blended, returns, today, target_vol=0.15)

    # 6. Risk check
    risk_agent = RiskAgent(config=config.risk)
    risk_sig = risk_agent.analyze(today, prices, returns, {
        "proposed_weights": vol_targeted,
        "universe": config.universe,
        "equity_value": 100000,
        "regime": rg_sig.data.get("regime_3way", "RISK_ON"),
    })

    # 7. PM decision
    pm = PMAgent()
    alloc_sig = Signal("Allocation", datetime.now(), "allocation",
                       {"target_weights": vol_targeted}, 0.7, "V4 blend")
    pm_sig = pm.analyze(today, prices, returns, {
        "research_signal_obj": r_sig,
        "regime_signal_obj": rg_sig,
        "allocation_signal_obj": alloc_sig,
        "risk_signal_obj": risk_sig,
        "equity_value": 100000,
        "universe": config.universe,
    })

    # === Output ===
    final_weights = pm_sig.data.get("final_weights", {})

    signal_log = {
        "date": str(today.date()),
        "regime": rg_sig.data.get("regime"),
        "regime_action": regime_action,
        "regime_confidence": rg_sig.confidence,
        "ml_bullish": dir_sig.data.get("bullish_tickers", [])[:5],
        "ml_bearish": dir_sig.data.get("bearish_tickers", [])[:5],
        "momentum_top5": r_sig.data.get("top_5", []),
        "target_weights": {t: round(w, 4) for t, w in final_weights.items() if w > 0.01},
        "conviction": pm_sig.data.get("conviction", 0),
        "rebalance": pm_sig.data.get("execute_rebalance", False),
    }

    print("\n" + "=" * 60)
    print("  GTAA V4 — Live Signal Output")
    print("=" * 60)
    print(json.dumps(signal_log, indent=2))
    print("=" * 60)

    # Execute via Alpaca if configured
    if pm_sig.data.get("execute_rebalance") and final_weights:
        dry_run = os.environ.get("DRY_RUN", "true").lower() == "true"
        api_key = os.environ.get("APCA_API_KEY_ID")

        if api_key:
            from trading.alpaca_trader import AlpacaTrader
            trader = AlpacaTrader()
            orders = trader.rebalance(final_weights, dry_run=dry_run)
            print(f"\nOrders executed (dry_run={dry_run}):")
            print(json.dumps(orders, indent=2))
        else:
            print("\nAlpaca not configured. Set APCA_API_KEY_ID to enable paper trading.")
    else:
        print("\nNo rebalance triggered.")

    # Save signal log
    os.makedirs("results", exist_ok=True)
    with open("results/latest_signal.json", "w") as f:
        json.dump(signal_log, f, indent=2)

    return signal_log


if __name__ == "__main__":
    run_live_signal()
