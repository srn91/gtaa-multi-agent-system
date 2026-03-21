#!/usr/bin/env python3
"""
Test Suite — Validates all agents produce correct signal types,
the engine runs without errors, and risk limits are enforced.
Run: python -m pytest tests/test_agents.py -v
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from agents.base_agent import Signal
from agents.research_agent import ResearchAgent
from agents.regime_agent import RegimeAgent
from agents.risk_agent import RiskAgent
from agents.allocation_agent import AllocationAgent
from agents.pm_agent import PMAgent
from agents.review_agent import ReviewAgent
from config.settings import GTAAConfig, ASSET_UNIVERSE, RiskConfig, AllocationConfig


def _make_fake_prices(n_days=500, n_tickers=10, trend=0.0003):
    """Generate fake price data for testing."""
    dates = pd.bdate_range(end=datetime.now(), periods=n_days)
    tickers = list(ASSET_UNIVERSE.keys())[:n_tickers]
    np.random.seed(42)

    data = {}
    for t in tickers:
        returns = np.random.normal(trend, 0.015, n_days)
        prices = 100 * np.cumprod(1 + returns)
        data[t] = prices

    prices = pd.DataFrame(data, index=dates)
    returns = prices.pct_change()
    return prices, returns


def test_research_agent():
    """Research agent produces valid momentum signal."""
    prices, returns = _make_fake_prices()
    agent = ResearchAgent()
    date = prices.index[-10]

    signal = agent.analyze(date, prices, returns, {"universe": ASSET_UNIVERSE})

    assert isinstance(signal, Signal)
    assert signal.signal_type == "momentum"
    assert "rankings" in signal.data
    assert "trend_flags" in signal.data
    assert "top_5" in signal.data
    assert 0 <= signal.confidence <= 1.0
    assert len(signal.reasoning) > 0

    # Rankings should have vol_adj_momentum and raw_momentum
    if len(signal.data["rankings"]) > 0:
        r = signal.data["rankings"][0]
        assert "ticker" in r
        assert "raw_momentum" in r
        assert "vol_adj_momentum" in r

    print(f"  Research: {len(signal.data['rankings'])} assets ranked, "
          f"top={signal.data['top_5'][:3]}, confidence={signal.confidence:.2f}")


def test_regime_agent():
    """Regime agent produces valid regime classification."""
    prices, returns = _make_fake_prices()
    agent = RegimeAgent()
    date = prices.index[-10]

    signal = agent.analyze(date, prices, returns, {
        "regime_data": pd.DataFrame(),
        "universe": ASSET_UNIVERSE,
    })

    assert isinstance(signal, Signal)
    assert signal.signal_type == "regime"
    assert signal.data["regime"] in ("RISK_ON", "RISK_OFF", "CRISIS")
    assert "composite_score" in signal.data
    assert "scores" in signal.data
    assert 0 <= signal.confidence <= 1.0

    print(f"  Regime: {signal.data['regime']}, "
          f"composite={signal.data['composite_score']:.2f}, "
          f"confidence={signal.confidence:.2f}")


def test_risk_agent_concentration_limits():
    """Risk agent enforces position and class limits."""
    agent = RiskAgent(config=RiskConfig(
        max_position_pct=0.25,
        max_asset_class_pct=0.50,
    ))

    prices, returns = _make_fake_prices()
    date = prices.index[-10]

    # Propose a portfolio that violates position limits
    proposed = {"SPY": 0.50, "QQQ": 0.30, "GLD": 0.20}

    signal = agent.analyze(date, prices, returns, {
        "proposed_weights": proposed,
        "universe": ASSET_UNIVERSE,
        "equity_value": 100000,
        "regime": "RISK_ON",
    })

    assert isinstance(signal, Signal)
    approved = signal.data["approved_weights"]

    # No single position should exceed 25%
    for ticker, weight in approved.items():
        assert weight <= 0.26, f"{ticker} weight {weight:.2%} exceeds 25% limit"

    assert len(signal.data["violations"]) > 0, "Should have detected violations"

    print(f"  Risk: {len(signal.data['violations'])} violations, "
          f"approved {len(approved)} positions")


def test_risk_agent_circuit_breaker():
    """Risk agent triggers circuit breaker on drawdown."""
    agent = RiskAgent(config=RiskConfig(
        drawdown_circuit_breaker=-0.15,
        drawdown_full_exit=-0.25,
    ))

    prices, returns = _make_fake_prices()
    date = prices.index[-10]

    # Set peak high then simulate a crash
    agent._peak_equity = 150000

    signal = agent.analyze(date, prices, returns, {
        "proposed_weights": {"SPY": 0.5, "QQQ": 0.5},
        "universe": ASSET_UNIVERSE,
        "equity_value": 100000,  # -33% drawdown from 150k
        "regime": "RISK_ON",
    })

    # Should trigger full exit
    assert signal.data["drawdown_action"] == "FULL_EXIT"
    assert signal.data["vetoed"] is True
    # Should be mostly cash
    assert signal.data["approved_weights"].get("BIL", 0) > 0.9

    print(f"  Circuit breaker: DD={signal.data['drawdown_level']:.1%}, "
          f"action={signal.data['drawdown_action']}, "
          f"BIL={signal.data['approved_weights'].get('BIL', 0):.0%}")


def test_allocation_agent():
    """Allocation agent produces valid weights."""
    agent = AllocationAgent(config=AllocationConfig(
        top_n_assets=5,
        risk_parity_blend=0.0,
        regime_tilt_strength=0.40,
    ))

    prices, returns = _make_fake_prices()
    date = prices.index[-10]

    # Fake research signal
    tickers = list(prices.columns)
    rankings = [
        {"ticker": t, "raw_momentum": 0.15 - i * 0.02, "vol_adj_momentum": 0.8 - i * 0.1}
        for i, t in enumerate(tickers)
    ]
    trend_flags = {t: True for t in tickers}

    research_signal = {
        "data": {"rankings": rankings, "trend_flags": trend_flags}
    }
    regime_signal = {"data": {"regime": "RISK_ON"}}

    signal = agent.analyze(date, prices, returns, {
        "research_signal": research_signal,
        "regime_signal": regime_signal,
        "universe": ASSET_UNIVERSE,
    })

    assert isinstance(signal, Signal)
    assert signal.signal_type == "allocation"
    weights = signal.data["target_weights"]

    # Weights should sum to ~1.0
    total = sum(weights.values())
    assert 0.95 <= total <= 1.05, f"Weights sum to {total:.3f}, expected ~1.0"

    # Should have <= top_n positions
    assert len(weights) <= 6  # top_n + maybe cash

    print(f"  Allocation: {len(weights)} positions, "
          f"sum={total:.3f}, regime={signal.data['regime_used']}")


def test_pm_agent_consensus():
    """PM agent gates on consensus and turnover."""
    pm = PMAgent(min_consensus_confidence=0.5, rebalance_threshold=0.05)

    prices, returns = _make_fake_prices()
    date = prices.index[-10]

    # Create mock signals
    research = Signal("Research", datetime.now(), "momentum", {
        "rankings": [], "top_5": ["SPY", "QQQ"]
    }, 0.8, "test")

    regime = Signal("Regime", datetime.now(), "regime", {
        "regime": "RISK_ON", "composite_score": 0.5
    }, 0.7, "test")

    allocation = Signal("Allocation", datetime.now(), "allocation", {
        "target_weights": {"SPY": 0.4, "QQQ": 0.3, "GLD": 0.3}
    }, 0.75, "test")

    risk = Signal("Risk", datetime.now(), "risk", {
        "approved_weights": {"SPY": 0.4, "QQQ": 0.3, "GLD": 0.3},
        "vetoed": False,
    }, 0.85, "test")

    signal = pm.analyze(date, prices, returns, {
        "research_signal_obj": research,
        "regime_signal_obj": regime,
        "allocation_signal_obj": allocation,
        "risk_signal_obj": risk,
        "equity_value": 100000,
        "universe": ASSET_UNIVERSE,
    })

    assert isinstance(signal, Signal)
    assert signal.signal_type == "decision"
    assert "execute_rebalance" in signal.data
    assert "conviction" in signal.data
    assert "consensus" in signal.data

    print(f"  PM: rebalance={signal.data['execute_rebalance']}, "
          f"conviction={signal.data['conviction']:.2f}, "
          f"consensus={signal.data['consensus']:.2f}")


def test_review_agent():
    """Review agent computes stats from equity curve."""
    agent = ReviewAgent()

    # Simulate an equity curve
    dates = pd.bdate_range(end=datetime.now(), periods=500)
    equity = 100000
    for d in dates:
        equity *= (1 + np.random.normal(0.0003, 0.01))
        agent.update_equity(d, equity)

    prices, returns = _make_fake_prices()
    date = dates[-1]

    signal = agent.analyze(date, prices, returns, {
        "current_weights": {"SPY": 0.5, "GLD": 0.5},
        "equity_value": equity,
    })

    assert isinstance(signal, Signal)
    assert signal.signal_type == "review"
    stats = signal.data["stats"]

    assert "cagr" in stats
    assert "sharpe" in stats
    assert "max_drawdown" in stats
    assert "sortino" in stats

    print(f"  Review: CAGR={stats['cagr']:.1%}, "
          f"Sharpe={stats['sharpe']:.2f}, "
          f"MaxDD={stats['max_drawdown']:.1%}")


def test_audit_trail():
    """All agents produce audit trail entries."""
    prices, returns = _make_fake_prices()
    date = prices.index[-10]

    research = ResearchAgent()
    research.analyze(date, prices, returns, {"universe": ASSET_UNIVERSE})

    trail = research.get_audit_trail()
    assert len(trail) > 0
    assert "timestamp" in trail[0]
    assert "agent" in trail[0]
    assert "action" in trail[0]

    print(f"  Audit: {len(trail)} entries, last action='{trail[-1]['action']}'")


def test_signal_serialization():
    """Signals serialize to dict and JSON."""
    signal = Signal(
        agent_name="Test",
        timestamp=datetime.now(),
        signal_type="test",
        data={"key": "value", "nested": {"a": 1}},
        confidence=0.85,
        reasoning="Test reasoning",
    )

    d = signal.to_dict()
    assert d["agent"] == "Test"
    assert d["confidence"] == 0.85

    j = signal.to_json()
    assert '"agent": "Test"' in j

    print(f"  Serialization: dict keys={list(d.keys())}")


def test_full_agent_pipeline():
    """Run all agents in sequence (mini integration test)."""
    prices, returns = _make_fake_prices(n_days=500, n_tickers=15)
    date = prices.index[-10]
    config = GTAAConfig()

    # 1. Research
    research = ResearchAgent()
    r_sig = research.analyze(date, prices, returns, {"universe": ASSET_UNIVERSE})

    # 2. Regime
    regime = RegimeAgent()
    rg_sig = regime.analyze(date, prices, returns, {
        "regime_data": pd.DataFrame(), "universe": ASSET_UNIVERSE
    })

    # 3. Allocation
    alloc = AllocationAgent()
    a_sig = alloc.analyze(date, prices, returns, {
        "research_signal": r_sig.to_dict(),
        "regime_signal": rg_sig.to_dict(),
        "universe": ASSET_UNIVERSE,
    })

    # 4. Risk
    risk = RiskAgent()
    rk_sig = risk.analyze(date, prices, returns, {
        "proposed_weights": a_sig.data["target_weights"],
        "universe": ASSET_UNIVERSE,
        "equity_value": 100000,
        "regime": rg_sig.data.get("regime", "RISK_ON"),
    })

    # 5. PM
    pm = PMAgent()
    pm_sig = pm.analyze(date, prices, returns, {
        "research_signal_obj": r_sig,
        "regime_signal_obj": rg_sig,
        "allocation_signal_obj": a_sig,
        "risk_signal_obj": rk_sig,
        "equity_value": 100000,
        "universe": ASSET_UNIVERSE,
    })

    # 6. Review
    review = ReviewAgent()
    review.update_equity(date, 100000)
    rv_sig = review.analyze(date, prices, returns, {
        "current_weights": pm_sig.data.get("final_weights", {}),
        "equity_value": 100000,
    })

    print(f"  Pipeline: Research→Regime({rg_sig.data['regime']})→"
          f"Alloc({len(a_sig.data['target_weights'])} pos)→"
          f"Risk(veto={rk_sig.data['vetoed']})→"
          f"PM(exec={pm_sig.data['execute_rebalance']})")

    # All signals should be valid
    for sig in [r_sig, rg_sig, a_sig, rk_sig, pm_sig, rv_sig]:
        assert isinstance(sig, Signal)
        assert 0 <= sig.confidence <= 1.0
        assert len(sig.reasoning) > 0


def test_ml_regime_agent():
    """ML Regime agent produces valid 5-regime classification."""
    from agents.ml_regime_agent import MLRegimeAgent, REGIME_META
    prices, returns = _make_fake_prices(n_days=500, n_tickers=10)

    # Create a fake VIX series
    vix_series = pd.Series(
        np.random.uniform(12, 35, len(prices)),
        index=prices.index,
    )

    agent = MLRegimeAgent(retrain_every=500)  # won't retrain during test
    date = prices.index[-10]

    signal = agent.analyze(date, prices, returns, {"vix_series": vix_series})

    assert isinstance(signal, Signal)
    assert signal.signal_type == "regime"
    assert signal.data["regime"] in [m["label"] for m in REGIME_META.values()]
    assert "action" in signal.data
    assert "var_multiplier" in signal.data
    assert "regime_3way" in signal.data
    assert signal.data["regime_3way"] in ("CRISIS", "RISK_OFF", "RISK_ON")

    print(f"  MLRegime: {signal.data['regime']} "
          f"(action={signal.data['action']}, conf={signal.confidence:.2f})")


def test_ml_direction_agent():
    """ML Direction agent produces per-asset predictions."""
    from agents.ml_direction_agent import MLDirectionAgent
    prices, returns = _make_fake_prices(n_days=500, n_tickers=8)
    date = prices.index[-10]

    agent = MLDirectionAgent(retrain_every=1000, min_train_samples=100)
    signal = agent.analyze(date, prices, returns, {})

    assert isinstance(signal, Signal)
    assert signal.signal_type == "direction"
    assert "predictions" in signal.data
    assert "bullish_tickers" in signal.data
    assert "bearish_tickers" in signal.data
    assert "n_models_trained" in signal.data

    n_bull = len(signal.data["bullish_tickers"])
    n_bear = len(signal.data["bearish_tickers"])
    n_models = signal.data["n_models_trained"]

    print(f"  MLDirection: {n_bull} bullish, {n_bear} bearish, {n_models} models")


def main():
    tests = [
        ("Signal Serialization", test_signal_serialization),
        ("Research Agent", test_research_agent),
        ("Regime Agent", test_regime_agent),
        ("Risk Agent (Concentration)", test_risk_agent_concentration_limits),
        ("Risk Agent (Circuit Breaker)", test_risk_agent_circuit_breaker),
        ("Allocation Agent", test_allocation_agent),
        ("PM Agent (Consensus)", test_pm_agent_consensus),
        ("Review Agent", test_review_agent),
        ("Audit Trail", test_audit_trail),
        ("Full Pipeline", test_full_agent_pipeline),
        ("ML Regime Agent", test_ml_regime_agent),
        ("ML Direction Agent", test_ml_direction_agent),
    ]

    print("\n🧪 GTAA Agent Test Suite\n")
    passed = 0
    failed = 0

    for name, test_fn in tests:
        try:
            test_fn()
            print(f"  ✅ {name}")
            passed += 1
        except Exception as e:
            print(f"  ❌ {name}: {e}")
            failed += 1

    print(f"\n  Results: {passed} passed, {failed} failed out of {len(tests)}")
    print()

    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
