# GTAA Multi-Agent System

A role-specialized Global Tactical Asset Allocation research system for regime-aware allocation, backtesting, paper trading, and portfolio review.

## What This Project Is

This project explores GTAA as a structured multi-agent workflow rather than a single monolithic model. Specialized agents handle research, regime classification, direction assessment, allocation, risk control, portfolio approval, and post-trade review.

The goal is to build a disciplined allocation framework that is:
- modular
- auditable
- regime-aware
- testable through backtesting and paper trading

---

## Current Status

| Area | Status |
|---|---|
| Multi-agent GTAA workflow (8 agents) | Implemented |
| Historical backtesting (6 versions) | Implemented |
| Monte Carlo robustness analysis (10,000 paths) | Implemented |
| ML regime classification (XGBoost) | Implemented |
| ML direction prediction (RF + KNN ensemble) | Implemented |
| Decision artifacts and audit trail | Implemented |
| Paper trading workflow | Implemented |
| GitHub Actions automation | Implemented |
| Walk-forward validation | In progress |
| Live deployment | Not started |

---

## Performance (2020–2025, net of transaction costs)

In the reported test window, the prototype outperformed SPY on the displayed risk-adjusted metrics under the stated assumptions.

| Metric | GTAA V5 | SPY Buy & Hold | Delta |
|---|---|---|---|
| CAGR | 20.9% | 16.3% | +4.6% |
| Sharpe Ratio | 0.92 | 0.81 | +0.11 |
| Sortino Ratio | 1.13 | 0.92 | +0.21 |
| Max Drawdown | -21.3% | -24.5% | +3.2% better |
| Calmar Ratio | 0.98 | 0.67 | +0.32 |
| Total Return | 168% | 119% | +49% |

> Results are from historical simulation under stated assumptions. The backtest landed in the ~80th percentile of Monte Carlo outcomes. The median expected CAGR across 10,000 simulated paths is 13.5%. Plan around the median, not the peak.

![Equity Curve](reports/01_equity_curve.png)

![Drawdown](reports/02_drawdown.png)

---

## Monte Carlo Validation — 10,000 Simulated Paths

Instead of trusting a single backtest, we ran 10,000 bootstrapped simulations using 21-day block sampling to preserve momentum structure.

| Metric | Worst Case (5th) | Expected (50th) | Best Case (95th) |
|---|---|---|---|
| CAGR | 3.8% | 13.5% | 24.6% |
| Sharpe Ratio | 0.32 | 0.89 | 1.48 |
| Max Drawdown | -33.4% | -20.9% | -10.9% |
| Final Value ($100k) | $132k | $258k | $521k |

| Probability | Value |
|---|---|
| Beat SPY | 63.1% |
| CAGR > 15% | 39.8% |
| CAGR > 20% | 16.1% |
| Lose money | 1.0% |

![Monte Carlo](reports/07_monte_carlo.png)

---

## Architecture — 8 Specialized Agents

The system operates as a fund team where each agent has a defined role. Every rebalance period, all agents are consulted in sequence. The PM Agent only executes when there is sufficient consensus.

![Architecture](reports/08_architecture.png)

| Agent | Role | ML Model | Key Function |
|---|---|---|---|
| **Research Agent** | Multi-timeframe momentum scanner | — | Raw + vol-adjusted momentum across 31 assets |
| **ML Regime Agent** | 5-regime market classifier | XGBoost | PANIC → COMPLACENT classification with LAG-1 protocol |
| **ML Direction Agent** | Per-asset direction prediction | RF + KNN | When models agree → high conviction; disagree → default to KNN |
| **Risk Agent** | Position limits and circuit breakers | — | VaR-based sizing, per-regime multipliers, drawdown protection |
| **Allocation Agent** | Cross-asset weighting | — | Momentum + regime tilt + sector rotation bonuses |
| **PM Agent** | Final decision maker | — | Consensus gating, conviction scoring, turnover control |
| **Review Agent** | Post-trade attribution | — | Sharpe, CAGR, drawdown attribution, performance flags |
| **Rule Regime Agent** | Fallback regime detection | — | SMA/breadth/VIX rules when ML is unavailable |

### Decision Flow

1. Research Agent scans all 31 assets → momentum rankings with trend flags
2. ML Regime Agent classifies market into 5 regimes using XGBoost
3. ML Direction Agent predicts direction for each asset using RF + KNN ensemble
4. Allocation Agent blends momentum scores with ML direction and regime tilts
5. Risk Agent enforces position limits, correlation checks, and drawdown circuit breakers
6. PM Agent checks consensus → execute or hold
7. Review Agent logs the trade and computes attribution

---

## Example Rebalance Decision

Below is a real decision from the system, not a mockup:

```json
{
  "rebalance_date": "2023-03-15",
  "research_agent": {
    "top_ranked_assets": ["QQQ", "GLD", "TLT", "XLE"],
    "scan_universe": 31,
    "confidence": 0.76
  },
  "regime_agent": {
    "regime": "NORMAL",
    "vix": 18.2,
    "confidence": 0.81,
    "risk_budget": "high"
  },
  "direction_agent": {
    "bullish": ["QQQ", "GLD"],
    "neutral": ["TLT"],
    "bearish": ["XLE"],
    "rf_knn_agreement": 0.72
  },
  "risk_agent": {
    "target_volatility": 0.14,
    "risk_budget": "normal",
    "warnings": [],
    "drawdown_from_peak": "-4.2%"
  },
  "allocation_agent": {
    "weights": {
      "QQQ": 0.30,
      "GLD": 0.18,
      "TLT": 0.12,
      "BTC-USD": 0.10,
      "XLK": 0.08,
      "CASH": 0.22
    }
  },
  "pm_agent": {
    "decision": "approve_rebalance",
    "turnover": "12%",
    "reason": "consensus across research, regime, and direction agents"
  }
}
```

Full rebalance artifacts are stored in [`reports/sample_decisions/`](reports/sample_decisions/).

---

## Asset Universe — 31 Instruments Across 7 Classes

| Class | Tickers |
|---|---|
| US Equity (broad) | SPY, QQQ, IWM, MDY |
| US Equity (sectors) | XLK, XLF, XLE, XLV, XLY, XLP, XLI, XLU |
| International Equity | EFA, EEM, VGK, EWJ |
| Fixed Income | TLT, IEF, SHY, HYG, LQD, TIP |
| Commodities | GLD, SLV, USO, DBC |
| Real Estate | VNQ, VNQI |
| Crypto | BTC-USD, ETH-USD |
| Cash | BIL |

![Allocation Over Time](reports/05_allocation_over_time.png)

---

## Regime Classification — XGBoost 5-Regime System

| Regime | VIX Range | Action | Risk Budget |
|---|---|---|---|
| PANIC | > 30 | Defensive (bonds + gold + defensive equity) | Minimal |
| HIGH_FEAR | 25–30 | Conservative equity + hedges | Low |
| ELEVATED | 20–25 | Equity-heavy with defensive tilt | Moderate |
| NORMAL | 15–20 | Full risk deployment, sector rotation | High |
| COMPLACENT | ≤ 15 | Maximum conviction, concentrated bets | Aggressive |

![Regime Timeline](reports/04_regime_timeline.png)

---

## Version Evolution

| Version | CAGR | Sharpe | Max DD | Calmar | Key Fix |
|---|---|---|---|---|---|
| V1 | 7.4% | 0.75 | -14% | 0.52 | Vol-adjusted momentum killed equity exposure |
| V2 | 8.0% | 0.66 | -16% | 0.49 | Raised vol target, reduced risk parity blend |
| V3 | 10.1% | 0.64 | -21% | 0.47 | Regime-aware raw momentum during RISK_ON |
| V4 | 17.8% | 0.68 | -27% | 0.65 | Added XGBoost 5-regime + RF/KNN direction ensemble |
| **V5** | **20.9%** | **0.92** | **-21%** | **0.98** | **Fixed cash drag, ART vol targeting, risk bypass in bull regimes** |
| V6 | 20.0% | 0.62 | -46% | 0.43 | Over-concentrated — worse risk-adjusted (rejected) |

---

## Monthly Returns

![Monthly Heatmap](reports/03_monthly_heatmap.png)

---

## Rolling Risk Metrics

![Rolling Sharpe](reports/06_rolling_sharpe.png)

---

## Reproducibility

Run the canonical GTAA backtest with:

```bash
python3 run_backtest.py
```

Reference configuration:
- **Strategy**: GTAA V5 (regime-aware tactical allocation)
- **Universe**: 31 instruments across 7 asset classes
- **Benchmark**: SPY
- **Period**: 2020-01-01 to 2025-12-31
- **Rebalance**: monthly
- **Costs**: transaction costs included per config
- **ML models**: XGBoost regime classifier, RF + KNN direction ensemble

---

## Quick Start

```bash
pip install -r requirements.txt
python3 run_backtest.py                    # production backtest
python3 -m engine.monte_carlo              # 10,000 Monte Carlo simulations
python3 reports/generate_reports.py        # generate all visualizations
python3 tests/test_agents.py              # 12 tests covering all agents
python3 -m streamlit run dashboard/app.py  # interactive dashboard
```

---

## Project Structure

```
├── run_backtest.py             # Canonical backtest entry point
├── agents/                     # 8 agent implementations
├── config/                     # Settings + production config + strategy YAML
├── dashboard/                  # Streamlit interactive dashboard
├── data/                       # yfinance data fetching with caching
├── engine/                     # Backtester + Monte Carlo simulation
├── trading/                    # Alpaca paper trading execution
├── reports/
│   ├── *.png                   # 9 publication-quality charts
│   ├── sample_decisions/       # Agent decision artifacts
│   └── generate_reports.py     # Report generator
├── research/archive/           # V1-V6 historical backtest scripts
├── tests/                      # 12-test suite (all agents + ML)
├── docs/                       # Architecture documentation
├── .github/workflows/          # GitHub Actions automation
├── notebooks/                  # Interactive walkthrough
└── CHANGELOG.md                # Version history
```

---

## Known Limitations

- Results are based on historical simulation and not live trading.
- The current implementation is a research prototype, not a production allocation engine.
- Monte Carlo analysis improves robustness assessment but does not guarantee future performance.
- Regime classification and allocation logic may behave differently in live markets than in historical simulation.
- Crypto exposure averages 14.5% — BTC/ETH captured meaningful alpha during 2020–2025, but the edge is partially period-dependent.
- The 2020–2025 period included a historically strong US equity bull market.
- Further walk-forward testing and paper trading are needed before any real-capital deployment.

---

## Roadmap

- [x] Multi-agent GTAA workflow (8 agents)
- [x] ML regime classification (XGBoost)
- [x] ML direction prediction (RF + KNN ensemble)
- [x] Monte Carlo validation (10,000 paths)
- [x] Interactive dashboard
- [x] Paper trading integration
- [x] GitHub Actions automation
- [ ] 90-day paper trading validation
- [ ] Walk-forward out-of-sample testing
- [ ] Agent-level performance attribution
- [ ] Live deployment with small capital

---

## License

```
Apache License Version 2.0, January 2004
Copyright 2026 Sathwik Rao Nadipelli
```

This repository contains a research and infrastructure framework only. The authors make no claims regarding financial performance. Example results shown are for research and educational purposes only. This software is not intended to be used as financial advice or as a production trading system without independent validation.
