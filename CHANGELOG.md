# Changelog

All notable changes to this project are documented here.

## v0.5.0 — Production Release (2025-03)

### Added
- XGBoost 5-regime classifier (PANIC → COMPLACENT) with LAG-1 protocol
- RF + KNN per-asset direction ensemble with ATR and SMA20 filters
- Monte Carlo simulation engine (10,000 block-bootstrapped paths)
- Auto-Regressive Risk Targeting (ART) at 20% vol target
- Professional report generator (9 publication-quality charts)
- Streamlit interactive dashboard
- Alpaca paper trading integration with GitHub Actions automation
- Apache 2.0 license with research-only notice

### Changed
- Reduced cash drag from 26% to 2.5% average by bypassing risk agent cash padding in bullish regimes
- Raised equity exposure from 12% to 62% through regime-aware allocation floors
- Sector ETF rotation (XLK, XLF, XLE, XLV, XLY, XLP, XLI, XLU) for finer equity granularity

### Performance
- CAGR: 20.9% (vs SPY 16.3%)
- Sharpe: 0.92 (vs SPY 0.81)
- Max DD: -21.3% (vs SPY -24.5%)
- Monte Carlo median CAGR: 13.5%, 63% probability of beating SPY

## v0.4.0 — ML Integration (2025-03)

### Added
- XGBoost regime agent and RF+KNN direction agent
- Ideas fused from 6 class projects (Regime-Aware Options Engine, LSTM project, Metal Futures RF+KNN, Options Framework, ATPM Portfolio Engine, Liquidity-Driven Trading)

### Performance
- CAGR: 17.8%, Sharpe: 0.68, Max DD: -27.4%

## v0.3.0 — Regime-Aware Momentum (2025-03)

### Changed
- Raw momentum during RISK_ON regimes (fixed structural bond overweight)
- Vol scaling disabled during bullish regimes

### Performance
- CAGR: 10.1%, Sharpe: 0.64, Max DD: -21.3%

## v0.1.0 — Initial Prototype (2025-03)

### Added
- 6-agent architecture (Research, Regime, Risk, Allocation, PM, Review)
- 23-instrument universe across 7 asset classes
- Monthly rebalancing with transaction costs
- Backtesting engine

### Performance
- CAGR: 7.4%, Sharpe: 0.75, Max DD: -14.2%
