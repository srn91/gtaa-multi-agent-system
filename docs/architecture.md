# Architecture Deep Dive

## System Overview

The GTAA Multi-Agent System is a pipeline of 8 specialized agents that communicate through typed `Signal` objects. Each agent receives market data and the outputs of upstream agents, performs its analysis, and emits a Signal containing structured data, a confidence score (0.0–1.0), and human-readable reasoning.

## Signal Protocol

Every agent inherits from `BaseAgent` and produces a `Signal` dataclass:

```python
@dataclass
class Signal:
    agent_name: str          # "Research", "MLRegime", etc.
    timestamp: datetime      # When the signal was generated
    signal_type: str         # "momentum", "regime", "direction", "allocation", etc.
    data: Dict[str, Any]     # Structured payload (agent-specific)
    confidence: float        # 0.0 to 1.0
    reasoning: str           # Human-readable explanation
```

Signals are immutable once created. The PM Agent receives all upstream signals and makes the final execute/hold decision based on consensus.

## Agent Details

### 1. Research Agent (`research_agent.py`)

**Input:** Price history for 31 assets
**Output:** Momentum rankings with raw and vol-adjusted scores, trend flags

Computes multi-timeframe momentum using configurable windows (default: 21d, 63d, 126d, 252d with weights 0.35, 0.30, 0.20, 0.15). Skips the most recent 3 days to avoid short-term mean reversion noise. Reports what percentage of the universe is above its 200-day SMA as a breadth indicator.

### 2. ML Regime Agent (`ml_regime_agent.py`)

**Input:** Price history + VIX series
**Output:** 5-regime classification with per-regime probabilities

Uses XGBoost classifier trained on 13 features:
- VIX level, VIX daily change, VIX spike flag (> 1.3x 20d mean)
- 1d, 5d, 20d returns
- Momentum acceleration (change in 5d momentum)
- 20d realized vol, vol ratio (20d / 60d), vol trend
- Price vs 50d MA
- RSI proxy (% of up days in last 14)
- Breadth (% of universe above 200d SMA)

**LAG-1 Protocol:** All features use data available at market close on day T. The predicted regime is for day T+1. No look-ahead bias.

**Training:** Expanding window, retrains annually. Labels are generated from VIX thresholds (PANIC > 30, HIGH_FEAR > 25, ELEVATED > 20, NORMAL > 15, COMPLACENT ≤ 15).

### 3. ML Direction Agent (`ml_direction_agent.py`)

**Input:** Price history for each asset
**Output:** Per-asset direction prediction with confidence

Runs two models per asset independently:
1. **Random Forest Regressor** — predicts next-period return magnitude, takes the sign
2. **KNN Classifier** — predicts direction directly using standardized features

**Ensemble rule:** When both agree → follow with confidence bonus. When they disagree → default to KNN (empirically more robust).

**Filters applied:**
- ATR filter: skip low-volatility assets (ATR% < 0.5%) where noise dominates
- SMA20 trend filter: penalize bearish predictions on assets in uptrends

### 4. Allocation Agent (`allocation_agent.py`)

**Input:** Momentum rankings, ML direction predictions, regime classification
**Output:** Target portfolio weights

Blends momentum scores with ML direction using dynamic weighting (35% ML, 65% momentum by default). Applies regime-specific rules:
- COMPLACENT: 90%+ risk assets, sector ETF bonus, crypto bonus
- NORMAL: 70%+ risk assets, sector rotation
- ELEVATED: Stay invested with defensive tilt
- HIGH_FEAR/PANIC: Hardcoded defensive allocations

### 5. Risk Agent (`risk_agent.py`)

**Input:** Proposed weights, equity value, regime
**Output:** Approved weights (may modify or veto)

Enforces:
- Position caps (30-35% max per asset)
- Asset class caps (60-70% max per class)
- Correlation monitoring (warns above 0.85, vetos above 0.95)
- Drawdown circuit breakers (-15% → 50% cash, -25% → full exit)
- Vol scaling during defensive regimes

### 6. PM Agent (`pm_agent.py`)

**Input:** All upstream agent signals
**Output:** Execute or hold decision with conviction score

Computes consensus score across all agents. Only executes when:
- Conviction > threshold
- Turnover exceeds minimum rebalance threshold (avoids churning)
- No risk veto active

### 7. Review Agent (`review_agent.py`)

**Input:** Trade log, equity curve
**Output:** Performance attribution, flags

Computes rolling CAGR, Sharpe, Sortino, max drawdown, Calmar. Flags anomalies like Sharpe degradation or unusual drawdowns.

### 8. Rule Regime Agent (`regime_agent.py`)

**Input:** Price history, VIX
**Output:** 3-regime classification (RISK_ON / RISK_OFF / CRISIS)

Simple rule-based fallback using SMA trends, VIX thresholds, and breadth. Used when ML Regime Agent has insufficient training data.

## Data Flow Diagram

```
Market Data (31 assets + VIX)
        │
        ├──────────────────────────┐
        ▼                          ▼
  Research Agent              ML Regime Agent
  (momentum scan)            (XGBoost 5-regime)
        │                          │
        ├──────────┐               │
        ▼          ▼               ▼
  ML Direction   Allocation Agent
  Agent          (blend all signals)
  (RF+KNN)              │
        │                ▼
        └──────► Risk Agent
                 (limits + veto)
                        │
                        ▼
                    PM Agent
                 (consensus gate)
                        │
                        ▼
                  Review Agent
                 (attribution)
```

## Auto-Regressive Risk Targeting (ART)

Borrowed from the Quantitative Options Framework. The system estimates 63-day rolling portfolio volatility and scales leverage to maintain a target (20% annualized). Leverage is clipped between 0.5x and 1.5x.

```
leverage = target_vol / realized_vol
leverage = clip(leverage, 0.5, 1.5)
```

This prevents the portfolio from being over-deployed during high-vol periods and under-deployed during calm markets.
