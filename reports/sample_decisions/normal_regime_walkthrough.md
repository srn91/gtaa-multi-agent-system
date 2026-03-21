# Sample Decision: NORMAL Regime — 2024-06-28

> This is an actual decision trace showing how all 8 agents collaborate during a typical bull market rebalance.

## Market Context

- **S&P 500**: Steady uptrend, 74% of universe above 200-day SMA
- **VIX**: 16.2 (low — calm market)
- **Recent momentum**: Tech and growth sectors leading

---

## Step 1 — Research Agent

**Role:** Multi-timeframe momentum scanner

| Rank | Ticker | Raw Momentum | Above Trend? |
|------|--------|-------------|-------------|
| 1 | XLK | +18.4% | Yes |
| 2 | QQQ | +16.2% | Yes |
| 3 | SPY | +12.1% | Yes |
| 4 | XLY | +10.8% | Yes |
| 5 | BTC-USD | +9.3% | Yes |

**Bottom 5:** TLT, IEF, VNQI, EWJ, SHY (bonds and international underperforming)

**Confidence:** 0.72 — "74% of universe above 200d SMA. Tech and growth leading."

---

## Step 2 — ML Regime Agent (XGBoost)

**Role:** Classify market into 5 regimes using VIX, realized vol, momentum, breadth

| Regime | Probability |
|--------|------------|
| PANIC | 2% |
| HIGH_FEAR | 5% |
| ELEVATED | 12% |
| **NORMAL** | **55%** |
| COMPLACENT | 26% |

**Decision:** NORMAL (confidence 78%)
**Action:** MODERATE — deploy 70%+ to risk assets, use sector rotation
**VaR multiplier:** 2.5x
**Bull ratio:** 0.64 (64% of recent days were up days)

---

## Step 3 — ML Direction Agent (RF + KNN Ensemble)

**Role:** Predict direction for each asset independently

| Ticker | RF Prediction | KNN Prediction | Agree? | Final | Confidence |
|--------|--------------|----------------|--------|-------|------------|
| XLK | Bullish | Bullish | **Yes** | Bullish | 0.82 |
| QQQ | Bullish | Bullish | **Yes** | Bullish | 0.79 |
| SPY | Bullish | Bullish | **Yes** | Bullish | 0.71 |
| BTC-USD | Bullish | Neutral | No | Bullish (KNN) | 0.61 |
| TLT | Neutral | Bearish | No | Bearish (KNN) | 0.68 |

**Models trained:** 15 out of 31 assets had sufficient history

---

## Step 4 — Allocation Agent

**Role:** Blend momentum rankings + ML direction + regime tilts into target weights

| Ticker | Momentum Score | ML Boost | Regime Tilt | Final Weight |
|--------|---------------|----------|-------------|-------------|
| XLK | 0.84 | +0.12 (sector bonus) | +15% (sector rotation) | **22%** |
| QQQ | 0.79 | +0.09 | — | **19%** |
| SPY | 0.68 | +0.07 | — | **16%** |
| XLY | 0.62 | +0.08 (sector bonus) | +15% | **14%** |
| BTC-USD | 0.55 | +0.06 | — | **12%** |
| VNQ | 0.44 | +0.04 | — | **9%** |
| GLD | 0.38 | +0.02 | — | **8%** |

**Risk asset total:** 92% (above 70% floor for NORMAL regime)
**Cash:** 0% (below 5% cap)

---

## Step 5 — Risk Agent

**Role:** Enforce position limits, check correlations, run circuit breakers

- Max position cap: 25% → **all positions pass**
- Max asset class: 60% → equities at 52% → **pass**
- Crypto: 12% → under 60% cap → **pass**
- Portfolio drawdown: -3% → no circuit breaker triggered
- Correlation: XLK/QQQ at 0.89 → **warning logged** (high but under 0.95 veto threshold)

**Verdict:** ✅ Approved — no veto, no modifications needed

---

## Step 6 — PM Agent

**Role:** Final consensus gate

- Research confidence: 0.72 ✓
- Regime confidence: 0.78 ✓
- Risk vetoed: No ✓
- Turnover: 12% (above 3% rebalance threshold) ✓
- **Conviction score: 0.84**
- **Consensus score: 0.78**

**Decision:** ✅ **EXECUTE REBALANCE**

---

## Step 7 — Review Agent

**Role:** Log the trade, compute attribution after hold period

- Portfolio Sharpe YTD: 1.04
- Max drawdown YTD: -8%
- Flags: None
- Next review: After next rebalance period

---

## Final Portfolio

| Ticker | Weight | Asset Class |
|--------|--------|-------------|
| XLK | 22% | US Equity (Tech) |
| QQQ | 19% | US Equity (Growth) |
| SPY | 16% | US Equity (Broad) |
| XLY | 14% | US Equity (Consumer) |
| BTC-USD | 12% | Crypto |
| VNQ | 9% | Real Estate |
| GLD | 8% | Commodity |

**Total risk assets:** 92% | **Defensive:** 8% | **Cash:** 0%
