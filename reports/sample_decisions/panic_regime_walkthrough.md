# Sample Decision: PANIC Regime — 2022-06-30

> This trace shows the system automatically rotating to defensive positioning when the XGBoost regime classifier detects panic conditions.

## Market Context

- **S&P 500**: Down 20% YTD, in confirmed bear market
- **VIX**: 31.4 (elevated fear)
- **Only 29% of universe above 200d SMA**

---

## Step 2 — ML Regime Agent

| Regime | Probability |
|--------|------------|
| **PANIC** | **61%** |
| HIGH_FEAR | 22% |
| ELEVATED | 12% |
| NORMAL | 4% |
| COMPLACENT | 1% |

**Decision:** PANIC with 83% confidence → **NO_TRADE action** → hardcoded defensive allocation

---

## Step 3 — ML Direction Agent

| Ticker | Direction | Confidence | Note |
|--------|-----------|-----------|------|
| SPY | Bearish | 0.74 | RF and KNN both bearish |
| QQQ | Bearish | 0.81 | Strong consensus |
| BTC-USD | Bearish | 0.72 | Crypto selling off with equities |
| GLD | Bullish | 0.56 | Mild safe-haven demand |
| USO | Bullish | 0.63 | Energy decoupled from equity rout |

---

## Step 4 — Allocation Agent (PANIC override)

During PANIC, momentum-based selection is **bypassed entirely**. The system deploys a hardcoded defensive mix:

| Ticker | Weight | Rationale |
|--------|--------|-----------|
| GLD | 30% | Safe haven — gold rallies during equity stress |
| TLT | 20% | Long treasuries — flight to quality |
| XLP | 15% | Consumer staples — defensive equity exposure |
| XLV | 15% | Healthcare — non-cyclical, maintains equity exposure |
| IEF | 10% | Intermediate bonds — low vol anchor |
| BIL | 10% | Cash — dry powder for re-entry |

**Key design choice:** The system does NOT go 100% cash. It keeps 30% in defensive equities (XLP + XLV) to avoid missing the recovery inflection.

---

## Step 5 — Risk Agent

- Portfolio drawdown: -14% → approaching circuit breaker at -15%
- **Warning:** "If drawdown exceeds -15%, will move to 50% cash"
- All positions within caps

**Verdict:** ✅ Approved

---

## Step 6 — PM Agent

**Decision:** ✅ **EXECUTE REBALANCE** — "Regime PANIC with high confidence. Defensive rotation mandatory."

**Conviction:** 0.71 — lower than normal because regime uncertainty is elevated

---

## Outcome

This defensive positioning protected the portfolio during the continued selloff in H2 2022. When the regime later shifted back to ELEVATED → NORMAL, the system automatically re-entered risk assets.
