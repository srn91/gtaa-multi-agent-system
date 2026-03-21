"""
Regime Agent — Determines the current macro regime.
Uses SPY trend, VIX levels, market breadth, and cross-asset signals.
Outputs: RISK_ON, RISK_OFF, or CRISIS.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Any, Dict
from enum import Enum

from agents.base_agent import BaseAgent, Signal


class Regime(str, Enum):
    RISK_ON = "RISK_ON"
    RISK_OFF = "RISK_OFF"
    CRISIS = "CRISIS"


class RegimeAgent(BaseAgent):
    """
    Determines macro regime from multiple indicators.
    Uses a scoring system — not a single indicator.
    """

    def __init__(
        self,
        sma_short: int = 50,
        sma_long: int = 200,
        vix_risk_off: float = 25.0,
        vix_crisis: float = 35.0,
    ):
        super().__init__("Regime")
        self.sma_short = sma_short
        self.sma_long = sma_long
        self.vix_risk_off = vix_risk_off
        self.vix_crisis = vix_crisis

    def _score_trend(self, prices: pd.DataFrame, date: pd.Timestamp, ticker: str = "SPY") -> float:
        """
        Score SPY trend: +1 for strong uptrend, -1 for strong downtrend.
        Uses dual SMA crossover + price vs SMA.
        """
        if ticker not in prices.columns:
            return 0.0

        loc = prices.index.get_loc(date)
        series = prices[ticker].iloc[:loc + 1]

        if len(series) < self.sma_long:
            return 0.0

        sma_s = series.iloc[-self.sma_short:].mean()
        sma_l = series.iloc[-self.sma_long:].mean()
        current = series.iloc[-1]

        score = 0.0

        # Price above long SMA = bullish
        if current > sma_l:
            score += 0.5
        else:
            score -= 0.5

        # Short SMA above long SMA (golden cross territory)
        if sma_s > sma_l:
            score += 0.5
        else:
            score -= 0.5

        return np.clip(score, -1.0, 1.0)

    def _score_vix(self, regime_data: pd.DataFrame, date: pd.Timestamp) -> float:
        """
        Score VIX level: low VIX = risk-on, high = risk-off, extreme = crisis.
        Returns -1.0 to +1.0
        """
        vix_col = None
        for col in regime_data.columns:
            if "VIX" in str(col).upper():
                vix_col = col
                break

        if vix_col is None or date not in regime_data.index:
            return 0.0

        loc = regime_data.index.get_loc(date)
        vix = regime_data[vix_col].iloc[loc]

        if pd.isna(vix):
            # Backfill
            recent = regime_data[vix_col].iloc[max(0, loc - 5):loc + 1].dropna()
            if len(recent) == 0:
                return 0.0
            vix = recent.iloc[-1]

        if vix >= self.vix_crisis:
            return -1.0
        elif vix >= self.vix_risk_off:
            return -0.5
        elif vix <= 15.0:
            return 1.0
        elif vix <= 20.0:
            return 0.5
        else:
            return 0.0

    def _score_breadth(self, prices: pd.DataFrame, date: pd.Timestamp,
                       equity_tickers: list) -> float:
        """
        Market breadth: what % of equity ETFs are above their 200d SMA?
        High breadth = risk-on.
        """
        loc = prices.index.get_loc(date)
        above = 0
        total = 0

        for ticker in equity_tickers:
            if ticker not in prices.columns:
                continue
            series = prices[ticker].iloc[:loc + 1]
            if len(series) < 200:
                continue
            sma200 = series.iloc[-200:].mean()
            total += 1
            if series.iloc[-1] > sma200:
                above += 1

        if total == 0:
            return 0.0

        breadth = above / total
        # Map 0-1 to -1 to +1
        return (breadth - 0.5) * 2.0

    def _score_momentum_regime(self, prices: pd.DataFrame, date: pd.Timestamp) -> float:
        """
        Are risk assets (SPY, QQQ) outperforming safe havens (TLT, GLD)?
        Positive = risk-on momentum.
        """
        loc = prices.index.get_loc(date)
        lookback = 63  # 3 months

        risk_tickers = ["SPY", "QQQ"]
        safe_tickers = ["TLT", "GLD"]

        def _ret(ticker):
            if ticker not in prices.columns:
                return 0.0
            s = prices[ticker].iloc[:loc + 1]
            if len(s) < lookback:
                return 0.0
            return s.iloc[-1] / s.iloc[-lookback] - 1.0

        risk_ret = np.mean([_ret(t) for t in risk_tickers])
        safe_ret = np.mean([_ret(t) for t in safe_tickers])

        spread = risk_ret - safe_ret
        return np.clip(spread * 5.0, -1.0, 1.0)  # scale to [-1, 1]

    def analyze(self, date: pd.Timestamp, prices: pd.DataFrame,
                returns: pd.DataFrame, context: Dict[str, Any]) -> Signal:
        """
        Combine all regime indicators into a final regime call.
        """
        regime_data = context.get("regime_data", pd.DataFrame())
        universe = context.get("universe", {})

        equity_tickers = [
            t for t, info in universe.items()
            if info.get("class", "").startswith("equity") and t in prices.columns
        ]

        # Score each indicator
        trend_score = self._score_trend(prices, date)
        vix_score = self._score_vix(regime_data, date) if len(regime_data) > 0 else 0.0
        breadth_score = self._score_breadth(prices, date, equity_tickers)
        mom_score = self._score_momentum_regime(prices, date)

        # Weighted composite
        weights = {"trend": 0.30, "vix": 0.25, "breadth": 0.25, "momentum": 0.20}
        composite = (
            trend_score * weights["trend"]
            + vix_score * weights["vix"]
            + breadth_score * weights["breadth"]
            + mom_score * weights["momentum"]
        )

        # Map composite to regime
        if composite <= -0.4:
            regime = Regime.CRISIS
        elif composite <= -0.1:
            regime = Regime.RISK_OFF
        else:
            regime = Regime.RISK_ON

        # Confidence: how decisive is the signal?
        confidence = min(abs(composite) + 0.3, 0.95)

        reasoning = (
            f"Regime: {regime.value}. "
            f"Composite={composite:.2f} "
            f"(trend={trend_score:.2f}, vix={vix_score:.2f}, "
            f"breadth={breadth_score:.2f}, momentum={mom_score:.2f}). "
        )

        signal = Signal(
            agent_name=self.name,
            timestamp=datetime.now(),
            signal_type="regime",
            data={
                "regime": regime.value,
                "composite_score": round(composite, 4),
                "scores": {
                    "trend": round(trend_score, 4),
                    "vix": round(vix_score, 4),
                    "breadth": round(breadth_score, 4),
                    "momentum": round(mom_score, 4),
                },
                "weights": weights,
            },
            confidence=round(confidence, 4),
            reasoning=reasoning,
        )

        self._last_signal = signal
        self._log_audit("regime_call", {
            "date": str(date.date()),
            "regime": regime.value,
            "composite": round(composite, 4),
        })

        return signal
