"""
Research Agent — Scans all asset classes for momentum signals.
Uses multi-timeframe volatility-adjusted momentum.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Any, Dict, List

from agents.base_agent import BaseAgent, Signal


class ResearchAgent(BaseAgent):
    """
    Scans the full universe for momentum.
    Produces ranked list of assets by composite momentum score.
    """

    def __init__(
        self,
        lookback_windows: List[int] = None,
        lookback_weights: List[float] = None,
        skip_recent: int = 5,
        vol_lookback: int = 63,
    ):
        super().__init__("Research")
        self.lookback_windows = lookback_windows or [21, 63, 126, 252]
        self.lookback_weights = lookback_weights or [0.35, 0.30, 0.20, 0.15]
        self.skip_recent = skip_recent
        self.vol_lookback = vol_lookback

    def _compute_momentum(self, prices: pd.DataFrame, date: pd.Timestamp) -> pd.DataFrame:
        """
        Compute volatility-adjusted momentum for each ticker at a given date.
        Returns DataFrame with columns: ticker, raw_mom, vol, vol_adj_mom, rank
        """
        loc = prices.index.get_loc(date)
        results = []

        for ticker in prices.columns:
            series = prices[ticker].iloc[:loc + 1]
            if series.isna().sum() / len(series) > 0.3:
                continue  # skip if too much missing data

            # Skip recent days to avoid short-term reversal
            if self.skip_recent > 0:
                series = series.iloc[:-self.skip_recent]

            if len(series) < max(self.lookback_windows):
                continue

            # Multi-timeframe momentum (simple return over each window)
            mom_scores = []
            for window, weight in zip(self.lookback_windows, self.lookback_weights):
                if len(series) >= window:
                    ret = series.iloc[-1] / series.iloc[-window] - 1.0
                    mom_scores.append(ret * weight)
                else:
                    mom_scores.append(0.0)

            raw_mom = sum(mom_scores)

            # Volatility (annualized)
            daily_ret = series.pct_change().dropna()
            vol = daily_ret.iloc[-self.vol_lookback:].std() * np.sqrt(252) if len(daily_ret) >= self.vol_lookback else daily_ret.std() * np.sqrt(252)

            if vol < 0.001:
                vol = 0.001  # floor to avoid division by zero

            vol_adj_mom = raw_mom / vol

            results.append({
                "ticker": ticker,
                "raw_momentum": round(raw_mom, 6),
                "volatility": round(vol, 6),
                "vol_adj_momentum": round(vol_adj_mom, 6),
            })

        df = pd.DataFrame(results)
        if len(df) > 0:
            df["rank"] = df["vol_adj_momentum"].rank(ascending=False).astype(int)
            df = df.sort_values("rank")

        return df

    def _compute_trend_flags(self, prices: pd.DataFrame, date: pd.Timestamp) -> Dict[str, bool]:
        """Check absolute momentum: is each asset above its 200-day SMA?"""
        loc = prices.index.get_loc(date)
        flags = {}
        for ticker in prices.columns:
            series = prices[ticker].iloc[:loc + 1]
            if len(series) >= 200:
                sma200 = series.iloc[-200:].mean()
                flags[ticker] = bool(series.iloc[-1] > sma200)
            else:
                flags[ticker] = True  # insufficient data, assume ok
        return flags

    def analyze(self, date: pd.Timestamp, prices: pd.DataFrame,
                returns: pd.DataFrame, context: Dict[str, Any]) -> Signal:
        """
        Produce momentum rankings and trend flags for the universe.
        Dual ranking: raw momentum (for RISK_ON) and vol-adjusted (for defensive).
        """
        momentum_df = self._compute_momentum(prices, date)
        trend_flags = self._compute_trend_flags(prices, date)

        # Add trend flag to momentum DataFrame
        if len(momentum_df) > 0:
            momentum_df["above_trend"] = momentum_df["ticker"].map(
                lambda t: trend_flags.get(t, False)
            )
            # Add raw momentum rank (not vol-adjusted) — for RISK_ON selection
            momentum_df["raw_rank"] = momentum_df["raw_momentum"].rank(ascending=False).astype(int)

        # Top and bottom performers
        top_5 = momentum_df.head(5)["ticker"].tolist() if len(momentum_df) >= 5 else momentum_df["ticker"].tolist()
        bottom_5 = momentum_df.tail(5)["ticker"].tolist() if len(momentum_df) >= 5 else []

        # Count how many assets are in uptrend
        uptrend_pct = sum(trend_flags.values()) / max(len(trend_flags), 1)

        reasoning = (
            f"Scanned {len(momentum_df)} assets. "
            f"Top momentum: {', '.join(top_5)}. "
            f"Bottom: {', '.join(bottom_5)}. "
            f"{uptrend_pct:.0%} of universe above 200d SMA."
        )

        signal = Signal(
            agent_name=self.name,
            timestamp=datetime.now(),
            signal_type="momentum",
            data={
                "rankings": momentum_df.to_dict("records") if len(momentum_df) > 0 else [],
                "trend_flags": trend_flags,
                "uptrend_pct": round(uptrend_pct, 4),
                "top_5": top_5,
                "bottom_5": bottom_5,
            },
            confidence=min(0.5 + uptrend_pct * 0.3, 0.95),
            reasoning=reasoning,
        )

        self._last_signal = signal
        self._log_audit("momentum_scan", {"date": str(date.date()), "n_assets": len(momentum_df)})

        return signal
