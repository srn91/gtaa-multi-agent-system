"""
Review Agent — Post-trade attribution and performance review.
Runs after each rebalance period to assess what worked and what didn't.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Any, Dict, List, Optional

from agents.base_agent import BaseAgent, Signal


class ReviewAgent(BaseAgent):
    """
    Post-mortem on portfolio performance.
    Computes attribution, Sharpe, drawdown, and flags problems.
    """

    def __init__(self):
        super().__init__("Review")
        self._equity_curve: List[float] = []
        self._dates: List[pd.Timestamp] = []
        self._trade_log: List[Dict] = []

    def log_trade(self, date: pd.Timestamp, weights_before: Dict, weights_after: Dict,
                  equity_value: float, turnover: float):
        """Record a trade for later review."""
        self._trade_log.append({
            "date": str(date.date()),
            "weights_before": weights_before,
            "weights_after": weights_after,
            "equity": equity_value,
            "turnover": turnover,
        })

    def update_equity(self, date: pd.Timestamp, equity: float):
        """Track equity curve."""
        self._equity_curve.append(equity)
        self._dates.append(date)

    def _compute_stats(self) -> Dict:
        """Compute portfolio statistics from equity curve."""
        if len(self._equity_curve) < 2:
            return {}

        equity = pd.Series(self._equity_curve, index=self._dates)
        returns = equity.pct_change().dropna()

        if len(returns) == 0:
            return {}

        # CAGR
        years = (equity.index[-1] - equity.index[0]).days / 365.25
        if years > 0 and equity.iloc[0] > 0:
            cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1.0 / years) - 1.0
        else:
            cagr = 0.0

        # Sharpe
        annual_ret = returns.mean() * 252
        annual_vol = returns.std() * np.sqrt(252)
        sharpe = annual_ret / annual_vol if annual_vol > 0 else 0.0

        # Sortino
        downside = returns[returns < 0]
        downside_vol = downside.std() * np.sqrt(252) if len(downside) > 0 else 0.001
        sortino = annual_ret / downside_vol

        # Max drawdown
        cummax = equity.cummax()
        drawdown = (equity - cummax) / cummax
        max_dd = drawdown.min()

        # Calmar
        calmar = cagr / abs(max_dd) if max_dd != 0 else 0.0

        # Win rate (monthly)
        monthly = returns.resample("ME").sum()
        win_rate = (monthly > 0).sum() / max(len(monthly), 1)

        # Total trades
        total_trades = len(self._trade_log)

        return {
            "cagr": round(cagr, 4),
            "annual_vol": round(annual_vol, 4),
            "sharpe": round(sharpe, 4),
            "sortino": round(sortino, 4),
            "max_drawdown": round(max_dd, 4),
            "calmar": round(calmar, 4),
            "monthly_win_rate": round(win_rate, 4),
            "total_return": round(equity.iloc[-1] / equity.iloc[0] - 1, 4),
            "total_trades": total_trades,
            "years": round(years, 2),
        }

    def _compute_attribution(self, returns: pd.DataFrame, date: pd.Timestamp,
                              weights: Dict[str, float], lookback: int = 21) -> Dict[str, float]:
        """
        Simple attribution: how much did each position contribute over lookback period.
        """
        loc = returns.index.get_loc(date)
        start = max(0, loc - lookback)
        subset = returns.iloc[start:loc + 1]

        attribution = {}
        for ticker, weight in weights.items():
            if ticker in subset.columns:
                ret = (1 + subset[ticker]).prod() - 1
                attribution[ticker] = round(weight * ret, 6)

        return attribution

    def analyze(self, date: pd.Timestamp, prices: pd.DataFrame,
                returns: pd.DataFrame, context: Dict[str, Any]) -> Signal:
        """
        Produce performance review signal.
        """
        current_weights = context.get("current_weights", {})
        equity_value = context.get("equity_value", 0)

        stats = self._compute_stats()
        attribution = self._compute_attribution(returns, date, current_weights)

        # Identify top contributors and detractors
        sorted_attr = sorted(attribution.items(), key=lambda x: x[1], reverse=True)
        top_contributors = sorted_attr[:3] if len(sorted_attr) >= 3 else sorted_attr
        top_detractors = sorted_attr[-3:] if len(sorted_attr) >= 3 else []

        # Flags
        flags = []
        if stats.get("max_drawdown", 0) < -0.20:
            flags.append(f"WARNING: Max DD {stats['max_drawdown']:.1%} exceeds -20%")
        if stats.get("sharpe", 0) < 0.5 and stats.get("years", 0) > 1:
            flags.append(f"WARNING: Sharpe {stats['sharpe']:.2f} below 0.5")
        if stats.get("monthly_win_rate", 0) < 0.45 and stats.get("years", 0) > 1:
            flags.append(f"WARNING: Monthly win rate {stats['monthly_win_rate']:.0%} below 45%")

        reasoning = (
            f"Portfolio review: CAGR={stats.get('cagr', 0):.1%}, "
            f"Sharpe={stats.get('sharpe', 0):.2f}, "
            f"MaxDD={stats.get('max_drawdown', 0):.1%}. "
            f"{'Flags: ' + '; '.join(flags) if flags else 'No warnings.'}"
        )

        signal = Signal(
            agent_name=self.name,
            timestamp=datetime.now(),
            signal_type="review",
            data={
                "stats": stats,
                "attribution": attribution,
                "top_contributors": [{"ticker": t, "contribution": c} for t, c in top_contributors],
                "top_detractors": [{"ticker": t, "contribution": c} for t, c in top_detractors],
                "flags": flags,
                "trade_count": len(self._trade_log),
            },
            confidence=0.9,
            reasoning=reasoning,
        )

        self._last_signal = signal
        self._log_audit("review", {"date": str(date.date()), "stats": stats})

        return signal

    def get_equity_curve(self) -> pd.Series:
        """Return equity curve as pandas Series."""
        if self._equity_curve and self._dates:
            return pd.Series(self._equity_curve, index=self._dates)
        return pd.Series(dtype=float)

    def get_trade_log(self) -> List[Dict]:
        return self._trade_log
