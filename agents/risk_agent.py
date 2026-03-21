"""
Risk Agent — Position sizing, correlation monitoring, veto power.
Has circuit breakers for drawdown protection.
Can VETO any trade that violates risk limits.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from agents.base_agent import BaseAgent, Signal
from config.settings import RiskConfig


class RiskAgent(BaseAgent):
    """
    Enforces risk limits on the portfolio.
    Inputs: proposed allocation + current portfolio state.
    Outputs: approved allocation (possibly modified) + risk metrics.
    """

    def __init__(self, config: RiskConfig = None):
        super().__init__("Risk")
        self.config = config or RiskConfig()
        self._peak_equity: float = 0.0
        self._current_drawdown: float = 0.0

    def _compute_correlation_matrix(self, returns: pd.DataFrame, date: pd.Timestamp) -> pd.DataFrame:
        """Rolling correlation matrix."""
        loc = returns.index.get_loc(date)
        window = self.config.correlation_lookback
        start = max(0, loc - window)
        subset = returns.iloc[start:loc + 1].dropna(axis=1, how="all")
        return subset.corr()

    def _check_concentration(self, weights: Dict[str, float],
                              universe: Dict) -> Tuple[Dict[str, float], List[str]]:
        """
        Enforce position and asset-class concentration limits.
        Returns adjusted weights and list of violations.
        """
        violations = []
        adjusted = dict(weights)

        # Single position cap
        for ticker, w in adjusted.items():
            if w > self.config.max_position_pct:
                violations.append(f"{ticker}: {w:.1%} > max {self.config.max_position_pct:.0%}")
                adjusted[ticker] = self.config.max_position_pct

        # Asset class cap
        class_weights: Dict[str, float] = {}
        for ticker, w in adjusted.items():
            ac = universe.get(ticker, {}).get("class", "unknown")
            class_weights[ac] = class_weights.get(ac, 0.0) + w

        for ac, cw in class_weights.items():
            if cw > self.config.max_asset_class_pct and ac != "cash":
                violations.append(f"Class {ac}: {cw:.1%} > max {self.config.max_asset_class_pct:.0%}")
                # Proportionally reduce all positions in this class
                scale = self.config.max_asset_class_pct / cw
                for ticker, w in adjusted.items():
                    if universe.get(ticker, {}).get("class") == ac:
                        adjusted[ticker] = w * scale

        return adjusted, violations

    def _check_correlation_clusters(self, weights: Dict[str, float],
                                      corr_matrix: pd.DataFrame) -> List[str]:
        """Flag highly correlated pairs in the portfolio."""
        warnings = []
        tickers = [t for t in weights if t in corr_matrix.columns and weights[t] > 0.01]

        for i, t1 in enumerate(tickers):
            for t2 in tickers[i + 1:]:
                if t1 in corr_matrix.columns and t2 in corr_matrix.columns:
                    c = corr_matrix.loc[t1, t2]
                    if abs(c) > self.config.max_correlated_exposure:
                        combined = weights.get(t1, 0) + weights.get(t2, 0)
                        warnings.append(
                            f"{t1}/{t2} corr={c:.2f}, combined weight={combined:.1%}"
                        )
        return warnings

    def _apply_vol_scaling(self, weights: Dict[str, float], returns: pd.DataFrame,
                            date: pd.Timestamp) -> Dict[str, float]:
        """
        Scale portfolio to target volatility.
        Only scale when predicted vol exceeds target by >30% (buffer zone).
        This prevents constant dampening that kills returns.
        """
        loc = returns.index.get_loc(date)
        window = self.config.correlation_lookback
        start = max(0, loc - window)
        subset = returns.iloc[start:loc + 1]

        tickers = [t for t in weights if t in subset.columns and weights[t] > 0.001]
        if len(tickers) < 2:
            return weights

        w_vec = np.array([weights.get(t, 0) for t in tickers])
        cov = subset[tickers].cov() * 252  # annualized
        port_var = w_vec @ cov.values @ w_vec
        port_vol = np.sqrt(max(port_var, 0))

        # Only scale if vol exceeds target by >30% — prevent constant dampening
        vol_buffer = self.config.vol_target * 1.3
        if port_vol > vol_buffer and port_vol > 0:
            scale = self.config.vol_target / port_vol
            scale = max(scale, 0.5)  # don't scale below 50%
            return {t: w * scale if t in tickers else w for t, w in weights.items()}

        return weights

    def _check_drawdown(self, equity_value: float) -> Tuple[str, float]:
        """
        Check drawdown from peak. Returns action and current DD.
        """
        self._peak_equity = max(self._peak_equity, equity_value)
        if self._peak_equity > 0:
            dd = (equity_value - self._peak_equity) / self._peak_equity
        else:
            dd = 0.0

        self._current_drawdown = dd

        if dd <= self.config.drawdown_full_exit:
            return "FULL_EXIT", dd
        elif dd <= self.config.drawdown_circuit_breaker:
            return "HALF_CASH", dd
        else:
            return "NORMAL", dd

    def analyze(self, date: pd.Timestamp, prices: pd.DataFrame,
                returns: pd.DataFrame, context: Dict[str, Any]) -> Signal:
        """
        Review proposed allocation, apply risk limits, return approved allocation.
        """
        proposed_weights = context.get("proposed_weights", {})
        universe = context.get("universe", {})
        equity_value = context.get("equity_value", 100000.0)
        regime = context.get("regime", "RISK_ON")

        # 1. Drawdown check
        dd_action, dd_level = self._check_drawdown(equity_value)

        # 2. Concentration limits
        adjusted, violations = self._check_concentration(proposed_weights, universe)

        # 3. Correlation check
        corr_matrix = self._compute_correlation_matrix(returns, date)
        corr_warnings = self._check_correlation_clusters(adjusted, corr_matrix)

        # 4. Vol scaling — ONLY during RISK_OFF/CRISIS, not during RISK_ON
        if regime != "RISK_ON":
            adjusted = self._apply_vol_scaling(adjusted, returns, date)

        # 5. Apply drawdown overrides
        if dd_action == "FULL_EXIT":
            # Move everything to cash
            cash_ticker = "BIL"
            adjusted = {cash_ticker: 1.0}
            violations.append(f"CIRCUIT BREAKER: DD={dd_level:.1%} — full exit to cash")
        elif dd_action == "HALF_CASH":
            # Cut all positions in half, add to cash
            cash_added = sum(adjusted.values()) * 0.5
            adjusted = {t: w * 0.5 for t, w in adjusted.items()}
            cash_ticker = "BIL"
            adjusted[cash_ticker] = adjusted.get(cash_ticker, 0) + cash_added
            violations.append(f"CIRCUIT BREAKER: DD={dd_level:.1%} — 50% to cash")

        # 6. Ensure minimum cash
        total_risky = sum(w for t, w in adjusted.items() if universe.get(t, {}).get("class") != "cash")
        if total_risky > (1.0 - self.config.min_cash_pct):
            scale = (1.0 - self.config.min_cash_pct) / max(total_risky, 0.001)
            for t in adjusted:
                if universe.get(t, {}).get("class") != "cash":
                    adjusted[t] *= scale
            adjusted["BIL"] = adjusted.get("BIL", 0) + self.config.min_cash_pct

        # Normalize and enforce position cap
        total = sum(adjusted.values())
        if total > 0:
            adjusted = {t: w / total for t, w in adjusted.items()}

        # Cap positions iteratively — excess goes to BIL (cash)
        for _ in range(5):
            overflow = 0.0
            for t in list(adjusted.keys()):
                cap = self.config.max_position_pct
                if universe.get(t, {}).get("class") == "cash":
                    continue  # no cap on cash
                if adjusted[t] > cap:
                    overflow += adjusted[t] - cap
                    adjusted[t] = cap

            if overflow <= 0.001:
                break
            # Push overflow to cash
            adjusted["BIL"] = adjusted.get("BIL", 0) + overflow

        # Final normalize
        total = sum(adjusted.values())
        if total > 0:
            adjusted = {t: w / total for t, w in adjusted.items()}

        # Veto decision
        vetoed = dd_action in ("FULL_EXIT", "HALF_CASH")
        confidence = 0.9 if not vetoed else 0.95

        reasoning = (
            f"Risk check: DD={dd_level:.1%}, action={dd_action}. "
            f"Violations: {len(violations)}. Corr warnings: {len(corr_warnings)}. "
            f"{'VETO APPLIED. ' if vetoed else ''}"
            f"Final positions: {len([w for w in adjusted.values() if w > 0.01])}."
        )

        signal = Signal(
            agent_name=self.name,
            timestamp=datetime.now(),
            signal_type="risk",
            data={
                "approved_weights": {t: round(w, 6) for t, w in adjusted.items() if w > 0.001},
                "vetoed": vetoed,
                "drawdown_action": dd_action,
                "drawdown_level": round(dd_level, 4),
                "violations": violations,
                "correlation_warnings": corr_warnings,
                "peak_equity": round(self._peak_equity, 2),
            },
            confidence=confidence,
            reasoning=reasoning,
        )

        self._last_signal = signal
        self._log_audit("risk_review", {
            "date": str(date.date()),
            "dd": round(dd_level, 4),
            "action": dd_action,
            "n_violations": len(violations),
        })

        return signal
