"""
Allocation Agent — Converts momentum rankings + regime into target weights.
Blends momentum-based allocation with inverse-volatility (risk parity lite).
Regime tilts the allocation toward/away from risk assets.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Any, Dict, List

from agents.base_agent import BaseAgent, Signal
from config.settings import AllocationConfig


class AllocationAgent(BaseAgent):
    """
    Takes momentum rankings + regime signal → produces target portfolio weights.
    """

    def __init__(self, config: AllocationConfig = None):
        super().__init__("Allocation")
        self.config = config or AllocationConfig()

    def _momentum_weights(self, rankings: List[Dict], trend_flags: Dict[str, bool],
                           top_n: int, regime: str = "RISK_ON") -> Dict[str, float]:
        """
        Assign weights based on momentum.
        RISK_ON: Use raw momentum (favors high-return assets like equities/crypto).
        RISK_OFF/CRISIS: Use vol-adjusted momentum (favors safe havens).
        Only include assets above trend, UNLESS regime is strong RISK_ON.
        """
        if regime == "RISK_ON":
            # Sort by raw momentum — pick the highest-returning assets
            sorted_rankings = sorted(rankings, key=lambda r: r.get("raw_momentum", 0), reverse=True)
            # Relax trend filter: include assets with positive raw momentum
            eligible = [
                r for r in sorted_rankings
                if r.get("raw_momentum", 0) > 0 or trend_flags.get(r["ticker"], False)
            ]
        else:
            # Defensive: sort by vol-adjusted momentum, strict trend filter
            sorted_rankings = sorted(rankings, key=lambda r: r.get("vol_adj_momentum", 0), reverse=True)
            eligible = [
                r for r in sorted_rankings
                if trend_flags.get(r["ticker"], False)
            ]

        if len(eligible) == 0:
            return {"BIL": 1.0}

        # Take top N
        top = eligible[:top_n]

        # Rank-based weights (inverse rank)
        n = len(top)
        raw_weights = [(n - i) for i in range(n)]
        total = sum(raw_weights)

        weights = {}
        for i, asset in enumerate(top):
            weights[asset["ticker"]] = raw_weights[i] / total

        return weights

    def _inverse_vol_weights(self, returns: pd.DataFrame, date: pd.Timestamp,
                              tickers: List[str], lookback: int = 63) -> Dict[str, float]:
        """
        Inverse volatility weighting (risk parity lite).
        Each asset gets weight proportional to 1/vol.
        """
        loc = returns.index.get_loc(date)
        start = max(0, loc - lookback)
        subset = returns.iloc[start:loc + 1]

        vols = {}
        for t in tickers:
            if t in subset.columns:
                v = subset[t].std() * np.sqrt(252)
                if v > 0.001:
                    vols[t] = v

        if len(vols) == 0:
            return {t: 1.0 / len(tickers) for t in tickers}

        inv_vols = {t: 1.0 / v for t, v in vols.items()}
        total = sum(inv_vols.values())

        return {t: iv / total for t, iv in inv_vols.items()}

    def _apply_regime_tilt(self, weights: Dict[str, float], regime: str,
                            universe: Dict, tilt_strength: float) -> Dict[str, float]:
        """
        Tilt allocation based on regime.
        RISK_ON: overweight equities, underweight bonds
        RISK_OFF: overweight bonds/gold, underweight equities
        CRISIS: max defensive — bonds, gold, cash
        """
        tilted = dict(weights)

        risk_assets = {"equity_us", "equity_intl", "crypto", "real_estate"}
        safe_assets = {"fixed_income", "commodity", "cash"}

        if regime == "RISK_OFF":
            for t in tilted:
                ac = universe.get(t, {}).get("class", "")
                if ac in risk_assets:
                    tilted[t] *= (1.0 - tilt_strength)
                elif ac in safe_assets:
                    tilted[t] *= (1.0 + tilt_strength)

        elif regime == "CRISIS":
            for t in tilted:
                ac = universe.get(t, {}).get("class", "")
                if ac in risk_assets:
                    tilted[t] *= (1.0 - tilt_strength * 2)
                elif ac in safe_assets:
                    tilted[t] *= (1.0 + tilt_strength * 2)

        # Re-normalize
        total = sum(tilted.values())
        if total > 0:
            tilted = {t: w / total for t, w in tilted.items()}

        return tilted

    def analyze(self, date: pd.Timestamp, prices: pd.DataFrame,
                returns: pd.DataFrame, context: Dict[str, Any]) -> Signal:
        """
        Produce target allocation weights.
        """
        research_signal = context.get("research_signal", {})
        regime_signal = context.get("regime_signal", {})
        universe = context.get("universe", {})

        rankings = research_signal.get("data", {}).get("rankings", [])
        trend_flags = research_signal.get("data", {}).get("trend_flags", {})
        regime = regime_signal.get("data", {}).get("regime", "RISK_ON")

        # 1. Momentum-based weights (regime-aware)
        mom_weights = self._momentum_weights(
            rankings, trend_flags, self.config.top_n_assets, regime
        )

        # 2. Inverse vol weights for the same tickers
        mom_tickers = list(mom_weights.keys())
        ivol_weights = self._inverse_vol_weights(returns, date, mom_tickers)

        # 3. Blend momentum + risk parity
        blend = self.config.risk_parity_blend
        blended = {}
        all_tickers = set(mom_weights.keys()) | set(ivol_weights.keys())
        for t in all_tickers:
            mw = mom_weights.get(t, 0.0)
            rw = ivol_weights.get(t, 0.0)
            blended[t] = mw * (1.0 - blend) + rw * blend

        # Normalize
        total = sum(blended.values())
        if total > 0:
            blended = {t: w / total for t, w in blended.items()}

        # 4. Apply regime tilt
        tilted = self._apply_regime_tilt(
            blended, regime, universe, self.config.regime_tilt_strength
        )

        # 5. Enforce minimum weight — drop positions below floor
        filtered = {t: w for t, w in tilted.items() if w >= self.config.min_weight}
        if len(filtered) == 0:
            filtered = {"BIL": 1.0}
        total = sum(filtered.values())
        filtered = {t: w / total for t, w in filtered.items()}

        reasoning = (
            f"Regime={regime}. Selected {len(filtered)} assets from top {self.config.top_n_assets}. "
            f"Blend: {(1-blend)*100:.0f}% momentum + {blend*100:.0f}% risk parity. "
            f"Tilt strength={self.config.regime_tilt_strength:.0%}."
        )

        signal = Signal(
            agent_name=self.name,
            timestamp=datetime.now(),
            signal_type="allocation",
            data={
                "target_weights": {t: round(w, 6) for t, w in filtered.items()},
                "momentum_weights": {t: round(w, 4) for t, w in mom_weights.items()},
                "ivol_weights": {t: round(w, 4) for t, w in ivol_weights.items()},
                "regime_used": regime,
                "n_positions": len(filtered),
            },
            confidence=0.75,
            reasoning=reasoning,
        )

        self._last_signal = signal
        self._log_audit("allocation", {
            "date": str(date.date()),
            "regime": regime,
            "n_positions": len(filtered),
        })

        return signal
