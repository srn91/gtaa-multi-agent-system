"""
PM Agent — Portfolio Manager. Makes final decisions.
Requires consensus from Research, Regime, Risk, and Allocation agents.
Scores conviction and decides whether to execute rebalance.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Any, Dict, List, Optional

from agents.base_agent import BaseAgent, Signal


class PMAgent(BaseAgent):
    """
    Final decision maker. Synthesizes all agent signals.
    Only executes when there's sufficient consensus.
    """

    def __init__(
        self,
        min_consensus_confidence: float = 0.5,
        rebalance_threshold: float = 0.05,
    ):
        super().__init__("PM")
        self.min_consensus_confidence = min_consensus_confidence
        self.rebalance_threshold = rebalance_threshold
        self._current_weights: Dict[str, float] = {}

    def _compute_consensus(self, signals: Dict[str, Signal]) -> float:
        """
        Average confidence across all agents.
        Each agent must have confidence > threshold.
        """
        confidences = [s.confidence for s in signals.values() if s is not None]
        if len(confidences) == 0:
            return 0.0
        return np.mean(confidences)

    def _compute_turnover(self, current: Dict[str, float],
                           target: Dict[str, float]) -> float:
        """Compute one-way turnover between current and target portfolios."""
        all_tickers = set(current.keys()) | set(target.keys())
        turnover = sum(
            abs(current.get(t, 0) - target.get(t, 0))
            for t in all_tickers
        ) / 2.0
        return turnover

    def _score_conviction(self, research_signal: Signal, regime_signal: Signal,
                           allocation_signal: Signal, risk_signal: Signal) -> float:
        """
        Conviction score 0-1. Higher = more confident to act.
        Factors: agent consensus, regime clarity, momentum strength.
        """
        scores = []

        # Research conviction: how strong is top momentum?
        rankings = research_signal.data.get("rankings", [])
        if len(rankings) >= 2:
            top_mom = rankings[0].get("vol_adj_momentum", 0)
            scores.append(min(abs(top_mom) / 2.0, 1.0))

        # Regime clarity: how decisive is the regime score?
        regime_composite = abs(regime_signal.data.get("composite_score", 0))
        scores.append(min(regime_composite / 0.5, 1.0))

        # Risk agent confidence
        scores.append(risk_signal.confidence)

        # Not vetoed bonus
        if not risk_signal.data.get("vetoed", False):
            scores.append(0.8)
        else:
            scores.append(0.3)

        return np.mean(scores) if scores else 0.5

    def analyze(self, date: pd.Timestamp, prices: pd.DataFrame,
                returns: pd.DataFrame, context: Dict[str, Any]) -> Signal:
        """
        Synthesize all agent signals into final portfolio decision.
        """
        research_signal = context.get("research_signal_obj")
        regime_signal = context.get("regime_signal_obj")
        allocation_signal = context.get("allocation_signal_obj")
        risk_signal = context.get("risk_signal_obj")

        # Get risk-approved weights
        if risk_signal and risk_signal.data.get("approved_weights"):
            target_weights = risk_signal.data["approved_weights"]
        elif allocation_signal:
            target_weights = allocation_signal.data.get("target_weights", {})
        else:
            target_weights = self._current_weights

        # Compute consensus
        agent_signals = {
            "research": research_signal,
            "regime": regime_signal,
            "allocation": allocation_signal,
            "risk": risk_signal,
        }
        consensus = self._compute_consensus(
            {k: v for k, v in agent_signals.items() if v is not None}
        )

        # Compute turnover
        turnover = self._compute_turnover(self._current_weights, target_weights)

        # Conviction scoring
        conviction = 0.5
        if all(v is not None for v in agent_signals.values()):
            conviction = self._score_conviction(
                research_signal, regime_signal, allocation_signal, risk_signal
            )

        # Decision logic
        execute_rebalance = False
        reasoning_parts = []

        if consensus < self.min_consensus_confidence:
            reasoning_parts.append(f"Low consensus ({consensus:.2f}) — HOLD current portfolio.")
        elif turnover < self.rebalance_threshold:
            reasoning_parts.append(f"Turnover {turnover:.1%} below threshold — no rebalance needed.")
        else:
            execute_rebalance = True
            reasoning_parts.append(f"Rebalance approved. Turnover={turnover:.1%}, conviction={conviction:.2f}.")

        # Was there a veto?
        vetoed = risk_signal.data.get("vetoed", False) if risk_signal else False
        if vetoed:
            execute_rebalance = True  # Force execution of risk override
            reasoning_parts.append("Risk agent VETO active — executing risk override.")

        # Build final decision
        if execute_rebalance:
            final_weights = target_weights
            self._current_weights = dict(target_weights)
        else:
            final_weights = self._current_weights

        regime = "UNKNOWN"
        if regime_signal:
            regime = regime_signal.data.get("regime", "UNKNOWN")

        reasoning = " ".join(reasoning_parts)

        signal = Signal(
            agent_name=self.name,
            timestamp=datetime.now(),
            signal_type="decision",
            data={
                "final_weights": {t: round(w, 6) for t, w in final_weights.items() if w > 0.001},
                "execute_rebalance": execute_rebalance,
                "turnover": round(turnover, 4),
                "conviction": round(conviction, 4),
                "consensus": round(consensus, 4),
                "regime": regime,
                "vetoed": vetoed,
            },
            confidence=conviction,
            reasoning=reasoning,
        )

        self._last_signal = signal
        self._log_audit("pm_decision", {
            "date": str(date.date()),
            "rebalance": execute_rebalance,
            "turnover": round(turnover, 4),
            "conviction": round(conviction, 4),
            "n_positions": len(final_weights),
        })

        return signal

    def set_current_weights(self, weights: Dict[str, float]):
        """Set current portfolio weights (used at init and after trades)."""
        self._current_weights = dict(weights)
