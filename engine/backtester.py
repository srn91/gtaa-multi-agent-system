"""
Backtesting Engine — Orchestrates all agents through historical data.
Handles rebalancing schedule, transaction costs, and performance tracking.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import logging
import json

from config.settings import GTAAConfig, ASSET_UNIVERSE
from data.data_loader import DataLoader
from agents.research_agent import ResearchAgent
from agents.regime_agent import RegimeAgent
from agents.risk_agent import RiskAgent
from agents.allocation_agent import AllocationAgent
from agents.pm_agent import PMAgent
from agents.review_agent import ReviewAgent

logger = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    """Container for backtest results."""
    equity_curve: pd.Series
    benchmark_curve: pd.Series
    weights_history: List[Dict]
    trade_log: List[Dict]
    stats: Dict[str, float]
    agent_audit: Dict[str, List]
    config: Dict


class BacktestEngine:
    """
    Runs the full GTAA simulation.
    Monthly rebalance. All agents consulted each period.
    Transaction costs applied realistically.
    """

    def __init__(self, config: GTAAConfig = None):
        self.config = config or GTAAConfig()

        # Initialize agents
        self.research = ResearchAgent(
            lookback_windows=self.config.momentum.lookback_windows,
            lookback_weights=self.config.momentum.lookback_weights,
            skip_recent=self.config.momentum.skip_recent_days,
            vol_lookback=self.config.momentum.vol_lookback,
        )
        self.regime = RegimeAgent(
            sma_short=self.config.regime.sma_short,
            sma_long=self.config.regime.sma_long,
            vix_risk_off=self.config.regime.vix_risk_off_threshold,
            vix_crisis=self.config.regime.vix_extreme_threshold,
        )
        self.risk = RiskAgent(config=self.config.risk)
        self.allocation = AllocationAgent(config=self.config.allocation)
        self.pm = PMAgent(
            rebalance_threshold=self.config.risk.rebalance_threshold,
        )
        self.review = ReviewAgent()

        # Data
        self.data_loader: Optional[DataLoader] = None
        self.regime_data: Optional[pd.DataFrame] = None

    def _get_rebalance_dates(self, dates: pd.DatetimeIndex) -> List[pd.Timestamp]:
        """Get rebalance dates based on frequency config."""
        freq = self.config.allocation.rebalance_frequency
        if freq == "weekly":
            # Last trading day of each week
            groups = dates.to_series().groupby(dates.to_period("W"))
            return [g.iloc[-1] for _, g in groups if len(g) > 0]
        else:
            # Last trading day of each month (default)
            groups = dates.to_series().groupby(dates.to_period("M"))
            return [g.iloc[-1] for _, g in groups if len(g) > 0]

    def _apply_transaction_costs(self, old_weights: Dict[str, float],
                                  new_weights: Dict[str, float],
                                  equity: float) -> float:
        """
        Compute and return transaction cost in dollars.
        """
        all_tickers = set(old_weights.keys()) | set(new_weights.keys())
        total_cost = 0.0

        for t in all_tickers:
            old_w = old_weights.get(t, 0)
            new_w = new_weights.get(t, 0)
            trade_value = abs(new_w - old_w) * equity

            if trade_value < self.config.costs.min_trade_value:
                continue

            # Slippage
            info = self.config.universe.get(t, {})
            if info.get("class") == "crypto":
                slippage_bps = self.config.costs.crypto_slippage_bps
            else:
                slippage_bps = self.config.costs.slippage_bps

            cost = trade_value * (slippage_bps / 10000.0)
            cost += self.config.costs.commission_per_trade
            total_cost += cost

        return total_cost

    def _simulate_day(self, date: pd.Timestamp, weights: Dict[str, float],
                       returns: pd.DataFrame) -> Tuple[Dict[str, float], float]:
        """
        Simulate one day of returns with current weights.
        Returns updated (drifted) weights and portfolio return.
        """
        if date not in returns.index:
            return weights, 0.0

        port_return = 0.0
        new_values = {}

        for ticker, weight in weights.items():
            if ticker in returns.columns:
                ret = returns.loc[date, ticker]
                if pd.notna(ret):
                    new_values[ticker] = weight * (1.0 + ret)
                    port_return += weight * ret
                else:
                    new_values[ticker] = weight
            else:
                new_values[ticker] = weight

        # Re-derive drifted weights
        total = sum(new_values.values())
        if total > 0:
            drifted = {t: v / total for t, v in new_values.items()}
        else:
            drifted = weights

        return drifted, port_return

    def run(self, progress_callback=None) -> BacktestResult:
        """
        Execute full backtest.
        """
        # Load data
        self.data_loader = DataLoader(
            universe=self.config.universe,
            start=self.config.backtest.start_date,
            end=self.config.backtest.end_date,
        )
        prices = self.data_loader.load()
        returns = self.data_loader.returns

        # Load regime data
        try:
            self.regime_data = self.data_loader.get_regime_data()
        except Exception as e:
            logger.warning(f"Could not load regime data: {e}")
            self.regime_data = pd.DataFrame()

        # Get rebalance dates
        # Start after warmup (need 252 days of history)
        warmup = max(self.config.momentum.lookback_windows) + self.config.momentum.skip_recent_days + 10
        valid_dates = prices.index[warmup:]
        rebalance_dates = set(self._get_rebalance_dates(valid_dates))

        # Initialize
        equity = self.config.backtest.initial_capital
        current_weights: Dict[str, float] = {"BIL": 1.0}  # start in cash
        self.pm.set_current_weights(current_weights)

        equity_curve = []
        dates_out = []
        weights_history = []
        benchmark_values = []

        # Benchmark tracking
        bench = self.config.backtest.benchmark
        bench_equity = self.config.backtest.initial_capital

        total_days = len(valid_dates)
        logger.info(f"Starting backtest: {valid_dates[0].date()} to {valid_dates[-1].date()} ({total_days} days)")

        for i, date in enumerate(valid_dates):
            # Track benchmark
            if bench in returns.columns and date in returns.index:
                bench_ret = returns.loc[date, bench]
                if pd.notna(bench_ret):
                    bench_equity *= (1.0 + bench_ret)

            # Simulate daily portfolio return
            drifted_weights, port_ret = self._simulate_day(date, current_weights, returns)
            equity *= (1.0 + port_ret)
            current_weights = drifted_weights

            # Record
            equity_curve.append(equity)
            dates_out.append(date)
            benchmark_values.append(bench_equity)
            self.review.update_equity(date, equity)

            # Rebalance check
            if date in rebalance_dates:
                try:
                    # 1. Research Agent
                    research_signal = self.research.analyze(
                        date, prices, returns, {"universe": self.config.universe}
                    )

                    # 2. Regime Agent
                    regime_signal = self.regime.analyze(
                        date, prices, returns,
                        {"regime_data": self.regime_data, "universe": self.config.universe}
                    )

                    # 3. Allocation Agent
                    alloc_signal = self.allocation.analyze(
                        date, prices, returns,
                        {
                            "research_signal": research_signal.to_dict(),
                            "regime_signal": regime_signal.to_dict(),
                            "universe": self.config.universe,
                        }
                    )

                    # 4. Risk Agent (reviews proposed allocation)
                    risk_signal = self.risk.analyze(
                        date, prices, returns,
                        {
                            "proposed_weights": alloc_signal.data["target_weights"],
                            "universe": self.config.universe,
                            "equity_value": equity,
                            "regime": regime_signal.data.get("regime", "RISK_ON"),
                        }
                    )

                    # 5. PM Agent (final decision)
                    pm_signal = self.pm.analyze(
                        date, prices, returns,
                        {
                            "research_signal_obj": research_signal,
                            "regime_signal_obj": regime_signal,
                            "allocation_signal_obj": alloc_signal,
                            "risk_signal_obj": risk_signal,
                            "equity_value": equity,
                            "universe": self.config.universe,
                        }
                    )

                    # Execute if rebalance approved
                    if pm_signal.data.get("execute_rebalance"):
                        new_weights = pm_signal.data["final_weights"]

                        # Transaction costs
                        cost = self._apply_transaction_costs(current_weights, new_weights, equity)
                        equity -= cost

                        # Log trade
                        turnover = pm_signal.data.get("turnover", 0)
                        self.review.log_trade(date, current_weights, new_weights, equity, turnover)

                        weights_history.append({
                            "date": str(date.date()),
                            "weights": new_weights,
                            "regime": regime_signal.data.get("regime", "UNKNOWN"),
                            "turnover": turnover,
                            "cost": round(cost, 2),
                            "equity": round(equity, 2),
                            "conviction": pm_signal.data.get("conviction", 0),
                        })

                        current_weights = new_weights
                        self.pm.set_current_weights(current_weights)

                except Exception as e:
                    logger.error(f"Error on rebalance date {date.date()}: {e}")
                    continue

            # Progress
            if progress_callback and i % 50 == 0:
                progress_callback(i / total_days)

        # Final review
        review_signal = self.review.analyze(
            valid_dates[-1], prices, returns,
            {
                "current_weights": current_weights,
                "equity_value": equity,
            }
        )

        stats = review_signal.data.get("stats", {})

        # Build result
        eq_series = pd.Series(equity_curve, index=dates_out)
        bench_series = pd.Series(benchmark_values, index=dates_out)

        result = BacktestResult(
            equity_curve=eq_series,
            benchmark_curve=bench_series,
            weights_history=weights_history,
            trade_log=self.review.get_trade_log(),
            stats=stats,
            agent_audit={
                "research": self.research.get_audit_trail(50),
                "regime": self.regime.get_audit_trail(50),
                "risk": self.risk.get_audit_trail(50),
                "allocation": self.allocation.get_audit_trail(50),
                "pm": self.pm.get_audit_trail(50),
                "review": self.review.get_audit_trail(50),
            },
            config={
                "momentum": self.config.momentum.__dict__,
                "regime": {k: v for k, v in self.config.regime.__dict__.items()},
                "risk": self.config.risk.__dict__,
                "allocation": self.config.allocation.__dict__,
                "costs": self.config.costs.__dict__,
                "backtest": self.config.backtest.__dict__,
            },
        )

        logger.info(f"Backtest complete. CAGR={stats.get('cagr', 0):.2%}, Sharpe={stats.get('sharpe', 0):.2f}")

        return result
