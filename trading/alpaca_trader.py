"""
Alpaca Paper Trading — Executes portfolio rebalances via Alpaca API.
Designed for GitHub Actions automation.
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

try:
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import MarketOrderRequest
    from alpaca.trading.enums import OrderSide, TimeInForce
    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False
    logger.info("Alpaca SDK not installed. Paper trading disabled.")


class AlpacaTrader:
    """
    Executes GTAA portfolio via Alpaca paper trading.
    """

    # Tickers that need to be mapped from yfinance format to Alpaca format
    TICKER_MAP = {
        "BTC-USD": "BTCUSD",
        "ETH-USD": "ETHUSD",
    }

    # Tickers not tradeable on Alpaca
    SKIP_TICKERS = {"^VIX", "^TNX", "^IRX"}

    def __init__(self):
        if not ALPACA_AVAILABLE:
            raise RuntimeError("alpaca-py not installed. Run: pip install alpaca-py")

        api_key = os.environ.get("APCA_API_KEY_ID")
        secret = os.environ.get("APCA_API_SECRET_KEY")

        if not api_key or not secret:
            raise RuntimeError("Set APCA_API_KEY_ID and APCA_API_SECRET_KEY env vars.")

        self.client = TradingClient(api_key, secret, paper=True)
        self._validate_connection()

    def _validate_connection(self):
        """Verify API connection works."""
        account = self.client.get_account()
        logger.info(
            f"Alpaca connected. Account: {account.account_number}, "
            f"Equity: ${float(account.equity):,.2f}, "
            f"Cash: ${float(account.cash):,.2f}"
        )

    def _map_ticker(self, ticker: str) -> Optional[str]:
        """Map yfinance ticker to Alpaca symbol."""
        if ticker in self.SKIP_TICKERS:
            return None
        return self.TICKER_MAP.get(ticker, ticker)

    def get_account_info(self) -> Dict:
        """Get current account state."""
        account = self.client.get_account()
        return {
            "equity": float(account.equity),
            "cash": float(account.cash),
            "buying_power": float(account.buying_power),
            "portfolio_value": float(account.portfolio_value),
        }

    def get_current_positions(self) -> Dict[str, float]:
        """Get current positions as {ticker: market_value_pct}."""
        positions = self.client.get_all_positions()
        account = self.client.get_account()
        total_equity = float(account.equity)

        pos_dict = {}
        for p in positions:
            mv = float(p.market_value)
            pos_dict[p.symbol] = mv / total_equity if total_equity > 0 else 0

        return pos_dict

    def rebalance(self, target_weights: Dict[str, float], dry_run: bool = False) -> List[Dict]:
        """
        Rebalance portfolio to target weights.
        Returns list of executed orders.
        """
        account_info = self.get_account_info()
        total_equity = account_info["equity"]
        current_positions = self.get_current_positions()

        orders = []

        # Calculate target dollar amounts
        targets = {}
        for ticker, weight in target_weights.items():
            symbol = self._map_ticker(ticker)
            if symbol is None:
                continue
            targets[symbol] = weight * total_equity

        # Calculate current dollar amounts
        current = {}
        for symbol, pct in current_positions.items():
            current[symbol] = pct * total_equity

        # Determine trades needed
        all_symbols = set(targets.keys()) | set(current.keys())

        # Sell first (free up cash)
        sell_orders = []
        buy_orders = []

        for symbol in all_symbols:
            target_val = targets.get(symbol, 0)
            current_val = current.get(symbol, 0)
            diff = target_val - current_val

            if abs(diff) < 50:  # ignore tiny differences
                continue

            if diff < 0:
                sell_orders.append((symbol, abs(diff)))
            else:
                buy_orders.append((symbol, diff))

        # Execute sells
        for symbol, amount in sell_orders:
            order_info = {"symbol": symbol, "side": "sell", "amount": round(amount, 2)}
            if not dry_run:
                try:
                    order = self.client.submit_order(
                        MarketOrderRequest(
                            symbol=symbol,
                            notional=round(amount, 2),
                            side=OrderSide.SELL,
                            time_in_force=TimeInForce.DAY,
                        )
                    )
                    order_info["order_id"] = str(order.id)
                    order_info["status"] = str(order.status)
                except Exception as e:
                    order_info["error"] = str(e)
            else:
                order_info["status"] = "DRY_RUN"

            orders.append(order_info)

        # Execute buys
        for symbol, amount in buy_orders:
            order_info = {"symbol": symbol, "side": "buy", "amount": round(amount, 2)}
            if not dry_run:
                try:
                    order = self.client.submit_order(
                        MarketOrderRequest(
                            symbol=symbol,
                            notional=round(amount, 2),
                            side=OrderSide.BUY,
                            time_in_force=TimeInForce.DAY,
                        )
                    )
                    order_info["order_id"] = str(order.id)
                    order_info["status"] = str(order.status)
                except Exception as e:
                    order_info["error"] = str(e)
            else:
                order_info["status"] = "DRY_RUN"

            orders.append(order_info)

        logger.info(f"Rebalance complete: {len(sell_orders)} sells, {len(buy_orders)} buys")

        return orders


def run_live_signal():
    """
    Entry point for GitHub Actions.
    Runs the full agent pipeline on current data and executes via Alpaca.
    """
    from config.settings import GTAAConfig, ASSET_UNIVERSE
    from data.data_loader import DataLoader
    from agents import ResearchAgent, RegimeAgent, RiskAgent, AllocationAgent, PMAgent

    config = GTAAConfig()

    # Load recent data
    loader = DataLoader(universe=config.universe, start="2023-01-01")
    prices = loader.load(use_cache=False)
    returns = loader.returns
    today = prices.index[-1]

    # Load regime data
    try:
        regime_data = loader.get_regime_data()
    except Exception:
        regime_data = pd.DataFrame()

    # Run agent pipeline
    research = ResearchAgent()
    research_signal = research.analyze(today, prices, returns, {"universe": config.universe})

    regime = RegimeAgent()
    regime_signal = regime.analyze(
        today, prices, returns, {"regime_data": regime_data, "universe": config.universe}
    )

    allocation = AllocationAgent(config=config.allocation)
    alloc_signal = allocation.analyze(
        today, prices, returns,
        {
            "research_signal": research_signal.to_dict(),
            "regime_signal": regime_signal.to_dict(),
            "universe": config.universe,
        }
    )

    risk = RiskAgent(config=config.risk)
    risk_signal = risk.analyze(
        today, prices, returns,
        {
            "proposed_weights": alloc_signal.data["target_weights"],
            "universe": config.universe,
            "equity_value": 100000,  # will be overridden by actual account
        }
    )

    pm = PMAgent()
    pm_signal = pm.analyze(
        today, prices, returns,
        {
            "research_signal_obj": research_signal,
            "regime_signal_obj": regime_signal,
            "allocation_signal_obj": alloc_signal,
            "risk_signal_obj": risk_signal,
            "equity_value": 100000,
            "universe": config.universe,
        }
    )

    # Log signals
    signals_log = {
        "date": str(today.date()),
        "regime": regime_signal.data.get("regime"),
        "research_top5": research_signal.data.get("top_5"),
        "target_weights": pm_signal.data.get("final_weights"),
        "conviction": pm_signal.data.get("conviction"),
        "rebalance": pm_signal.data.get("execute_rebalance"),
    }

    print(json.dumps(signals_log, indent=2))

    # Execute via Alpaca
    if pm_signal.data.get("execute_rebalance"):
        dry_run = os.environ.get("DRY_RUN", "true").lower() == "true"
        trader = AlpacaTrader()
        orders = trader.rebalance(pm_signal.data["final_weights"], dry_run=dry_run)
        print(f"\nOrders executed (dry_run={dry_run}):")
        print(json.dumps(orders, indent=2))
    else:
        print("\nNo rebalance needed.")


if __name__ == "__main__":
    import pandas as pd
    logging.basicConfig(level=logging.INFO)
    run_live_signal()
