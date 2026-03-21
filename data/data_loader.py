"""
Data Loader - Fetches and caches price data for the full asset universe.
Uses yfinance. No paid APIs needed.
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional, List
import logging
import json
import hashlib

logger = logging.getLogger(__name__)

CACHE_DIR = Path("data/cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)


class DataLoader:
    """Loads, caches, and serves price data for the GTAA universe."""

    def __init__(self, universe: Dict[str, Dict], start: str = "2007-01-01", end: Optional[str] = None):
        self.universe = universe
        self.tickers = list(universe.keys())
        self.start = start
        self.end = end or datetime.now().strftime("%Y-%m-%d")
        self._prices: Optional[pd.DataFrame] = None
        self._returns: Optional[pd.DataFrame] = None
        self._volumes: Optional[pd.DataFrame] = None

    def _cache_key(self) -> str:
        key_str = f"{sorted(self.tickers)}_{self.start}_{self.end}"
        return hashlib.md5(key_str.encode()).hexdigest()[:12]

    def _cache_path(self) -> Path:
        return CACHE_DIR / f"prices_{self._cache_key()}.parquet"

    def load(self, use_cache: bool = True) -> pd.DataFrame:
        """Load adjusted close prices for all tickers."""
        cache_path = self._cache_path()

        if use_cache and cache_path.exists():
            cache_age = datetime.now().timestamp() - cache_path.stat().st_mtime
            if cache_age < 86400:  # 24hr cache
                logger.info(f"Loading cached data from {cache_path}")
                self._prices = pd.read_parquet(cache_path)
                self._compute_returns()
                return self._prices

        logger.info(f"Downloading data for {len(self.tickers)} tickers: {self.start} to {self.end}")

        # Download in batch - much faster
        raw = yf.download(
            self.tickers,
            start=self.start,
            end=self.end,
            auto_adjust=True,
            threads=True,
            progress=False,
        )

        if isinstance(raw.columns, pd.MultiIndex):
            prices = raw["Close"].copy()
        else:
            # Single ticker edge case
            prices = raw[["Close"]].copy()
            prices.columns = self.tickers[:1]

        # Handle tickers that failed to download
        missing = [t for t in self.tickers if t not in prices.columns]
        if missing:
            logger.warning(f"Missing tickers (no data): {missing}")

        # Forward fill gaps (max 5 days), then drop remaining NaN columns
        prices = prices.ffill(limit=5)

        # Drop tickers with > 30% missing data
        threshold = len(prices) * 0.3
        valid_cols = prices.columns[prices.notna().sum() > threshold]
        dropped = set(prices.columns) - set(valid_cols)
        if dropped:
            logger.warning(f"Dropped tickers (too much missing data): {dropped}")
        prices = prices[valid_cols]

        self._prices = prices
        self._compute_returns()

        # Cache
        try:
            prices.to_parquet(cache_path)
        except Exception as e:
            logger.warning(f"Cache write failed: {e}")

        return self._prices

    def _compute_returns(self):
        """Compute daily returns from prices."""
        if self._prices is not None:
            self._returns = self._prices.pct_change()

    @property
    def prices(self) -> pd.DataFrame:
        if self._prices is None:
            self.load()
        return self._prices

    @property
    def returns(self) -> pd.DataFrame:
        if self._returns is None:
            self.load()
        return self._returns

    def get_ticker_class(self, ticker: str) -> str:
        """Get asset class for a ticker."""
        info = self.universe.get(ticker, {})
        return info.get("class", "unknown")

    def get_tickers_by_class(self, asset_class: str) -> List[str]:
        """Get all tickers in an asset class."""
        return [
            t for t, info in self.universe.items()
            if info.get("class") == asset_class and t in self.prices.columns
        ]

    def get_regime_data(self, vix_ticker: str = "^VIX") -> pd.DataFrame:
        """Load VIX and other regime indicators."""
        regime_tickers = [vix_ticker, "^TNX"]
        raw = yf.download(
            regime_tickers,
            start=self.start,
            end=self.end,
            auto_adjust=True,
            threads=True,
            progress=False,
        )
        if isinstance(raw.columns, pd.MultiIndex):
            return raw["Close"]
        return raw[["Close"]]

    def get_available_tickers(self) -> List[str]:
        """Return tickers that actually have data."""
        return list(self.prices.columns)

    def summary(self) -> Dict:
        """Quick summary of loaded data."""
        p = self.prices
        return {
            "tickers": len(p.columns),
            "date_range": f"{p.index[0].date()} to {p.index[-1].date()}",
            "trading_days": len(p),
            "missing_pct": round(p.isna().sum().sum() / p.size * 100, 2),
            "available": list(p.columns),
        }
