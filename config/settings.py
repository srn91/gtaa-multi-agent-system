"""
GTAA System Configuration
Global Tactical Asset Allocation - Multi-Agent Momentum Rotation Engine
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

# ─────────────────────────────────────────────
# ASSET UNIVERSE - All free ETFs/tickers
# ─────────────────────────────────────────────

ASSET_UNIVERSE: Dict[str, Dict[str, str]] = {
    # US Equities - Broad
    "SPY":  {"name": "S&P 500",              "class": "equity_us",      "sub": "large_cap"},
    "QQQ":  {"name": "Nasdaq 100",           "class": "equity_us",      "sub": "tech"},
    "IWM":  {"name": "Russell 2000",         "class": "equity_us",      "sub": "small_cap"},
    "MDY":  {"name": "S&P MidCap 400",       "class": "equity_us",      "sub": "mid_cap"},

    # US Equities - Sectors (NEW: enables sector rotation)
    "XLK":  {"name": "Technology",           "class": "equity_sector",  "sub": "tech"},
    "XLF":  {"name": "Financials",           "class": "equity_sector",  "sub": "financials"},
    "XLE":  {"name": "Energy",               "class": "equity_sector",  "sub": "energy"},
    "XLV":  {"name": "Healthcare",           "class": "equity_sector",  "sub": "healthcare"},
    "XLY":  {"name": "Consumer Disc.",       "class": "equity_sector",  "sub": "cons_disc"},
    "XLP":  {"name": "Consumer Staples",     "class": "equity_sector",  "sub": "cons_staples"},
    "XLI":  {"name": "Industrials",          "class": "equity_sector",  "sub": "industrials"},
    "XLU":  {"name": "Utilities",            "class": "equity_sector",  "sub": "utilities"},

    # International Equities
    "EFA":  {"name": "EAFE Developed",        "class": "equity_intl",    "sub": "developed"},
    "EEM":  {"name": "Emerging Markets",      "class": "equity_intl",    "sub": "emerging"},
    "VGK":  {"name": "Europe",                "class": "equity_intl",    "sub": "europe"},
    "EWJ":  {"name": "Japan",                 "class": "equity_intl",    "sub": "japan"},

    # Fixed Income
    "TLT":  {"name": "20+ Year Treasury",     "class": "fixed_income",   "sub": "long_govt"},
    "IEF":  {"name": "7-10 Year Treasury",    "class": "fixed_income",   "sub": "mid_govt"},
    "SHY":  {"name": "1-3 Year Treasury",     "class": "fixed_income",   "sub": "short_govt"},
    "HYG":  {"name": "High Yield Corporate",  "class": "fixed_income",   "sub": "high_yield"},
    "LQD":  {"name": "Inv Grade Corporate",   "class": "fixed_income",   "sub": "inv_grade"},
    "TIP":  {"name": "TIPS",                  "class": "fixed_income",   "sub": "inflation"},

    # Commodities
    "GLD":  {"name": "Gold",                  "class": "commodity",      "sub": "precious"},
    "SLV":  {"name": "Silver",                "class": "commodity",      "sub": "precious"},
    "USO":  {"name": "Crude Oil",             "class": "commodity",      "sub": "energy"},
    "DBC":  {"name": "Commodity Index",       "class": "commodity",      "sub": "broad"},

    # Real Estate
    "VNQ":  {"name": "US REITs",              "class": "real_estate",    "sub": "us"},
    "VNQI": {"name": "Intl REITs",            "class": "real_estate",    "sub": "intl"},

    # Crypto (via ETF or direct)
    "BTC-USD": {"name": "Bitcoin",            "class": "crypto",         "sub": "btc"},
    "ETH-USD": {"name": "Ethereum",           "class": "crypto",         "sub": "eth"},

    # Cash proxy
    "BIL":  {"name": "1-3 Month T-Bill",      "class": "cash",           "sub": "tbill"},
}

# Benchmark
BENCHMARK_TICKER = "SPY"

# Asset class groupings for correlation & allocation constraints
ASSET_CLASSES = [
    "equity_us", "equity_sector", "equity_intl", "fixed_income",
    "commodity", "real_estate", "crypto", "cash"
]


# ─────────────────────────────────────────────
# MOMENTUM PARAMETERS
# ─────────────────────────────────────────────

@dataclass
class MomentumConfig:
    """Momentum signal configuration."""
    lookback_windows: List[int] = field(default_factory=lambda: [21, 63, 126, 252])  # 1m, 3m, 6m, 12m
    lookback_weights: List[float] = field(default_factory=lambda: [0.35, 0.30, 0.20, 0.15])  # recency bias
    skip_recent_days: int = 5  # skip most recent week (mean-reversion noise)
    vol_lookback: int = 63  # for volatility adjustment
    min_history_days: int = 252  # require 1yr data minimum


# ─────────────────────────────────────────────
# REGIME DETECTION PARAMETERS
# ─────────────────────────────────────────────

@dataclass
class RegimeConfig:
    """Macro regime detection settings."""
    sma_short: int = 50
    sma_long: int = 200
    vix_ticker: str = "^VIX"
    vix_risk_off_threshold: float = 25.0
    vix_extreme_threshold: float = 35.0
    trend_ema_span: int = 10  # EMA on breadth signals
    regime_lookback: int = 252
    # Yield curve: 10yr - 2yr spread
    yield_10y_ticker: str = "^TNX"
    yield_2y_ticker: str = "^IRX"  # proxy for short rates


# ─────────────────────────────────────────────
# RISK MANAGEMENT PARAMETERS
# ─────────────────────────────────────────────

@dataclass
class RiskConfig:
    """Risk management constraints."""
    max_position_pct: float = 0.25          # no single position > 25%
    max_asset_class_pct: float = 0.50       # no single class > 50%
    max_correlated_exposure: float = 0.60   # correlation threshold for grouping
    min_cash_pct: float = 0.02              # always hold 2% cash minimum
    max_portfolio_vol: float = 0.20         # target annual vol cap 20%
    drawdown_circuit_breaker: float = -0.15 # go 50% cash if DD > 15%
    drawdown_full_exit: float = -0.25       # go 100% cash if DD > 25%
    vol_target: float = 0.12               # target portfolio vol 12% annualized
    correlation_lookback: int = 63          # rolling correlation window
    rebalance_threshold: float = 0.05       # rebalance if drift > 5%


# ─────────────────────────────────────────────
# ALLOCATION PARAMETERS
# ─────────────────────────────────────────────

@dataclass
class AllocationConfig:
    """Portfolio allocation rules."""
    top_n_assets: int = 8                   # hold top N momentum assets
    min_weight: float = 0.03                # minimum position 3%
    rebalance_frequency: str = "monthly"    # weekly | monthly
    risk_parity_blend: float = 0.50         # 50% momentum + 50% risk parity
    regime_tilt_strength: float = 0.30      # how much regime shifts allocation


# ─────────────────────────────────────────────
# TRANSACTION COST MODEL
# ─────────────────────────────────────────────

@dataclass
class CostConfig:
    """Realistic transaction costs."""
    commission_per_trade: float = 0.0       # most brokers are zero now
    slippage_bps: float = 5.0              # 5 basis points slippage
    etf_expense_ratio_avg: float = 0.0020  # ~20bps avg ETF expense
    crypto_slippage_bps: float = 15.0      # crypto has wider spreads
    min_trade_value: float = 100.0         # don't trade tiny positions


# ─────────────────────────────────────────────
# BACKTEST PARAMETERS
# ─────────────────────────────────────────────

@dataclass
class BacktestConfig:
    """Backtest settings."""
    start_date: str = "2008-01-01"
    end_date: str = "2025-12-31"
    initial_capital: float = 100_000.0
    benchmark: str = "SPY"


# ─────────────────────────────────────────────
# ALPACA / PAPER TRADING
# ─────────────────────────────────────────────

@dataclass
class AlpacaConfig:
    """Alpaca paper trading config. Set via env vars."""
    base_url: str = "https://paper-api.alpaca.markets"
    # APCA_API_KEY_ID and APCA_API_SECRET_KEY from env


# ─────────────────────────────────────────────
# MASTER CONFIG
# ─────────────────────────────────────────────

@dataclass
class GTAAConfig:
    """Master configuration object."""
    momentum: MomentumConfig = field(default_factory=MomentumConfig)
    regime: RegimeConfig = field(default_factory=RegimeConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    allocation: AllocationConfig = field(default_factory=AllocationConfig)
    costs: CostConfig = field(default_factory=CostConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    alpaca: AlpacaConfig = field(default_factory=AlpacaConfig)
    universe: Dict = field(default_factory=lambda: ASSET_UNIVERSE)
