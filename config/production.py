"""
Production Configuration — V3d Aggressive (Best CAGR variant)
CAGR: 10.1% | Sharpe: 0.64 | Max DD: -21.3% | Calmar: 0.47
"""

from config.settings import GTAAConfig


def get_production_config() -> GTAAConfig:
    """Returns the tuned production config (V3d aggressive)."""
    config = GTAAConfig()

    # Momentum
    config.momentum.lookback_windows = [21, 63, 126, 252]
    config.momentum.lookback_weights = [0.35, 0.30, 0.20, 0.15]
    config.momentum.skip_recent_days = 2
    config.momentum.vol_lookback = 63

    # Regime
    config.regime.sma_short = 50
    config.regime.sma_long = 200
    config.regime.vix_risk_off_threshold = 25.0
    config.regime.vix_extreme_threshold = 35.0

    # Risk
    config.risk.vol_target = 0.25
    config.risk.max_position_pct = 0.45
    config.risk.max_asset_class_pct = 0.70
    config.risk.min_cash_pct = 0.01
    config.risk.drawdown_circuit_breaker = -0.15
    config.risk.drawdown_full_exit = -0.25
    config.risk.rebalance_threshold = 0.05

    # Allocation
    config.allocation.top_n_assets = 5
    config.allocation.risk_parity_blend = 0.0  # pure momentum
    config.allocation.regime_tilt_strength = 0.50
    config.allocation.rebalance_frequency = "monthly"
    config.allocation.min_weight = 0.03

    # Costs (realistic)
    config.costs.slippage_bps = 5.0
    config.costs.crypto_slippage_bps = 15.0
    config.costs.commission_per_trade = 0.0

    return config
