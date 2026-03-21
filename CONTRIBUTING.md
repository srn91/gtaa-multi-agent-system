# Contributing

Contributions are welcome. This is a research project, so the bar is intellectual rigor, not perfection.

## How to contribute

1. Fork the repo
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Make your changes
4. Run the test suite: `python3 tests/test_agents.py` (all 12 must pass)
5. Run the backtest to verify no regression: `python3 run_backtest.py`
6. Commit with a clear message
7. Open a pull request

## What would be valuable

- **Walk-forward retraining** — implement expanding-window ML retraining during backtests
- **Additional ML models** — LSTM, Transformer, or other architectures for regime/direction
- **Options overlay** — tail risk hedging using the Regime-Aware Options Engine concepts
- **Alternative data** — sentiment, order flow, or macro indicators as agent inputs
- **Live paper-trade monitoring** — dashboards and alerting for the Alpaca pipeline
- **Improved transaction cost model** — market impact, partial fills, timing slippage

## Code standards

- Type hints on all public functions
- Docstrings on all classes and public methods
- All agents must produce typed `Signal` objects via `BaseAgent.analyze()`
- New agents must have corresponding tests in `tests/test_agents.py`
- No hardcoded magic numbers — use `config/settings.py` or YAML configs

## Running tests

```bash
python3 tests/test_agents.py
```

All 12 tests must pass before submitting a PR.
