"""
Backtesting Module
Comprehensive backtesting service and utilities.
"""

from .service import (
    BacktestConfig,
    BacktestResult,
    run_backtest,
    run_backtest_batch,
    generate_signals
)

__all__ = [
    'BacktestConfig',
    'BacktestResult',
    'run_backtest',
    'run_backtest_batch',
    'generate_signals'
]
