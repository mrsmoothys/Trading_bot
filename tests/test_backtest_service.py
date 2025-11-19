"""
Tests for Backtest Service
Unit tests for the backtesting service API.
"""

import pytest
from datetime import datetime, timedelta
from typing import Dict, Any
import pandas as pd
import numpy as np
import json

from backtesting.service import (
    BacktestConfig,
    BacktestResult,
    run_backtest,
    run_backtest_batch,
    generate_signals,
    load_market_data
)


class TestBacktestConfig:
    """Test BacktestConfig dataclass."""

    def test_config_creation(self):
        """Test creating a BacktestConfig."""
        config = BacktestConfig(
            symbol="BTCUSDT",
            timeframe="1h",
            start=datetime(2024, 1, 1),
            end=datetime(2024, 1, 31),
            strategy="sma",
            params={"fast_period": 20, "slow_period": 50}
        )

        assert config.symbol == "BTCUSDT"
        assert config.timeframe == "1h"
        assert config.strategy == "sma"
        assert config.params == {"fast_period": 20, "slow_period": 50}

    def test_config_to_json(self):
        """Test converting config to JSON."""
        config = BacktestConfig(
            symbol="BTCUSDT",
            timeframe="1h",
            start=datetime(2024, 1, 1),
            end=datetime(2024, 1, 31),
            strategy="sma",
            params={}
        )

        json_str = config.to_json()
        assert "BTCUSDT" in json_str
        assert "2024-01-01" in json_str

    def test_config_from_dict(self):
        """Test creating config from dictionary."""
        data = {
            "symbol": "BTCUSDT",
            "timeframe": "1h",
            "start": "2024-01-01T00:00:00",
            "end": "2024-01-31T00:00:00",
            "strategy": "sma",
            "params": {},
            "initial_capital": 10000.0
        }

        config = BacktestConfig.from_dict(data)
        assert config.symbol == "BTCUSDT"
        assert isinstance(config.start, datetime)

    def test_config_hash(self):
        """Test config hash generation."""
        config1 = BacktestConfig(
            symbol="BTCUSDT",
            timeframe="1h",
            start=datetime(2024, 1, 1),
            end=datetime(2024, 1, 31),
            strategy="sma",
            params={}
        )

        config2 = BacktestConfig(
            symbol="BTCUSDT",
            timeframe="1h",
            start=datetime(2024, 1, 1),
            end=datetime(2024, 1, 31),
            strategy="sma",
            params={}
        )

        hash1 = config1.get_config_hash()
        hash2 = config2.get_config_hash()

        assert hash1 == hash2

        # Different params should give different hash
        config3 = BacktestConfig(
            symbol="BTCUSDT",
            timeframe="1h",
            start=datetime(2024, 1, 1),
            end=datetime(2024, 1, 31),
            strategy="sma",
            params={"fast_period": 10}
        )

        hash3 = config3.get_config_hash()
        assert hash1 != hash3


class TestGenerateSignals:
    """Test signal generation for different strategies."""

    def create_sample_data(self, periods: int = 100) -> pd.DataFrame:
        """Create sample OHLCV data."""
        dates = pd.date_range(start='2024-01-01', periods=periods, freq='H')
        np.random.seed(42)

        # Generate price data with trend
        price_base = 50000
        returns = np.random.normal(0.001, 0.02, periods)
        prices = price_base * (1 + returns).cumprod()

        # Create OHLCV data
        data = pd.DataFrame(index=dates)
        data['open'] = prices
        data['high'] = prices * (1 + np.abs(np.random.normal(0, 0.01, periods)))
        data['low'] = prices * (1 - np.abs(np.random.normal(0, 0.01, periods)))
        data['close'] = prices
        data['volume'] = np.random.lognormal(10, 1, periods)

        return data

    def test_sma_strategy(self):
        """Test SMA crossover strategy."""
        df = self.create_sample_data()
        signals = generate_signals(df, "sma", {"fast_period": 10, "slow_period": 20})

        assert len(signals) == len(df)
        assert 'signal' in signals.columns
        assert 'strength' in signals.columns
        assert 'confidence' in signals.columns

        # Signals should be -1, 0, or 1
        assert signals['signal'].isin([-1, 0, 1]).all()

    def test_rsi_strategy(self):
        """Test RSI strategy."""
        df = self.create_sample_data()
        signals = generate_signals(df, "rsi", {"period": 14, "oversold": 30, "overbought": 70})

        assert len(signals) == len(df)
        assert signals['signal'].isin([-1, 0, 1]).all()

    def test_macd_strategy(self):
        """Test MACD strategy."""
        df = self.create_sample_data()
        signals = generate_signals(df, "macd", {})

        assert len(signals) == len(df)
        assert signals['signal'].isin([-1, 0, 1]).all()

    def test_convergence_strategy(self):
        """Test convergence strategy."""
        df = self.create_sample_data()
        signals = generate_signals(df, "convergence", {})

        assert len(signals) == len(df)
        assert signals['signal'].isin([-1, 0, 1]).all()

        # Convergence should have higher confidence
        assert signals['confidence'].mean() >= 0.7

    def test_unknown_strategy(self):
        """Test unknown strategy defaults to passive."""
        df = self.create_sample_data()
        signals = generate_signals(df, "unknown_strategy", {})

        assert len(signals) == len(df)
        # Should be all zeros for unknown strategy
        assert (signals['signal'] == 0).all()


class TestRunBacktest:
    """Test running backtests."""

    def create_sample_data(self, periods: int = 200) -> pd.DataFrame:
        """Create sample OHLCV data."""
        dates = pd.date_range(start='2024-01-01', periods=periods, freq='H')
        np.random.seed(42)

        price_base = 50000
        returns = np.random.normal(0.001, 0.02, periods)
        prices = price_base * (1 + returns).cumprod()

        data = pd.DataFrame(index=dates)
        data['open'] = prices
        data['high'] = prices * (1 + np.abs(np.random.normal(0, 0.01, periods)))
        data['low'] = prices * (1 - np.abs(np.random.normal(0, 0.01, periods)))
        data['close'] = prices
        data['volume'] = np.random.lognormal(10, 1, periods)

        return data

    def test_run_sma_backtest(self):
        """Test running SMA backtest."""
        # Mock load_market_data
        import backtesting.service
        original_load = backtesting.service.load_market_data

        df = self.create_sample_data()

        def mock_load(symbol, timeframe, start, end):
            return df

        backtesting.service.load_market_data = mock_load

        try:
            config = BacktestConfig(
                symbol="BTCUSDT",
                timeframe="1h",
                start=datetime(2024, 1, 1),
                end=datetime(2024, 1, 15),
                strategy="sma",
                params={"fast_period": 10, "slow_period": 20}
            )

            result = run_backtest(config)

            assert isinstance(result, BacktestResult)
            assert result.symbol == "BTCUSDT"
            assert result.timeframe == "1h"
            assert result.strategy == "sma"
            assert result.total_trades >= 0
            assert 0 <= result.win_rate <= 1
            assert result.sharpe_ratio is not None
            assert result.max_drawdown >= 0

            # Check result structure
            assert result.initial_capital == config.initial_capital
            assert isinstance(result.equity_curve, list)
            assert isinstance(result.trades, list)

        finally:
            backtesting.service.load_market_data = original_load

    def test_run_rsi_backtest(self):
        """Test running RSI backtest."""
        import backtesting.service
        original_load = backtesting.service.load_market_data

        df = self.create_sample_data()

        def mock_load(symbol, timeframe, start, end):
            return df

        backtesting.service.load_market_data = mock_load

        try:
            config = BacktestConfig(
                symbol="BTCUSDT",
                timeframe="1h",
                start=datetime(2024, 1, 1),
                end=datetime(2024, 1, 15),
                strategy="rsi",
                params={"period": 14}
            )

            result = run_backtest(config)

            assert isinstance(result, BacktestResult)
            assert result.strategy == "rsi"

        finally:
            backtesting.service.load_market_data = original_load

    def test_run_backtest_batch(self):
        """Test running batch backtests."""
        import backtesting.service
        original_load = backtesting.service.load_market_data

        df = self.create_sample_data()

        def mock_load(symbol, timeframe, start, end):
            return df

        backtesting.service.load_market_data = mock_load

        try:
            configs = [
                BacktestConfig(
                    symbol="BTCUSDT",
                    timeframe="1h",
                    start=datetime(2024, 1, 1),
                    end=datetime(2024, 1, 15),
                    strategy="sma",
                    params={"fast_period": 10, "slow_period": 20}
                ),
                BacktestConfig(
                    symbol="BTCUSDT",
                    timeframe="1h",
                    start=datetime(2024, 1, 1),
                    end=datetime(2024, 1, 15),
                    strategy="rsi",
                    params={"period": 14}
                )
            ]

            results = run_backtest_batch(configs)

            assert len(results) == 2

            # Check first result
            config1, result1, error1 = results[0]
            assert config1.strategy == "sma"
            assert result1 is not None
            assert error1 is None

            # Check second result
            config2, result2, error2 = results[1]
            assert config2.strategy == "rsi"
            assert result2 is not None
            assert error2 is None

        finally:
            backtesting.service.load_market_data = original_load

    def test_result_serialization(self):
        """Test BacktestResult serialization."""
        import backtesting.service
        original_load = backtesting.service.load_market_data

        df = self.create_sample_data()

        def mock_load(symbol, timeframe, start, end):
            return df

        backtesting.service.load_market_data = mock_load

        try:
            config = BacktestConfig(
                symbol="BTCUSDT",
                timeframe="1h",
                start=datetime(2024, 1, 1),
                end=datetime(2024, 1, 15),
                strategy="sma",
                params={}
            )

            result = run_backtest(config)

            # Test to_dict
            result_dict = result.to_dict()
            assert isinstance(result_dict, dict)
            assert "symbol" in result_dict
            assert "total_return" in result_dict

            # Test to_json
            result_json = result.to_json()
            assert isinstance(result_json, str)
            parsed = json.loads(result_json)
            assert parsed["symbol"] == "BTCUSDT"

        finally:
            backtesting.service.load_market_data = original_load


class TestBacktestResult:
    """Test BacktestResult dataclass."""

    def test_from_backtest_results(self):
        """Test creating BacktestResult from BacktestResults."""
        from models.backtester import BacktestResults, Trade

        config = BacktestConfig(
            symbol="BTCUSDT",
            timeframe="1h",
            start=datetime(2024, 1, 1),
            end=datetime(2024, 1, 31),
            strategy="sma",
            params={},
            initial_capital=10000.0
        )

        # Create mock BacktestResults
        mock_results = BacktestResults(
            trades=[],
            total_return=0.1,
            annualized_return=0.15,
            max_drawdown=0.05,
            sharpe_ratio=1.2,
            win_rate=0.6,
            profit_factor=1.5,
            avg_win=100.0,
            avg_loss=-50.0,
            total_trades=10,
            winning_trades=6,
            losing_trades=4
        )

        result = BacktestResult.from_backtest_results(config, mock_results)

        assert result.symbol == "BTCUSDT"
        assert result.strategy == "sma"
        assert result.total_return == 0.1
        assert result.total_return_pct == 10.0
        assert result.win_rate == 0.6
        assert result.sharpe_ratio == 1.2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
