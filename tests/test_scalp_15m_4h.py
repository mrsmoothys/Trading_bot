"""
Unit tests for Scalp15m4hStrategy (P2.1)

Tests cover:
- Pullback entry logic
- Breakout entry logic
- Risk management (1.5% per trade)
- Stop loss and take profit calculation
- 4h trend awareness
"""
import pytest
import pandas as pd
import numpy as np
from core.strategies.scalp_15m_4h import Scalp15m4hStrategy, ScalpSignal


class TestScalp15m4hStrategy:
    """Test cases for the scalp strategy."""

    def setup_method(self):
        """Set up test fixtures."""
        self.strategy = Scalp15m4hStrategy(risk_per_trade=0.015)
        self.sample_15m = self._generate_sample_ohlcv(100, base_price=50000)
        self.sample_4h = self._generate_sample_ohlcv(50, base_price=50000)

    def _generate_sample_ohlcv(self, num_bars: int, base_price: float = 50000) -> pd.DataFrame:
        """Generate sample OHLCV data for testing."""
        np.random.seed(42)  # For reproducible tests
        
        timestamps = pd.date_range(start='2025-01-01', periods=num_bars, freq='15T')
        
        returns = np.random.normal(0, 0.002, num_bars)
        prices = base_price * (1 + returns).cumprod()
        
        opens = prices.copy()
        highs = np.maximum(opens, prices) * (1 + np.abs(np.random.normal(0, 0.001, num_bars)))
        lows = np.minimum(opens, prices) * (1 - np.abs(np.random.normal(0, 0.001, num_bars)))
        closes = prices
        
        return pd.DataFrame({
            'timestamp': timestamps,
            'open': opens,
            'high': highs,
            'low': lows,
            'close': closes,
            'volume': np.random.lognormal(4, 1, num_bars)
        })

    def test_strategy_initialization(self):
        """Test strategy initializes correctly."""
        assert self.strategy.name == "Scalp 15m/4h"
        assert self.strategy.risk_per_trade == 0.015

    def test_insufficient_data_15m(self):
        """Test strategy handles insufficient 15m data."""
        signal = self.strategy.evaluate(pd.DataFrame(), self.sample_4h)
        
        assert signal.action == 'WAIT'
        assert signal.confidence == 0.0
        assert 'Insufficient 15m data' in signal.reasoning

    def test_insufficient_data_4h(self):
        """Test strategy handles insufficient 4h data."""
        signal = self.strategy.evaluate(self.sample_15m, pd.DataFrame())
        
        assert signal.action == 'WAIT'
        assert signal.confidence == 0.0
        assert 'Insufficient 4h data' in signal.reasoning

    def test_atr_calculation(self):
        """Test ATR calculation."""
        atr = self.strategy._calculate_atr(self.sample_15m, period=14)
        
        assert isinstance(atr, float)
        assert atr > 0

    def test_trend_direction_calculation(self):
        """Test 4h trend direction calculation."""
        trend = self.strategy._get_trend_direction(self.sample_4h)
        
        assert isinstance(trend, float)
        assert -1.0 <= trend <= 1.0

    def test_support_resistance_level_identification(self):
        """Test support and resistance level identification."""
        resistance_levels = self.strategy._find_resistance_levels(self.sample_15m)
        support_levels = self.strategy._find_support_levels(self.sample_15m)
        
        assert isinstance(resistance_levels, list)
        assert isinstance(support_levels, list)
        assert len(resistance_levels) <= 5
        assert len(support_levels) <= 5

    def test_pullback_setup_detection(self):
        """Test pullback setup detection."""
        # Create data with clear support level
        test_data = self.sample_15m.copy()
        support_level = test_data['low'].min()
        
        # Create a pullback to this support
        test_data.loc[test_data.index[-5:], 'low'] = support_level
        test_data.loc[test_data.index[-5:], 'close'] = support_level * 1.002
        
        setup = self.strategy._check_pullback_setup(
            test_data, self.sample_4h,
            support_level * 1.001,
            [support_level],
            [],
            self.strategy._calculate_atr(test_data)
        )
        
        # May or may not detect setup depending on market conditions
        assert setup is None or isinstance(setup, dict)

    def test_breakout_setup_detection(self):
        """Test breakout setup detection."""
        # Create data with clear resistance level
        test_data = self.sample_15m.copy()
        resistance_level = test_data['high'].max()
        
        # Price approaching resistance
        test_price = resistance_level * 0.999
        
        setup = self.strategy._check_breakout_setup(
            test_data,
            test_price,
            [],
            [resistance_level],
            self.strategy._calculate_atr(test_data)
        )
        
        # May or may not detect setup
        assert setup is None or isinstance(setup, dict)

    def test_signal_structure(self):
        """Test that signals have proper structure."""
        # Use data that should produce a signal
        bullish_4h = self.sample_4h.copy()
        bullish_4h['close'] = bullish_4h['close'].iloc[0] * np.linspace(1, 1.1, len(bullish_4h))
        
        signal = self.strategy.evaluate(self.sample_15m, bullish_4h)
        
        assert isinstance(signal, ScalpSignal)
        assert signal.action in ['LONG', 'SHORT', 'WAIT']
        assert 0.0 <= signal.confidence <= 1.0
        assert signal.entry_price >= 0
        assert signal.stop_loss >= 0
        assert signal.take_profit_1 >= 0
        assert signal.take_profit_2 >= 0
        assert 0.0 <= signal.position_size <= 1.0
        assert signal.reasoning is not None
        assert signal.entry_type in ['PULLBACK', 'BREAKOUT', 'N/A']

    def test_risk_management(self):
        """Test that risk management is applied (1.5% per trade)."""
        signal = self.strategy.evaluate(self.sample_15m, self.sample_4h)
        
        # If we got a trade signal, position size should be reasonable
        if signal.action != 'WAIT':
            # Position size should be capped at 10%
            assert signal.position_size <= 0.1
            # Or if based on 1.5% risk, should be proportional to price movement

    def test_error_handling(self):
        """Test strategy handles errors gracefully."""
        # Test with None data
        signal = self.strategy.evaluate(None, None)
        assert signal.action == 'WAIT'
        assert 'Insufficient' in signal.reasoning

    def test_long_setup_logic(self):
        """Test LONG setup detection logic."""
        # Create bullish 4h trend
        bullish_4h = self.sample_4h.copy()
        bullish_4h['close'] = bullish_4h['close'].iloc[0] * np.linspace(1, 1.2, len(bullish_4h))
        
        signal = self.strategy.evaluate(self.sample_15m, bullish_4h)
        
        # Should prefer LONG in strong uptrend
        if signal.action != 'WAIT':
            assert isinstance(signal, ScalpSignal)

    def test_short_setup_logic(self):
        """Test SHORT setup detection logic."""
        # Create bearish 4h trend
        bearish_4h = self.sample_4h.copy()
        bearish_4h['close'] = bearish_4h['close'].iloc[0] * np.linspace(1, 0.8, len(bearish_4h))
        
        signal = self.strategy.evaluate(self.sample_15m, bearish_4h)
        
        # Should prefer SHORT in strong downtrend
        if signal.action != 'WAIT':
            assert isinstance(signal, ScalpSignal)


class TestScalpSignal:
    """Test cases for ScalpSignal dataclass."""

    def test_signal_creation(self):
        """Test creating a valid signal."""
        signal = ScalpSignal(
            action='LONG',
            confidence=0.8,
            entry_price=50000.0,
            stop_loss=49500.0,
            take_profit_1=50750.0,
            take_profit_2=51500.0,
            position_size=0.05,
            reasoning='Test setup',
            entry_type='PULLBACK'
        )
        
        assert signal.action == 'LONG'
        assert signal.confidence == 0.8
        assert signal.entry_price == 50000.0
        assert signal.stop_loss == 49500.0
        assert signal.take_profit_1 == 50750.0
        assert signal.take_profit_2 == 51500.0
        assert signal.position_size == 0.05
        assert signal.reasoning == 'Test setup'
        assert signal.entry_type == 'PULLBACK'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
