"""
Test P3.3 - Conditional Feature Updates
Tests that heavy features are only recalculated on bar close, while lightweight features update every tick.
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, MagicMock
from core.signal_generator import SignalGenerator
from core.system_context import SystemContext
from features.engine import FeatureEngine


class TestConditionalFeatureUpdates:
    """Test conditional feature calculation logic."""

    def setup_method(self):
        """Set up test fixtures."""
        # Create mocks
        self.system_context = Mock(spec=SystemContext)
        self.deepseek_brain = Mock()
        self.feature_engine = Mock(spec=FeatureEngine)

        # Create SignalGenerator instance
        self.signal_gen = SignalGenerator(
            system_context=self.system_context,
            deepseek_brain=self.deepseek_brain,
            feature_engine=self.feature_engine,
            enable_db_logging=False,
            memory_manager=None,
            enable_convergence_strategy=False,
            enable_scalp_strategy=False
        )

        # Generate sample market data
        self.market_data = self._generate_sample_data()

    def _generate_sample_data(self, num_bars=100, use_same_timestamp=False):
        """Generate sample OHLCV data."""
        timestamps = pd.date_range(start='2025-01-01', periods=num_bars, freq='15T')
        np.random.seed(42)

        prices = 50000 * (1 + np.random.normal(0, 0.01, num_bars)).cumprod()

        # Create series with timestamps as index for bar close detection
        data = {
            'open': pd.Series(prices, index=timestamps),
            'high': pd.Series(prices * (1 + np.abs(np.random.normal(0, 0.005, num_bars))), index=timestamps),
            'low': pd.Series(prices * (1 - np.abs(np.random.normal(0, 0.005, num_bars))), index=timestamps),
            'close': pd.Series(prices, index=timestamps),
            'volume': pd.Series(np.random.lognormal(4, 1, num_bars), index=timestamps)
        }

        if use_same_timestamp:
            # For testing cache, use the same last timestamp
            same_timestamp = timestamps[-1]
            data['open'].index = pd.Index([same_timestamp] * num_bars)
            data['high'].index = pd.Index([same_timestamp] * num_bars)
            data['low'].index = pd.Index([same_timestamp] * num_bars)
            data['close'].index = pd.Index([same_timestamp] * num_bars)
            data['volume'].index = pd.Index([same_timestamp] * num_bars)

        return data

    def test_should_recalculate_heavy_features_first_call(self):
        """Test that heavy features are recalculated on first call."""
        # Mock feature calculations
        self.signal_gen.features.calculate_liquidity_zones = Mock(return_value={'zones': [50000, 50500]})
        self.signal_gen.features.calculate_advanced_supertrend = Mock(return_value={'supertrend': [49000]})
        self.signal_gen.features.calculate_enhanced_chandelier_exit = Mock(return_value={'long_exit': [49500]})
        self.signal_gen.features.calculate_market_regime_classification = Mock(return_value={'regime': 'TRENDING'})
        self.signal_gen.features.calculate_order_flow_imbalance = Mock(return_value={'imbalance': 0.5})

        # First call should always recalculate
        result = self.signal_gen._calculate_features_conditionally(self.market_data)

        # Verify heavy features were calculated
        assert self.signal_gen.features.calculate_liquidity_zones.called
        assert self.signal_gen.features.calculate_advanced_supertrend.called
        assert self.signal_gen.features.calculate_enhanced_chandelier_exit.called
        assert self.signal_gen.features.calculate_market_regime_classification.called

        # Verify lightweight features were calculated
        assert self.signal_gen.features.calculate_order_flow_imbalance.called

        # Verify caching flag is correct
        assert result['cached'] is False  # First call should not be cached

    def test_should_use_cached_heavy_features_on_subsequent_calls(self):
        """Test that cached heavy features are used on subsequent calls within same bar."""
        # Mock feature calculations
        self.signal_gen.features.calculate_liquidity_zones = Mock(return_value={'zones': [50000, 50500]})
        self.signal_gen.features.calculate_advanced_supertrend = Mock(return_value={'supertrend': [49000]})
        self.signal_gen.features.calculate_enhanced_chandelier_exit = Mock(return_value={'long_exit': [49500]})
        self.signal_gen.features.calculate_market_regime_classification = Mock(return_value={'regime': 'TRENDING'})
        self.signal_gen.features.calculate_order_flow_imbalance = Mock(return_value={'imbalance': 0.5})

        # First call
        self.signal_gen._calculate_features_conditionally(self.market_data)

        # Reset call counts
        self.signal_gen.features.calculate_liquidity_zones.reset_mock()
        self.signal_gen.features.calculate_advanced_supertrend.reset_mock()
        self.signal_gen.features.calculate_enhanced_chandelier_exit.reset_mock()
        self.signal_gen.features.calculate_market_regime_classification.reset_mock()
        self.signal_gen.features.calculate_order_flow_imbalance.reset_mock()

        # Second call with same timestamp (same bar) - use cached heavy features
        same_bar_data = self._generate_sample_data(use_same_timestamp=True)
        result = self.signal_gen._calculate_features_conditionally(same_bar_data)

        # Verify heavy features were NOT recalculated (using cache)
        assert not self.signal_gen.features.calculate_liquidity_zones.called
        assert not self.signal_gen.features.calculate_advanced_supertrend.called
        assert not self.signal_gen.features.calculate_enhanced_chandelier_exit.called
        assert not self.signal_gen.features.calculate_market_regime_classification.called

        # Verify lightweight features WERE still calculated
        assert self.signal_gen.features.calculate_order_flow_imbalance.called

        # Verify caching flag is correct
        assert result['cached'] is True  # Should be cached

    def test_force_calculation_overrides_cache(self):
        """Test that force_calculation=True bypasses cache."""
        # Mock feature calculations
        self.signal_gen.features.calculate_liquidity_zones = Mock(return_value={'zones': [50000, 50500]})
        self.signal_gen.features.calculate_advanced_supertrend = Mock(return_value={'supertrend': [49000]})
        self.signal_gen.features.calculate_enhanced_chandelier_exit = Mock(return_value={'long_exit': [49500]})
        self.signal_gen.features.calculate_market_regime_classification = Mock(return_value={'regime': 'TRENDING'})
        self.signal_gen.features.calculate_order_flow_imbalance = Mock(return_value={'imbalance': 0.5})

        # First call
        self.signal_gen._calculate_features_conditionally(self.market_data)

        # Reset call counts
        self.signal_gen.features.calculate_liquidity_zones.reset_mock()
        self.signal_gen.features.calculate_advanced_supertrend.reset_mock()
        self.signal_gen.features.calculate_enhanced_chandelier_exit.reset_mock()
        self.signal_gen.features.calculate_market_regime_classification.reset_mock()
        self.signal_gen.features.calculate_order_flow_imbalance.reset_mock()

        # Second call with force_calculation=True
        result = self.signal_gen._calculate_features_conditionally(
            self.market_data,
            force_calculation=True
        )

        # Verify ALL features were recalculated despite cache
        assert self.signal_gen.features.calculate_liquidity_zones.called
        assert self.signal_gen.features.calculate_advanced_supertrend.called
        assert self.signal_gen.features.calculate_enhanced_chandelier_exit.called
        assert self.signal_gen.features.calculate_market_regime_classification.called
        assert self.signal_gen.features.calculate_order_flow_imbalance.called

        # Verify caching flag is correct
        assert result['cached'] is False  # Force calculation means not cached

    def test_feature_cache_state_tracking(self):
        """Test that feature cache state is properly tracked."""
        # Verify initial state
        assert self.signal_gen.feature_cache['last_bar_timestamp'] is None

        # First call should set the timestamp
        self.signal_gen.features.calculate_liquidity_zones = Mock(return_value={'zones': []})
        self.signal_gen.features.calculate_advanced_supertrend = Mock(return_value={})
        self.signal_gen.features.calculate_enhanced_chandelier_exit = Mock(return_value={})
        self.signal_gen.features.calculate_market_regime_classification = Mock(return_value={})
        self.signal_gen.features.calculate_order_flow_imbalance = Mock(return_value={})

        result = self.signal_gen._calculate_features_conditionally(self.market_data)

        # Verify timestamp was set
        assert self.signal_gen.feature_cache['last_bar_timestamp'] is not None

        # Verify cached features were stored
        assert 'liquidity' in self.signal_gen.feature_cache['cached_heavy_features']

    def test_heavy_vs_lightweight_feature_separation(self):
        """Test that heavy and lightweight features are properly separated."""
        # Mock feature calculations
        self.signal_gen.features.calculate_liquidity_zones = Mock(return_value={'zones': [50000]})
        self.signal_gen.features.calculate_advanced_supertrend = Mock(return_value={'supertrend': [49000]})
        self.signal_gen.features.calculate_enhanced_chandelier_exit = Mock(return_value={'long_exit': [49500]})
        self.signal_gen.features.calculate_market_regime_classification = Mock(return_value={'regime': 'TRENDING'})
        self.signal_gen.features.calculate_order_flow_imbalance = Mock(return_value={'imbalance': 0.5})

        result = self.signal_gen._calculate_features_conditionally(self.market_data)

        # Verify result structure
        assert 'heavy_features' in result
        assert 'lightweight_features' in result
        assert 'cached' in result

        # Verify heavy features
        assert 'liquidity' in result['heavy_features']
        assert 'supertrend' in result['heavy_features']
        assert 'chandelier' in result['heavy_features']
        assert 'regime' in result['heavy_features']

        # Verify lightweight features
        assert 'orderflow' in result['lightweight_features']


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
