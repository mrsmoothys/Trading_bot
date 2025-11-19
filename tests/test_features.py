"""
Unit tests for the Feature Engineering Engine.
Tests all feature calculation functions and the FeatureEngine class.
"""

import sys
sys.path.insert(0, '/Users/mrsmoothy/Downloads/Trading_bot')

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from features.engine import (
    FeatureEngine,
    calculate_liquidity_zones,
    calculate_order_flow_imbalance,
    calculate_enhanced_chandelier_exit,
    calculate_advanced_supertrend,
    calculate_market_regime,
    calculate_multi_timeframe_convergence,
    calculate_rsi,
)


def create_sample_ohlcv_data(num_periods: int = 200) -> pd.DataFrame:
    """
    Create sample OHLCV data for testing.

    Args:
        num_periods: Number of periods to generate

    Returns:
        DataFrame with OHLCV data
    """
    # Generate timestamps
    start_date = datetime.now() - timedelta(days=num_periods)
    timestamps = pd.date_range(start=start_date, periods=num_periods, freq='1h')

    # Generate price data with some trend and volatility
    np.random.seed(42)  # For reproducible tests
    returns = np.random.normal(0.001, 0.02, num_periods)
    close_prices = pd.Series(45000 * (1 + returns).cumprod())

    # Generate OHLC from close
    high_noise = np.random.uniform(0.005, 0.015, num_periods)
    low_noise = np.random.uniform(0.005, 0.015, num_periods)

    high_prices = close_prices * (1 + high_noise)
    low_prices = close_prices * (1 - low_noise)
    open_prices = close_prices.shift(1).fillna(close_prices.iloc[0])
    open_prices.iloc[0] = close_prices.iloc[0]

    # Ensure open, high, low make sense
    high_prices = np.maximum(high_prices, open_prices, close_prices)
    low_prices = np.minimum(low_prices, open_prices, close_prices)

    # Generate volume
    base_volume = 1000
    volume = pd.Series(np.random.uniform(base_volume * 0.5, base_volume * 2.0, num_periods))

    df = pd.DataFrame({
        'timestamp': timestamps,
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': close_prices,
        'volume': volume
    })

    return df


class TestFeatureCalculations:
    """Test individual feature calculation functions."""

    def test_calculate_liquidity_zones_basic(self):
        """Test basic liquidity zone calculation."""
        df = create_sample_ohlcv_data(200)

        result = calculate_liquidity_zones(
            high=df['high'],
            low=df['low'],
            close=df['close'],
            volume=df['volume'],
            lookback_periods=100,
            price_bins=50
        )

        # Verify structure
        assert 'liquidity_zones' in result
        assert 'nearest_zone' in result
        assert 'distance_to_zone_pct' in result
        assert 'zone_strength' in result
        assert 'above_below_zone' in result
        assert 'total_zones' in result

        # Verify types
        assert isinstance(result['liquidity_zones'], list)
        assert isinstance(result['nearest_zone'], float)
        assert isinstance(result['distance_to_zone_pct'], float)
        assert isinstance(result['zone_strength'], float)
        assert isinstance(result['total_zones'], int)

        # Verify values make sense
        assert result['total_zones'] >= 0
        assert result['distance_to_zone_pct'] >= 0
        assert result['zone_strength'] >= 0
        assert result['above_below_zone'] in ['above', 'below', 'at']

        print("✅ test_calculate_liquidity_zones_basic passed")

    def test_calculate_liquidity_zones_insufficient_data(self):
        """Test with insufficient data."""
        df = create_sample_ohlcv_data(5)

        result = calculate_liquidity_zones(
            high=df['high'],
            low=df['low'],
            close=df['close'],
            volume=df['volume'],
            lookback_periods=100
        )

        # Should handle gracefully
        assert 'liquidity_zones' in result
        print("✅ test_calculate_liquidity_zones_insufficient_data passed")

    def test_calculate_order_flow_imbalance_basic(self):
        """Test basic order flow imbalance calculation."""
        df = create_sample_ohlcv_data(100)

        result = calculate_order_flow_imbalance(
            open_price=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            volume=df['volume'],
            period=20
        )

        # Verify structure
        assert 'order_flow_imbalance' in result
        assert 'buying_pressure_20ma' in result
        assert 'selling_pressure_20ma' in result
        assert 'imbalance_trend' in result

        # Verify types
        assert isinstance(result['order_flow_imbalance'], float)
        assert isinstance(result['buying_pressure_20ma'], float)
        assert isinstance(result['selling_pressure_20ma'], float)
        assert isinstance(result['imbalance_trend'], float)

        # Values should be reasonable
        assert abs(result['order_flow_imbalance']) < 10  # Sanity check
        assert 0 <= result['buying_pressure_20ma'] <= 1
        assert 0 <= result['selling_pressure_20ma'] <= 1

        print("✅ test_calculate_order_flow_imbalance_basic passed")

    def test_calculate_enhanced_chandelier_exit(self):
        """Test enhanced chandelier exit calculation."""
        df = create_sample_ohlcv_data(100)

        result = calculate_enhanced_chandelier_exit(
            high=df['high'],
            low=df['low'],
            close=df['close'],
            period=22,
            multiplier=3.0
        )

        # Verify structure
        assert 'chandelier_long_stop' in result
        assert 'chandelier_short_stop' in result
        assert 'chandelier_trend' in result
        assert 'distance_to_stop_pct' in result
        assert 'adaptive_multiplier' in result

        # Verify types
        assert isinstance(result['chandelier_long_stop'], float)
        assert isinstance(result['chandelier_short_stop'], float)
        assert isinstance(result['chandelier_trend'], str)
        assert isinstance(result['distance_to_stop_pct'], float)
        assert isinstance(result['adaptive_multiplier'], float)

        # Verify trend values
        assert result['chandelier_trend'] in ['bullish', 'bearish', 'neutral']

        print("✅ test_calculate_enhanced_chandelier_exit passed")

    def test_calculate_advanced_supertrend(self):
        """Test advanced supertrend calculation."""
        df = create_sample_ohlcv_data(100)

        result = calculate_advanced_supertrend(
            high=df['high'],
            low=df['low'],
            close=df['close'],
            atr_period=10,
            multiplier=3.0
        )

        # Verify structure
        assert 'supertrend_value' in result
        assert 'supertrend_trend' in result
        assert 'supertrend_strength' in result
        assert 'trend_consistency' in result
        assert 'price_vs_supertrend' in result

        # Verify types
        assert isinstance(result['supertrend_value'], float)
        assert isinstance(result['supertrend_trend'], str)
        assert isinstance(result['supertrend_strength'], float)
        assert isinstance(result['trend_consistency'], float)
        assert isinstance(result['price_vs_supertrend'], float)

        # Verify trend values
        assert result['supertrend_trend'] in ['uptrend', 'downtrend', 'neutral']

        # Values should be reasonable
        assert result['trend_consistency'] >= 0
        assert result['trend_consistency'] <= 1

        print("✅ test_calculate_advanced_supertrend passed")

    def test_calculate_market_regime(self):
        """Test market regime classification."""
        df = create_sample_ohlcv_data(200)

        result = calculate_market_regime(
            close=df['close'],
            high=df['high'],
            low=df['low'],
            volume=df['volume'],
            period=50
        )

        # Verify structure
        assert 'market_regime' in result
        assert 'volatility_percentile' in result
        assert 'trend_strength' in result
        assert 'range_percentile' in result
        assert 'volume_anomaly' in result
        assert 'regime_confidence' in result

        # Verify types
        assert isinstance(result['market_regime'], str)
        assert isinstance(result['volatility_percentile'], float)
        assert isinstance(result['trend_strength'], float)
        assert isinstance(result['range_percentile'], float)
        assert isinstance(result['volume_anomaly'], float)
        assert isinstance(result['regime_confidence'], float)

        # Verify regime values
        valid_regimes = [
            'TRENDING_HIGH_VOL', 'TRENDING_LOW_VOL', 'RANGING_COMPRESSION',
            'RANGING_EXPANSION', 'TRANSITION'
        ]
        assert result['market_regime'] in valid_regimes

        # Percentiles should be 0-1
        assert 0 <= result['volatility_percentile'] <= 1
        assert 0 <= result['range_percentile'] <= 1
        assert 0 <= result['regime_confidence'] <= 1

        print("✅ test_calculate_market_regime passed")

    def test_calculate_multi_timeframe_convergence(self):
        """Test multi-timeframe convergence calculation."""
        # Create sample data for different timeframes
        timeframe_data = {
            '5m': create_sample_ohlcv_data(100),
            '15m': create_sample_ohlcv_data(100),
            '1h': create_sample_ohlcv_data(100),
            '4h': create_sample_ohlcv_data(100),
        }

        result = calculate_multi_timeframe_convergence(
            symbol='BTCUSDT',
            timeframe_data=timeframe_data,
            timeframes=['5m', '15m', '1h', '4h']
        )

        # Verify structure
        assert 'timeframe_alignment' in result
        assert 'primary_trend' in result
        assert 'convergence_details' in result
        assert 'trading_timeframe_recommendation' in result

        # Verify types
        assert isinstance(result['timeframe_alignment'], float)
        assert isinstance(result['primary_trend'], str)
        assert isinstance(result['convergence_details'], dict)
        assert isinstance(result['trading_timeframe_recommendation'], str)

        # Verify alignment score
        assert 0 <= result['timeframe_alignment'] <= 1

        # Verify trend value
        assert result['primary_trend'] in ['BULLISH', 'BEARISH', 'NEUTRAL']

        # Verify recommendation
        assert result['trading_timeframe_recommendation'] in ['SWING', 'INTRADAY']

        print("✅ test_calculate_multi_timeframe_convergence passed")

    def test_calculate_multi_timeframe_convergence_missing_tf(self):
        """Test with missing timeframes."""
        timeframe_data = {
            '5m': create_sample_ohlcv_data(100),
            # Missing other timeframes
        }

        result = calculate_multi_timeframe_convergence(
            symbol='BTCUSDT',
            timeframe_data=timeframe_data,
            timeframes=['5m', '15m', '1h', '4h']
        )

        # Should handle missing timeframes gracefully
        assert 'timeframe_alignment' in result
        print("✅ test_calculate_multi_timeframe_convergence_missing_tf passed")

    def test_calculate_rsi(self):
        """Test RSI calculation."""
        df = create_sample_ohlcv_data(100)
        close = df['close']

        rsi = calculate_rsi(close, period=14)

        # Verify it's a Series
        assert isinstance(rsi, pd.Series)
        assert len(rsi) == len(close)

        # RSI should be between 0 and 100 (mostly)
        assert rsi.min() >= 0
        assert rsi.max() <= 100

        print("✅ test_calculate_rsi passed")


class TestFeatureEngine:
    """Test the FeatureEngine class."""

    def test_feature_engine_initialization(self):
        """Test FeatureEngine initialization."""
        engine = FeatureEngine()

        assert engine.feature_cache == {}
        assert engine.cache_ttl == 300

        print("✅ test_feature_engine_initialization passed")

    def test_compute_all_features(self):
        """Test computing all features."""
        import asyncio

        engine = FeatureEngine()
        df = create_sample_ohlcv_data(200)

        # Prepare market data
        market_data = {
            'open': df['open'],
            'high': df['high'],
            'low': df['low'],
            'close': df['close'],
            'volume': df['volume']
        }

        # Compute features
        features = asyncio.run(engine.compute_all_features('BTCUSDT', market_data))

        # Verify all expected features are present
        expected_features = [
            'liquidity_zones', 'nearest_zone', 'distance_to_zone_pct', 'zone_strength',
            'order_flow_imbalance', 'buying_pressure_20ma', 'selling_pressure_20ma',
            'chandelier_long_stop', 'chandelier_short_stop', 'chandelier_trend',
            'supertrend_value', 'supertrend_trend', 'supertrend_strength',
            'market_regime', 'volatility_percentile', 'trend_strength',
            'timeframe_alignment', 'primary_trend',
            'symbol', 'timestamp'
        ]

        for feature in expected_features:
            assert feature in features, f"Missing feature: {feature}"

        # Verify metadata
        assert features['symbol'] == 'BTCUSDT'
        assert 'timestamp' in features

        print("✅ test_compute_all_features passed")

    def test_compute_all_features_with_timeframes(self):
        """Test computing features with multi-timeframe data."""
        import asyncio

        engine = FeatureEngine()
        df = create_sample_ohlcv_data(200)

        market_data = {
            'open': df['open'],
            'high': df['high'],
            'low': df['low'],
            'close': df['close'],
            'volume': df['volume']
        }

        # Add multi-timeframe data
        timeframe_data = {
            '1h': create_sample_ohlcv_data(200),
            '4h': create_sample_ohlcv_data(200),
        }

        features = asyncio.run(engine.compute_all_features('BTCUSDT', market_data, timeframe_data))

        # Should have convergence details
        assert 'convergence_details' in features
        assert 'trading_timeframe_recommendation' in features

        print("✅ test_compute_all_features_with_timeframes passed")

    def test_compute_all_features_insufficient_data(self):
        """Test with insufficient data."""
        import asyncio

        engine = FeatureEngine()
        df = create_sample_ohlcv_data(10)  # Very little data

        market_data = {
            'open': df['open'],
            'high': df['high'],
            'low': df['low'],
            'close': df['close'],
            'volume': df['volume']
        }

        # Should handle gracefully
        features = asyncio.run(engine.compute_all_features('BTCUSDT', market_data))

        assert 'symbol' in features
        assert features['symbol'] == 'BTCUSDT'

        print("✅ test_compute_all_features_insufficient_data passed")

    def test_get_feature_highlights(self):
        """Test extracting feature highlights."""
        engine = FeatureEngine()

        # Create sample features
        features = {
            'market_regime': 'TRENDING_HIGH_VOL',
            'distance_to_zone_pct': 0.005,  # Near zone
            'order_flow_imbalance': 0.15,  # Strong buying
            'trend_consistency': 0.9,  # Very consistent
            'supertrend_trend': 'uptrend',
            'timeframe_alignment': 0.8,  # High alignment
            'some_other_feature': 'value'
        }

        highlights = engine.get_feature_highlights(features)

        # Should have extracted relevant highlights
        assert isinstance(highlights, list)
        assert len(highlights) > 0

        # Check for expected highlights
        highlights_text = ' '.join(highlights).lower()
        assert 'regime' in highlights_text or 'trending' in highlights_text
        assert 'near' in highlights_text or 'liquidity' in highlights_text

        print("✅ test_get_feature_highlights passed")

    def test_get_feature_highlights_no_matches(self):
        """Test with features that don't match any highlights."""
        engine = FeatureEngine()

        # Neutral features
        features = {
            'market_regime': 'TRANSITION',
            'distance_to_zone_pct': 0.1,  # Far from zone
            'order_flow_imbalance': 0.0,  # Neutral
            'trend_consistency': 0.2,  # Low consistency
            'timeframe_alignment': 0.3,  # Low alignment
        }

        highlights = engine.get_feature_highlights(features)

        # Should return empty or minimal highlights
        assert isinstance(highlights, list)

        print("✅ test_get_feature_highlights_no_matches passed")


class TestFeatureEdgeCases:
    """Test edge cases and error conditions."""

    def test_with_nan_values(self):
        """Test handling of NaN values in data."""
        df = create_sample_ohlcv_data(100)
        df.loc[50, 'close'] = np.nan
        df.loc[60, 'volume'] = np.nan

        result = calculate_order_flow_imbalance(
            open_price=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            volume=df['volume']
        )

        # Should handle NaN gracefully
        assert 'order_flow_imbalance' in result
        assert isinstance(result['order_flow_imbalance'], float)

        print("✅ test_with_nan_values passed")

    def test_with_constant_prices(self):
        """Test with flat/constant prices."""
        df = pd.DataFrame({
            'open': [45000] * 100,
            'high': [45000] * 100,
            'low': [45000] * 100,
            'close': [45000] * 100,
            'volume': [1000] * 100
        })

        result = calculate_market_regime(
            close=df['close'],
            high=df['high'],
            low=df['low'],
            volume=df['volume']
        )

        # Should handle constant prices gracefully
        assert 'market_regime' in result

        print("✅ test_with_constant_prices passed")

    def test_with_zero_volume(self):
        """Test with zero volume."""
        df = create_sample_ohlcv_data(100)
        df['volume'] = 0

        result = calculate_liquidity_zones(
            high=df['high'],
            low=df['low'],
            close=df['close'],
            volume=df['volume']
        )

        # Should handle zero volume
        assert 'liquidity_zones' in result

        print("✅ test_with_zero_volume passed")


if __name__ == '__main__':
    print("\n" + "="*70)
    print("Running Feature Engine Tests")
    print("="*70 + "\n")

    # Test individual feature calculations
    print("Testing Feature Calculations...")
    test_calc = TestFeatureCalculations()
    test_calc.test_calculate_liquidity_zones_basic()
    test_calc.test_calculate_liquidity_zones_insufficient_data()
    test_calc.test_calculate_order_flow_imbalance_basic()
    test_calc.test_calculate_enhanced_chandelier_exit()
    test_calc.test_calculate_advanced_supertrend()
    test_calc.test_calculate_market_regime()
    test_calc.test_calculate_multi_timeframe_convergence()
    test_calc.test_calculate_multi_timeframe_convergence_missing_tf()
    test_calc.test_calculate_rsi()

    print("\nTesting FeatureEngine Class...")
    test_engine = TestFeatureEngine()
    test_engine.test_feature_engine_initialization()
    test_engine.test_compute_all_features()
    test_engine.test_compute_all_features_with_timeframes()
    test_engine.test_compute_all_features_insufficient_data()
    test_engine.test_get_feature_highlights()
    test_engine.test_get_feature_highlights_no_matches()

    print("\nTesting Edge Cases...")
    test_edge = TestFeatureEdgeCases()
    test_edge.test_with_nan_values()
    test_edge.test_with_constant_prices()
    test_edge.test_with_zero_volume()

    print("\n" + "="*70)
    print("✅ All feature engine tests passed successfully!")
    print("="*70)
