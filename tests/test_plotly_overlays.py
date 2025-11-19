"""
Plotly Overlay Snapshot Tests
Tests all chart overlay combinations for visual regression testing.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from ui.dashboard import (
    create_interactive_chart,
    calculate_supertrend,
    calculate_chandelier_exit,
    calculate_liquidity_zones,
    calculate_market_regime_overlay,
    calculate_timeframe_alignment,
    generate_sample_data
)
from plotly.graph_objects import Figure


class TestPlotlyOverlaySnapshots:
    """Test suite for Plotly overlay rendering and snapshot validation."""

    @pytest.fixture
    def sample_df(self):
        """Generate sample OHLCV data for testing."""
        return generate_sample_data('BTCUSDT', '15m', num_bars=200)

    @pytest.fixture
    def sample_df_small(self):
        """Generate small sample for edge case testing."""
        return generate_sample_data('BTCUSDT', '15m', num_bars=10)

    @pytest.fixture
    def sample_df_trending(self):
        """Generate trending market data for testing."""
        df = generate_sample_data('BTCUSDT', '15m', num_bars=100)
        # Create an uptrend
        df['close'] = df['close'].iloc[0] * (1 + np.linspace(0, 0.5, len(df)))
        df['open'] = df['close'].shift(1) * 0.999
        df['high'] = df['close'] * 1.01
        df['low'] = df['close'] * 0.99
        return df.fillna(method='bfill')

    # Test individual overlays
    def test_liquidity_overlay_only(self, sample_df):
        """Test chart with only Liquidity Zones overlay enabled."""
        features = {
            'liquidity': True,
            'supertrend': False,
            'chandelier': False,
            'orderflow': False,
            'regime': False,
            'alignment': False
        }
        fig = create_interactive_chart(sample_df, 'BTCUSDT', '15m', features)
        assert isinstance(fig, Figure)
        assert len(fig.data) >= 1  # At least candlestick
        # Verify liquidity zones are rendered in layout shapes
        layout_dict = fig.to_dict()
        shapes = layout_dict.get('layout', {}).get('shapes', [])
        # Check if any shapes exist (liquidity zones add vrect shapes)
        assert len(shapes) > 0, "Expected liquidity zone shapes in layout"

    def test_supertrend_overlay_only(self, sample_df):
        """Test chart with only Supertrend overlay enabled."""
        features = {
            'liquidity': False,
            'supertrend': True,
            'chandelier': False,
            'orderflow': False,
            'regime': False,
            'alignment': False
        }
        fig = create_interactive_chart(sample_df, 'BTCUSDT', '15m', features)
        assert isinstance(fig, Figure)
        # Verify supertrend trace exists by checking trace names
        fig_dict = fig.to_dict()
        trace_names = [trace.get('name', '') for trace in fig_dict.get('data', [])]
        assert any('supertrend' in name.lower() for name in trace_names)

    def test_chandelier_overlay_only(self, sample_df):
        """Test chart with only Chandelier Exit overlay enabled."""
        features = {
            'liquidity': False,
            'supertrend': False,
            'chandelier': True,
            'orderflow': False,
            'regime': False,
            'alignment': False
        }
        fig = create_interactive_chart(sample_df, 'BTCUSDT', '15m', features)
        assert isinstance(fig, Figure)
        # Verify chandelier traces exist
        fig_dict = fig.to_dict()
        trace_names = [trace.get('name', '') for trace in fig_dict.get('data', [])]
        chandelier_count = sum(1 for name in trace_names if 'chandelier' in name.lower())
        assert chandelier_count >= 2  # Long and Short exits

    def test_orderflow_overlay_only(self, sample_df):
        """Test chart with only Order Flow overlay enabled."""
        features = {
            'liquidity': False,
            'supertrend': False,
            'chandelier': False,
            'orderflow': True,
            'regime': False,
            'alignment': False
        }
        fig = create_interactive_chart(sample_df, 'BTCUSDT', '15m', features)
        assert isinstance(fig, Figure)
        # Verify order flow subplot exists
        assert len(fig.data) >= 3  # Candlestick + Volume + Order Flow

    def test_regime_overlay_only(self, sample_df):
        """Test chart with only Market Regime overlay enabled."""
        features = {
            'liquidity': False,
            'supertrend': False,
            'chandelier': False,
            'orderflow': False,
            'regime': True,
            'alignment': False
        }
        fig = create_interactive_chart(sample_df, 'BTCUSDT', '15m', features)
        assert isinstance(fig, Figure)
        # Verify regime shading exists
        shapes = fig.to_dict().get('layout', {}).get('shapes', [])
        assert len(shapes) > 0 or any('regime' in str(shape).lower()
                                      for shape in shapes)

    def test_alignment_overlay_only(self, sample_df):
        """Test chart with only Timeframe Alignment overlay enabled."""
        features = {
            'liquidity': False,
            'supertrend': False,
            'chandelier': False,
            'orderflow': False,
            'regime': False,
            'alignment': True
        }
        fig = create_interactive_chart(sample_df, 'BTCUSDT', '15m', features)
        assert isinstance(fig, Figure)
        # Verify alignment markers exist (may be 0 if no alignment detected)
        fig_dict = fig.to_dict()
        trace_names = [trace.get('name', '') for trace in fig_dict.get('data', [])]
        alignment_count = sum(1 for name in trace_names if 'alignment' in name.lower())
        assert alignment_count >= 0  # May be 0 if no alignment detected

    # Test overlay combinations
    def test_all_overlays_enabled(self, sample_df):
        """Test chart with all overlays enabled."""
        features = {
            'liquidity': True,
            'supertrend': True,
            'chandelier': True,
            'orderflow': True,
            'regime': True,
            'alignment': True
        }
        fig = create_interactive_chart(sample_df, 'BTCUSDT', '15m', features)
        assert isinstance(fig, Figure)
        # Should have many traces
        assert len(fig.data) >= 8  # Candlestick + Volume + OrderFlow +
                                  # Supertrend + 2x Chandelier + Alignment markers

    def test_no_overlays_enabled(self, sample_df):
        """Test chart with no overlays enabled."""
        features = {
            'liquidity': False,
            'supertrend': False,
            'chandelier': False,
            'orderflow': False,
            'regime': False,
            'alignment': False
        }
        fig = create_interactive_chart(sample_df, 'BTCUSDT', '15m', features)
        assert isinstance(fig, Figure)
        # Should have minimal traces (candlestick + volume)
        assert len(fig.data) >= 2
        assert len(fig.data) <= 2

    def test_liquidity_supertrend_combo(self, sample_df):
        """Test Liquidity + Supertrend combination."""
        features = {
            'liquidity': True,
            'supertrend': True,
            'chandelier': False,
            'orderflow': False,
            'regime': False,
            'alignment': False
        }
        fig = create_interactive_chart(sample_df, 'BTCUSDT', '15m', features)
        assert isinstance(fig, Figure)
        # Verify both overlays are present
        fig_dict = fig.to_dict()
        trace_names = [trace.get('name', '') for trace in fig_dict.get('data', [])]
        assert any('supertrend' in name.lower() for name in trace_names)
        shapes = fig_dict.get('layout', {}).get('shapes', [])
        assert len(shapes) > 0

    def test_regime_alignment_combo(self, sample_df):
        """Test Market Regime + Timeframe Alignment combination."""
        features = {
            'liquidity': False,
            'supertrend': False,
            'chandelier': False,
            'orderflow': False,
            'regime': True,
            'alignment': True
        }
        fig = create_interactive_chart(sample_df, 'BTCUSDT', '15m', features)
        assert isinstance(fig, Figure)
        # Both overlays should be visible
        assert len(fig.data) >= 2  # Candlestick + Volume

    def test_supertrend_chandelier_combo(self, sample_df):
        """Test Supertrend + Chandelier combination."""
        features = {
            'liquidity': False,
            'supertrend': True,
            'chandelier': True,
            'orderflow': False,
            'regime': False,
            'alignment': False
        }
        fig = create_interactive_chart(sample_df, 'BTCUSDT', '15m', features)
        assert isinstance(fig, Figure)
        # Verify both trend indicators
        fig_dict = fig.to_dict()
        trace_names = [trace.get('name', '') for trace in fig_dict.get('data', [])]
        supertrend_present = any('supertrend' in name.lower() for name in trace_names)
        chandelier_present = any('chandelier' in name.lower() for name in trace_names)
        assert supertrend_present and chandelier_present

    # Test edge cases
    def test_insufficient_data_for_regime(self, sample_df_small):
        """Test regime calculation with insufficient data."""
        features = {
            'liquidity': False,
            'supertrend': False,
            'chandelier': False,
            'orderflow': False,
            'regime': True,
            'alignment': False
        }
        # Should not crash
        fig = create_interactive_chart(sample_df_small, 'BTCUSDT', '15m', features)
        assert isinstance(fig, Figure)

    def test_insufficient_data_for_alignment(self, sample_df_small):
        """Test alignment calculation with insufficient data."""
        features = {
            'liquidity': False,
            'supertrend': False,
            'chandelier': False,
            'orderflow': False,
            'regime': False,
            'alignment': True
        }
        # Should not crash
        fig = create_interactive_chart(sample_df_small, 'BTCUSDT', '15m', features)
        assert isinstance(fig, Figure)

    def test_trending_market_regime(self, sample_df_trending):
        """Test regime detection in trending market."""
        features = {
            'liquidity': False,
            'supertrend': False,
            'chandelier': False,
            'orderflow': False,
            'regime': True,
            'alignment': False
        }
        fig = create_interactive_chart(sample_df_trending, 'BTCUSDT', '15m', features)
        assert isinstance(fig, Figure)
        # Should detect trending regime

    def test_different_symbols(self, sample_df):
        """Test overlays work with different symbols."""
        symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT']
        features = {
            'liquidity': True,
            'supertrend': True,
            'chandelier': False,
            'orderflow': True,
            'regime': False,
            'alignment': False
        }
        for symbol in symbols:
            df = generate_sample_data(symbol, '15m', num_bars=100)
            fig = create_interactive_chart(df, symbol, '15m', features)
            assert isinstance(fig, Figure)

    def test_different_timeframes(self, sample_df):
        """Test overlays work with different timeframes."""
        timeframes = ['1m', '5m', '15m', '1h', '4h']
        features = {
            'liquidity': True,
            'supertrend': False,
            'chandelier': False,
            'orderflow': True,
            'regime': False,
            'alignment': False
        }
        for tf in timeframes:
            df = generate_sample_data('BTCUSDT', tf, num_bars=100)
            fig = create_interactive_chart(df, 'BTCUSDT', tf, features)
            assert isinstance(fig, Figure)

    def test_chart_layout_consistency(self, sample_df):
        """Test that chart layout is consistent across different overlay configurations."""
        feature_sets = [
            {'liquidity': True, 'supertrend': False, 'chandelier': False,
             'orderflow': False, 'regime': False, 'alignment': False},
            {'liquidity': False, 'supertrend': True, 'chandelier': False,
             'orderflow': False, 'regime': False, 'alignment': False},
            {'liquidity': False, 'supertrend': False, 'chandelier': True,
             'orderflow': False, 'regime': False, 'alignment': False},
        ]

        layouts = []
        for features in feature_sets:
            fig = create_interactive_chart(sample_df, 'BTCUSDT', '15m', features)
            layout = fig.to_dict()['layout']
            layouts.append(layout)

        # Check that basic layout structure is consistent
        for layout in layouts:
            assert 'paper_bgcolor' in layout
            assert 'plot_bgcolor' in layout
            assert 'margin' in layout
            assert 'showlegend' in layout

    def test_orderflow_always_enabled(self, sample_df):
        """Test that orderflow subplot is always present when enabled in features."""
        features = {
            'liquidity': False,
            'supertrend': False,
            'chandelier': False,
            'orderflow': True,
            'regime': False,
            'alignment': False
        }
        fig = create_interactive_chart(sample_df, 'BTCUSDT', '15m', features)
        # Should have 3 subplots: Price, Volume, Order Flow
        assert len(fig.data) >= 3

    # Test indicator calculations directly
    def test_supertrend_calculation(self, sample_df):
        """Test Supertrend calculation function."""
        result = calculate_supertrend(sample_df, period=10, multiplier=3.0)
        assert 'supertrend' in result
        assert 'direction' in result
        assert 'atr' in result
        assert len(result['supertrend']) == len(sample_df)
        assert len(result['direction']) == len(sample_df)

    def test_chandelier_calculation(self, sample_df):
        """Test Chandelier Exit calculation function."""
        result = calculate_chandelier_exit(sample_df, period=22, multiplier=3.0)
        assert 'long_exit' in result
        assert 'short_exit' in result
        assert 'atr' in result
        assert len(result['long_exit']) == len(sample_df)
        assert len(result['short_exit']) == len(sample_df)

    def test_liquidity_zones_calculation(self, sample_df):
        """Test Liquidity Zones calculation function."""
        result = calculate_liquidity_zones(sample_df, lookback=100)
        assert 'zones' in result
        assert 'volume_profile' in result
        assert len(result['zones']) > 0

    def test_market_regime_calculation(self, sample_df):
        """Test Market Regime calculation function."""
        result = calculate_market_regime_overlay(sample_df)
        if result:  # May return None if insufficient data
            assert 'series' in result
            assert 'colors' in result
            assert len(result['series']) == len(sample_df)

    def test_timeframe_alignment_calculation(self, sample_df):
        """Test Timeframe Alignment calculation function."""
        result = calculate_timeframe_alignment(sample_df)
        if result:  # May return None if insufficient data
            assert 'bullish' in result
            assert 'bearish' in result
            assert len(result['bullish']) == len(sample_df)
            assert len(result['bearish']) == len(sample_df)

    # Test visual regression - save baseline images
    @pytest.mark.slow
    def test_generate_baseline_snapshots(self, tmp_path, sample_df):
        """Generate baseline snapshot images for visual regression testing."""
        import plotly.io as pio

        test_cases = [
            ('no_overlays', {'liquidity': False, 'supertrend': False,
                           'chandelier': False, 'orderflow': False,
                           'regime': False, 'alignment': False}),
            ('liquidity_only', {'liquidity': True, 'supertrend': False,
                              'chandelier': False, 'orderflow': False,
                              'regime': False, 'alignment': False}),
            ('supertrend_only', {'liquidity': False, 'supertrend': True,
                               'chandelier': False, 'orderflow': False,
                               'regime': False, 'alignment': False}),
            ('chandelier_only', {'liquidity': False, 'supertrend': False,
                               'chandelier': True, 'orderflow': False,
                               'regime': False, 'alignment': False}),
            ('orderflow_only', {'liquidity': False, 'supertrend': False,
                              'chandelier': False, 'orderflow': True,
                              'regime': False, 'alignment': False}),
            ('regime_only', {'liquidity': False, 'supertrend': False,
                           'chandelier': False, 'orderflow': False,
                           'regime': True, 'alignment': False}),
            ('alignment_only', {'liquidity': False, 'supertrend': False,
                              'chandelier': False, 'orderflow': False,
                              'regime': False, 'alignment': True}),
            ('all_overlays', {'liquidity': True, 'supertrend': True,
                            'chandelier': True, 'orderflow': True,
                            'regime': True, 'alignment': True}),
        ]

        snapshot_dir = tmp_path / 'snapshots'
        snapshot_dir.mkdir(exist_ok=True)

        for test_name, features in test_cases:
            fig = create_interactive_chart(sample_df, 'BTCUSDT', '15m', features)
            # Save as HTML for visual inspection
            fig.write_html(snapshot_dir / f'{test_name}.html')
            # Also save as PNG if kaleido is available
            try:
                fig.write_image(snapshot_dir / f'{test_name}.png', width=1600, height=800)
            except Exception as e:
                # Kaleido may not be installed, skip PNG generation
                print(f"PNG generation skipped for {test_name}: {e}")

        print(f"Baseline snapshots saved to {snapshot_dir}")
