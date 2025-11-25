"""
Test script to verify chart and controls functionality.
This ensures the dashboard can handle various interactions without errors.
"""

import sys
sys.path.insert(0, '/Users/mrsmoothy/Downloads/Trading_bot')

from ui.dashboard import (
    fetch_market_data,
    create_interactive_chart,
    calculate_supertrend,
    calculate_chandelier_exit,
    calculate_liquidity_zones,
    calculate_market_regime_overlay,
    calculate_timeframe_alignment,
)
import pandas as pd
from datetime import datetime, timedelta

def test_chart_components():
    """Test all chart components and overlays."""
    print("Testing chart components...")

    # Test 1: Fetch market data
    print("\n1. Testing fetch_market_data()...")
    try:
        df, meta = fetch_market_data('BTCUSDT', '15m', num_bars=100)
        print(f"   ✓ Data fetched: {len(df)} rows")
        print(f"   ✓ Metadata: cache_hit={meta.get('cache_hit')}, sample_data={meta.get('used_sample_data')}")
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return False

    # Test 2: Create interactive chart
    print("\n2. Testing create_interactive_chart()...")
    try:
        features = {
            'liquidity': True,
            'supertrend': True,
            'chandelier': True,
            'orderflow': True,
            'regime': True,
            'alignment': False
        }
        fig = create_interactive_chart(df, 'BTCUSDT', '15m', features)
        print(f"   ✓ Chart created with {len(fig.data)} traces")
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return False

    # Test 3: Supertrend calculation
    print("\n3. Testing calculate_supertrend()...")
    try:
        st = calculate_supertrend(df, period=10, multiplier=3.0)
        print(f"   ✓ Supertrend calculated: {len(st['supertrend'])} values")
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return False

    # Test 4: Chandelier Exit calculation
    print("\n4. Testing calculate_chandelier_exit()...")
    try:
        ce = calculate_chandelier_exit(df, period=22, multiplier=3.0)
        print(f"   ✓ Chandelier Exit calculated: {len(ce['long_exit'])} values")
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return False

    # Test 5: Liquidity zones calculation
    print("\n5. Testing calculate_liquidity_zones()...")
    try:
        lz = calculate_liquidity_zones(df, lookback=100)
        print(f"   ✓ Liquidity zones calculated: {len(lz.get('zones', []))} zones")
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return False

    # Test 6: Market regime overlay
    print("\n6. Testing calculate_market_regime_overlay()...")
    try:
        regime = calculate_market_regime_overlay(df, short_window=20, long_window=60)
        if regime:
            print(f"   ✓ Market regime calculated: {len(regime['series'])} values")
        else:
            print(f"   ⚠ Not enough data for regime calculation")
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return False

    # Test 7: Timeframe alignment
    print("\n7. Testing calculate_timeframe_alignment()...")
    try:
        alignment = calculate_timeframe_alignment(df)
        if alignment:
            print(f"   ✓ Timeframe alignment calculated")
        else:
            print(f"   ⚠ Not enough data for alignment calculation")
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return False

    return True

def test_different_timeframes():
    """Test multiple timeframes."""
    print("\n\nTesting multiple timeframes...")
    timeframes = ['1m', '5m', '15m', '1h', '4h', '1d']

    for tf in timeframes:
        try:
            df, meta = fetch_market_data('BTCUSDT', tf, num_bars=50)
            print(f"   ✓ {tf}: {len(df)} rows, sample={meta.get('used_sample_data')}")
        except Exception as e:
            print(f"   ✗ {tf}: Error - {e}")
            return False

    return True

def test_overlay_combinations():
    """Test different overlay combinations."""
    print("\n\nTesting overlay combinations...")
    df, _ = fetch_market_data('BTCUSDT', '15m', num_bars=100)

    combinations = [
        {'liquidity': True, 'supertrend': False, 'chandelier': False, 'orderflow': True, 'regime': False, 'alignment': False},
        {'liquidity': False, 'supertrend': True, 'chandelier': True, 'orderflow': False, 'regime': True, 'alignment': True},
        {'liquidity': True, 'supertrend': True, 'chandelier': True, 'orderflow': True, 'regime': True, 'alignment': True},
    ]

    for i, features in enumerate(combinations, 1):
        try:
            fig = create_interactive_chart(df, 'BTCUSDT', '15m', features)
            print(f"   ✓ Combination {i}: {len(fig.data)} traces")
        except Exception as e:
            print(f"   ✗ Combination {i}: Error - {e}")
            return False

    return True

if __name__ == '__main__':
    print("=" * 60)
    print("Dashboard Chart & Controls Verification")
    print("=" * 60)

    all_passed = True

    # Run tests
    all_passed &= test_chart_components()
    all_passed &= test_different_timeframes()
    all_passed &= test_overlay_combinations()

    print("\n" + "=" * 60)
    if all_passed:
        print("✓ ALL TESTS PASSED")
    else:
        print("✗ SOME TESTS FAILED")
    print("=" * 60)

    sys.exit(0 if all_passed else 1)
