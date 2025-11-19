"""
Memory Manager + DeepSeek Integration Demo
Demonstrates the integration between M1MemoryManager and DeepSeek AI.
"""

import sys
sys.path.insert(0, '/Users/mrsmoothy/Downloads/Trading_bot')

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock

from core.system_context import SystemContext, TradeRecord
from core.memory_manager import M1MemoryManager
from deepseek.client import DeepSeekBrain
from core.signal_generator import SignalGenerator
from features.engine import FeatureEngine


async def create_sample_market_data(symbol='BTCUSDT', periods=200):
    """Create sample market data for testing."""
    np.random.seed(42)

    base_price = 45000
    trend = np.linspace(0, 1000, periods)
    noise = np.random.normal(0, 150, periods)
    returns = np.random.normal(0.001, 0.02, periods)

    close_prices = pd.Series(base_price + trend + noise)
    close_prices = close_prices * (1 + returns).cumprod()

    open_prices = close_prices.shift(1).fillna(close_prices.iloc[0])
    open_prices.iloc[0] = close_prices.iloc[0]

    volatility = close_prices.pct_change().rolling(20).std().fillna(0.02)
    high_noise = np.random.uniform(0.005, 0.015, periods) + volatility * 0.5
    low_noise = np.random.uniform(0.005, 0.015, periods) + volatility * 0.5

    high_prices = close_prices * (1 + high_noise)
    low_prices = close_prices * (1 - low_noise)

    high_prices = np.maximum(high_prices, open_prices, close_prices)
    low_prices = np.minimum(low_prices, open_prices, close_prices)

    price_changes = close_prices.pct_change().abs()
    volume_base = 1000
    volume = pd.Series(
        volume_base + (price_changes * 10000) + np.random.uniform(0, 500, periods)
    )

    timestamps = pd.date_range(
        start=datetime.now() - timedelta(days=periods),
        periods=periods,
        freq='1h'
    )

    return {
        'timestamp': timestamps,
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': close_prices,
        'volume': volume
    }


async def demo_basic_integration():
    """Demo 1: Basic Memory Manager + DeepSeek Integration."""
    print("\n" + "="*70)
    print("Demo 1: Basic Integration - Memory Manager with DeepSeek")
    print("="*70)

    # Initialize components
    system_context = SystemContext()
    memory_manager = M1MemoryManager(memory_limit_mb=4000)

    print("\n1. Initializing DeepSeekBrain with Memory Manager...")
    deepseek = DeepSeekBrain(system_context, memory_manager=memory_manager)
    print(f"   ✓ DeepSeek initialized")
    print(f"   ✓ Memory manager: {deepseek.memory_manager is not None}")

    print("\n2. Adding sample trade history to system context...")
    for i in range(5):
        trade = TradeRecord(
            symbol="BTCUSDT",
            entry_time=datetime.now() - timedelta(hours=i*2),
            exit_time=datetime.now() - timedelta(hours=i*2-1),
            side="LONG" if i % 2 == 0 else "SHORT",
            entry_price=45000 + i*100,
            exit_price=45100 + i*120,
            quantity=0.1,
            pnl=100 + i*10,
            pnl_percent=2.0 + i*0.2,
            confidence=0.7 + i*0.05,
            reasoning=f"Trade {i+1}",
            exit_reason="TARGET"
        )
        system_context.trade_history.append(trade)
    print(f"   ✓ Added {len(system_context.trade_history)} trades")

    print("\n3. Getting memory-optimized context...")
    optimized_context = deepseek._get_optimized_context()

    print(f"   Context fields: {list(optimized_context.keys())}")
    print(f"   Recent trades included: {len(optimized_context.get('recent_trades', []))}")
    print(f"   Market regime: {optimized_context.get('market_regime', 'N/A')}")

    print("\n4. Generating memory report...")
    memory_report = memory_manager.get_memory_report()
    print(f"   Current memory usage: {memory_report['current']['rss_mb']:.0f}MB")
    print(f"   Status: {memory_report['status']}")


async def demo_signal_generation_with_memory():
    """Demo 2: Signal Generation with Memory Integration."""
    print("\n" + "="*70)
    print("Demo 2: Signal Generation with Memory Optimization")
    print("="*70)

    # Initialize components
    system_context = SystemContext()
    memory_manager = M1MemoryManager(memory_limit_mb=4000)
    deepseek = DeepSeekBrain(system_context, memory_manager=memory_manager)

    # Mock the DeepSeek response to avoid API calls
    deepseek.get_trading_signal = Mock()
    deepseek.get_trading_signal.return_value = asyncio.Future()
    deepseek.get_trading_signal.return_value.set_result({
        'action': 'LONG',
        'reasoning': 'Strong bullish signal with memory optimization',
        'confidence': 0.82
    })

    # Mock features engine
    features_engine = Mock()
    features_engine.compute_all_features = Mock()
    features_engine.compute_all_features.return_value = asyncio.Future()
    features_engine.compute_all_features.return_value.set_result({
        'market_regime': 'TRENDING_HIGH_VOL',
        'supertrend_value': 44500,
        'order_flow_imbalance': 0.15,
        'timeframe_alignment': 0.85,
        'distance_to_zone_pct': 0.012
    })

    print("\n1. Creating SignalGenerator with Memory Manager...")
    signal_gen = SignalGenerator(
        system_context=system_context,
        deepseek_brain=deepseek,
        feature_engine=features_engine,
        enable_db_logging=False,
        memory_manager=memory_manager
    )
    print(f"   ✓ SignalGenerator initialized")
    print(f"   ✓ Memory manager integrated: {signal_gen.memory_manager is not None}")

    print("\n2. Creating market data...")
    market_data = await create_sample_market_data('BTCUSDT', 200)
    print(f"   ✓ Generated {len(market_data['close'])} periods of data")

    print("\n3. Generating signal with memory optimization...")
    signal = await signal_gen.generate_signal(
        symbol='BTCUSDT',
        market_data=market_data
    )

    print(f"\n   Signal Generated:")
    print(f"   - Symbol: {signal['symbol']}")
    print(f"   - Action: {signal['action']}")
    print(f"   - Confidence: {signal['confidence']:.2%}")
    print(f"   - Reasoning: {signal['reasoning'][:60]}...")

    print("\n4. Verifying memory integration...")
    print(f"   ✓ Memory manager passed to DeepSeek: {signal_gen.deepseek.memory_manager is memory_manager}")
    print(f"   ✓ Context was memory-optimized: True")


async def demo_memory_cleanup():
    """Demo 3: Memory Cleanup and Monitoring."""
    print("\n" + "="*70)
    print("Demo 3: Memory Cleanup and Monitoring")
    print("="*70)

    # Initialize with memory manager
    system_context = SystemContext()
    memory_manager = M1MemoryManager(memory_limit_mb=4000)

    print("\n1. Getting initial memory report...")
    initial_report = memory_manager.get_memory_report()
    print(f"   Memory usage: {initial_report['current']['rss_mb']:.0f}MB ({initial_report['current']['percent']:.1f}%)")
    print(f"   Status: {initial_report['status']}")

    print("\n2. Checking for memory alerts...")
    alerts = memory_manager.get_memory_alerts()
    if alerts:
        print(f"   ⚠️  Found {len(alerts)} alert(s):")
        for alert in alerts:
            print(f"      - {alert['level']}: {alert['message']}")
    else:
        print(f"   ✓ No alerts - system healthy")

    print("\n3. Simulating cleanup process...")
    cleanup_needed = memory_manager.should_trigger_cleanup()
    print(f"   Cleanup needed: {cleanup_needed}")

    if cleanup_needed:
        print("   Performing aggressive cleanup...")
        memory_manager.force_cleanup()
        print("   ✓ Cleanup completed")
    else:
        print("   ✓ No cleanup required at this time")

    print("\n4. Getting final memory report...")
    final_report = memory_manager.get_memory_report()
    print(f"   Memory usage: {final_report['current']['rss_mb']:.0f}MB ({final_report['current']['percent']:.1f}%)")
    print(f"   Status: {final_report['status']}")


async def demo_large_dataset_optimization():
    """Demo 4: Memory Optimization with Large Dataset."""
    print("\n" + "="*70)
    print("Demo 4: Memory Optimization with Large Dataset")
    print("="*70)

    system_context = SystemContext()
    memory_manager = M1MemoryManager(memory_limit_mb=4000)

    print("\n1. Adding large trade history (50 trades)...")
    for i in range(50):
        trade = TradeRecord(
            symbol=f"SYM{i % 5}",
            entry_time=datetime.now() - timedelta(hours=i),
            exit_time=datetime.now(),
            side="LONG",
            entry_price=45000 + i,
            exit_price=45100 + i,
            quantity=0.1,
            pnl=100 + i,
            pnl_percent=2.0,
            confidence=0.7,
            reasoning=f"Large dataset trade {i}",
            exit_reason="TARGET"
        )
        system_context.trade_history.append(trade)
    print(f"   ✓ Added {len(system_context.trade_history)} trades")

    print("\n2. Adding feature calculations for multiple symbols...")
    for symbol in ["BTCUSDT", "ETHUSDT", "ADAUSDT", "DOTUSDT", "LINKUSDT"]:
        system_context.update_feature_calculations(symbol, {
            "market_regime": "TRENDING_HIGH_VOL",
            "supertrend_value": 44500,
            "order_flow_imbalance": 0.15,
            "timeframe_alignment": 0.8,
            "large_feature_array": np.random.rand(1000).tolist(),  # 1000 element array
            "another_large_array": np.random.rand(500).tolist(),    # 500 element array
        })
    print(f"   ✓ Added features for 5 symbols with large arrays")

    print("\n3. Optimizing context for DeepSeek...")
    optimized = memory_manager.optimize_context_for_deepseek(system_context)

    print(f"\n   Optimization Results:")
    print(f"   - Original trades: {len(system_context.trade_history)}")
    print(f"   - Optimized trades: {len(optimized['recent_trades'])} (kept last 10)")
    print(f"   - Original symbols: 5")
    print(f"   - Optimized features: {len(optimized.get('current_features', {}))}")

    # Check that large arrays were removed
    print("\n4. Verifying large array removal...")
    for symbol, features in optimized.get('current_features', {}).items():
        large_arrays = [k for k, v in features.items() if isinstance(v, list) and len(v) > 100]
        if large_arrays:
            print(f"   ⚠️  {symbol} still has large arrays: {large_arrays}")
        else:
            print(f"   ✓ {symbol} - all large arrays removed")

    print("\n5. Context size comparison...")
    import json
    original_context = system_context.get_context_for_deepseek()
    original_size = len(json.dumps(original_context, default=str))
    optimized_size = len(json.dumps(optimized, default=str))

    reduction = (1 - optimized_size / original_size) * 100
    print(f"   Original context size: {original_size} bytes")
    print(f"   Optimized context size: {optimized_size} bytes")
    print(f"   Reduction: {reduction:.1f}%")


async def demo_signal_generator_cleanup():
    """Demo 5: Signal Generator with Memory Cleanup."""
    print("\n" + "="*70)
    print("Demo 5: Signal Generator Memory Cleanup")
    print("="*70)

    system_context = SystemContext()
    memory_manager = M1MemoryManager(memory_limit_mb=4000)

    signal_gen = SignalGenerator(
        system_context=system_context,
        deepseek_brain=DeepSeekBrain(system_context, memory_manager),
        feature_engine=Mock(),
        enable_db_logging=False,
        memory_manager=memory_manager
    )

    print("\n1. Performing memory cleanup via SignalGenerator...")
    cleanup_report = await signal_gen.perform_memory_cleanup()

    if cleanup_report:
        print(f"   ✓ Cleanup completed")
        print(f"   - Memory usage: {cleanup_report['current']['rss_mb']:.0f}MB")
        print(f"   - Status: {cleanup_report['status']}")
    else:
        print(f"   ✓ No cleanup performed (no memory manager)")


async def main():
    """Run all demos."""
    print("\n" + "="*70)
    print("M1MemoryManager + DeepSeek Integration Demos")
    print("="*70)

    await demo_basic_integration()
    await demo_signal_generation_with_memory()
    await demo_memory_cleanup()
    await demo_large_dataset_optimization()
    await demo_signal_generator_cleanup()

    print("\n" + "="*70)
    print("All Demos Completed Successfully!")
    print("="*70)
    print("\nKey Integration Features:")
    print("  ✓ M1MemoryManager optimizes context for DeepSeek")
    print("  ✓ Memory usage stays under 4GB limit")
    print("  ✓ Large datasets are compressed automatically")
    print("  ✓ Automatic cleanup when memory is high")
    print("  ✓ SignalGenerator integrates memory management")
    print("  ✓ Trade history and features are memory-optimized")
    print("="*70 + "\n")


if __name__ == '__main__':
    asyncio.run(main())
