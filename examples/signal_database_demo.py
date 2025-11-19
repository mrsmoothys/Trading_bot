"""
Signal Generator Database Integration Demo
Demonstrates how signals are now persisted to database.
"""

import sys
import asyncio
sys.path.insert(0, '/Users/mrsmoothy/Downloads/Trading_bot')

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock

from core.signal_generator import SignalGenerator
from core.system_context import SystemContext
from features.engine import FeatureEngine
from deepseek.client import DeepSeekBrain
from ops.db import DatabaseManager, close_database


def create_sample_market_data(symbol: str = 'BTCUSDT', periods: int = 200) -> dict:
    """Create sample market data for testing."""
    start_date = datetime.now() - timedelta(days=periods)

    # Generate price data with trend
    np.random.seed(42)
    returns = np.random.normal(0.001, 0.02, periods)
    close_prices = pd.Series(45000 * (1 + returns).cumprod())

    # Generate OHLC
    high_prices = close_prices * (1 + np.random.uniform(0.005, 0.015, periods))
    low_prices = close_prices * (1 - np.random.uniform(0.005, 0.015, periods))
    open_prices = close_prices.shift(1).fillna(close_prices.iloc[0])
    open_prices.iloc[0] = close_prices.iloc[0]
    high_prices = np.maximum(high_prices, open_prices, close_prices)
    low_prices = np.minimum(low_prices, open_prices, close_prices)

    # Generate volume
    volume = pd.Series(np.random.uniform(500, 2000, periods))

    # Create DataFrame
    df = pd.DataFrame({
        'timestamp': pd.date_range(start=start_date, periods=periods, freq='1h'),
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': close_prices,
        'volume': volume
    })

    # Convert to dict of Series for signal generator
    return {
        'open': df['open'],
        'high': df['high'],
        'low': df['low'],
        'close': df['close'],
        'volume': df['volume']
    }


async def demo_signal_database_integration():
    """Demonstrate signal generation and database persistence."""
    print("\n" + "="*70)
    print("Signal Generator + Database Integration Demo")
    print("="*70)

    # Step 1: Create sample data
    print("\n1. Creating sample market data...")
    market_data = create_sample_market_data('BTCUSDT', 200)
    timeframe_data = {
        '1h': create_sample_market_data('BTCUSDT', 200),
        '4h': create_sample_market_data('BTCUSDT', 200)
    }
    print(f"   ✓ Generated {len(market_data['close'])} periods of OHLCV data")

    # Step 2: Initialize components
    print("\n2. Initializing trading system components...")
    system_context = SystemContext()
    feature_engine = FeatureEngine()
    deepseek_brain = DeepSeekBrain(system_context)
    print("   ✓ SystemContext initialized")
    print("   ✓ FeatureEngine initialized")
    print("   ✓ DeepSeekBrain initialized")

    # Step 3: Create SignalGenerator with database logging enabled
    print("\n3. Creating SignalGenerator with database logging...")
    signal_gen = SignalGenerator(
        system_context=system_context,
        deepseek_brain=deepseek_brain,
        feature_engine=feature_engine,
        enable_db_logging=True
    )
    print("   ✓ SignalGenerator initialized with database logging enabled")
    print(f"   ✓ Database location: data/trading_signals.db")

    # Step 4: Mock DeepSeek response
    print("\n4. Mocking DeepSeek AI analysis...")
    original_get_signal = deepseek_brain.get_trading_signal
    deepseek_brain.get_trading_signal = Mock(return_value=asyncio.Future())
    deepseek_brain.get_trading_signal.return_value.set_result({
        'action': 'LONG',
        'reasoning': 'Strong bullish signal based on technical analysis. Price is above supertrend with high timeframe alignment.',
        'confidence': 0.78
    })
    print("   ✓ DeepSeek response mocked for demo")

    # Step 5: Generate signal
    print("\n5. Generating trading signal...")
    signal = await signal_gen.generate_signal(
        symbol='BTCUSDT',
        market_data=market_data,
        timeframe_data=timeframe_data
    )

    print(f"\n   Signal Details:")
    print(f"   Symbol: {signal['symbol']}")
    print(f"   Action: {signal['action']}")
    print(f"   Confidence: {signal['confidence']:.2%}")
    print(f"   Position Size: {signal['position_size']:.2%}")
    print(f"   Market Regime: {signal.get('market_regime', 'N/A')}")
    print(f"   Signal Strength: {signal.get('signal_strength', 'N/A')}")
    print(f"   Is Tradeable: {signal.get('is_tradeable', False)}")

    # Step 6: Retrieve signal from database
    print("\n6. Retrieving signal from database...")
    history = await signal_gen.get_signal_history(limit=5)
    if history:
        latest = history[0]
        print(f"   ✓ Retrieved {len(history)} signal(s) from database")
        print(f"   ✓ Latest signal ID: {latest.get('id')}")
        print(f"   ✓ Symbol: {latest.get('symbol')}")
        print(f"   ✓ Action: {latest.get('action')}")
        print(f"   ✓ Confidence: {latest.get('confidence', 0):.2%}")

    # Step 7: Get signal statistics
    print("\n7. Signal Statistics:")
    stats = await signal_gen.get_signal_statistics(days=30)
    if stats:
        print(f"   Total Signals: {stats.get('total_signals', 0)}")
        print(f"   Average Confidence: {stats.get('avg_confidence', 0):.2%}")
        print(f"   Tradeable Signals: {stats.get('tradeable_count', 0)}")
        print(f"   Tradeable Percentage: {stats.get('tradeable_percentage', 0):.2f}%")
        print(f"   Action Breakdown: {stats.get('action_breakdown', {})}")

    # Step 8: Generate more signals for statistics
    print("\n8. Generating additional signals for demo...")
    for i in range(3):
        # Vary the mock slightly
        deepseek_brain.get_trading_signal = Mock(return_value=asyncio.Future())
        deepseek_brain.get_trading_signal.return_value.set_result({
            'action': 'SHORT' if i % 2 == 0 else 'HOLD',
            'reasoning': f'Market analysis for signal {i+2}',
            'confidence': 0.65 + (i * 0.05)
        })

        await signal_gen.generate_signal(
            symbol='BTCUSDT',
            market_data=market_data,
            timeframe_data=timeframe_data
        )

    print(f"   ✓ Generated 3 additional signals")

    # Step 9: Show updated statistics
    print("\n9. Updated Statistics:")
    stats = await signal_gen.get_signal_statistics(days=30)
    if stats:
        print(f"   Total Signals: {stats.get('total_signals', 0)}")
        print(f"   Average Confidence: {stats.get('avg_confidence', 0):.2%}")
        print(f"   Action Breakdown: {stats.get('action_breakdown', {})}")

    # Step 10: Demo database query features
    print("\n10. Database Query Features:")

    # Query by confidence
    high_conf_signals = await signal_gen.db.get_signals(
        min_confidence=0.70,
        limit=10
    )
    print(f"   ✓ High confidence signals (>= 70%): {len(high_conf_signals)}")

    # Query tradeable signals
    tradeable_signals = await signal_gen.db.get_signals(
        only_tradeable=True,
        limit=10
    )
    print(f"   ✓ Tradeable signals: {len(tradeable_signals)}")

    # Query by date range
    start_date = datetime.now() - timedelta(days=1)
    recent_signals = await signal_gen.db.get_signals(
        start_date=start_date,
        limit=10
    )
    print(f"   ✓ Recent signals (last 24h): {len(recent_signals)}")

    # Step 11: Demonstrate signal execution tracking
    if history and len(history) > 0:
        print("\n11. Signal Execution Tracking:")
        signal_id = history[0].get('id')
        if signal_id:
            # Mark as executed
            success = await signal_gen.mark_signal_executed(signal_id, True)
            print(f"   ✓ Marked signal {signal_id} as executed: {success}")

            # Verify update
            updated_signal = await signal_gen.db.get_signal(signal_id)
            if updated_signal:
                print(f"   ✓ Verified: is_executed = {updated_signal.get('is_executed')}")

    # Cleanup
    await close_database()
    print("\n12. Cleanup complete")

    print("\n" + "="*70)
    print("✅ Signal Database Integration Demo Complete!")
    print("="*70)
    print("\nKey Features Demonstrated:")
    print("  • Automatic signal persistence to SQLite database")
    print("  • Signal retrieval with various filters")
    print("  • Statistics calculation from stored signals")
    print("  • Signal execution tracking")
    print("  • Query by confidence, date, tradeable status")
    print("\nDatabase Schema:")
    print("  • signals table with 20+ columns")
    print("  • Indexed for fast queries")
    print("  • Supports JSON for complex data")
    print("  • Automatic timestamp tracking")
    print("="*70 + "\n")


async def demo_database_features():
    """Demonstrate advanced database features."""
    print("\n" + "="*70)
    print("Advanced Database Features Demo")
    print("="*70)

    # Create a separate database manager
    db = DatabaseManager("data/advanced_demo.db")

    print("\n1. Creating test signals with various attributes...")

    # Create diverse signals
    test_signals = [
        {
            'symbol': 'BTCUSDT',
            'action': 'LONG',
            'confidence': 0.85,
            'position_size': 0.03,
            'reasoning': 'Strong bullish momentum',
            'deepseek_confidence': 0.90,
            'feature_confidence': 0.80,
            'current_price': 45000.0,
            'timestamp': datetime.now().isoformat(),
            'market_regime': 'TRENDING_LOW_VOL',
            'risk_adjustment': {'volatility_adj': 1.1},
            'volatility_percentile': 0.30,
            'trend_strength': 0.85,
            'entry_conditions': {'preferred_entry': 44900.0},
            'exit_strategy': {'stop_loss': 44000.0},
            'feature_highlights': ['Near liquidity zone', 'Strong supertrend'],
            'is_tradeable': True,
            'signal_strength': 'STRONG'
        },
        {
            'symbol': 'ETHUSDT',
            'action': 'SHORT',
            'confidence': 0.72,
            'position_size': 0.02,
            'reasoning': 'Bearish divergence detected',
            'deepseek_confidence': 0.75,
            'feature_confidence': 0.69,
            'current_price': 2800.0,
            'timestamp': (datetime.now() - timedelta(hours=1)).isoformat(),
            'market_regime': 'RANGING_EXPANSION',
            'risk_adjustment': {'volatility_adj': 0.8},
            'volatility_percentile': 0.75,
            'trend_strength': 0.45,
            'is_tradeable': True,
            'signal_strength': 'MEDIUM'
        },
        {
            'symbol': 'BTCUSDT',
            'action': 'HOLD',
            'confidence': 0.45,
            'position_size': 0.0,
            'reasoning': 'Unclear market conditions',
            'deepseek_confidence': 0.50,
            'feature_confidence': 0.40,
            'current_price': 45100.0,
            'timestamp': (datetime.now() - timedelta(hours=2)).isoformat(),
            'market_regime': 'TRANSITION',
            'is_tradeable': False,
            'signal_strength': 'NO_SIGNAL'
        }
    ]

    # Save signals
    signal_ids = []
    for signal in test_signals:
        signal_id = await db.save_signal(signal)
        signal_ids.append(signal_id)
        print(f"   ✓ Saved {signal['symbol']} {signal['action']} signal (ID: {signal_id})")

    print("\n2. Querying with filters...")

    # Filter by symbol
    btc_signals = await db.get_signals(symbol='BTCUSDT', limit=10)
    print(f"   ✓ BTCUSDT signals: {len(btc_signals)}")

    # Filter by confidence
    high_conf = await db.get_signals(min_confidence=0.70, limit=10)
    print(f"   ✓ High confidence signals (>= 70%): {len(high_conf)}")

    # Filter by tradeable
    tradeable = await db.get_signals(only_tradeable=True, limit=10)
    print(f"   ✓ Tradeable signals: {len(tradeable)}")

    print("\n3. Statistics by timeframe...")

    # Overall statistics
    all_stats = await db.get_signal_statistics(days=30)
    print(f"   All symbols - Total: {all_stats.get('total_signals')}, Avg confidence: {all_stats.get('avg_confidence', 0):.2%}")

    # BTC-specific statistics
    btc_stats = await db.get_signal_statistics(symbol='BTCUSDT', days=30)
    print(f"   BTCUSDT - Total: {btc_stats.get('total_signals')}, Avg confidence: {btc_stats.get('avg_confidence', 0):.2%}")

    print("\n4. Updating signal execution status...")

    # Mark first signal as executed
    if signal_ids:
        await db.update_signal_execution(signal_ids[0], True)
        print(f"   ✓ Marked signal {signal_ids[0]} as executed")

        # Verify
        updated = await db.get_signal(signal_ids[0])
        print(f"   ✓ Verified execution status: {updated.get('is_executed')}")

    print("\n5. Retrieving latest signal for each symbol...")

    symbols = ['BTCUSDT', 'ETHUSDT']
    for symbol in symbols:
        latest = await db.get_latest_signal(symbol)
        if latest:
            print(f"   ✓ {symbol} latest: {latest.get('action')} (confidence: {latest.get('confidence', 0):.2%})")

    # Cleanup
    await db.close()

    print("\n" + "="*70)
    print("✅ Advanced Database Features Demo Complete!")
    print("="*70 + "\n")


if __name__ == '__main__':
    print("\n" + "🚀" * 35)
    print("SIGNAL DATABASE INTEGRATION DEMONSTRATION")
    print("🚀" * 35)

    # Run demos
    asyncio.run(demo_signal_database_integration())
    asyncio.run(demo_database_features())

    print("\n" + "✨" * 35)
    print("ALL DEMONSTRATIONS COMPLETE!")
    print("✨" * 35)
    print("\n" + "="*70)
    print("SUMMARY - Task #4 Completed: Signal Generator Database Logging")
    print("="*70)
    print("\nWhat Was Implemented:")
    print("  1. Created ops/db.py:")
    print("     • DatabaseManager class with SQLite + SQLAlchemy")
    print("     • SignalRecord model with comprehensive schema")
    print("     • CRUD operations for signals")
    print("     • Statistics and analytics queries")
    print("     • Connection pooling and error handling")
    print("\n  2. Enhanced core/signal_generator.py:")
    print("     • Added enable_db_logging parameter")
    print("     • Automatic signal persistence after generation")
    print("     • Database query methods (history, statistics)")
    print("     • Signal execution tracking")
    print("     • Graceful fallback if database unavailable")
    print("\n  3. Created comprehensive tests:")
    print("     • test_signal_database.py (600+ lines)")
    print("     • Tests for all database operations")
    print("     • Tests for signal integration")
    print("     • Edge cases and error handling")
    print("\n  4. Database Features:")
    print("     • 20+ columns for complete signal tracking")
    print("     • Indexed for fast queries")
    print("     • JSON fields for complex data")
    print("     • Automatic cleanup methods")
    print("     • Statistics generation")
    print("\n" + "="*70)
    print("Ready for Task #5: E2E Tests for Signal Flow")
    print("="*70 + "\n")
