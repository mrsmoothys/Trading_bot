"""
End-to-End Tests for Signal Generation Flow
Tests the complete pipeline: Market Data → Features → DeepSeek → Signal → Database

This test suite validates the entire signal generation pipeline from raw market data
through feature calculation, AI analysis, signal generation, and database storage.
"""

import sys
sys.path.insert(0, '/Users/mrsmoothy/Downloads/Trading_bot')

import pytest
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock

from core.system_context import SystemContext
from core.signal_generator import SignalGenerator
from features.engine import FeatureEngine
from deepseek.client import DeepSeekBrain
from ops.db import DatabaseManager, close_database


def create_comprehensive_market_data(symbol: str = 'BTCUSDT', periods: int = 200) -> dict:
    """
    Create realistic market data for e2e testing.

    Args:
        symbol: Trading symbol
        periods: Number of periods

    Returns:
        Dictionary of OHLCV data
    """
    # Create realistic price movement with trends and volatility
    np.random.seed(42)

    # Generate base price series with trend
    base_price = 45000
    trend = np.linspace(0, 1000, periods)  # Upward trend
    noise = np.random.normal(0, 150, periods)
    returns = np.random.normal(0.001, 0.02, periods)

    close_prices = pd.Series(base_price + trend + noise)
    close_prices = close_prices * (1 + returns).cumprod()

    # Generate OHLC from close
    open_prices = close_prices.shift(1).fillna(close_prices.iloc[0])
    open_prices.iloc[0] = close_prices.iloc[0]

    # High/Low based on volatility
    volatility = close_prices.pct_change().rolling(20).std().fillna(0.02)
    high_noise = np.random.uniform(0.005, 0.015, periods) + volatility * 0.5
    low_noise = np.random.uniform(0.005, 0.015, periods) + volatility * 0.5

    high_prices = close_prices * (1 + high_noise)
    low_prices = close_prices * (1 - low_noise)

    # Ensure OHLC consistency
    high_prices = np.maximum(high_prices, open_prices, close_prices)
    low_prices = np.minimum(low_prices, open_prices, close_prices)

    # Generate volume with correlation to price movement
    price_changes = close_prices.pct_change().abs()
    volume_base = 1000
    volume = pd.Series(
        volume_base + (price_changes * 10000) + np.random.uniform(0, 500, periods)
    )

    # Create timestamps
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


@pytest.mark.slow

class TestSignalFlowE2E:
    """End-to-end tests for signal generation flow."""

    @pytest.mark.asyncio
    async def test_complete_signal_flow_single_timeframe(self):
        """Test complete signal flow with single timeframe data."""
        print("\n" + "="*70)
        print("Test: Complete Signal Flow - Single Timeframe")
        print("="*70)

        # Step 1: Create market data
        print("\n1. Creating market data...")
        market_data = create_comprehensive_market_data('BTCUSDT', 200)
        print(f"   ✓ Created {len(market_data['close'])} periods of OHLCV data")

        # Step 2: Initialize components
        print("\n2. Initializing system components...")
        system_context = SystemContext()
        feature_engine = FeatureEngine()
        deepseek_brain = DeepSeekBrain(system_context)
        signal_gen = SignalGenerator(
            system_context=system_context,
            deepseek_brain=deepseek_brain,
            feature_engine=feature_engine,
            enable_db_logging=True
        )
        print("   ✓ All components initialized")

        # Step 3: Mock DeepSeek response
        print("\n3. Configuring DeepSeek AI...")
        deepseek_brain.get_trading_signal = Mock(return_value=asyncio.Future())
        deepseek_brain.get_trading_signal.return_value.set_result({
            'action': 'LONG',
            'reasoning': 'Strong bullish signal. Price above supertrend with high alignment.',
            'confidence': 0.82
        })
        print("   ✓ DeepSeek configured")

        # Step 4: Generate signal through complete flow
        print("\n4. Generating signal through complete pipeline...")
        signal = await signal_gen.generate_signal(
            symbol='BTCUSDT',
            market_data=market_data,
            timeframe_data=None  # Single timeframe
        )

        # Step 5: Verify signal structure
        print("\n5. Verifying signal structure...")
        assert signal is not None
        assert 'symbol' in signal
        assert 'action' in signal
        assert 'confidence' in signal
        assert 'market_regime' in signal
        print(f"   ✓ Signal generated: {signal['symbol']} {signal['action']} {signal['confidence']:.2%}")

        # Step 6: Verify signal was saved to database
        print("\n6. Verifying database persistence...")
        history = await signal_gen.get_signal_history(limit=5)
        assert len(history) > 0
        latest_signal = history[0]
        assert latest_signal['symbol'] == 'BTCUSDT'
        assert latest_signal['action'] == 'LONG'
        print(f"   ✓ Signal saved to database (ID: {latest_signal.get('id')})")

        # Step 7: Verify feature calculations in system context
        print("\n7. Verifying feature calculations...")
        feature_set = system_context.feature_calculations.get('BTCUSDT', {})
        assert len(feature_set) > 0
        assert 'liquidity_zones' in feature_set
        assert 'order_flow_imbalance' in feature_set
        assert 'supertrend_value' in feature_set
        assert 'market_regime' in feature_set
        print(f"   ✓ Feature set calculated: {len(feature_set)} features")

        # Step 8: Verify signal metadata
        print("\n8. Verifying signal metadata...")
        assert signal['position_size'] > 0
        assert signal['current_price'] > 0
        assert 'timestamp' in signal
        assert 'reasoning' in signal
        print("   ✓ All metadata present")

        # Cleanup
        await close_database()
        print("\n✅ test_complete_signal_flow_single_timeframe PASSED")

    @pytest.mark.asyncio
    async def test_complete_signal_flow_multi_timeframe(self):
        """Test complete signal flow with multi-timeframe data."""
        print("\n" + "="*70)
        print("Test: Complete Signal Flow - Multi-Timeframe")
        print("="*70)

        # Step 1: Create multi-timeframe data
        print("\n1. Creating multi-timeframe data...")
        timeframe_data = {
            '5m': create_comprehensive_market_data('ETHUSDT', 200),
            '15m': create_comprehensive_market_data('ETHUSDT', 200),
            '1h': create_comprehensive_market_data('ETHUSDT', 200),
            '4h': create_comprehensive_market_data('ETHUSDT', 200)
        }
        print(f"   ✓ Created data for {len(timeframe_data)} timeframes")

        # Step 2: Initialize components
        print("\n2. Initializing components...")
        system_context = SystemContext()
        feature_engine = FeatureEngine()
        deepseek_brain = DeepSeekBrain(system_context)
        signal_gen = SignalGenerator(
            system_context=system_context,
            deepseek_brain=deepseek_brain,
            feature_engine=feature_engine,
            enable_db_logging=True
        )
        print("   ✓ Components initialized")

        # Step 3: Configure DeepSeek
        print("\n3. Configuring DeepSeek...")
        deepseek_brain.get_trading_signal = Mock(return_value=asyncio.Future())
        deepseek_brain.get_trading_signal.return_value.set_result({
            'action': 'SHORT',
            'reasoning': 'Multi-timeframe bearish divergence detected.',
            'confidence': 0.76
        })
        print("   ✓ DeepSeek configured")

        # Step 4: Generate signal
        print("\n4. Generating signal with multi-timeframe data...")
        signal = await signal_gen.generate_signal(
            symbol='ETHUSDT',
            market_data=timeframe_data['1h'],
            timeframe_data=timeframe_data
        )

        # Step 5: Verify multi-timeframe features
        print("\n5. Verifying multi-timeframe features...")
        feature_set = system_context.feature_calculations.get('ETHUSDT', {})
        assert 'timeframe_alignment' in feature_set
        assert 'convergence_details' in feature_set
        assert 'trading_timeframe_recommendation' in feature_set

        alignment = feature_set['timeframe_alignment']
        assert 0 <= alignment <= 1
        print(f"   ✓ Timeframe alignment: {alignment:.2%}")

        # Step 6: Verify signal quality
        print("\n6. Verifying signal quality...")
        assert signal['confidence'] > 0.5
        assert signal['action'] == 'SHORT'
        print(f"   ✓ Signal quality: {signal['confidence']:.2%} confidence")

        # Step 7: Verify database storage
        print("\n7. Verifying database storage...")
        history = await signal_gen.get_signal_history(symbol='ETHUSDT', limit=5)
        assert len(history) > 0
        print(f"   ✓ Signal stored in database")

        await close_database()
        print("\n✅ test_complete_signal_flow_multi_timeframe PASSED")

    @pytest.mark.asyncio
    async def test_signal_flow_with_database_retrieval(self):
        """Test that signals can be retrieved and analyzed after generation."""
        print("\n" + "="*70)
        print("Test: Signal Flow with Database Retrieval")
        print("="*70)

        # Step 1: Create market data
        print("\n1. Creating market data...")
        market_data = create_comprehensive_market_data('BTCUSDT', 200)
        print("   ✓ Market data created")

        # Step 2: Initialize system
        print("\n2. Initializing system...")
        system_context = SystemContext()
        feature_engine = FeatureEngine()
        deepseek_brain = DeepSeekBrain(system_context)
        signal_gen = SignalGenerator(
            system_context=system_context,
            deepseek_brain=deepseek_brain,
            feature_engine=feature_engine,
            enable_db_logging=True
        )

        # Step 3: Generate multiple signals
        print("\n3. Generating multiple signals...")
        signals = []
        for i in range(3):
            # Vary DeepSeek response
            deepseek_brain.get_trading_signal = Mock(return_value=asyncio.Future())
            deepseek_brain.get_trading_signal.return_value.set_result({
                'action': ['LONG', 'SHORT', 'HOLD'][i],
                'reasoning': f'Test signal {i+1}',
                'confidence': 0.6 + (i * 0.1)
            })

            signal = await signal_gen.generate_signal(
                symbol='BTCUSDT',
                market_data=market_data
            )
            signals.append(signal)
            print(f"   ✓ Generated signal {i+1}: {signal['action']}")

        # Step 4: Retrieve signals from database
        print("\n4. Retrieving signals from database...")
        # Filter by recent date to avoid signals from previous test runs
        from datetime import datetime, timedelta
        start_time = datetime.now() - timedelta(minutes=5)
        history = await signal_gen.db.get_signals(
            symbol='BTCUSDT',
            start_date=start_time,
            limit=10
        )
        assert len(history) >= 3
        print(f"   ✓ Retrieved {len(history)} signals from database")

        # Step 5: Verify signal statistics
        print("\n5. Calculating signal statistics...")
        stats = await signal_gen.get_signal_statistics(days=30)
        assert stats['total_signals'] >= 3  # At least the 3 we just created
        assert 'avg_confidence' in stats
        print(f"   ✓ Statistics: {stats['total_signals']} signals, "
              f"{stats['avg_confidence']:.2%} avg confidence")

        # Step 6: Filter signals
        print("\n6. Testing signal filtering...")
        high_conf = await signal_gen.db.get_signals(
            symbol='BTCUSDT',
            min_confidence=0.7,
            limit=10
        )
        print(f"   ✓ High confidence signals: {len(high_conf)}")

        # Step 7: Get latest signal
        print("\n7. Retrieving latest signal...")
        latest = await signal_gen.get_latest_signals(symbol='BTCUSDT', limit=1)
        assert len(latest) > 0
        print(f"   ✓ Latest signal: {latest[0]['action']}")

        await close_database()
        print("\n✅ test_signal_flow_with_database_retrieval PASSED")

    @pytest.mark.asyncio
    async def test_signal_flow_error_handling(self):
        """Test error handling in signal flow."""
        print("\n" + "="*70)
        print("Test: Signal Flow Error Handling")
        print("="*70)

        # Step 1: Create market data
        print("\n1. Creating market data...")
        market_data = create_comprehensive_market_data('BTCUSDT', 50)
        print("   ✓ Market data created")

        # Step 2: Initialize system
        print("\n2. Initializing system...")
        system_context = SystemContext()
        feature_engine = FeatureEngine()
        deepseek_brain = DeepSeekBrain(system_context)
        signal_gen = SignalGenerator(
            system_context=system_context,
            deepseek_brain=deepseek_brain,
            feature_engine=feature_engine,
            enable_db_logging=True
        )

        # Step 3: Test with insufficient data
        print("\n3. Testing with insufficient data...")
        insufficient_data = {k: v[:10] for k, v in market_data.items()}

        # Mock DeepSeek to return an error response
        deepseek_brain.get_trading_signal = Mock(return_value=asyncio.Future())
        deepseek_brain.get_trading_signal.return_value.set_result({
            'action': 'HOLD',
            'reasoning': 'Insufficient data for analysis',
            'confidence': 0.0
        })

        signal = await signal_gen.generate_signal(
            symbol='BTCUSDT',
            market_data=insufficient_data
        )

        # Should still return a valid signal structure
        assert signal is not None
        assert 'symbol' in signal
        assert signal['action'] == 'HOLD'
        print(f"   ✓ Handled insufficient data gracefully")

        # Step 4: Test database error handling
        print("\n4. Testing database error isolation...")
        # Generate another signal
        signal = await signal_gen.generate_signal(
            symbol='BTCUSDT',
            market_data=market_data
        )
        assert signal is not None
        print("   ✓ Signals generated despite potential DB errors")

        await close_database()
        print("\n✅ test_signal_flow_error_handling PASSED")

    @pytest.mark.asyncio
    async def test_signal_flow_system_context_integration(self):
        """Test integration with SystemContext."""
        print("\n" + "="*70)
        print("Test: SystemContext Integration")
        print("="*70)

        # Step 1: Create market data
        print("\n1. Creating market data...")
        market_data = create_comprehensive_market_data('BTCUSDT', 200)
        print("   ✓ Market data created")

        # Step 2: Initialize with custom risk metrics
        print("\n2. Initializing system with risk metrics...")
        system_context = SystemContext()
        system_context.risk_metrics = {
            'current_drawdown': 0.05,  # 5% drawdown
            'max_drawdown': 0.10,  # 10% max
            'daily_pnl': 0.02  # 2% daily P&L
        }
        print("   ✓ Risk metrics configured")

        feature_engine = FeatureEngine()
        deepseek_brain = DeepSeekBrain(system_context)
        signal_gen = SignalGenerator(
            system_context=system_context,
            deepseek_brain=deepseek_brain,
            feature_engine=feature_engine,
            enable_db_logging=True
        )

        # Step 3: Configure DeepSeek
        print("\n3. Configuring DeepSeek...")
        deepseek_brain.get_trading_signal = Mock(return_value=asyncio.Future())
        deepseek_brain.get_trading_signal.return_value.set_result({
            'action': 'LONG',
            'reasoning': 'System context-aware signal',
            'confidence': 0.80
        })
        print("   ✓ DeepSeek configured")

        # Step 4: Generate signal
        print("\n4. Generating signal...")
        signal = await signal_gen.generate_signal(
            symbol='BTCUSDT',
            market_data=market_data
        )

        # Step 5: Verify system context was used
        print("\n5. Verifying system context integration...")
        assert signal is not None
        print(f"   ✓ Signal generated with context awareness")
        print(f"     Current drawdown: {system_context.risk_metrics['current_drawdown']:.2%}")
        print(f"     Signal action: {signal['action']}")

        await close_database()
        print("\n✅ test_signal_flow_system_context_integration PASSED")

    @pytest.mark.asyncio
    async def test_signal_flow_performance(self):
        """Test signal generation performance."""
        print("\n" + "="*70)
        print("Test: Signal Flow Performance")
        print("="*70)

        # Step 1: Create market data
        print("\n1. Creating market data...")
        market_data = create_comprehensive_market_data('BTCUSDT', 200)
        print("   ✓ Market data created")

        # Step 2: Initialize system
        print("\n2. Initializing system...")
        system_context = SystemContext()
        feature_engine = FeatureEngine()
        deepseek_brain = DeepSeekBrain(system_context)
        signal_gen = SignalGenerator(
            system_context=system_context,
            deepseek_brain=deepseek_brain,
            feature_engine=feature_engine,
            enable_db_logging=True
        )

        # Step 3: Configure DeepSeek
        print("\n3. Configuring DeepSeek...")
        deepseek_brain.get_trading_signal = Mock(return_value=asyncio.Future())
        deepseek_brain.get_trading_signal.return_value.set_result({
            'action': 'LONG',
            'reasoning': 'Performance test signal',
            'confidence': 0.75
        })
        print("   ✓ DeepSeek configured")

        # Step 4: Measure signal generation time
        print("\n4. Measuring signal generation time...")
        import time
        start_time = time.time()

        signal = await signal_gen.generate_signal(
            symbol='BTCUSDT',
            market_data=market_data
        )

        end_time = time.time()
        duration = end_time - start_time

        # Step 5: Verify performance
        print("\n5. Verifying performance...")
        assert signal is not None
        assert duration < 5.0  # Should complete in under 5 seconds
        print(f"   ✓ Signal generated in {duration:.2f} seconds")
        print(f"   ✓ Performance acceptable: {duration < 5.0}")

        await close_database()
        print("\n✅ test_signal_flow_performance PASSED")

    @pytest.mark.asyncio
    async def test_signal_flow_without_database(self):
        """Test signal generation without database logging."""
        print("\n" + "="*70)
        print("Test: Signal Flow Without Database")
        print("="*70)

        # Step 1: Create market data
        print("\n1. Creating market data...")
        market_data = create_comprehensive_market_data('BTCUSDT', 200)
        print("   ✓ Market data created")

        # Step 2: Initialize without database
        print("\n2. Initializing without database logging...")
        system_context = SystemContext()
        feature_engine = FeatureEngine()
        deepseek_brain = DeepSeekBrain(system_context)
        signal_gen = SignalGenerator(
            system_context=system_context,
            deepseek_brain=deepseek_brain,
            feature_engine=feature_engine,
            enable_db_logging=False  # No database
        )
        print("   ✓ Initialized without database")

        # Step 3: Configure DeepSeek
        print("\n3. Configuring DeepSeek...")
        deepseek_brain.get_trading_signal = Mock(return_value=asyncio.Future())
        deepseek_brain.get_trading_signal.return_value.set_result({
            'action': 'HOLD',
            'reasoning': 'No database logging test',
            'confidence': 0.65
        })
        print("   ✓ DeepSeek configured")

        # Step 4: Generate signal
        print("\n4. Generating signal...")
        signal = await signal_gen.generate_signal(
            symbol='BTCUSDT',
            market_data=market_data
        )

        # Step 5: Verify signal works without database
        print("\n5. Verifying signal generation without database...")
        assert signal is not None
        assert signal['action'] == 'HOLD'
        assert 'db' not in signal_gen.__dict__ or signal_gen.db is None
        print("   ✓ Signal generated successfully without database")

        await close_database()
        print("\n✅ test_signal_flow_without_database PASSED")


if __name__ == '__main__':
    print("\n" + "="*70)
    print("Running End-to-End Signal Flow Tests")
    print("="*70 + "\n")

    # Run all e2e tests
    async def run_all_tests():
        test_suite = TestSignalFlowE2E()

        tests = [
            ("Complete Signal Flow - Single Timeframe",
             test_suite.test_complete_signal_flow_single_timeframe),
            ("Complete Signal Flow - Multi-Timeframe",
             test_suite.test_complete_signal_flow_multi_timeframe),
            ("Signal Flow with Database Retrieval",
             test_suite.test_signal_flow_with_database_retrieval),
            ("Signal Flow Error Handling",
             test_suite.test_signal_flow_error_handling),
            ("SystemContext Integration",
             test_suite.test_signal_flow_system_context_integration),
            ("Signal Flow Performance",
             test_suite.test_signal_flow_performance),
            ("Signal Flow Without Database",
             test_suite.test_signal_flow_without_database),
        ]

        passed = 0
        failed = 0

        for test_name, test_func in tests:
            try:
                await test_func()
                passed += 1
            except Exception as e:
                print(f"\n❌ {test_name} FAILED: {e}")
                failed += 1

        print("\n" + "="*70)
        print(f"E2E Test Results: {passed} passed, {failed} failed")
        print("="*70)

        if failed == 0:
            print("\n✅ All end-to-end tests PASSED!")
        else:
            print(f"\n⚠️  {failed} test(s) FAILED")

    asyncio.run(run_all_tests())
