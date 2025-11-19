"""
Tests for M1MemoryManager + DeepSeek Integration
Tests the integration between memory manager and DeepSeekBrain.
"""

import sys
sys.path.insert(0, '/Users/mrsmoothy/Downloads/Trading_bot')

import pytest
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock

from core.system_context import SystemContext, TradeRecord
from core.memory_manager import M1MemoryManager
from deepseek.client import DeepSeekBrain
from core.signal_generator import SignalGenerator


@pytest.mark.slow

class TestMemoryManagerIntegration:
    """Tests for memory manager integration with DeepSeek."""

    @pytest.mark.asyncio
    async def test_deepseek_initialization_with_memory_manager(self):
        """Test DeepSeekBrain initialization with memory manager."""
        print("\n" + "="*70)
        print("Test: DeepSeekBrain Initialization with Memory Manager")
        print("="*70)

        system_context = SystemContext()
        memory_manager = M1MemoryManager(memory_limit_mb=4000)

        # Initialize DeepSeek with memory manager
        deepseek = DeepSeekBrain(system_context, memory_manager=memory_manager)

        assert deepseek.memory_manager is memory_manager
        assert deepseek.system_context is system_context
        print("✅ DeepSeekBrain initialized with memory manager")

    @pytest.mark.asyncio
    async def test_memory_optimized_context_generation(self):
        """Test memory-optimized context generation."""
        print("\n" + "="*70)
        print("Test: Memory-Optimized Context Generation")
        print("="*70)

        system_context = SystemContext()
        memory_manager = M1MemoryManager(memory_limit_mb=4000)

        # Add some trade history
        trade1 = TradeRecord(
            symbol="BTCUSDT",
            entry_time=datetime.now() - timedelta(hours=1),
            exit_time=datetime.now(),
            side="LONG",
            entry_price=45000,
            exit_price=46000,
            quantity=0.1,
            pnl=100,
            pnl_percent=2.2,
            confidence=0.8,
            reasoning="Test trade",
            exit_reason="PROFIT_TARGET"
        )
        system_context.trade_history.append(trade1)

        # Add feature calculations
        system_context.update_feature_calculations("BTCUSDT", {
            "market_regime": "TRENDING_HIGH_VOL",
            "supertrend_value": 44500,
            "order_flow_imbalance": 0.15,
        })

        # Get optimized context
        optimized_context = memory_manager.optimize_context_for_deepseek(system_context)

        assert optimized_context is not None
        assert "timestamp" in optimized_context
        assert "market_regime" in optimized_context
        assert "recent_trades" in optimized_context
        assert "current_features" in optimized_context
        assert len(optimized_context["recent_trades"]) == 1
        assert optimized_context["recent_trades"][0]["symbol"] == "BTCUSDT"

        print(f"✅ Generated optimized context with {len(optimized_context)} fields")
        print(f"   Recent trades: {len(optimized_context['recent_trades'])}")
        print(f"   Current features: {len(optimized_context.get('current_features', {}))}")

    @pytest.mark.asyncio
    async def test_get_optimized_context_method(self):
        """Test _get_optimized_context method in DeepSeekBrain."""
        print("\n" + "="*70)
        print("Test: _get_optimized_context Method")
        print("="*70)

        system_context = SystemContext()
        memory_manager = M1MemoryManager(memory_limit_mb=4000)
        deepseek = DeepSeekBrain(system_context, memory_manager=memory_manager)

        # Mock the memory manager's optimize method
        mock_optimized = {
            "timestamp": datetime.now().isoformat(),
            "market_regime": "TRENDING_LOW_VOL",
            "memory_optimized": True
        }
        memory_manager.optimize_context_for_deepseek = Mock(return_value=mock_optimized)

        # Call _get_optimized_context
        result = deepseek._get_optimized_context()

        assert result == mock_optimized
        memory_manager.optimize_context_for_deepseek.assert_called_once_with(system_context)

        print("✅ _get_optimized_context calls memory manager correctly")

    @pytest.mark.asyncio
    async def test_fallback_when_no_memory_manager(self):
        """Test fallback when memory manager is not provided."""
        print("\n" + "="*70)
        print("Test: Fallback When No Memory Manager")
        print("="*70)

        system_context = SystemContext()
        # No memory manager provided
        deepseek = DeepSeekBrain(system_context)

        # Should use system_context.get_context_for_deepseek()
        result = deepseek._get_optimized_context()

        assert result is not None
        assert isinstance(result, dict)

        print("✅ Falls back to system_context.get_context_for_deepseek()")

    @pytest.mark.asyncio
    async def test_signal_generator_with_memory_manager(self):
        """Test SignalGenerator with memory manager integration."""
        print("\n" + "="*70)
        print("Test: SignalGenerator with Memory Manager")
        print("="*70)

        system_context = SystemContext()
        memory_manager = M1MemoryManager(memory_limit_mb=4000)
        deepseek = DeepSeekBrain(system_context, memory_manager=memory_manager)

        # Mock get_trading_signal to avoid API call
        deepseek.get_trading_signal = AsyncMock(return_value={
            "action": "LONG",
            "confidence": 0.75,
            "reasoning": "Test signal with memory optimization"
        })

        # Create SignalGenerator with memory_manager
        signal_gen = SignalGenerator(
            system_context=system_context,
            deepseek_brain=deepseek,
            feature_engine=Mock(),
            enable_db_logging=False,
            memory_manager=memory_manager
        )

        assert signal_gen.memory_manager is memory_manager
        assert signal_gen.deepseek.memory_manager is memory_manager

        print("✅ SignalGenerator initialized with memory manager")

    @pytest.mark.asyncio
    async def test_memory_cleanup_method(self):
        """Test perform_memory_cleanup method."""
        print("\n" + "="*70)
        print("Test: Memory Cleanup Method")
        print("="*70)

        system_context = SystemContext()
        memory_manager = M1MemoryManager(memory_limit_mb=4000)
        signal_gen = SignalGenerator(
            system_context=system_context,
            deepseek_brain=DeepSeekBrain(system_context),
            feature_engine=Mock(),
            enable_db_logging=False,
            memory_manager=memory_manager
        )

        # Mock memory manager methods
        memory_manager.get_memory_report = Mock(return_value={
            "current": {"rss_mb": 1500, "percent": 37.5},
            "status": "HEALTHY"
        })
        memory_manager.cleanup_cache = AsyncMock()
        memory_manager.should_trigger_cleanup = Mock(return_value=False)

        # Perform cleanup
        report = await signal_gen.perform_memory_cleanup()

        assert report is not None
        assert "current" in report
        assert report["current"]["rss_mb"] == 1500

        print("✅ Memory cleanup completed successfully")
        print(f"   Memory usage: {report['current']['rss_mb']}MB ({report['current']['percent']:.1f}%)")

    @pytest.mark.asyncio
    async def test_memory_cleanup_without_memory_manager(self):
        """Test cleanup when no memory manager is available."""
        print("\n" + "="*70)
        print("Test: Memory Cleanup Without Memory Manager")
        print("="*70)

        system_context = SystemContext()
        signal_gen = SignalGenerator(
            system_context=system_context,
            deepseek_brain=DeepSeekBrain(system_context),
            feature_engine=Mock(),
            enable_db_logging=False,
            memory_manager=None
        )

        # Perform cleanup without memory manager
        report = await signal_gen.perform_memory_cleanup()

        assert report is None

        print("✅ Gracefully handles missing memory manager")

    @pytest.mark.asyncio
    async def test_signal_generation_with_memory_integration(self):
        """Test complete signal generation with memory integration."""
        print("\n" + "="*70)
        print("Test: Signal Generation with Memory Integration")
        print("="*70)

        system_context = SystemContext()
        memory_manager = M1MemoryManager(memory_limit_mb=4000)
        deepseek = DeepSeekBrain(system_context, memory_manager=memory_manager)

        # Mock features engine
        mock_features = Mock()
        mock_features.compute_all_features = AsyncMock(return_value={
            "market_regime": "TRENDING_HIGH_VOL",
            "supertrend_value": 44500,
            "order_flow_imbalance": 0.15,
            "timeframe_alignment": 0.8
        })

        # Mock DeepSeek
        deepseek.get_trading_signal = AsyncMock(return_value={
            "action": "LONG",
            "confidence": 0.78,
            "reasoning": "Strong trend with good alignment"
        })

        signal_gen = SignalGenerator(
            system_context=system_context,
            deepseek_brain=deepseek,
            feature_engine=mock_features,
            enable_db_logging=False,
            memory_manager=memory_manager
        )

        # Create market data
        market_data = {
            'close': pd.Series([45000, 45100, 45200, 45300, 45400]),
            'volume': pd.Series([100, 120, 110, 130, 115])
        }

        # Generate signal
        signal = await signal_gen.generate_signal(
            symbol="BTCUSDT",
            market_data=market_data
        )

        # Verify signal was generated
        assert signal is not None
        assert signal["symbol"] == "BTCUSDT"
        assert signal["action"] == "LONG"
        assert signal["confidence"] > 0

        # Verify memory manager was integrated
        assert signal_gen.deepseek.memory_manager is memory_manager

        print(f"✅ Signal generated with memory integration")
        print(f"   Symbol: {signal['symbol']}")
        print(f"   Action: {signal['action']}")
        print(f"   Confidence: {signal['confidence']:.2%}")

    @pytest.mark.asyncio
    async def test_memory_optimization_with_large_dataset(self):
        """Test memory optimization with larger dataset."""
        print("\n" + "="*70)
        print("Test: Memory Optimization with Large Dataset")
        print("="*70)

        system_context = SystemContext()
        memory_manager = M1MemoryManager(memory_limit_mb=4000)

        # Add many trades to history
        for i in range(20):
            trade = TradeRecord(
                symbol=f"SYMBOL{i % 3}",
                entry_time=datetime.now() - timedelta(hours=i),
                exit_time=datetime.now(),
                side="LONG" if i % 2 == 0 else "SHORT",
                entry_price=45000 + i,
                exit_price=45100 + i,
                quantity=0.1,
                pnl=100 + i,
                pnl_percent=2.0 + i,
                confidence=0.7,
                reasoning=f"Trade {i}",
                exit_reason="TARGET"
            )
            system_context.trade_history.append(trade)

        # Add feature calculations for multiple symbols
        for symbol in ["BTCUSDT", "ETHUSDT", "ADAUSDT"]:
            system_context.update_feature_calculations(symbol, {
                "market_regime": "TRENDING_HIGH_VOL",
                "supertrend_value": 44500,
                "order_flow_imbalance": 0.15,
                "large_array": np.random.rand(100).tolist(),  # Large data
            })

        # Get optimized context
        optimized_context = memory_manager.optimize_context_for_deepseek(system_context)

        # Should keep only last 10 trades
        assert len(optimized_context["recent_trades"]) == 10

        # Features should be compressed (no large arrays)
        for symbol, features in optimized_context.get("current_features", {}).items():
            for key, value in features.items():
                if key == "large_array":
                    # Should be removed or compressed
                    assert not isinstance(value, list) or len(value) < 100

        print(f"✅ Optimized large dataset")
        print(f"   Trade history: {len(system_context.trade_history)} → {len(optimized_context['recent_trades'])}")
        print(f"   Symbols with features: {len(optimized_context.get('current_features', {}))}")

    @pytest.mark.asyncio
    async def test_memory_alerts(self):
        """Test memory alert generation."""
        print("\n" + "="*70)
        print("Test: Memory Alert Generation")
        print("="*70)

        memory_manager = M1MemoryManager(memory_limit_mb=4000)

        # Mock high memory usage (slightly above 95% critical threshold)
        memory_manager.get_current_memory_usage = Mock(return_value={
            "rss_mb": 3850,  # Above 95% critical threshold (3800MB)
            "percent": 96.25,
            "available_mb": 150,
            "timestamp": datetime.now().isoformat()
        })

        alerts = memory_manager.get_memory_alerts()

        assert len(alerts) > 0
        assert any(alert["level"] == "CRITICAL" for alert in alerts)

        print(f"✅ Generated {len(alerts)} memory alert(s)")
        for alert in alerts:
            print(f"   {alert['level']}: {alert['message']}")


if __name__ == '__main__':
    print("\n" + "="*70)
    print("Running M1MemoryManager + DeepSeek Integration Tests")
    print("="*70 + "\n")

    # Run all tests
    async def run_all_tests():
        test_suite = TestMemoryManagerIntegration()

        tests = [
            ("DeepSeek Initialization with Memory Manager",
             test_suite.test_deepseek_initialization_with_memory_manager),
            ("Memory-Optimized Context Generation",
             test_suite.test_memory_optimized_context_generation),
            ("_get_optimized_context Method",
             test_suite.test_get_optimized_context_method),
            ("Fallback When No Memory Manager",
             test_suite.test_fallback_when_no_memory_manager),
            ("SignalGenerator with Memory Manager",
             test_suite.test_signal_generator_with_memory_manager),
            ("Memory Cleanup Method",
             test_suite.test_memory_cleanup_method),
            ("Memory Cleanup Without Memory Manager",
             test_suite.test_memory_cleanup_without_memory_manager),
            ("Signal Generation with Memory Integration",
             test_suite.test_signal_generation_with_memory_integration),
            ("Memory Optimization with Large Dataset",
             test_suite.test_memory_optimization_with_large_dataset),
            ("Memory Alert Generation",
             test_suite.test_memory_alerts),
        ]

        passed = 0
        failed = 0

        for test_name, test_func in tests:
            try:
                await test_func()
                passed += 1
            except Exception as e:
                print(f"\n❌ {test_name} FAILED: {e}")
                import traceback
                traceback.print_exc()
                failed += 1

        print("\n" + "="*70)
        print(f"Integration Test Results: {passed} passed, {failed} failed")
        print("="*70)

        if failed == 0:
            print("\n✅ All integration tests PASSED!")
        else:
            print(f"\n⚠️  {failed} test(s) FAILED")

    asyncio.run(run_all_tests())
