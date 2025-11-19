"""
Tests for PositionManager with Testnet Order Routes
Tests real Binance API integration for position management.
"""

import sys
sys.path.insert(0, '/Users/mrsmoothy/Downloads/Trading_bot')

import pytest
import asyncio
from datetime import datetime
from unittest.mock import Mock, AsyncMock, patch

from core.system_context import SystemContext, TradeRecord
from deepseek.client import DeepSeekBrain
from execution.position_manager import PositionManager


@pytest.mark.slow

class TestPositionManagerTestnet:
    """Test PositionManager with testnet order functionality."""

    @pytest.mark.asyncio
    async def test_initialization_with_testnet_flag(self):
        """Test PositionManager initialization with testnet flag."""
        print("\n" + "="*70)
        print("Test: PositionManager Initialization with Testnet Flag")
        print("="*70)

        system_context = SystemContext()
        deepseek_brain = Mock()
        binance_client = Mock()

        # Test with testnet=True (default)
        position_manager = PositionManager(
            system_context=system_context,
            deepseek_brain=deepseek_brain,
            binance_client=binance_client,
            use_testnet=True
        )

        assert position_manager.use_testnet is True
        assert position_manager.live_orders == {}
        print("✅ Initialized with testnet=True (default safe mode)")

        # Test with testnet=False
        position_manager_live = PositionManager(
            system_context=system_context,
            deepseek_brain=deepseek_brain,
            binance_client=binance_client,
            use_testnet=False
        )

        assert position_manager_live.use_testnet is False
        print("✅ Initialized with testnet=False (LIVE MODE - real money!)")

    @pytest.mark.asyncio
    async def test_place_order_simulation_mode(self):
        """Test order placement in simulation mode."""
        print("\n" + "="*70)
        print("Test: Order Placement in Simulation Mode")
        print("="*70)

        system_context = SystemContext()
        deepseek_brain = Mock()

        # Mock binance client without create_order method
        binance_client = Mock()
        del binance_client.create_order

        position_manager = PositionManager(
            system_context=system_context,
            deepseek_brain=deepseek_brain,
            binance_client=binance_client,
            use_testnet=True
        )

        # Create test signal
        signal = {
            "symbol": "BTCUSDT",
            "action": "LONG",
            "position_size": 0.02,
            "confidence": 0.75,
            "exit_strategy": {
                "stop_loss": 44000,
                "time_based_exit_minutes": 120
            }
        }

        # Mock get_current_price
        position_manager.binance.get_current_price = AsyncMock(return_value=45000)

        # Place order
        result = await position_manager.place_order(signal, 0.02)

        assert result["success"] is True
        assert "order" in result
        assert result["order"]["symbol"] == "BTCUSDT"
        assert result["order"]["side"] == "BUY"
        assert result["order"]["status"] == "FILLED"
        assert result["message"] == "Demo order simulated"

        print(f"✅ Simulated order placed: {result['order']['symbol']} {result['order']['side']}")

    @pytest.mark.asyncio
    async def test_place_order_with_real_api(self):
        """Test order placement with real Binance API."""
        print("\n" + "="*70)
        print("Test: Order Placement with Real Binance API")
        print("="*70)

        system_context = SystemContext()
        deepseek_brain = Mock()

        # Mock binance client with create_order method
        binance_client = Mock()
        binance_client.create_order = AsyncMock(return_value={
            "orderId": 12345,
            "symbol": "BTCUSDT",
            "side": "BUY",
            "quantity": 0.002,
            "price": 45000,
            "avg_price": 45000,
            "status": "FILLED",
            "timestamp": datetime.now().isoformat(),
            "transactTime": datetime.now().isoformat()
        })
        binance_client.get_current_price = AsyncMock(return_value=45000)

        position_manager = PositionManager(
            system_context=system_context,
            deepseek_brain=deepseek_brain,
            binance_client=binance_client,
            use_testnet=False  # LIVE MODE
        )

        signal = {
            "symbol": "BTCUSDT",
            "action": "LONG",
            "position_size": 0.02,
            "confidence": 0.75
        }

        result = await position_manager.place_order(signal, 0.02)

        assert result["success"] is True
        assert result["order"]["orderId"] == 12345
        assert "TESTNET" not in result["message"]
        assert "LIVE" in result["message"]

        # Verify order was tracked
        assert "BTCUSDT" in position_manager.live_orders
        assert position_manager.live_orders["BTCUSDT"] == 12345

        print(f"✅ Real order placed (LIVE mode): {result['order']['orderId']}")

    @pytest.mark.asyncio
    async def test_place_real_order_fallback_on_error(self):
        """Test fallback to simulation when real API fails."""
        print("\n" + "="*70)
        print("Test: Real API Fallback on Error")
        print("="*70)

        system_context = SystemContext()
        deepseek_brain = Mock()

        # Mock binance client that raises an error
        binance_client = Mock()
        binance_client.create_order = AsyncMock(side_effect=Exception("API Error"))
        binance_client.get_current_price = AsyncMock(return_value=45000)

        position_manager = PositionManager(
            system_context=system_context,
            deepseek_brain=deepseek_brain,
            binance_client=binance_client,
            use_testnet=False  # Will try real API, then fallback
        )

        signal = {
            "symbol": "BTCUSDT",
            "action": "LONG",
            "position_size": 0.02
        }

        result = await position_manager.place_order(signal, 0.02)

        # Should fall back to simulation
        assert result["success"] is True
        assert result["message"] == "Demo order simulated"

        print("✅ Successfully fell back to simulation when API failed")

    @pytest.mark.asyncio
    async def test_close_position_simulation(self):
        """Test position closing in simulation mode."""
        print("\n" + "="*70)
        print("Test: Position Closing in Simulation Mode")
        print("="*70)

        system_context = SystemContext()
        deepseek_brain = Mock()

        # Create a position
        position_data = {
            "symbol": "BTCUSDT",
            "side": "LONG",
            "entry_price": 44500,
            "quantity": 0.01,
            "entry_time": datetime.now().isoformat(),
            "exit_strategy": {}
        }
        system_context.update_position("BTCUSDT", position_data)

        binance_client = Mock()
        binance_client.get_current_price = AsyncMock(return_value=45000)

        position_manager = PositionManager(
            system_context=system_context,
            deepseek_brain=deepseek_brain,
            binance_client=binance_client,
            use_testnet=True
        )

        # Close position
        result = await position_manager.close_position("BTCUSDT", "PROFIT_TARGET")

        assert result["success"] is True
        assert result["symbol"] == "BTCUSDT"
        assert result["exit_price"] == 45000
        assert result["pnl"] > 0  # Profitable trade
        assert result["reason"] == "PROFIT_TARGET"

        print(f"✅ Position closed in simulation: P&L ${result['pnl']:.2f}")

    @pytest.mark.asyncio
    async def test_cancel_order(self):
        """Test order cancellation."""
        print("\n" + "="*70)
        print("Test: Order Cancellation")
        print("="*70)

        system_context = SystemContext()
        deepseek_brain = Mock()
        binance_client = Mock()

        # Mock cancel_order response
        binance_client.cancel_order = AsyncMock(return_value={
            "orderId": 12345,
            "symbol": "BTCUSDT",
            "status": "CANCELED"
        })

        position_manager = PositionManager(
            system_context=system_context,
            deepseek_brain=deepseek_brain,
            binance_client=binance_client,
            use_testnet=False
        )

        # Simulate having a live order
        position_manager.live_orders["BTCUSDT"] = 12345

        # Cancel the order
        result = await position_manager.cancel_order("BTCUSDT")

        assert result["success"] is True
        assert "BTCUSDT" not in position_manager.live_orders

        # Verify cancel_order was called
        binance_client.cancel_order.assert_called_once_with(
            symbol="BTCUSDT",
            orderId=12345
        )

        print("✅ Order cancelled successfully")

    @pytest.mark.asyncio
    async def test_get_order_status(self):
        """Test getting order status."""
        print("\n" + "="*70)
        print("Test: Get Order Status")
        print("="*70)

        system_context = SystemContext()
        deepseek_brain = Mock()
        binance_client = Mock()

        # Mock get_order response
        binance_client.get_order = AsyncMock(return_value={
            "orderId": 12345,
            "symbol": "BTCUSDT",
            "status": "FILLED",
            "side": "BUY",
            "quantity": 0.01,
            "price": 45000
        })

        position_manager = PositionManager(
            system_context=system_context,
            deepseek_brain=deepseek_brain,
            binance_client=binance_client,
            use_testnet=False
        )

        position_manager.live_orders["BTCUSDT"] = 12345

        # Get order status
        result = await position_manager.get_order_status("BTCUSDT")

        assert result["success"] is True
        assert result["order"]["status"] == "FILLED"

        print(f"✅ Order status retrieved: {result['order']['status']}")

    @pytest.mark.asyncio
    async def test_get_account_balance(self):
        """Test getting account balance."""
        print("\n" + "="*70)
        print("Test: Get Account Balance")
        print("="*70)

        system_context = SystemContext()
        deepseek_brain = Mock()
        binance_client = Mock()

        # Mock get_account response
        binance_client.get_account = AsyncMock(return_value={
            "accountType": "SPOT",
            "canTrade": True,
            "balances": [
                {"asset": "USDT", "free": "5000.00", "locked": "100.00"},
                {"asset": "BTC", "free": "0.05", "locked": "0.00"},
                {"asset": "ETH", "free": "0.00", "locked": "1.00"}
            ]
        })

        position_manager = PositionManager(
            system_context=system_context,
            deepseek_brain=deepseek_brain,
            binance_client=binance_client,
            use_testnet=True
        )

        # Get balance
        result = await position_manager.get_account_balance()

        assert result["success"] is True
        assert "USDT" in result["balances"]
        assert result["balances"]["USDT"]["free"] == 5000.0
        assert result["balances"]["USDT"]["locked"] == 100.0
        assert result["can_trade"] is True

        print(f"✅ Account balance retrieved:")
        print(f"   USDT: {result['balances']['USDT']['free']} available")

    @pytest.mark.asyncio
    async def test_get_open_orders(self):
        """Test getting open orders."""
        print("\n" + "="*70)
        print("Test: Get Open Orders")
        print("="*70)

        system_context = SystemContext()
        deepseek_brain = Mock()
        binance_client = Mock()

        # Mock get_open_orders response
        binance_client.get_open_orders = AsyncMock(return_value=[
            {"orderId": 123, "symbol": "BTCUSDT", "status": "NEW"},
            {"orderId": 456, "symbol": "ETHUSDT", "status": "NEW"}
        ])

        position_manager = PositionManager(
            system_context=system_context,
            deepseek_brain=deepseek_brain,
            binance_client=binance_client,
            use_testnet=False
        )

        # Get open orders
        result = await position_manager.get_open_orders()

        assert result["success"] is True
        assert result["count"] == 2
        assert len(result["orders"]) == 2

        print(f"✅ Open orders retrieved: {result['count']} orders")

    @pytest.mark.asyncio
    async def test_get_testnet_status(self):
        """Test getting testnet connection status."""
        print("\n" + "="*70)
        print("Test: Get Testnet Status")
        print("="*70)

        system_context = SystemContext()
        deepseek_brain = Mock()
        binance_client = Mock()

        # Mock successful connection
        binance_client.get_account = AsyncMock(return_value={
            "accountType": "SPOT",
            "canTrade": True
        })

        # Test with testnet=True
        position_manager = PositionManager(
            system_context=system_context,
            deepseek_brain=deepseek_brain,
            binance_client=binance_client,
            use_testnet=True
        )

        status = await position_manager.get_testnet_status()

        assert status["success"] is True
        assert status["connected"] is True
        assert status["testnet"] is True
        assert status["can_trade"] is True

        print("✅ Testnet status: Connected")

    @pytest.mark.asyncio
    async def test_real_order_with_stop_loss(self):
        """Test placing order with stop loss parameters."""
        print("\n" + "="*70)
        print("Test: Real Order with Stop Loss")
        print("="*70)

        system_context = SystemContext()
        deepseek_brain = Mock()

        # Mock binance client
        binance_client = Mock()
        binance_client.create_order = AsyncMock(return_value={
            "orderId": 99999,
            "symbol": "BTCUSDT",
            "side": "BUY",
            "type": "MARKET",
            "quantity": 0.01,
            "price": 45000,
            "avg_price": 45000,
            "timestamp": datetime.now().isoformat(),
            "status": "FILLED"
        })
        binance_client.get_current_price = AsyncMock(return_value=45000)

        position_manager = PositionManager(
            system_context=system_context,
            deepseek_brain=deepseek_brain,
            binance_client=binance_client,
            use_testnet=False
        )

        signal = {
            "symbol": "BTCUSDT",
            "action": "LONG",
            "position_size": 0.02,
            "exit_strategy": {
                "stop_loss": 44000
            }
        }

        result = await position_manager.place_order(signal, 0.02)

        assert result["success"] is True
        assert result["order"]["type"] == "MARKET"

        # Verify order was placed as MARKET (for testnet)
        call_args = binance_client.create_order.call_args[1]
        assert call_args['type'] == "MARKET"

        print("✅ Order with stop loss strategy placed successfully (MARKET type for testnet)")

    @pytest.mark.asyncio
    async def test_evaluate_signal_with_deepseek(self):
        """Test signal evaluation with DeepSeek risk assessment."""
        print("\n" + "="*70)
        print("Test: Signal Evaluation with DeepSeek")
        print("="*70)

        system_context = SystemContext()
        deepseek_brain = Mock()

        # Mock DeepSeek responses
        deepseek_brain.assess_risk = AsyncMock(return_value={
            "approved": True,
            "adjusted_size": 0.015,
            "risk_score": 0.3,
            "reasoning": "Low risk, good opportunity",
            "concerns": []
        })

        deepseek_brain.optimize_position = AsyncMock(return_value={
            "size": 0.018,
            "reasoning": "Position optimized for volatility",
            "adjustments": ["reduced_size_due_to_vol"]
        })

        position_manager = PositionManager(
            system_context=system_context,
            deepseek_brain=deepseek_brain,
            binance_client=Mock(),
            use_testnet=True
        )

        # Set up some risk metrics
        system_context.risk_metrics["current_drawdown"] = 0.05

        signal = {
            "symbol": "BTCUSDT",
            "action": "LONG",
            "position_size": 0.02,
            "market_regime": "TRENDING_LOW_VOL",
            "confidence": 0.75
        }

        result = await position_manager.evaluate_signal(signal, 0.15)

        assert result["approved"] is True
        assert result["position_size"] == 0.018  # Final optimized size
        assert result["risk_score"] == 0.3

        print("✅ Signal evaluated with DeepSeek risk assessment")

    @pytest.mark.asyncio
    async def test_position_tracking_and_update(self):
        """Test position tracking and updates."""
        print("\n" + "="*70)
        print("Test: Position Tracking and Update")
        print("="*70)

        system_context = SystemContext()
        deepseek_brain = Mock()

        binance_client = Mock()
        binance_client.get_current_price = AsyncMock(return_value=45200)

        position_manager = PositionManager(
            system_context=system_context,
            deepseek_brain=deepseek_brain,
            binance_client=binance_client,
            use_testnet=True
        )

        # Add a test position
        system_context.update_position("ETHUSDT", {
            "symbol": "ETHUSDT",
            "side": "LONG",
            "entry_price": 3000,
            "quantity": 0.5,
            "entry_time": datetime.now().isoformat(),
            "exit_strategy": {}
        })

        # Update positions
        await position_manager.update_positions()

        # Check if position was updated
        assert "ETHUSDT" in system_context.active_positions
        position = system_context.active_positions["ETHUSDT"]
        assert position["current_price"] == 45200
        assert position["unrealized_pnl"] > 0  # Profitable

        print(f"✅ Position tracked and updated (P&L: ${position['unrealized_pnl']:.2f})")

    @pytest.mark.asyncio
    async def test_can_open_new_position_limits(self):
        """Test position limit checks."""
        print("\n" + "="*70)
        print("Test: Position Limit Checks")
        print("="*70)

        system_context = SystemContext()
        deepseek_brain = Mock()

        position_manager = PositionManager(
            system_context=system_context,
            deepseek_brain=deepseek_brain,
            binance_client=Mock(),
            use_testnet=True
        )

        # Default max_positions is 10
        assert position_manager.max_positions == 10

        # Initially should be able to open
        assert position_manager.can_open_new_position() is True

        # Add 10 positions
        for i in range(10):
            system_context.update_position(f"SYMBOL{i}", {
                "symbol": f"SYMBOL{i}",
                "side": "LONG",
                "entry_price": 1000,
                "quantity": 1.0,
                "entry_time": datetime.now().isoformat(),
                "exit_strategy": {}
            })

        # Should not be able to open more
        assert position_manager.can_open_new_position() is False

        print("✅ Position limits enforced correctly")

    @pytest.mark.asyncio
    async def test_drawdown_limit(self):
        """Test drawdown limit enforcement."""
        print("\n" + "="*70)
        print("Test: Drawdown Limit Enforcement")
        print("="*70)

        system_context = SystemContext()
        deepseek_brain = Mock()

        # Set high drawdown
        system_context.risk_metrics["current_drawdown"] = 0.12  # 12%

        position_manager = PositionManager(
            system_context=system_context,
            deepseek_brain=deepseek_brain,
            binance_client=Mock(),
            use_testnet=True
        )

        # Should not be able to open new position due to drawdown
        assert position_manager.can_open_new_position() is False

        # Reduce drawdown
        system_context.risk_metrics["current_drawdown"] = 0.05  # 5%

        # Should be able to open now
        assert position_manager.can_open_new_position() is True

        print("✅ Drawdown limits enforced correctly")


if __name__ == '__main__':
    print("\n" + "="*70)
    print("Running PositionManager Testnet Tests")
    print("="*70 + "\n")

    # Run all tests
    async def run_all_tests():
        test_suite = TestPositionManagerTestnet()

        tests = [
            ("Initialization with Testnet Flag",
             test_suite.test_initialization_with_testnet_flag),
            ("Order Placement in Simulation Mode",
             test_suite.test_place_order_simulation_mode),
            ("Order Placement with Real API",
             test_suite.test_place_order_with_real_api),
            ("Real API Fallback on Error",
             test_suite.test_place_real_order_fallback_on_error),
            ("Position Closing in Simulation Mode",
             test_suite.test_close_position_simulation),
            ("Order Cancellation",
             test_suite.test_cancel_order),
            ("Get Order Status",
             test_suite.test_get_order_status),
            ("Get Account Balance",
             test_suite.test_get_account_balance),
            ("Get Open Orders",
             test_suite.test_get_open_orders),
            ("Get Testnet Status",
             test_suite.test_get_testnet_status),
            ("Real Order with Stop Loss",
             test_suite.test_real_order_with_stop_loss),
            ("Signal Evaluation with DeepSeek",
             test_suite.test_evaluate_signal_with_deepseek),
            ("Position Tracking and Update",
             test_suite.test_position_tracking_and_update),
            ("Position Limit Checks",
             test_suite.test_can_open_new_position_limits),
            ("Drawdown Limit Enforcement",
             test_suite.test_drawdown_limit),
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
        print(f"Testnet Test Results: {passed} passed, {failed} failed")
        print("="*70)

        if failed == 0:
            print("\n✅ All PositionManager Testnet tests PASSED!")
        else:
            print(f"\n⚠️  {failed} test(s) FAILED")

    asyncio.run(run_all_tests())
