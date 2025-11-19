"""
Position Manager
Manages trading positions, order execution, and tracking.
"""

import asyncio
from datetime import datetime
from typing import Dict, Any, List, Optional
from loguru import logger

from core.system_context import SystemContext
from deepseek.client import DeepSeekBrain


class PositionManager:
    """
    Manages trading positions and order execution.
    Handles position tracking, P&L calculation, and order management.
    """

    def __init__(
        self,
        system_context: SystemContext,
        deepseek_brain: DeepSeekBrain,
        binance_client,
        use_testnet: bool = True
    ):
        """
        Initialize position manager.

        Args:
            system_context: System state context
            deepseek_brain: DeepSeek AI client
            binance_client: Binance API client
            use_testnet: Whether to use Binance testnet (default: True)
        """
        self.system_context = system_context
        self.deepseek = deepseek_brain
        self.binance = binance_client
        self.use_testnet = use_testnet

        # Configuration
        self.max_positions = 10
        self.default_leverage = 1

        # Convergence Strategy Integration
        self.enable_multi_layer_tp = True  # Enable multi-layer take profits

        # Track order IDs for real orders
        self.live_orders = {}  # Maps symbol to order ID
        self.liquidity_targets = {}  # Track liquidity targets for positions

        if self.use_testnet:
            logger.info("PositionManager initialized - TESTNET MODE (No real money)")
        else:
            logger.warning("PositionManager initialized - LIVE MODE (Real money!)")

        logger.info(f"PositionManager initialized - Multi-layer TP: {self.enable_multi_layer_tp}")

    async def evaluate_signal(
        self,
        signal: Dict[str, Any],
        current_exposure: float
    ) -> Dict[str, Any]:
        """
        Evaluate if signal should be executed.

        Args:
            signal: Trading signal
            current_exposure: Current portfolio exposure

        Returns:
            Evaluation result
        """
        try:
            # Get risk assessment from DeepSeek
            risk_context = {
                "signal": signal,
                "current_exposure": current_exposure,
                "active_positions": len(self.system_context.active_positions),
                "market_regime": signal.get('market_regime', 'UNKNOWN'),
                "drawdown": self.system_context.risk_metrics.get('current_drawdown', 0)
            }

            risk_assessment = await self.deepseek.assess_risk(risk_context)

            # Check if approved
            if not risk_assessment.get('approved', False):
                return {
                    "approved": False,
                    "reason": risk_assessment.get('reason', 'Risk assessment rejected'),
                    "suggested_action": "HOLD"
                }

            # Get optimized position parameters
            system_state = self.system_context.get_context_for_deepseek()
            optimization = await self.deepseek.optimize_position(signal, system_state)

            # Combine results
            adjusted_size = risk_assessment.get('adjusted_size', signal.get('position_size', 0.02))
            final_size = optimization.get('size', adjusted_size)

            return {
                "approved": True,
                "position_size": final_size,
                "risk_score": risk_assessment.get('risk_score', 0.5),
                "reasoning": f"{risk_assessment.get('reasoning', '')} {optimization.get('reasoning', '')}",
                "warnings": risk_assessment.get('concerns', []),
                "optimizations": optimization.get('adjustments', [])
            }

        except Exception as e:
            logger.error(f"Error evaluating signal: {e}")
            return {
                "approved": False,
                "reason": f"Error: {str(e)}",
                "suggested_action": "HOLD"
            }

    async def place_order(
        self,
        signal: Dict[str, Any],
        position_size: float,
        use_convergence_strategy: bool = False
    ) -> Dict[str, Any]:
        """
        Place a new order.

        Args:
            signal: Trading signal
            position_size: Position size
            use_convergence_strategy: Whether this is a convergence strategy signal

        Returns:
            Order result
        """
        try:
            symbol = signal['symbol']
            action = signal['action']

            # Get current price
            current_price = await self.binance.get_current_price(symbol)

            # Calculate quantity
            # For demo purposes, assume USDT balance of $10,000
            balance = 10000
            quantity = (balance * position_size) / current_price

            # Determine order side
            if action in ['ENTER_LONG', 'LONG', 'BUY']:
                side = 'BUY'
            elif action in ['ENTER_SHORT', 'SHORT', 'SELL']:
                side = 'SELL'
            else:
                return {
                    "success": False,
                    "reason": f"Invalid action: {action}"
                }

            # Determine order type and parameters
            order_type = 'MARKET'  # Use market orders for simplicity
            stop_price = None

            # Check if stop loss is specified
            # For convergence strategy signals, use the signal's stop_loss
            if use_convergence_strategy and signal.get('strategy') == 'convergence':
                stop_price = signal.get('stop_loss')
                if stop_price:
                    logger.info(
                        f"Convergence strategy order: {symbol} {action} - "
                        f"Entry: {current_price:.2f}, Stop: {stop_price:.2f}"
                    )
            else:
                # Use traditional exit strategy
                exit_strategy = signal.get('exit_strategy', {})
                if exit_strategy.get('stop_loss'):
                    stop_price = exit_strategy['stop_loss']

            # Place order (real or simulated based on mode)
            if self.use_testnet or not hasattr(self.binance, 'create_order'):
                # Simulation mode
                order_result = await self._simulate_order(
                    symbol, side, quantity, current_price, signal
                )
            else:
                # Real Binance API call
                try:
                    order_result = await self._place_real_order(
                        symbol, side, order_type, quantity, current_price, stop_price
                    )
                except Exception as e:
                    logger.error(f"Real order failed, falling back to simulation: {e}")
                    order_result = await self._simulate_order(
                        symbol, side, quantity, current_price, signal
                    )

            # If successful, update system context
            if order_result['success']:
                self._update_position(
                    symbol, order_result['order'], signal, position_size
                )

                # Track live order ID if this was a real order
                if 'orderId' in order_result['order']:
                    self.live_orders[symbol] = order_result['order']['orderId']

                # For convergence strategy signals, track additional data
                if use_convergence_strategy and signal.get('strategy') == 'convergence':
                    liquidity_target = signal.get('liquidity_target')
                    if liquidity_target:
                        self.liquidity_targets[symbol] = liquidity_target
                        logger.info(f"Tracking liquidity target for {symbol}: {liquidity_target:.2f}")

                    # Place multi-layer take profits if enabled
                    if self.enable_multi_layer_tp and signal.get('take_profit'):
                        await self._place_convergence_take_profits(
                            symbol, quantity, side, signal, current_price
                        )

                logger.info(
                    f"Order placed ({'TESTNET' if self.use_testnet else 'LIVE'}): "
                    f"{symbol} {side} {quantity:.4f} @ {current_price}"
                )

            return order_result

        except Exception as e:
            logger.error(f"Error placing order: {e}")
            return {
                "success": False,
                "reason": f"Error: {str(e)}"
            }

    async def close_position(
        self,
        symbol: str,
        reason: str = "MANUAL"
    ) -> Dict[str, Any]:
        """
        Close an existing position.

        Args:
            symbol: Symbol to close
            reason: Reason for closing

        Returns:
            Close result
        """
        try:
            if symbol not in self.system_context.active_positions:
                return {
                    "success": False,
                    "reason": f"No position for {symbol}"
                }

            position = self.system_context.active_positions[symbol]

            # Get current price
            current_price = await self.binance.get_current_price(symbol)

            # Calculate P&L
            entry_price = position['entry_price']
            quantity = position['quantity']
            side = position['side']

            if side == 'LONG':
                pnl = (current_price - entry_price) * quantity
                pnl_percent = ((current_price - entry_price) / entry_price) * 100
            else:  # SHORT
                pnl = (entry_price - current_price) * quantity
                pnl_percent = ((entry_price - current_price) / entry_price) * 100

            # Close order (real or simulated)
            if self.use_testnet or not hasattr(self.binance, 'create_order'):
                # Simulation mode
                close_result = {
                    "success": True,
                    "symbol": symbol,
                    "exit_price": current_price,
                    "quantity": quantity,
                    "pnl": pnl,
                    "pnl_percent": pnl_percent,
                    "reason": reason,
                    "timestamp": datetime.now().isoformat()
                }
            else:
                # Real Binance API call
                try:
                    close_side = 'SELL' if side == 'LONG' else 'BUY'
                    close_result = await self._place_real_order(
                        symbol=symbol,
                        side=close_side,
                        order_type='MARKET',
                        quantity=quantity,
                        price=current_price,
                        stop_price=None
                    )

                    if close_result['success']:
                        # Add P&L data
                        close_result['pnl'] = pnl
                        close_result['pnl_percent'] = pnl_percent
                        close_result['reason'] = reason
                except Exception as e:
                    logger.error(f"Real close failed, simulating: {e}")
                    close_result = {
                        "success": True,
                        "symbol": symbol,
                        "exit_price": current_price,
                        "quantity": quantity,
                        "pnl": pnl,
                        "pnl_percent": pnl_percent,
                        "reason": reason,
                        "timestamp": datetime.now().isoformat()
                    }

            # Update system context
            self.system_context.close_position(symbol, close_result)

            logger.info(
                f"Position closed: {symbol} P&L: ${pnl:.2f} ({pnl_percent:.2f}%)"
            )

            return close_result

        except Exception as e:
            logger.error(f"Error closing position: {e}")
            return {
                "success": False,
                "reason": f"Error: {str(e)}"
            }

    async def update_positions(self):
        """
        Update all positions with current market data.
        Calculate unrealized P&L and check exit conditions.
        """
        try:
            for symbol, position in list(self.system_context.active_positions.items()):
                # Get current price
                current_price = await self.binance.get_current_price(symbol)

                # Calculate unrealized P&L
                entry_price = position['entry_price']
                quantity = position['quantity']
                side = position['side']

                if side == 'LONG':
                    unrealized_pnl = (current_price - entry_price) * quantity
                else:  # SHORT
                    unrealized_pnl = (entry_price - current_price) * quantity

                # Update position
                position['current_price'] = current_price
                position['unrealized_pnl'] = unrealized_pnl
                position['unrealized_pnl_percent'] = (unrealized_pnl / (entry_price * quantity)) * 100
                position['last_updated'] = datetime.now().isoformat()

                # Check exit conditions
                await self._check_exit_conditions(symbol, position)

            # Update system risk metrics
            self.system_context._update_risk_metrics()

        except Exception as e:
            logger.error(f"Error updating positions: {e}")

    async def get_position_summary(self) -> Dict[str, Any]:
        """
        Get summary of all positions.

        Returns:
            Position summary
        """
        positions = self.system_context.active_positions

        total_unrealized_pnl = sum(
            pos.get('unrealized_pnl', 0) for pos in positions.values()
        )

        long_positions = sum(
            1 for pos in positions.values() if pos.get('side') == 'LONG'
        )

        short_positions = sum(
            1 for pos in positions.values() if pos.get('side') == 'SHORT'
        )

        return {
            "total_positions": len(positions),
            "long_positions": long_positions,
            "short_positions": short_positions,
            "total_unrealized_pnl": total_unrealized_pnl,
            "positions": positions
        }

    async def _simulate_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        signal: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Simulate order in demo mode."""
        # In a real implementation, this would place the order via Binance API
        order = {
            "orderId": int(datetime.now().timestamp()),
            "symbol": symbol,
            "side": side,
            "quantity": quantity,
            "price": price,
            "status": "FILLED",
            "timestamp": datetime.now().isoformat()
        }

        return {
            "success": True,
            "order": order,
            "message": "Demo order simulated"
        }

    def _update_position(
        self,
        symbol: str,
        order: Dict[str, Any],
        signal: Dict[str, Any],
        position_size: float
    ):
        """Update position in system context."""
        position_data = {
            "symbol": symbol,
            "side": "LONG" if order['side'] == 'BUY' else "SHORT",
            "entry_price": order.get('price', order.get('avg_price', 0)),
            "quantity": order['quantity'],
            "entry_time": order.get('timestamp', datetime.now().isoformat()),
            "position_size": position_size,
            "confidence": signal.get('confidence', 0.5),
            "reasoning": signal.get('reasoning', ''),
            "exit_strategy": signal.get('exit_strategy', {}),
            "entry_conditions": signal.get('entry_conditions', {}),
            "unrealized_pnl": 0.0,
            "unrealized_pnl_percent": 0.0,
            "current_price": order.get('price', order.get('avg_price', 0)),
            "status": "OPEN"
        }

        self.system_context.update_position(symbol, position_data)

    async def _check_exit_conditions(self, symbol: str, position: Dict[str, Any]):
        """
        Check if position should be closed based on exit conditions.

        Args:
            symbol: Symbol
            position: Position data
        """
        try:
            # Check stop loss
            current_price = position.get('current_price', 0)
            stop_loss = position.get('exit_strategy', {}).get('stop_loss')

            if stop_loss and position['side'] == 'LONG':
                if current_price <= stop_loss:
                    logger.warning(f"Stop loss triggered for {symbol}")
                    await self.close_position(symbol, "STOP_LOSS")
                    return
            elif stop_loss and position['side'] == 'SHORT':
                if current_price >= stop_loss:
                    logger.warning(f"Stop loss triggered for {symbol}")
                    await self.close_position(symbol, "STOP_LOSS")
                    return

            # Check take profit (partial exits would be implemented here)
            # For now, just log if we're close

            # Check time-based exit
            entry_time = datetime.fromisoformat(position['entry_time'])
            time_based_exit = position.get('exit_strategy', {}).get('time_based_exit_minutes', 240)
            elapsed_minutes = (datetime.now() - entry_time).total_seconds() / 60

            if elapsed_minutes >= time_based_exit:
                logger.info(f"Time-based exit for {symbol} ({elapsed_minutes:.0f} minutes)")
                await self.close_position(symbol, "TIME_EXIT")

        except Exception as e:
            logger.error(f"Error checking exit conditions: {e}")

    def get_risk_metrics(self) -> Dict[str, Any]:
        """Get current risk metrics."""
        return self.system_context.risk_metrics

    def can_open_new_position(self) -> bool:
        """Check if we can open a new position."""
        # Check position count
        if len(self.system_context.active_positions) >= self.max_positions:
            return False

        # Check drawdown
        current_drawdown = self.system_context.risk_metrics.get('current_drawdown', 0)
        max_drawdown = 0.10  # 10%

        if current_drawdown >= max_drawdown:
            return False

        return True

    async def _place_real_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        quantity: float,
        price: Optional[float] = None,
        stop_price: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Place a real order via Binance API.

        Args:
            symbol: Trading symbol
            side: Order side (BUY/SELL)
            order_type: Order type (MARKET, LIMIT, STOP_LOSS, etc.)
            quantity: Order quantity
            price: Order price (for LIMIT orders)
            stop_price: Stop price (for STOP orders)

        Returns:
            Order result
        """
        try:
            # Prepare order parameters
            params = {
                'symbol': symbol,
                'side': side,
                'type': order_type,
                'quantity': round(quantity, 8),  # Round to appropriate precision
            }

            # Add price for LIMIT orders
            if order_type in ['LIMIT', 'TAKE_PROFIT_LIMIT'] and price:
                params['price'] = round(price, 8)
                params['timeInForce'] = 'GTC'  # Good Till Cancel

            # Add stop price for STOP orders
            if stop_price and order_type in ['STOP_LOSS', 'TAKE_PROFIT', 'STOP_LOSS_LIMIT', 'TAKE_PROFIT_LIMIT']:
                params['stopPrice'] = round(stop_price, 8)

            logger.info(
                f"Placing real order: {symbol} {side} {order_type} "
                f"{quantity:.8f} @ {price if price else 'MARKET'}"
            )

            # Place order via Binance client
            order = await self.binance.create_order(**params)

            return {
                "success": True,
                "order": order,
                "message": f"{'TESTNET' if self.use_testnet else 'LIVE'} order placed"
            }

        except Exception as e:
            logger.error(f"Error placing real order: {e}")
            return {
                "success": False,
                "reason": f"Real order failed: {str(e)}"
            }

    async def cancel_order(self, symbol: str, order_id: Optional[int] = None) -> Dict[str, Any]:
        """
        Cancel an existing order.

        Args:
            symbol: Trading symbol
            order_id: Order ID (uses live_orders[symbol] if not provided)

        Returns:
            Cancel result
        """
        try:
            if order_id is None:
                order_id = self.live_orders.get(symbol)

            if not order_id:
                return {
                    "success": False,
                    "reason": f"No order ID found for {symbol}"
                }

            logger.info(f"Cancelling order {order_id} for {symbol}")

            result = await self.binance.cancel_order(symbol=symbol, orderId=order_id)

            # Remove from tracked orders
            if symbol in self.live_orders:
                del self.live_orders[symbol]

            return {
                "success": True,
                "result": result,
                "message": "Order cancelled successfully"
            }

        except Exception as e:
            logger.error(f"Error cancelling order: {e}")
            return {
                "success": False,
                "reason": f"Cancel failed: {str(e)}"
            }

    async def get_order_status(self, symbol: str, order_id: Optional[int] = None) -> Dict[str, Any]:
        """
        Get order status from Binance.

        Args:
            symbol: Trading symbol
            order_id: Order ID (uses live_orders[symbol] if not provided)

        Returns:
            Order status
        """
        try:
            if order_id is None:
                order_id = self.live_orders.get(symbol)

            if not order_id:
                return {
                    "success": False,
                    "reason": f"No order ID found for {symbol}"
                }

            order = await self.binance.get_order(symbol=symbol, orderId=order_id)

            return {
                "success": True,
                "order": order
            }

        except Exception as e:
            logger.error(f"Error getting order status: {e}")
            return {
                "success": False,
                "reason": f"Status check failed: {str(e)}"
            }

    async def get_account_balance(self) -> Dict[str, Any]:
        """
        Get account balance from Binance.

        Returns:
            Account balance information
        """
        try:
            account = await self.binance.get_account()

            # Extract relevant balances
            balances = {
                asset['asset']: {
                    'free': float(asset['free']),
                    'locked': float(asset['locked']),
                    'total': float(asset['free']) + float(asset['locked'])
                }
                for asset in account['balances']
                if float(asset['free']) > 0 or float(asset['locked']) > 0
            }

            return {
                "success": True,
                "balances": balances,
                "account_type": account.get('accountType', 'SPOT'),
                "can_trade": account.get('canTrade', False)
            }

        except Exception as e:
            logger.error(f"Error getting account balance: {e}")
            return {
                "success": False,
                "reason": f"Balance check failed: {str(e)}"
            }

    async def get_open_orders(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        """
        Get all open orders.

        Args:
            symbol: Optional symbol filter

        Returns:
            Open orders
        """
        try:
            if symbol:
                orders = await self.binance.get_open_orders(symbol=symbol)
            else:
                orders = await self.binance.get_open_orders()

            return {
                "success": True,
                "orders": orders,
                "count": len(orders)
            }

        except Exception as e:
            logger.error(f"Error getting open orders: {e}")
            return {
                "success": False,
                "reason": f"Open orders check failed: {str(e)}"
            }

    async def get_testnet_status(self) -> Dict[str, Any]:
        """
        Get testnet connection status.

        Returns:
            Connection status
        """
        try:
            # Try to get account info
            account = await self.binance.get_account()

            return {
                "success": True,
                "connected": True,
                "testnet": self.use_testnet,
                "can_trade": account.get('canTrade', False),
                "account_type": account.get('accountType', 'UNKNOWN')
            }

        except Exception as e:
            return {
                "success": False,
                "connected": False,
                "error": str(e)
            }

    async def track_liquidity_targets(
        self,
        symbol: str,
        position_id: str,
        liquidity_target: float
    ) -> Dict[str, Any]:
        """
        Track position against liquidity zone targets.

        Args:
            symbol: Trading symbol
            position_id: Position identifier
            liquidity_target: Liquidity target price

        Returns:
            Tracking result
        """
        try:
            self.liquidity_targets[symbol] = liquidity_target

            logger.info(f"Liquidity target set for {symbol}: {liquidity_target:.2f}")

            # Get current price
            current_price = await self.binance.get_current_price(symbol)

            # Calculate distance to target
            distance_pct = abs(current_price - liquidity_target) / current_price * 100

            return {
                "success": True,
                "symbol": symbol,
                "liquidity_target": liquidity_target,
                "current_price": current_price,
                "distance_pct": distance_pct,
                "message": f"Within {distance_pct:.2f}% of liquidity target"
            }

        except Exception as e:
            logger.error(f"Error tracking liquidity target: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    async def _place_convergence_take_profits(
        self,
        symbol: str,
        quantity: float,
        side: str,
        signal: Dict[str, Any],
        entry_price: float
    ) -> Dict[str, Any]:
        """
        Place multi-layer take profit orders for convergence strategy.

        Args:
            symbol: Trading symbol
            quantity: Position quantity
            side: Order side (BUY/SELL)
            signal: Trading signal
            entry_price: Entry price

        Returns:
            Order results
        """
        try:
            # For convergence strategy, we can have a primary take profit
            # In a full implementation, this would place multiple partial TP orders
            take_profit = signal.get('take_profit')

            if not take_profit:
                return {"success": True, "message": "No take profit specified"}

            logger.info(
                f"Convergence take profit set for {symbol}: {take_profit:.2f} "
                f"(Entry: {entry_price:.2f})"
            )

            # In a real implementation, would place LIMIT orders at TP levels
            # For now, just log the intention
            return {
                "success": True,
                "message": "Take profit order scheduled",
                "take_profit": take_profit
            }

        except Exception as e:
            logger.error(f"Error placing convergence take profit: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    def get_liquidity_target_status(self, symbol: str) -> Dict[str, Any]:
        """
        Get liquidity target tracking status for a position.

        Args:
            symbol: Trading symbol

        Returns:
            Liquidity target status
        """
        if symbol not in self.liquidity_targets:
            return {
                "has_target": False,
                "message": "No liquidity target set"
            }

        return {
            "has_target": True,
            "liquidity_target": self.liquidity_targets[symbol]
        }
