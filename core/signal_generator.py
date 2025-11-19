"""
Signal Generator
Generates trading signals by combining technical features with DeepSeek AI analysis.
"""

import asyncio
from datetime import datetime
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from loguru import logger

from core.system_context import SystemContext
from features.engine import FeatureEngine
from deepseek.client import DeepSeekBrain
from ops.db import get_database
from core.strategies.convergence_system import ConvergenceStrategy, MarketRegime, AlignmentState
from core.strategies.scalp_15m_4h import Scalp15m4hStrategy


class SignalGenerator:
    """
    Generates trading signals using DeepSeek AI + Technical Analysis.
    Combines feature analysis with AI decision making.
    """

    def __init__(
        self,
        system_context: SystemContext,
        deepseek_brain: DeepSeekBrain,
        feature_engine: FeatureEngine,
        enable_db_logging: bool = True,
        memory_manager=None,
        enable_convergence_strategy: bool = True,
        enable_scalp_strategy: bool = True
    ):
        """
        Initialize signal generator.

        Args:
            system_context: System state context
            deepseek_brain: DeepSeek AI client
            feature_engine: Feature calculation engine
            enable_db_logging: Enable database signal logging
            memory_manager: M1MemoryManager for memory optimization
            enable_convergence_strategy: Enable convergence strategy signals
            enable_scalp_strategy: Enable scalp 15m/4h strategy signals
        """
        self.system_context = system_context
        self.deepseek = deepseek_brain
        self.features = feature_engine
        self.enable_db_logging = enable_db_logging
        self.memory_manager = memory_manager
        self.enable_convergence_strategy = enable_convergence_strategy
        self.enable_scalp_strategy = enable_scalp_strategy

        # Configuration
        self.confidence_threshold = 0.60
        self.base_position_size = 0.02

        # Initialize convergence strategy
        if self.enable_convergence_strategy:
            self.convergence_strategy = ConvergenceStrategy()
            logger.info("ConvergenceStrategy initialized and enabled")
        else:
            self.convergence_strategy = None
            logger.info("ConvergenceStrategy disabled")

        # Initialize scalp strategy
        if self.enable_scalp_strategy:
            self.scalp_strategy = Scalp15m4hStrategy(risk_per_trade=0.015)
            logger.info("Scalp15m4hStrategy initialized and enabled")
        else:
            self.scalp_strategy = None
            logger.info("Scalp15m4hStrategy disabled")

        # Initialize database if logging enabled
        if self.enable_db_logging:
            self.db = get_database()
            logger.info("SignalGenerator initialized with database logging")
        else:
            self.db = None
            logger.info("SignalGenerator initialized without database logging")

        # P3.3: Feature calculation state for performance optimization
        self.feature_cache = {
            'last_bar_timestamp': None,
            'cached_heavy_features': {},
            'heavy_feature_interval_seconds': 60  # Heavy features recalculated every 60s (bar close)
        }

    async def perform_memory_cleanup(self, data_store=None):
        """
        Perform memory cleanup using the memory manager.

        Args:
            data_store: Optional DataStore instance for cache cleanup
        """
        if self.memory_manager:
            logger.info("Performing memory cleanup via M1MemoryManager")

            # Get current memory usage
            memory_report = self.memory_manager.get_memory_report()
            logger.info(
                f"Memory status: {memory_report['current']['rss_mb']:.0f}MB "
                f"({memory_report['current']['percent']:.1f}%)"
            )

            # Clean up data store cache if provided
            if data_store:
                await self.memory_manager.cleanup_cache(data_store)

            # Check if cleanup is needed
            if self.memory_manager.should_trigger_cleanup():
                logger.warning("Triggering aggressive memory cleanup")
                self.memory_manager.force_cleanup()

            return memory_report
        else:
            logger.debug("No memory manager available, skipping cleanup")
            return None

    def _should_recalculate_heavy_features(self, market_data: Dict[str, pd.Series]) -> bool:
        """
        P3.3: Check if heavy features should be recalculated (bar close detection).

        Heavy features (liquidity, supertrend, chandelier, regime) are only
        recalculated when a new bar closes to optimize CPU usage.

        Args:
            market_data: OHLCV market data

        Returns:
            True if heavy features should be recalculated
        """
        current_timestamp = market_data['close'].index[-1] if len(market_data['close']) > 0 else None
        last_timestamp = self.feature_cache['last_bar_timestamp']

        # Always recalculate on first run
        if last_timestamp is None:
            self.feature_cache['last_bar_timestamp'] = current_timestamp
            self.feature_cache['last_calc_time'] = datetime.now()
            return True

        # Check if timestamp changed (new bar)
        if current_timestamp != last_timestamp:
            self.feature_cache['last_bar_timestamp'] = current_timestamp
            self.feature_cache['last_calc_time'] = datetime.now()
            logger.debug(f"New bar detected at {current_timestamp}, recalculating heavy features")
            return True

        # If same bar, use cached features (don't recalculate based on time)
        # This ensures heavy features are only calculated once per bar
        logger.debug(f"Same bar detected (timestamp: {current_timestamp}), using cached heavy features")
        return False

    def _calculate_features_conditionally(
        self,
        market_data: Dict[str, pd.Series],
        force_calculation: bool = False
    ) -> Dict[str, Any]:
        """
        P3.3: Calculate features with conditional logic for performance.

        - Heavy features (liquidity, supertrend, chandelier, regime): Only on bar close
        - Lightweight features (orderflow): Every tick

        Args:
            market_data: OHLCV market data
            force_calculation: Force recalculation (for testing)

        Returns:
            Dictionary with heavy_features and lightweight_features
        """
        should_recalc_heavy = force_calculation or self._should_recalculate_heavy_features(market_data)

        result = {
            'heavy_features': {},
            'lightweight_features': {},
            'cached': not should_recalc_heavy
        }

        # Calculate heavy features only when needed
        if should_recalc_heavy or force_calculation:
            logger.debug("Calculating heavy features (liquidity, supertrend, chandelier, regime)")

            # Calculate liquidity zones
            result['heavy_features']['liquidity'] = self.features.calculate_liquidity_zones(
                high=market_data['high'],
                low=market_data['low'],
                close=market_data['close'],
                volume=market_data['volume']
            )

            # Calculate supertrend
            result['heavy_features']['supertrend'] = self.features.calculate_advanced_supertrend(
                high=market_data['high'],
                low=market_data['low'],
                close=market_data['close']
            )

            # Calculate chandelier exit
            result['heavy_features']['chandelier'] = self.features.calculate_enhanced_chandelier_exit(
                high=market_data['high'],
                low=market_data['low'],
                close=market_data['close']
            )

            # Calculate market regime
            result['heavy_features']['regime'] = self.features.calculate_market_regime_classification(
                close=market_data['close'],
                volume=market_data['volume']
            )

            # Cache heavy features
            self.feature_cache['cached_heavy_features'] = result['heavy_features'].copy()

        else:
            # Use cached heavy features
            logger.debug("Using cached heavy features")
            result['heavy_features'] = self.feature_cache['cached_heavy_features'].copy()

        # Always calculate lightweight features (orderflow)
        logger.debug("Calculating lightweight features (orderflow)")
        result['lightweight_features']['orderflow'] = self.features.calculate_order_flow_imbalance(
            open=market_data['open'],
            high=market_data['high'],
            low=market_data['low'],
            close=market_data['close'],
            volume=market_data['volume']
        )

        return result

    async def generate_convergence_signal(
        self,
        symbol: str,
        market_data: Dict[str, pd.Series],
        timeframe_data: Dict[str, pd.DataFrame] = None
    ) -> Dict[str, Any]:
        """
        Generate trading signal using the convergence strategy.

        Args:
            symbol: Trading symbol
            market_data: OHLCV data
            timeframe_data: Multi-timeframe data

        Returns:
            Convergence strategy signal
        """
        if not self.convergence_strategy:
            return {
                "symbol": symbol,
                "action": "HOLD",
                "confidence": 0.0,
                "strategy": "convergence",
                "reasoning": "Convergence strategy not enabled",
                "error": False
            }

        try:
            # Prepare market data as DataFrame
            df = pd.DataFrame({
                'open': market_data['open'],
                'high': market_data['high'],
                'low': market_data['low'],
                'close': market_data['close'],
                'volume': market_data['volume']
            })

            # P3.3: Prepare multi-timeframe data using conditional feature calculation
            mtf_data = {}
            current_data = {}
            liquidity_zones = {}

            if timeframe_data:
                # Use feature engine's multi-timeframe preparation
                mtf_data, current_data, liquidity_zones = self.features.prepare_convergence_strategy_input(
                    symbol, timeframe_data, market_data
                )
            else:
                # P3.3: Use conditional feature calculation for better performance
                feature_data = self._calculate_features_conditionally(market_data)
                liquidity_data = feature_data['heavy_features'].get('liquidity', {})
                orderflow_data = feature_data['lightweight_features'].get('orderflow', {})

                # Calculate ATR
                tr1 = market_data['high'] - market_data['low']
                tr2 = abs(market_data['high'] - market_data['close'].shift(1))
                tr3 = abs(market_data['low'] - market_data['close'].shift(1))
                tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
                atr = tr.rolling(window=14).mean()

                current_data = {
                    'close': float(market_data['close'].iloc[-1]),
                    'atr': float(atr.iloc[-1]) if not pd.isna(atr.iloc[-1]) else 0.0,
                    'volatility': float(atr.iloc[-1] / market_data['close'].iloc[-1]) if market_data['close'].iloc[-1] > 0 else 0.0,
                    'orderflow_cumulative': float(orderflow_data.get('cumulative_imbalance', 0.0)),
                    'features_cached': feature_data.get('cached', False)
                }
                liquidity_zones = liquidity_data

            # Generate signal using convergence strategy
            convergence_signal = self.convergence_strategy.generate_signal(df, mtf_data)

            # Convert to standard signal format
            signal = {
                "symbol": symbol,
                "action": convergence_signal.action,
                "confidence": convergence_signal.confidence,
                "strategy": "convergence",
                "entry_price": convergence_signal.entry_price,
                "stop_loss": convergence_signal.stop_loss,
                "take_profit": convergence_signal.take_profit,
                "risk_reward_ratio": convergence_signal.risk_reward_ratio,
                "alignment_score": convergence_signal.alignment_score,
                "regime": convergence_signal.regime,
                "satisfied_conditions": convergence_signal.satisfied_conditions or [],
                "liquidity_target": convergence_signal.liquidity_target,
                "timestamp": datetime.now().isoformat(),
                "reasoning": self._format_convergence_reasoning(convergence_signal),
                "error": False
            }

            logger.info(
                f"Convergence signal generated: {symbol} {signal['action']} "
                f"(confidence: {signal['confidence']:.2f})"
            )

            return signal

        except Exception as e:
            logger.error(f"Error generating convergence signal: {e}")
            return {
                "symbol": symbol,
                "action": "HOLD",
                "confidence": 0.0,
                "strategy": "convergence",
                "reasoning": f"Error: {str(e)}",
                "error": True
            }

    def _format_convergence_reasoning(self, signal) -> str:
        """
        Format convergence signal reasoning for display.

        Args:
            signal: ConvergenceSignal object

        Returns:
            Formatted reasoning string
        """
        reasoning_parts = []

        reasoning_parts.append(f"Action: {signal.action}")
        reasoning_parts.append(f"Confidence: {signal.confidence:.2f}")

        if signal.regime:
            reasoning_parts.append(f"Market Regime: {signal.regime}")

        if signal.alignment_score is not None:
            reasoning_parts.append(f"Alignment Score: {signal.alignment_score:.2f}")

        if signal.satisfied_conditions:
            reasoning_parts.append(f"Satisfied Conditions ({len(signal.satisfied_conditions)}/6):")
            for condition in signal.satisfied_conditions:
                reasoning_parts.append(f"  - {condition.replace('_', ' ').title()}")

        if signal.liquidity_target:
            reasoning_parts.append(f"Liquidity Target: {signal.liquidity_target:.2f}")

        return " | ".join(reasoning_parts)

    async def generate_scalp_signal(
        self,
        symbol: str,
        market_data: Dict[str, pd.Series],
        timeframe_data: Dict[str, pd.DataFrame] = None
    ) -> Dict[str, Any]:
        """
        Generate scalp 15m/4h strategy signal.

        Args:
            symbol: Trading symbol
            market_data: OHLCV data for 15m timeframe
            timeframe_data: Multi-timeframe data including 4h

        Returns:
            Scalp signal dictionary with entry/exit details
        """
        if not self.enable_scalp_strategy or not self.scalp_strategy:
            return {
                "symbol": symbol,
                "action": "WAIT",
                "confidence": 0.0,
                "strategy": "scalp_15m_4h",
                "reasoning": "Scalp strategy disabled",
                "error": True
            }

        try:
            # Prepare 15m data
            data_15m = pd.DataFrame(market_data)
            if data_15m is None or len(data_15m) < 50:
                return {
                    "symbol": symbol,
                    "action": "WAIT",
                    "confidence": 0.0,
                    "strategy": "scalp_15m_4h",
                    "reasoning": "Insufficient 15m data (need 50+ candles)",
                    "error": True
                }

            # Prepare 4h data
            data_4h = None
            if timeframe_data and '4h' in timeframe_data:
                data_4h = timeframe_data['4h']

            # Generate signal using scalp strategy
            scalp_signal = self.scalp_strategy.evaluate(data_15m, data_4h)

            # Format signal for output
            signal = {
                "symbol": symbol,
                "action": scalp_signal.action,
                "confidence": scalp_signal.confidence,
                "strategy": "scalp_15m_4h",
                "entry_price": scalp_signal.entry_price,
                "stop_loss": scalp_signal.stop_loss,
                "take_profit_1": scalp_signal.take_profit_1,
                "take_profit_2": scalp_signal.take_profit_2,
                "position_size": scalp_signal.position_size,
                "entry_type": scalp_signal.entry_type,
                "entry_zone": scalp_signal.entry_zone,
                "entry_trigger": scalp_signal.entry_trigger,
                "timestamp": datetime.now().isoformat(),
                "reasoning": scalp_signal.reasoning,
                "error": False
            }

            logger.info(
                f"Scalp signal generated: {symbol} {signal['action']} "
                f"(confidence: {signal['confidence']:.2f}, type: {signal['entry_type']})"
            )

            return signal

        except Exception as e:
            logger.error(f"Error generating scalp signal: {e}")
            return {
                "symbol": symbol,
                "action": "WAIT",
                "confidence": 0.0,
                "strategy": "scalp_15m_4h",
                "reasoning": f"Error: {str(e)}",
                "error": True
            }

    async def generate_signal(
        self,
        symbol: str,
        market_data: Dict[str, pd.Series],
        timeframe_data: Dict[str, pd.DataFrame] = None
    ) -> Dict[str, Any]:
        """
        Generate comprehensive trading signal.

        Args:
            symbol: Trading symbol
            market_data: OHLCV data
            timeframe_data: Multi-timeframe data

        Returns:
            Complete trading signal
        """
        try:
            # Step 1: Calculate all features
            logger.info(f"Generating signal for {symbol}")
            feature_set = await self.features.compute_all_features(
                symbol, market_data, timeframe_data, system_context=self.system_context
            )

            # Step 2: Prepare market data for DeepSeek
            current_price = market_data['close'].iloc[-1]
            market_context = {
                "price": current_price,
                "change_24h": self._calculate_24h_change(market_data),
                "volume": market_data['volume'].iloc[-1] if 'volume' in market_data else 0,
                "volatility": self._calculate_volatility(market_data['close'])
            }

            # Step 3: Pass memory manager to DeepSeek if available
            if self.memory_manager and not self.deepseek.memory_manager:
                self.deepseek.memory_manager = self.memory_manager
                logger.debug("Integrated M1MemoryManager with DeepSeekBrain")

            # Step 4: Get DeepSeek analysis (with memory-optimized context)
            deepseek_signal = await self.deepseek.get_trading_signal(
                symbol, market_context, feature_set
            )

            # Step 5: Calculate feature confidence
            feature_confidence = self.calculate_feature_confidence(feature_set)

            # Step 6: Combine confidences
            deepseek_confidence = deepseek_signal.get('confidence', 0.5)
            final_confidence = (deepseek_confidence * 0.7) + (feature_confidence * 0.3)

            # Step 7: Generate signal
            signal = self.compile_signal(
                symbol,
                deepseek_signal,
                feature_set,
                final_confidence,
                current_price
            )

            # Step 8: Update system context
            self.system_context.update_feature_calculations(symbol, feature_set)
            self.system_context.update_feature_resource_metrics(
                getattr(self.features, "performance_metrics", {})
            )

            # Step 9: Save to database
            if self.enable_db_logging and self.db:
                await self._save_signal_to_database(signal, feature_set)

            logger.info(
                f"Signal generated: {symbol} {signal['action']} "
                f"(confidence: {signal['confidence']:.2f})"
            )

            return signal

        except Exception as e:
            logger.error(f"Error generating signal: {e}")
            return {
                "symbol": symbol,
                "action": "HOLD",
                "confidence": 0.0,
                "reasoning": f"Error generating signal: {str(e)}",
                "error": True
            }

    def calculate_feature_confidence(self, feature_set: Dict[str, Any]) -> float:
        """
        Calculate confidence based on feature alignment.

        Args:
            feature_set: Calculated features

        Returns:
            Confidence score (0-1)
        """
        confidence_factors = []

        # 1. Liquidity zone proximity
        distance = abs(feature_set.get('distance_to_zone_pct', 0))
        if distance < 0.01:  # Within 1%
            confidence_factors.append(0.9)
        elif distance < 0.02:  # Within 2%
            confidence_factors.append(0.75)
        else:
            confidence_factors.append(0.6)

        # 2. Market regime confidence
        regime_conf = feature_set.get('regime_confidence', 0.5)
        confidence_factors.append(regime_conf)

        # 3. Timeframe alignment
        alignment = feature_set.get('timeframe_alignment', 0.5)
        confidence_factors.append(alignment)

        # 4. Trend consistency
        consistency = feature_set.get('trend_consistency', 0.5)
        confidence_factors.append(consistency)

        # 5. Order flow strength
        imbalance = abs(feature_set.get('order_flow_imbalance', 0))
        flow_confidence = min(imbalance * 5, 1.0)  # Scale to 0-1
        confidence_factors.append(flow_confidence)

        # 6. Supertrend strength
        st_strength = abs(feature_set.get('supertrend_strength', 0))
        trend_confidence = min(st_strength / 2, 1.0)
        confidence_factors.append(trend_confidence)

        return np.mean(confidence_factors)

    def compile_signal(
        self,
        symbol: str,
        deepseek_analysis: Dict[str, Any],
        feature_set: Dict[str, Any],
        confidence: float,
        current_price: float
    ) -> Dict[str, Any]:
        """
        Compile final signal from DeepSeek and features.

        Args:
            symbol: Trading symbol
            deepseek_analysis: DeepSeek AI analysis
            feature_set: Technical features
            confidence: Combined confidence score
            current_price: Current market price

        Returns:
            Complete trading signal
        """
        # Extract DeepSeek decision
        action = deepseek_analysis.get('action', 'HOLD')
        reasoning = deepseek_analysis.get('reasoning', '')
        deepseek_confidence = deepseek_analysis.get('confidence', 0.5)

        # Calculate position size
        position_size = self.calculate_position_size(confidence, feature_set)

        # Generate entry conditions
        entry_conditions = self.generate_entry_conditions(feature_set, current_price)

        # Generate exit strategy
        exit_strategy = self.generate_exit_strategy(
            feature_set, action, current_price
        )

        # Get feature highlights
        highlights = self.features.get_feature_highlights(feature_set)

        # Get current overlay state from SystemContext
        overlay_state = self.system_context.overlay_state.copy()

        # Build signal
        signal = {
            "symbol": symbol,
            "action": action,
            "confidence": confidence,
            "position_size": position_size,
            "reasoning": reasoning,
            "deepseek_confidence": deepseek_confidence,
            "feature_confidence": self.calculate_feature_confidence(feature_set),
            "entry_conditions": entry_conditions,
            "exit_strategy": exit_strategy,
            "feature_highlights": highlights,
            "timestamp": datetime.now().isoformat(),
            "risk_adjustment": self.calculate_risk_adjustment(feature_set),
            "market_regime": feature_set.get('market_regime', 'UNKNOWN'),
            "current_price": current_price,
            "chart_overlays": overlay_state
        }

        return signal

    def calculate_position_size(
        self,
        confidence: float,
        feature_set: Dict[str, Any]
    ) -> float:
        """
        Calculate optimal position size based on confidence and risk.

        Args:
            confidence: Signal confidence
            feature_set: Market features

        Returns:
            Position size (as decimal, e.g., 0.05 for 5%)
        """
        base_size = self.base_position_size

        # Scale with confidence
        size_multiplier = min(confidence * 2, 1.0)

        # Adjust for market regime
        regime = feature_set.get('market_regime', 'UNKNOWN')
        if regime in ['TRENDING_HIGH_VOL', 'RANGING_EXPANSION']:
            size_multiplier *= 0.7  # Reduce size in high volatility
        elif regime in ['TRENDING_LOW_VOL']:
            size_multiplier *= 1.1  # Slightly increase in trending low vol

        # Adjust for timeframe alignment
        alignment = feature_set.get('timeframe_alignment', 0.5)
        if alignment > 0.75:
            size_multiplier *= 1.2  # Increase when timeframes align

        position_size = base_size * size_multiplier

        # Cap at maximum
        max_size = 0.05  # 5% maximum
        return min(position_size, max_size)

    def generate_entry_conditions(
        self,
        feature_set: Dict[str, Any],
        current_price: float
    ) -> Dict[str, Any]:
        """
        Generate specific entry conditions.

        Args:
            feature_set: Technical features
            current_price: Current price

        Returns:
            Entry conditions
        """
        liquidity_zone = feature_set.get('nearest_zone', current_price)

        conditions = {
            "preferred_entry": float(liquidity_zone),
            "max_entry_price": current_price * 1.005,  # 0.5% above current
            "min_entry_price": current_price * 0.995,  # 0.5% below current
            "entry_timeout_minutes": 15,
            "required_confirmations": []
        }

        # Add feature-based confirmations
        if feature_set.get('supertrend_trend') == 'uptrend':
            conditions["required_confirmations"].append("price_above_supertrend")
        elif feature_set.get('supertrend_trend') == 'downtrend':
            conditions["required_confirmations"].append("price_below_supertrend")

        # Volume confirmation
        conditions["required_confirmations"].append("volume_above_average")

        # Order flow confirmation
        imbalance = feature_set.get('order_flow_imbalance', 0)
        if imbalance > 0:
            conditions["required_confirmations"].append("order_flow_positive")
        else:
            conditions["required_confirmations"].append("order_flow_negative")

        return conditions

    def generate_exit_strategy(
        self,
        feature_set: Dict[str, Any],
        action: str,
        current_price: float
    ) -> Dict[str, Any]:
        """
        Generate exit strategy based on features.

        Args:
            feature_set: Technical features
            action: Trading action
            current_price: Current price

        Returns:
            Exit strategy
        """
        # Calculate ATR for stop loss
        atr = self._estimate_atr(feature_set)

        if action == 'ENTER_LONG' or action == 'LONG':
            stop_loss = current_price - (atr * 2.0)
            take_profit_levels = [
                {"price": current_price * 1.02, "size": 0.5},  # 50% at +2%
                {"price": current_price * 1.04, "size": 0.25},  # 25% at +4%
                {"price": current_price * 1.06, "size": 0.25}   # 25% at +6%
            ]
            trailing_activation = current_price * 1.015  # Activate at +1.5%

        elif action == 'ENTER_SHORT' or action == 'SHORT':
            stop_loss = current_price + (atr * 2.0)
            take_profit_levels = [
                {"price": current_price * 0.98, "size": 0.5},  # 50% at -2%
                {"price": current_price * 0.96, "size": 0.25},  # 25% at -4%
                {"price": current_price * 0.94, "size": 0.25}   # 25% at -6%
            ]
            trailing_activation = current_price * 0.985  # Activate at -1.5%

        else:  # HOLD or CLOSE
            stop_loss = None
            take_profit_levels = []
            trailing_activation = None

        return {
            "stop_loss": stop_loss,
            "take_profit_levels": take_profit_levels,
            "trailing_stop_activation": trailing_activation,
            "trailing_stop_distance": atr * 1.5,
            "time_based_exit_minutes": 240,  # 4 hours
            "confidence_drop_exit": 0.15  # Exit if confidence drops 15%
        }

    def calculate_risk_adjustment(self, feature_set: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate risk-based adjustments.

        Args:
            feature_set: Technical features

        Returns:
            Risk adjustments
        """
        adjustments = {
            "volatility_adj": 1.0,
            "correlation_adj": 1.0,
            "regime_adj": 1.0
        }

        # Volatility adjustment
        vol_percentile = feature_set.get('volatility_percentile', 0.5)
        if vol_percentile > 0.8:
            adjustments["volatility_adj"] = 0.7  # Reduce in high vol
        elif vol_percentile < 0.2:
            adjustments["volatility_adj"] = 1.1  # Increase in low vol

        # Regime adjustment
        regime = feature_set.get('market_regime', 'UNKNOWN')
        if 'TRENDING' in regime:
            adjustments["regime_adj"] = 1.0  # Neutral
        elif 'RANGING' in regime:
            adjustments["regime_adj"] = 0.8  # Reduce in ranging

        return adjustments

    def _calculate_24h_change(self, market_data: Dict[str, pd.Series]) -> float:
        """Calculate 24h price change percentage."""
        if len(market_data['close']) < 24:
            return 0.0
        current = market_data['close'].iloc[-1]
        previous_24h = market_data['close'].iloc[-24]
        return ((current - previous_24h) / previous_24h) * 100

    def _calculate_volatility(self, close_prices: pd.Series) -> float:
        """Calculate price volatility."""
        returns = close_prices.pct_change().dropna()
        return returns.std() * 100

    def _estimate_atr(self, feature_set: Dict[str, Any]) -> float:
        """Estimate Average True Range from features."""
        # This is a simplified estimation
        # In practice, you'd calculate ATR from actual price data
        supertrend_value = feature_set.get('supertrend_value', 0)
        current_price = feature_set.get('current_price', supertrend_value)

        # Rough ATR estimation
        atr = current_price * 0.02  # Assume 2% ATR
        return atr

    def get_signal_strength(self, signal: Dict[str, Any]) -> str:
        """
        Categorize signal strength.

        Args:
            signal: Trading signal

        Returns:
            Strength category
        """
        confidence = signal.get('confidence', 0)

        if confidence >= 0.8:
            return "STRONG"
        elif confidence >= 0.65:
            return "MEDIUM"
        elif confidence >= 0.5:
            return "WEAK"
        else:
            return "NO_SIGNAL"

    def should_trade(self, signal: Dict[str, Any]) -> bool:
        """
        Determine if signal should be traded.

        Args:
            signal: Trading signal

        Returns:
            Whether to execute trade
        """
        action = signal.get('action', 'HOLD')
        confidence = signal.get('confidence', 0)

        # Don't trade if action is HOLD or confidence is too low
        if action == 'HOLD' or confidence < self.confidence_threshold:
            return False

        # Check system health
        current_drawdown = self.system_context.risk_metrics.get('current_drawdown', 0)
        max_drawdown = 0.10  # 10% max

        if current_drawdown >= max_drawdown:
            return False

        return True

    async def _save_signal_to_database(
        self,
        signal: Dict[str, Any],
        feature_set: Dict[str, Any]
    ) -> Optional[int]:
        """
        Save signal to database with overlay metadata.

        Args:
            signal: Generated signal (includes chart_overlays from compile_signal)
            feature_set: Feature calculations

        Returns:
            Signal ID if successful, None otherwise
        """
        if not self.db:
            return None

        try:
            # Determine if signal is tradeable
            is_tradeable = self.should_trade(signal)

            # Get signal strength
            signal_strength = self.get_signal_strength(signal)

            # Add database-specific fields
            signal['is_tradeable'] = is_tradeable
            signal['signal_strength'] = signal_strength
            signal['volatility_percentile'] = feature_set.get('volatility_percentile')
            signal['trend_strength'] = feature_set.get('trend_strength')

            # Save signal with overlay metadata to database
            # Note: signal['chart_overlays'] contains the overlay state at signal generation time
            signal_id = await self.db.save_signal(signal)
            return signal_id

        except Exception as e:
            logger.error(f"Failed to save signal to database: {e}")
            return None

    async def get_signal_history(
        self,
        symbol: Optional[str] = None,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Retrieve signal history from database.

        Args:
            symbol: Filter by symbol
            limit: Maximum number of records

        Returns:
            List of signals
        """
        if not self.db:
            logger.warning("Database not initialized - cannot retrieve signal history")
            return []

        try:
            signals = await self.db.get_signals(symbol=symbol, limit=limit)
            return signals
        except Exception as e:
            logger.error(f"Failed to retrieve signal history: {e}")
            return []

    async def get_latest_signals(
        self,
        symbol: Optional[str] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Get latest signals from database.

        Args:
            symbol: Filter by symbol
            limit: Maximum number of records

        Returns:
            List of latest signals
        """
        if not self.db:
            return []

        try:
            signals = await self.db.get_signals(
                symbol=symbol,
                limit=limit,
                min_confidence=self.confidence_threshold
            )
            return signals
        except Exception as e:
            logger.error(f"Failed to retrieve latest signals: {e}")
            return []

    async def get_signal_statistics(
        self,
        symbol: Optional[str] = None,
        days: int = 30
    ) -> Dict[str, Any]:
        """
        Get signal statistics from database.

        Args:
            symbol: Filter by symbol
            days: Number of days to analyze

        Returns:
            Statistics dictionary
        """
        if not self.db:
            return {}

        try:
            stats = await self.db.get_signal_statistics(symbol=symbol, days=days)
            return stats
        except Exception as e:
            logger.error(f"Failed to retrieve signal statistics: {e}")
            return {}

    async def mark_signal_executed(self, signal_id: int, executed: bool = True) -> bool:
        """
        Mark signal as executed in database.

        Args:
            signal_id: Signal ID
            executed: Whether signal was executed

        Returns:
            Success status
        """
        if not self.db:
            return False

        try:
            return await self.db.update_signal_execution(signal_id, executed)
        except Exception as e:
            logger.error(f"Failed to mark signal as executed: {e}")
            return False
