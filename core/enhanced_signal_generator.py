"""
Enhanced Signal Generator with ML Integration
Combines DeepSeek AI + Technical Analysis + Machine Learning predictions.
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
from models.predictor import EnsemblePredictor
from models.backtester import Backtester, StrategyValidator


class EnhancedSignalGenerator:
    """
    Advanced signal generator integrating:
    - DeepSeek AI analysis
    - Technical feature analysis
    - Machine learning predictions
    - Risk assessment
    - Backtesting validation
    """

    def __init__(
        self,
        system_context: SystemContext,
        deepseek_brain: DeepSeekBrain,
        feature_engine: FeatureEngine
    ):
        """
        Initialize enhanced signal generator.

        Args:
            system_context: System state context
            deepseek_brain: DeepSeek AI client
            feature_engine: Feature calculation engine
        """
        self.system_context = system_context
        self.deepseek = deepseek_brain
        self.features = feature_engine

        # ML predictor
        self.ml_predictor = EnsemblePredictor()

        # Configuration
        self.confidence_threshold = 0.65
        self.base_position_size = 0.02
        self.min_trade_signals = 3  # Minimum signals needed to trade

        # Signal history
        self.signal_history = []
        self.prediction_cache = {}

        logger.info("Enhanced SignalGenerator initialized with ML integration")

    async def generate_signal(
        self,
        symbol: str,
        market_data: Dict[str, pd.Series],
        timeframe_data: Dict[str, pd.DataFrame] = None
    ) -> Dict[str, Any]:
        """
        Generate comprehensive trading signal with ML enhancement.

        Args:
            symbol: Trading symbol
            market_data: OHLCV data
            timeframe_data: Multi-timeframe data

        Returns:
            Complete trading signal with ML predictions
        """
        try:
            start_time = datetime.now()
            logger.info(f"🚀 Generating ENHANCED signal for {symbol}")

            # Step 1: Calculate all technical features
            logger.info(f"  → Computing technical features...")
            feature_set = await self.features.compute_all_features(
                symbol, market_data, timeframe_data
            )

            # Step 2: Get ML predictions
            logger.info(f"  → Getting ML predictions...")
            ml_predictions = self._get_ml_predictions(
                symbol, market_data, feature_set
            )

            # Step 3: Get DeepSeek AI analysis
            logger.info(f"  → Getting DeepSeek AI analysis...")
            ai_signal = await self._get_ai_signal(
                symbol, market_data, feature_set, ml_predictions
            )

            # Step 4: Combine all signals
            logger.info(f"  → Combining signals...")
            combined_signal = self._combine_all_signals(
                symbol, feature_set, ml_predictions, ai_signal, market_data
            )

            # Step 5: Validate signal
            logger.info(f"  → Validating signal...")
            validation = self._validate_signal(combined_signal, feature_set)

            # Step 6: Calculate position parameters
            logger.info(f"  → Calculating position parameters...")
            position_params = self._calculate_position_parameters(
                combined_signal, feature_set, ml_predictions
            )

            # Step 7: Build final signal
            final_signal = self._build_final_signal(
                symbol=symbol,
                feature_set=feature_set,
                ml_predictions=ml_predictions,
                ai_signal=ai_signal,
                combined_signal=combined_signal,
                validation=validation,
                position_params=position_params,
                market_data=market_data,
                execution_time=(datetime.now() - start_time).total_seconds()
            )

            # Step 8: Log signal
            self._log_signal(final_signal)

            # Step 9: Store in history
            self.signal_history.append(final_signal)

            logger.info(
                f"✅ Signal generated: {symbol} {final_signal['action']} "
                f"(confidence: {final_signal['confidence']:.2f}, "
                f"ML: {final_signal['ml_signal']['prediction']}, "
                f"AI: {final_signal['ai_signal']['action']})"
            )

            return final_signal

        except Exception as e:
            logger.error(f"Error generating enhanced signal: {e}")
            import traceback
            traceback.print_exc()
            return {
                "symbol": symbol,
                "action": "HOLD",
                "confidence": 0.0,
                "reasoning": f"Error generating signal: {str(e)}",
                "error": True
            }

    def _get_ml_predictions(
        self,
        symbol: str,
        market_data: Dict[str, pd.Series],
        feature_set: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Get ML model predictions."""
        try:
            # Convert market_data to DataFrame
            df = pd.DataFrame(market_data)

            # Get predictions
            predictions = self.ml_predictor.get_prediction(
                symbol=symbol,
                df=df,
                features=feature_set,
                horizon=5
            )

            return predictions

        except Exception as e:
            logger.error(f"Error getting ML predictions: {e}")
            return {
                'direction': {'prediction': 0, 'confidence': 0.5},
                'price_change': {'predicted_return': 0.0},
                'volatility': {'prediction': 0, 'confidence': 0.5},
                'combined_signal': {'action': 'HOLD', 'confidence': 0.5}
            }

    async def _get_ai_signal(
        self,
        symbol: str,
        market_data: Dict[str, pd.Series],
        feature_set: Dict[str, Any],
        ml_predictions: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Get DeepSeek AI analysis."""
        try:
            current_price = market_data['close'].iloc[-1]
            market_context = {
                "price": current_price,
                "change_24h": self._calculate_24h_change(market_data),
                "volume": market_data['volume'].iloc[-1] if 'volume' in market_data else 0,
                "volatility": self._calculate_volatility(market_data['close'])
            }

            # Add ML predictions to market context
            market_context["ml_direction"] = ml_predictions.get('direction', {})
            market_context["ml_confidence"] = ml_predictions.get('direction', {}).get('confidence', 0.5)

            system_state = self.system_context.get_context_for_deepseek()

            ai_signal = await self.deepseek.get_trading_signal(
                symbol, market_context, feature_set, system_state
            )

            return ai_signal

        except Exception as e:
            logger.error(f"Error getting AI signal: {e}")
            return {
                "action": "HOLD",
                "confidence": 0.0,
                "reasoning": f"AI error: {str(e)}"
            }

    def _combine_all_signals(
        self,
        symbol: str,
        feature_set: Dict[str, Any],
        ml_predictions: Dict[str, Any],
        ai_signal: Dict[str, Any],
        market_data: Dict[str, pd.Series]
    ) -> Dict[str, Any]:
        """Combine ML, AI, and technical signals."""
        # ML signal (0 to 1, where 0.5 is neutral)
        ml_dir = ml_predictions.get('direction', {})
        ml_score = ml_dir.get('prediction', 0)
        ml_confidence = ml_dir.get('confidence', 0.5)

        # Normalize ML score to 0-1 range
        if ml_score > 0:
            ml_normalized = 0.5 + (ml_score * ml_confidence * 0.5)
        elif ml_score < 0:
            ml_normalized = 0.5 + (ml_score * ml_confidence * 0.5)
        else:
            ml_normalized = 0.5

        # AI signal
        ai_action = ai_signal.get('action', 'HOLD')
        ai_confidence = ai_signal.get('confidence', 0.5)

        ai_score = 0.5
        if 'LONG' in ai_action.upper():
            ai_score = 0.5 + (ai_confidence * 0.5)
        elif 'SHORT' in ai_action.upper():
            ai_score = 0.5 - (ai_confidence * 0.5)

        # Technical signal
        tech_score = self._calculate_technical_score(feature_set, market_data)

        # Weighted combination
        weights = {
            'ml': 0.40,
            'ai': 0.35,
            'technical': 0.25
        }

        combined_score = (
            ml_normalized * weights['ml'] +
            ai_score * weights['ai'] +
            tech_score * weights['technical']
        )

        # Determine action
        if combined_score > 0.65:
            action = "BUY"
        elif combined_score < 0.35:
            action = "SELL"
        else:
            action = "HOLD"

        # Calculate confidence
        confidence_factors = [ml_confidence, ai_confidence]
        tech_confidence = self._calculate_technical_confidence(feature_set)
        confidence_factors.append(tech_confidence)

        final_confidence = np.mean(confidence_factors)

        return {
            'action': action,
            'confidence': final_confidence,
            'score': combined_score,
            'components': {
                'ml': {
                    'score': ml_normalized,
                    'raw_prediction': ml_score,
                    'confidence': ml_confidence
                },
                'ai': {
                    'score': ai_score,
                    'action': ai_action,
                    'confidence': ai_confidence
                },
                'technical': {
                    'score': tech_score,
                    'confidence': tech_confidence
                },
                'weights': weights
            }
        }

    def _calculate_technical_score(
        self,
        feature_set: Dict[str, Any],
        market_data: Dict[str, pd.Series]
    ) -> float:
        """Calculate technical analysis score (0-1)."""
        score = 0.5  # Start neutral

        # Supertrend
        st_trend = feature_set.get('supertrend_trend', 'neutral')
        st_strength = feature_set.get('supertrend_strength', 0)
        if st_trend == 'uptrend':
            score += 0.15 * st_strength
        elif st_trend == 'downtrend':
            score -= 0.15 * st_strength

        # Order flow
        order_flow = feature_set.get('order_flow_imbalance', 0)
        score += min(order_flow * 0.1, 0.15)

        # Timeframe alignment
        alignment = feature_set.get('timeframe_alignment', 0.5)
        if alignment > 0.75:
            score += 0.1

        # Market regime
        regime = feature_set.get('market_regime', 'UNKNOWN')
        if 'TRENDING' in regime and 'HIGH' in regime:
            score += 0.05

        # Liquidity zones
        distance = abs(feature_set.get('distance_to_zone_pct', 0))
        if distance < 0.01:
            score += 0.05

        return max(0, min(score, 1))

    def _calculate_technical_confidence(self, feature_set: Dict[str, Any]) -> float:
        """Calculate technical analysis confidence."""
        confidences = []

        # Supertrend confidence
        st_strength = abs(feature_set.get('supertrend_strength', 0))
        confidences.append(min(st_strength / 2, 1.0))

        # Regime confidence
        regime_conf = feature_set.get('regime_confidence', 0.5)
        confidences.append(regime_conf)

        # Timeframe alignment
        alignment = feature_set.get('timeframe_alignment', 0.5)
        confidences.append(alignment)

        # Trend consistency
        consistency = feature_set.get('trend_consistency', 0.5)
        confidences.append(consistency)

        return np.mean(confidences) if confidences else 0.5

    def _validate_signal(
        self,
        combined_signal: Dict[str, Any],
        feature_set: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate signal against system constraints."""
        validation = {
            'valid': True,
            'warnings': [],
            'rejected_reasons': []
        }

        # Check confidence threshold
        confidence = combined_signal['confidence']
        if confidence < self.confidence_threshold:
            validation['rejected_reasons'].append(
                f"Confidence too low: {confidence:.2f} < {self.confidence_threshold}"
            )

        # Check action
        action = combined_signal['action']
        if action == 'HOLD':
            validation['rejected_reasons'].append("Signal indicates HOLD")

        # Check market regime
        regime = feature_set.get('market_regime', 'UNKNOWN')
        if 'RANGING' in regime and action != 'HOLD':
            validation['warnings'].append("Trading in ranging market")

        # Check system risk
        current_drawdown = self.system_context.risk_metrics.get('current_drawdown', 0)
        max_drawdown = 0.10

        if current_drawdown >= max_drawdown * 0.8:
            validation['warnings'].append(
                f"High drawdown: {current_drawdown:.2%} of max"
            )

        # Check volume
        order_flow = feature_set.get('order_flow_imbalance', 0)
        if abs(order_flow) < 0.05:
            validation['warnings'].append("Low order flow - limited conviction")

        # Final validation
        if validation['rejected_reasons']:
            validation['valid'] = False

        return validation

    def _calculate_position_parameters(
        self,
        combined_signal: Dict[str, Any],
        feature_set: Dict[str, Any],
        ml_predictions: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate position size and risk parameters."""
        base_size = self.base_position_size
        confidence = combined_signal['confidence']

        # Scale with confidence
        size_multiplier = min(confidence * 1.5, 1.5)

        # ML prediction adjustment
        ml_confidence = ml_predictions.get('direction', {}).get('confidence', 0.5)
        size_multiplier *= (0.5 + ml_confidence)

        # Market regime adjustment
        regime = feature_set.get('market_regime', 'UNKNOWN')
        if 'HIGH_VOL' in regime:
            size_multiplier *= 0.7  # Reduce in high volatility
        elif 'LOW_VOL' in regime:
            size_multiplier *= 1.1  # Slightly increase in low volatility

        # Timeframe alignment adjustment
        alignment = feature_set.get('timeframe_alignment', 0.5)
        if alignment > 0.75:
            size_multiplier *= 1.2

        position_size = base_size * size_multiplier
        position_size = min(position_size, 0.05)  # Cap at 5%

        return {
            'position_size': position_size,
            'size_multiplier': size_multiplier,
            'risk_level': 'LOW' if position_size < 0.03 else 'MEDIUM' if position_size < 0.04 else 'HIGH'
        }

    def _build_final_signal(
        self,
        symbol: str,
        feature_set: Dict[str, Any],
        ml_predictions: Dict[str, Any],
        ai_signal: Dict[str, Any],
        combined_signal: Dict[str, Any],
        validation: Dict[str, Any],
        position_params: Dict[str, Any],
        market_data: Dict[str, pd.Series],
        execution_time: float
    ) -> Dict[str, Any]:
        """Build final comprehensive signal."""
        current_price = market_data['close'].iloc[-1]

        # Calculate entry/exit conditions
        entry_conditions = self._generate_entry_conditions(feature_set, current_price)
        exit_strategy = self._generate_exit_strategy(combined_signal['action'], current_price)

        # Get feature highlights
        highlights = self.features.get_feature_highlights(feature_set)

        # Build final signal
        signal = {
            # Core signal data
            'symbol': symbol,
            'action': combined_signal['action'],
            'confidence': combined_signal['confidence'],
            'position_size': position_params['position_size'],

            # Detailed analysis
            'reasoning': self._build_reasoning(combined_signal, ai_signal, ml_predictions),
            'feature_highlights': highlights,

            # Component breakdown
            'ml_signal': {
                'direction': ml_predictions.get('direction', {}),
                'price_prediction': ml_predictions.get('price_change', {}),
                'volatility': ml_predictions.get('volatility', {}),
                'combined': ml_predictions.get('combined_signal', {})
            },
            'ai_signal': ai_signal,
            'technical_signal': {
                'supertrend': {
                    'trend': feature_set.get('supertrend_trend', 'neutral'),
                    'strength': feature_set.get('supertrend_strength', 0)
                },
                'order_flow': feature_set.get('order_flow_imbalance', 0),
                'market_regime': feature_set.get('market_regime', 'UNKNOWN'),
                'timeframe_alignment': feature_set.get('timeframe_alignment', 0.5)
            },

            # Position management
            'entry_conditions': entry_conditions,
            'exit_strategy': exit_strategy,

            # Risk management
            'risk_assessment': self._assess_risk(combined_signal, feature_set, position_params),

            # Metadata
            'timestamp': datetime.now().isoformat(),
            'execution_time_ms': execution_time * 1000,
            'validation': validation,
            'version': '2.0-enhanced'
        }

        return signal

    def _build_reasoning(
        self,
        combined_signal: Dict[str, Any],
        ai_signal: Dict[str, Any],
        ml_predictions: Dict[str, Any]
    ) -> str:
        """Build detailed reasoning for the signal."""
        reasoning_parts = []

        # ML reasoning
        ml_dir = ml_predictions.get('direction', {})
        ml_pred = ml_dir.get('prediction', 0)
        ml_conf = ml_dir.get('confidence', 0)

        if abs(ml_pred) > 0:
            direction_str = "UPWARD" if ml_pred > 0 else "DOWNWARD"
            reasoning_parts.append(
                f"ML models predict {direction_str} movement "
                f"(confidence: {ml_conf:.2f})"
            )

        # AI reasoning
        ai_action = ai_signal.get('action', 'HOLD')
        ai_reasoning = ai_signal.get('reasoning', '')
        if ai_reasoning:
            reasoning_parts.append(f"DeepSeek AI: {ai_reasoning[:200]}...")

        # Technical reasoning
        components = combined_signal.get('components', {})
        tech_score = components.get('technical', {}).get('score', 0.5)
        reasoning_parts.append(
            f"Technical analysis score: {tech_score:.2f}/1.0"
        )

        # Final recommendation
        confidence = combined_signal['confidence']
        if confidence > 0.8:
            reasoning_parts.append("HIGH CONVICTION signal")
        elif confidence > 0.65:
            reasoning_parts.append("MODERATE CONVICTION signal")
        else:
            reasoning_parts.append("LOW CONVICTION signal")

        return " | ".join(reasoning_parts)

    def _generate_entry_conditions(
        self,
        feature_set: Dict[str, Any],
        current_price: float
    ) -> Dict[str, Any]:
        """Generate entry conditions."""
        return {
            "max_entry_price": current_price * 1.003,  # 0.3% above
            "min_entry_price": current_price * 0.997,  # 0.3% below
            "entry_timeout_minutes": 15,
            "required_confirmations": [
                "volume_above_average",
                "order_flow_alignment"
            ]
        }

    def _generate_exit_strategy(
        self,
        action: str,
        current_price: float
    ) -> Dict[str, Any]:
        """Generate exit strategy."""
        atr_estimate = current_price * 0.02  # Assume 2% ATR

        if 'BUY' in action.upper():
            return {
                "stop_loss": current_price - (atr_estimate * 1.5),
                "take_profit_levels": [
                    {"price": current_price * 1.02, "size": 0.5},  # 50% at +2%
                    {"price": current_price * 1.035, "size": 0.3},  # 30% at +3.5%
                    {"price": current_price * 1.05, "size": 0.2}   # 20% at +5%
                ],
                "trailing_stop_activation": current_price * 1.015
            }
        elif 'SELL' in action.upper():
            return {
                "stop_loss": current_price + (atr_estimate * 1.5),
                "take_profit_levels": [
                    {"price": current_price * 0.98, "size": 0.5},  # 50% at -2%
                    {"price": current_price * 0.965, "size": 0.3}, # 30% at -3.5%
                    {"price": current_price * 0.95, "size": 0.2}   # 20% at -5%
                ],
                "trailing_stop_activation": current_price * 0.985
            }
        else:
            return {}

    def _assess_risk(
        self,
        combined_signal: Dict[str, Any],
        feature_set: Dict[str, Any],
        position_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Assess signal risk."""
        return {
            "level": position_params['risk_level'],
            "volatility_risk": "MEDIUM",
            "market_condition_risk": "LOW",
            "max_loss_estimate": position_params['position_size'] * 0.02,  # 2% max loss
            "risk_score": 0.3  # 0-1 scale
        }

    def _calculate_24h_change(self, market_data: Dict[str, pd.Series]) -> float:
        """Calculate 24h price change."""
        if len(market_data['close']) < 24:
            return 0.0
        current = market_data['close'].iloc[-1]
        previous_24h = market_data['close'].iloc[-24]
        return ((current - previous_24h) / previous_24h) * 100

    def _calculate_volatility(self, close_prices: pd.Series) -> float:
        """Calculate price volatility."""
        returns = close_prices.pct_change().dropna()
        return returns.std() * 100

    def _log_signal(self, signal: Dict[str, Any]):
        """Log signal to file."""
        try:
            import os
            os.makedirs('logs', exist_ok=True)

            with open('logs/signal_generation.log', 'a') as f:
                f.write(f"\n{'='*80}\n")
                f.write(f"Timestamp: {signal['timestamp']}\n")
                f.write(f"Symbol: {signal['symbol']}\n")
                f.write(f"Action: {signal['action']}\n")
                f.write(f"Confidence: {signal['confidence']:.3f}\n")
                f.write(f"Position Size: {signal['position_size']:.4f}\n")
                f.write(f"ML Direction: {signal['ml_signal']['direction']}\n")
                f.write(f"AI Action: {signal['ai_signal'].get('action', 'N/A')}\n")
                f.write(f"Technical Score: {signal['technical_signal']}\n")
                f.write(f"Reasoning: {signal['reasoning']}\n")
                f.write(f"Execution Time: {signal['execution_time_ms']:.2f}ms\n")
                f.write(f"Valid: {signal['validation']['valid']}\n")

        except Exception as e:
            logger.error(f"Error logging signal: {e}")

    def get_signal_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent signal history."""
        return self.signal_history[-limit:]

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get signal performance metrics."""
        if not self.signal_history:
            return {}

        # Calculate metrics
        total_signals = len(self.signal_history)
        buy_signals = sum(1 for s in self.signal_history if 'BUY' in s['action'])
        sell_signals = sum(1 for s in self.signal_history if 'SELL' in s['action'])
        hold_signals = sum(1 for s in self.signal_history if s['action'] == 'HOLD')

        avg_confidence = np.mean([s['confidence'] for s in self.signal_history])

        # Get last 24h signals
        now = datetime.now()
        recent_signals = [
            s for s in self.signal_history
            if (now - datetime.fromisoformat(s['timestamp'])).total_seconds() < 86400
        ]

        return {
            'total_signals': total_signals,
            'buy_signals': buy_signals,
            'sell_signals': sell_signals,
            'hold_signals': hold_signals,
            'avg_confidence': avg_confidence,
            'recent_signals_24h': len(recent_signals),
            'last_signal': self.signal_history[-1] if self.signal_history else None
        }


class SignalAlertSystem:
    """Real-time signal alerts and notifications."""

    def __init__(self):
        self.active_alerts = []
        self.alert_history = []

    def check_for_alerts(self, signal: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Check if signal triggers any alerts."""
        alerts = []

        # High confidence signal
        if signal.get('confidence', 0) > 0.85:
            alerts.append({
                'type': 'HIGH_CONFIDENCE',
                'message': f"HIGH CONVICTION {signal['action']} signal for {signal['symbol']} "
                          f"(confidence: {signal['confidence']:.2f})",
                'priority': 'HIGH',
                'timestamp': datetime.now().isoformat()
            })

        # Signal change
        if len(self.signal_history) > 1:
            prev_signal = self.signal_history[-2]
            if prev_signal['action'] != signal['action']:
                alerts.append({
                    'type': 'SIGNAL_CHANGE',
                    'message': f"Signal changed from {prev_signal['action']} to {signal['action']} "
                              f"for {signal['symbol']}",
                    'priority': 'MEDIUM',
                    'timestamp': datetime.now().isoformat()
                })

        # ML vs AI disagreement
        ml_pred = signal['ml_signal']['direction'].get('prediction', 0)
        ai_action = signal['ai_signal'].get('action', 'HOLD')

        ml_bullish = ml_pred > 0.3
        ai_bullish = 'LONG' in ai_action.upper()
        ml_bearish = ml_pred < -0.3
        ai_bearish = 'SHORT' in ai_action.upper()

        if (ml_bullish and ai_bearish) or (ml_bearish and ai_bullish):
            alerts.append({
                'type': 'ML_AI_DISAGREEMENT',
                'message': f"ML and AI disagree: ML={ml_pred:.2f}, AI={ai_action}",
                'priority': 'LOW',
                'timestamp': datetime.now().isoformat()
            })

        self.active_alerts.extend(alerts)
        self.alert_history.extend(alerts)

        return alerts

    def get_active_alerts(self) -> List[Dict[str, Any]]:
        """Get currently active alerts."""
        return self.active_alerts

    def clear_alerts(self):
        """Clear all active alerts."""
        self.active_alerts = []
