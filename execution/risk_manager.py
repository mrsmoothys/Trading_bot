"""
Risk Manager
Manages portfolio risk, enforces limits, and monitors exposure.
AI-enhanced with DeepSeek integration for intelligent risk assessment.
"""

import asyncio
from datetime import datetime
from typing import Dict, Any, List
import numpy as np
from loguru import logger

from core.system_context import SystemContext


class RiskManager:
    """
    Manages trading risk and enforces risk controls.
    Monitors exposure, drawdown, and position sizing.
    """

    def __init__(self, system_context: SystemContext, deepseek_brain=None):
        """
        Initialize risk manager.

        Args:
            system_context: System state context
            deepseek_brain: Optional DeepSeek AI client for intelligent risk assessment
        """
        self.system_context = system_context
        self.deepseek = deepseek_brain

        # Risk limits from configuration
        self.max_position_size = 0.05  # 5% per position
        self.max_total_exposure = 0.20  # 20% total exposure
        self.max_drawdown_limit = 0.10  # 10% drawdown
        self.max_positions = 10

        # Correlation limits
        self.max_correlation_exposure = 0.15  # Max 15% in correlated assets

        # Convergence Strategy Integration
        self.enable_convergence_stops = True  # Enable ATR + liquidity stops
        self.convergence_size_multiplier = 1.0  # Size adjustment for convergence signals

        # AI-enhanced risk assessment
        self.use_ai_risk_assessment = self.deepseek is not None

        logger.info(f"RiskManager initialized - AI Risk Assessment: {self.use_ai_risk_assessment}")
        logger.info(f"Convergence Strategy Integration: {self.enable_convergence_stops}")

    async def validate_trade(
        self,
        signal: Dict[str, Any],
        proposed_size: float,
        convergence_data: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Validate a proposed trade against risk limits.

        Args:
            signal: Trading signal
            proposed_size: Proposed position size
            convergence_data: Optional convergence strategy data (ATR, liquidity zones, etc.)

        Returns:
            Validation result
        """
        try:
            symbol = signal['symbol']
            action = signal['action']
            confidence = signal.get('confidence', 0.5)

            # Check 1: Position size limit
            if proposed_size > self.max_position_size:
                return {
                    "approved": False,
                    "reason": f"Position size {proposed_size:.2%} exceeds limit {self.max_position_size:.2%}",
                    "adjusted_size": self.max_position_size
                }

            # Check 2: Maximum positions limit
            if len(self.system_context.active_positions) >= self.max_positions:
                return {
                    "approved": False,
                    "reason": f"Maximum positions ({self.max_positions}) reached",
                    "adjusted_size": 0
                }

            # Check 3: Drawdown limit
            current_drawdown = self.system_context.risk_metrics.get('current_drawdown', 0)
            if current_drawdown >= self.max_drawdown_limit:
                return {
                    "approved": False,
                    "reason": f"Drawdown {current_drawdown:.2%} exceeds limit {self.max_drawdown_limit:.2%}",
                    "adjusted_size": 0
                }

            # Check 4: Total exposure
            current_exposure = self.system_context.risk_metrics.get('total_exposure', 0)
            new_exposure = current_exposure + proposed_size

            if new_exposure > self.max_total_exposure:
                adjusted_size = self.max_total_exposure - current_exposure
                return {
                    "approved": adjusted_size > 0,
                    "reason": f"Total exposure would exceed limit",
                    "adjusted_size": max(adjusted_size, 0)
                }

            # Check 5: Confidence threshold
            min_confidence = 0.60
            if confidence < min_confidence:
                return {
                    "approved": False,
                    "reason": f"Confidence {confidence:.2%} below threshold {min_confidence:.2%}",
                    "adjusted_size": 0
                }

            # Check 6: Correlation risk
            correlation_risk = self._check_correlation_risk(symbol, proposed_size)
            if correlation_risk['exceeds_limit']:
                return {
                    "approved": False,
                    "reason": f"Correlation risk: {correlation_risk['reason']}",
                    "adjusted_size": correlation_risk['adjusted_size']
                }

            # Check 7: Market regime risk
            regime_risk = self._check_regime_risk(signal)
            if regime_risk['high_risk']:
                return {
                    "approved": confidence > 0.75,  # Require higher confidence
                    "reason": f"High risk market regime: {regime_risk['reason']}",
                    "adjusted_size": proposed_size * 0.8  # Reduce size
                }

            # Check 8: Convergence Strategy Stop Loss Validation
            if self.enable_convergence_stops and signal.get('strategy') == 'convergence':
                convergence_validation = self._validate_convergence_stops(
                    signal, proposed_size, convergence_data
                )
                if not convergence_validation['valid']:
                    return {
                        "approved": False,
                        "reason": f"Convergence stop validation failed: {convergence_validation['reason']}",
                        "adjusted_size": 0
                    }
                logger.info(
                    f"Convergence stops validated: ATR={convergence_validation.get('atr_stop', 'N/A')}, "
                    f"Liquidity={convergence_validation.get('liquidity_stop', 'N/A')}"
                )

            # Check 9: Convergence Strategy Position Size Validation
            if self.enable_convergence_stops and signal.get('strategy') == 'convergence':
                convergence_size = signal.get('position_size')
                if convergence_size and convergence_size > self.max_position_size * self.convergence_size_multiplier:
                    return {
                        "approved": False,
                        "reason": f"Convergence position size {convergence_size:.2%} exceeds adjusted limit",
                        "adjusted_size": self.max_position_size * self.convergence_size_multiplier
                    }

            # All checks passed - proceed to AI assessment if available
            ai_assessment = None
            if self.use_ai_risk_assessment:
                try:
                    ai_assessment = await self._assess_trade_with_ai(signal, proposed_size)
                    if ai_assessment and not ai_assessment.get('approved', True):
                        return ai_assessment
                except Exception as e:
                    logger.warning(f"AI risk assessment failed: {e}")

            # All checks passed
            result = {
                "approved": True,
                "reason": "All risk checks passed",
                "adjusted_size": proposed_size
            }

            if ai_assessment:
                result["ai_assessment"] = ai_assessment
                result["ai_enhanced"] = True

            return result

        except Exception as e:
            logger.error(f"Error validating trade: {e}")
            return {
                "approved": False,
                "reason": f"Error: {str(e)}",
                "adjusted_size": 0
            }

    def calculate_position_size(
        self,
        base_size: float,
        confidence: float,
        market_data: Dict[str, Any],
        signal: Dict[str, Any]
    ) -> float:
        """
        Calculate optimal position size considering risk.

        Args:
            base_size: Base position size
            confidence: Signal confidence
            market_data: Market data
            signal: Trading signal

        Returns:
            Adjusted position size
        """
        try:
            # Start with base size
            size = base_size

            # Scale with confidence
            confidence_multiplier = min(confidence * 2, 1.0)
            size *= confidence_multiplier

            # Adjust for volatility
            volatility = self._calculate_volatility(market_data)
            if volatility > 0.03:  # High volatility
                size *= 0.8
            elif volatility < 0.01:  # Low volatility
                size *= 1.1

            # Adjust for drawdown
            current_drawdown = self.system_context.risk_metrics.get('current_drawdown', 0)
            drawdown_multiplier = 1 - current_drawdown  # Reduce as drawdown increases
            size *= drawdown_multiplier

            # Adjust for market regime
            regime = signal.get('market_regime', 'UNKNOWN')
            if 'TRENDING_HIGH_VOL' in regime:
                size *= 0.7
            elif 'RANGING_COMPRESSION' in regime:
                size *= 0.9

            # Enforce limits
            size = min(size, self.max_position_size)
            size = max(size, 0)  # No negative sizes

            return size

        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return base_size * 0.5  # Conservative fallback

    def check_portfolio_risk(self) -> Dict[str, Any]:
        """
        Check overall portfolio risk.

        Returns:
            Portfolio risk assessment
        """
        try:
            risk_metrics = self.system_context.risk_metrics
            positions = self.system_context.active_positions

            # Calculate various risk metrics
            total_exposure = risk_metrics.get('total_exposure', 0)
            current_drawdown = risk_metrics.get('current_drawdown', 0)
            max_drawdown = risk_metrics.get('max_drawdown', 0)

            # Position concentration
            if positions:
                position_sizes = [
                    abs(pos.get('value', 0)) for pos in positions.values()
                ]
                largest_position = max(position_sizes) if position_sizes else 0
                concentration_risk = largest_position / sum(position_sizes) if sum(position_sizes) > 0 else 0
            else:
                concentration_risk = 0

            # Risk score (0-1, higher is riskier)
            risk_score = 0
            risk_score += min(total_exposure / self.max_total_exposure, 1.0) * 0.3
            risk_score += min(current_drawdown / self.max_drawdown_limit, 1.0) * 0.3
            risk_score += concentration_risk * 0.2
            risk_score += len(positions) / self.max_positions * 0.2

            # Overall risk level
            if risk_score < 0.3:
                risk_level = "LOW"
            elif risk_score < 0.6:
                risk_level = "MEDIUM"
            else:
                risk_level = "HIGH"

            # Warnings
            warnings = []
            if total_exposure > self.max_total_exposure * 0.8:
                warnings.append("Total exposure approaching limit")
            if current_drawdown > self.max_drawdown_limit * 0.8:
                warnings.append("Drawdown approaching limit")
            if concentration_risk > 0.4:
                warnings.append("High position concentration")
            if len(positions) > self.max_positions * 0.8:
                warnings.append("Approaching maximum positions")

            return {
                "risk_score": risk_score,
                "risk_level": risk_level,
                "total_exposure": total_exposure,
                "current_drawdown": current_drawdown,
                "max_drawdown": max_drawdown,
                "concentration_risk": concentration_risk,
                "position_count": len(positions),
                "warnings": warnings,
                "recommendations": self._get_risk_recommendations(risk_score, risk_level)
            }

        except Exception as e:
            logger.error(f"Error checking portfolio risk: {e}")
            return {
                "risk_score": 1.0,
                "risk_level": "HIGH",
                "error": str(e)
            }

    def emergency_stop(self) -> bool:
        """
        Check if emergency stop should be triggered.

        Returns:
            True if emergency stop should be triggered
        """
        # Check drawdown
        current_drawdown = self.system_context.risk_metrics.get('current_drawdown', 0)
        if current_drawdown >= self.max_drawdown_limit:
            logger.critical(f"EMERGENCY STOP: Drawdown {current_drawdown:.2%} exceeded limit")
            return True

        # Check system errors
        recent_errors = self.system_context.system_health.get('errors', [])[-5:]
        if len(recent_errors) > 10:
            logger.critical("EMERGENCY STOP: Too many system errors")
            return True

        return False

    def _check_correlation_risk(self, symbol: str, proposed_size: float) -> Dict[str, Any]:
        """Check correlation risk for proposed trade."""
        # Simplified correlation check
        # In production, you'd use actual correlation matrix

        # Group symbols by base asset
        crypto_groups = {
            'BTC': ['BTCUSDT', 'WBTC', 'BTCB'],
            'ETH': ['ETHUSDT', 'WETH', 'ETHB'],
            'SOL': ['SOLUSDT', 'SOLB'],
            'ADA': ['ADAUSDT'],
        }

        # Find group for symbol
        symbol_group = None
        for group, symbols in crypto_groups.items():
            if any(s in symbol for s in symbols):
                symbol_group = group
                break

        if not symbol_group:
            return {"exceeds_limit": False}

        # Calculate existing exposure in group
        group_exposure = 0
        for pos_symbol, position in self.system_context.active_positions.items():
            if any(s in pos_symbol for s in crypto_groups.get(symbol_group, [])):
                # This is a simplification
                group_exposure += abs(position.get('value', 0))

        # Check if adding proposed position exceeds limit
        if group_exposure + proposed_size > self.max_correlation_exposure:
            adjusted_size = self.max_correlation_exposure - group_exposure
            return {
                "exceeds_limit": True,
                "reason": f"Correlation exposure would exceed {self.max_correlation_exposure:.2%}",
                "adjusted_size": max(adjusted_size, 0)
            }

        return {"exceeds_limit": False}

    def _check_regime_risk(self, signal: Dict[str, Any]) -> Dict[str, Any]:
        """Check risk based on market regime."""
        regime = signal.get('market_regime', 'UNKNOWN')

        high_risk_regimes = [
            'TRANSITION',
            'RANGING_EXPANSION'
        ]

        if regime in high_risk_regimes:
            return {
                "high_risk": True,
                "reason": f"High risk regime: {regime}"
            }

        return {"high_risk": False}

    def _calculate_volatility(self, market_data: Dict[str, Any]) -> float:
        """Calculate market volatility."""
        try:
            if 'close' not in market_data:
                return 0.02  # Default

            close_prices = market_data['close']
            returns = close_prices.pct_change().dropna()

            if len(returns) == 0:
                return 0.02

            return returns.std()
        except:
            return 0.02

    def _get_risk_recommendations(self, risk_score: float, risk_level: str) -> List[str]:
        """Get risk management recommendations."""
        recommendations = []

        if risk_level == "HIGH":
            recommendations.append("Reduce position sizes")
            recommendations.append("Close some positions")
            recommendations.append("Wait for better opportunities")

        elif risk_level == "MEDIUM":
            recommendations.append("Monitor positions closely")
            recommendations.append("Consider taking partial profits")

        if risk_score > 0.7:
            recommendations.append("Review correlation exposure")
            recommendations.append("Check stop losses")

        return recommendations

    def get_risk_limits(self) -> Dict[str, float]:
        """Get current risk limits."""
        return {
            "max_position_size": self.max_position_size,
            "max_total_exposure": self.max_total_exposure,
            "max_drawdown_limit": self.max_drawdown_limit,
            "max_positions": self.max_positions,
            "max_correlation_exposure": self.max_correlation_exposure,
            "enable_convergence_stops": self.enable_convergence_stops,
            "convergence_size_multiplier": self.convergence_size_multiplier
        }

    def calculate_convergence_stops(
        self,
        entry_price: float,
        direction: str,
        atr: float,
        liquidity_zones: Dict[str, Any],
        supertrend_value: float
    ) -> Dict[str, float]:
        """
        Calculate stops using convergence strategy method.

        Args:
            entry_price: Entry price
            direction: 'LONG' or 'SHORT'
            atr: Average True Range
            liquidity_zones: Liquidity zone information
            supertrend_value: SuperTrend value

        Returns:
            Dict with atr_stop, liquidity_stop, supertrend_stop, final_stop
        """
        stops = {}

        # ATR-based stop (2 ATR)
        atr_stop = entry_price - (2 * atr) if direction == 'LONG' else entry_price + (2 * atr)
        stops['atr_stop'] = atr_stop

        # Liquidity-based stop (beyond nearest level)
        if direction == 'LONG':
            nearest_support = liquidity_zones.get('current_nearest_support')
            if nearest_support:
                liquidity_stop = nearest_support * 0.995  # 0.5% below support
                stops['liquidity_stop'] = liquidity_stop
        else:
            nearest_resistance = liquidity_zones.get('current_nearest_resistance')
            if nearest_resistance:
                liquidity_stop = nearest_resistance * 1.005  # 0.5% above resistance
                stops['liquidity_stop'] = liquidity_stop

        # SuperTrend-based stop (more conservative)
        if direction == 'LONG':
            if supertrend_value > entry_price:
                stops['supertrend_stop'] = supertrend_value
        else:
            if supertrend_value < entry_price:
                stops['supertrend_stop'] = supertrend_value

        # Final stop (most conservative)
        if direction == 'LONG':
            stop_values = [v for v in stops.values() if isinstance(v, (int, float))]
            stops['final_stop'] = max(stop_values) if stop_values else entry_price * 0.98
        else:
            stop_values = [v for v in stops.values() if isinstance(v, (int, float))]
            stops['final_stop'] = min(stop_values) if stop_values else entry_price * 1.02

        return stops

    def _validate_convergence_stops(
        self,
        signal: Dict[str, Any],
        proposed_size: float,
        convergence_data: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Validate convergence strategy stop loss levels.

        Args:
            signal: Trading signal
            proposed_size: Position size
            convergence_data: Convergence strategy data

        Returns:
            Validation result
        """
        try:
            if not convergence_data:
                return {"valid": True, "reason": "No convergence data provided"}

            entry_price = signal.get('entry_price')
            stop_loss = signal.get('stop_loss')
            action = signal.get('action', 'LONG')

            if not entry_price or not stop_loss:
                return {"valid": False, "reason": "Missing entry price or stop loss"}

            # Check stop loss distance
            if action == 'LONG':
                stop_distance = (entry_price - stop_loss) / entry_price
                atr_stop = convergence_data.get('atr_stop')
                if atr_stop and stop_loss < atr_stop:
                    return {
                        "valid": False,
                        "reason": f"Stop loss {stop_loss:.2f} too tight (below ATR stop {atr_stop:.2f})"
                    }
            else:  # SHORT
                stop_distance = (stop_loss - entry_price) / entry_price
                atr_stop = convergence_data.get('atr_stop')
                if atr_stop and stop_loss > atr_stop:
                    return {
                        "valid": False,
                        "reason": f"Stop loss {stop_loss:.2f} too tight (above ATR stop {atr_stop:.2f})"
                    }

            # Validate liquidity zone alignment
            liquidity_stop = convergence_data.get('liquidity_stop')
            if liquidity_stop and action == 'LONG' and stop_loss < liquidity_stop:
                return {
                    "valid": False,
                    "reason": f"Stop below liquidity zone ({liquidity_stop:.2f})"
                }
            elif liquidity_stop and action == 'SHORT' and stop_loss > liquidity_stop:
                return {
                    "valid": False,
                    "reason": f"Stop above liquidity zone ({liquidity_stop:.2f})"
                    }

            return {"valid": True, "reason": "Stops validated"}

        except Exception as e:
            logger.error(f"Error validating convergence stops: {e}")
            return {"valid": False, "reason": str(e)}

    def update_risk_limits(self, **kwargs):
        """Update risk limits."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
                logger.info(f"Updated {key} to {value}")

    async def _assess_trade_with_ai(
        self,
        signal: Dict[str, Any],
        proposed_size: float
    ) -> Dict[str, Any]:
        """
        Use DeepSeek AI to assess trade risk.

        Args:
            signal: Trading signal
            proposed_size: Proposed position size

        Returns:
            AI risk assessment
        """
        try:
            # Prepare context for AI analysis
            risk_context = {
                "signal": signal,
                "proposed_size": proposed_size,
                "current_exposure": self.system_context.risk_metrics.get('total_exposure', 0),
                "current_drawdown": self.system_context.risk_metrics.get('current_drawdown', 0),
                "active_positions": len(self.system_context.active_positions),
                "max_positions": self.max_positions,
                "max_position_size": self.max_position_size,
                "market_regime": signal.get('market_regime', 'UNKNOWN'),
                "confidence": signal.get('confidence', 0.5)
            }

            # Query DeepSeek for risk assessment
            assessment = await self.deepseek.assess_risk(risk_context)

            # Log AI assessment
            if assessment.get('approved'):
                logger.info(
                    f"AI Risk Assessment APPROVED: {assessment.get('reasoning', 'N/A')} "
                    f"(score: {assessment.get('risk_score', 'N/A')})"
                )
            else:
                logger.warning(
                    f"AI Risk Assessment REJECTED: {assessment.get('reason', 'N/A')} "
                    f"(score: {assessment.get('risk_score', 'N/A')})"
                )

            return assessment

        except Exception as e:
            logger.error(f"Error in AI risk assessment: {e}")
            # Fail-safe: reject the trade if AI assessment fails
            return {
                "approved": False,
                "reason": f"AI assessment error: {str(e)}",
                "risk_score": 1.0,
                "adjusted_size": 0
            }

    async def _assess_portfolio_risk_with_ai(self) -> Dict[str, Any]:
        """
        Use DeepSeek AI to assess overall portfolio risk.

        Returns:
            AI portfolio risk assessment
        """
        try:
            # Prepare portfolio context
            portfolio_context = {
                "positions": list(self.system_context.active_positions.values()),
                "risk_metrics": self.system_context.risk_metrics,
                "max_positions": self.max_positions,
                "max_total_exposure": self.max_total_exposure,
                "max_drawdown_limit": self.max_drawdown_limit,
                "correlation_limits": {
                    "max_correlation_exposure": self.max_correlation_exposure
                }
            }

            # Query DeepSeek for portfolio assessment
            assessment = await self.deepseek.assess_portfolio_risk(portfolio_context)

            logger.info(
                f"AI Portfolio Risk: {assessment.get('risk_level', 'UNKNOWN')} "
                f"(score: {assessment.get('risk_score', 'N/A')})"
            )

            return assessment

        except Exception as e:
            logger.error(f"Error in AI portfolio assessment: {e}")
            return {
                "risk_level": "UNKNOWN",
                "risk_score": 0.5,
                "error": str(e),
                "recommendations": ["Unable to assess portfolio risk"]
            }

    async def get_ai_risk_recommendations(self) -> List[str]:
        """
        Get AI-generated risk management recommendations.

        Returns:
            List of recommendations
        """
        try:
            # Get current system state
            system_state = self.system_context.get_context_for_deepseek()

            # Query DeepSeek for recommendations
            recommendations = await self.deepseek.get_risk_recommendations(system_state)

            return recommendations.get('recommendations', [])

        except Exception as e:
            logger.error(f"Error getting AI recommendations: {e}")
            return ["Unable to generate AI recommendations"]
