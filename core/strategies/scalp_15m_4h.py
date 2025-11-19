"""
15m Scalp with 4h Awareness Strategy (P2.1)

A professional scalping strategy that operates on 15-minute charts with context
from the 4-hour timeframe. Implements pullback and breakout entry logic with
precise risk management.

Strategy Logic:
- Pullback entries: Buy dips to support/resistance with 4h trend alignment
- Breakout entries: Enter momentum breaks above/below key levels
- Risk: 1.5% per trade
- Stop Loss: ATR-based with liquidity zone confluence
- Take Profit: 2-tier system (R:R 1:1.5 and 1:3)

Author: Trading Bot System
Date: 2025-11-17
"""
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
from loguru import logger


@dataclass
class ScalpSignal:
    """Signal from the 15m Scalp with 4h Awareness strategy."""
    action: str  # LONG, SHORT, WAIT
    confidence: float  # 0.0 to 1.0
    entry_price: float
    stop_loss: float
    take_profit_1: float  # Tier 1 (R:R 1:1.5)
    take_profit_2: float  # Tier 2 (R:R 1:3)
    position_size: float  # 0.0 to 1.0 (of account)
    reasoning: str
    entry_type: str  # PULLBACK or BREAKOUT
    entry_zone: Optional[Tuple[float, float]] = None  # (min, max) for pullback
    entry_trigger: Optional[float] = None  # Breakout trigger level


class Scalp15m4hStrategy:
    """
    15-minute scalp strategy with 4-hour awareness.

    Operates on a 15-minute chart while considering 4-hour trend and
    key levels for entry and risk management.
    """

    def __init__(self, risk_per_trade: float = 0.015):
        """
        Initialize the scalp strategy.

        Args:
            risk_per_trade: Risk percentage per trade (default: 1.5%)
        """
        self.risk_per_trade = risk_per_trade
        self.name = "Scalp 15m/4h"
        
    def evaluate(self, data_15m: pd.DataFrame, data_4h: pd.DataFrame) -> ScalpSignal:
        """
        Evaluate market conditions and generate a trading signal.

        Args:
            data_15m: 15-minute OHLCV DataFrame
            data_4h: 4-hour OHLCV DataFrame

        Returns:
            ScalpSignal with entry details or WAIT signal
        """
        try:
            if data_15m is None or len(data_15m) < 50:
                return ScalpSignal(
                    action='WAIT',
                    confidence=0.0,
                    entry_price=0.0,
                    stop_loss=0.0,
                    take_profit_1=0.0,
                    take_profit_2=0.0,
                    position_size=0.0,
                    reasoning='Insufficient 15m data (need 50+ candles)',
                    entry_type='N/A'
                )
            
            if data_4h is None or len(data_4h) < 20:
                return ScalpSignal(
                    action='WAIT',
                    confidence=0.0,
                    entry_price=0.0,
                    stop_loss=0.0,
                    take_profit_1=0.0,
                    take_profit_2=0.0,
                    position_size=0.0,
                    reasoning='Insufficient 4h data (need 20+ candles)',
                    entry_type='N/A'
                )

            # Calculate indicators
            current_price = data_15m['close'].iloc[-1]
            atr_15m = self._calculate_atr(data_15m, period=14)
            atr_4h = self._calculate_atr(data_4h, period=14)
            
            # Get 4h trend
            trend_4h = self._get_trend_direction(data_4h)
            
            # Identify key levels
            resistance_levels = self._find_resistance_levels(data_15m)
            support_levels = self._find_support_levels(data_15m)
            
            # Check for pullback setup
            pullback_signal = self._check_pullback_setup(
                data_15m, data_4h, current_price, support_levels, resistance_levels, atr_15m
            )
            
            # Check for breakout setup
            breakout_signal = self._check_breakout_setup(
                data_15m, current_price, support_levels, resistance_levels, atr_15m
            )
            
            # Compare signals and choose best
            if pullback_signal and breakout_signal:
                # Choose higher confidence
                if pullback_signal['confidence'] > breakout_signal['confidence']:
                    return self._build_signal('LONG' if trend_4h > 0 else 'SHORT', pullback_signal, current_price, atr_15m)
                else:
                    return self._build_signal('LONG' if trend_4h > 0 else 'SHORT', breakout_signal, current_price, atr_15m)
            elif pullback_signal:
                return self._build_signal('LONG' if trend_4h > 0 else 'SHORT', pullback_signal, current_price, atr_15m)
            elif breakout_signal:
                return self._build_signal('LONG' if trend_4h > 0 else 'SHORT', breakout_signal, current_price, atr_15m)
            else:
                return ScalpSignal(
                    action='WAIT',
                    confidence=0.0,
                    entry_price=0.0,
                    stop_loss=0.0,
                    take_profit_1=0.0,
                    take_profit_2=0.0,
                    position_size=0.0,
                    reasoning='No valid setup detected',
                    entry_type='N/A'
                )
                
        except Exception as e:
            logger.error(f"Error in scalp strategy evaluation: {e}")
            return ScalpSignal(
                action='WAIT',
                confidence=0.0,
                entry_price=0.0,
                stop_loss=0.0,
                take_profit_1=0.0,
                take_profit_2=0.0,
                position_size=0.0,
                reasoning=f'Error: {str(e)}',
                entry_type='N/A'
            )

    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> float:
        """Calculate Average True Range."""
        high = df['high']
        low = df['low']
        close = df['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        
        return atr.iloc[-1] if not pd.isna(atr.iloc[-1]) else 0.0

    def _get_trend_direction(self, df: pd.DataFrame) -> float:
        """
        Determine trend direction from 4h data.
        Returns: positive for bullish, negative for bearish
        """
        if len(df) < 20:
            return 0.0
        
        # Use EMA crossover and price position
        ema_fast = df['close'].ewm(span=21).mean()
        ema_slow = df['close'].ewm(span=50).mean()
        
        current_price = df['close'].iloc[-1]
        
        score = 0.0
        if current_price > ema_fast.iloc[-1]:
            score += 0.5
        if ema_fast.iloc[-1] > ema_slow.iloc[-1]:
            score += 0.5
        if ema_fast.iloc[-1] > ema_fast.iloc[-5]:
            score += 0.5
        if ema_fast.iloc[-1] > ema_slow.iloc[-5]:
            score += 0.5
        
        return score - 1.0  # Range: -1.0 to +1.0

    def _find_resistance_levels(self, df: pd.DataFrame, lookback: int = 100) -> list:
        """Find significant resistance levels."""
        df_subset = df.tail(lookback)
        highs = df_subset['high']
        
        # Find local maxima
        resistance_levels = []
        for i in range(5, len(highs) - 5):
            if highs.iloc[i] > highs.iloc[i-5:i].max() and highs.iloc[i] > highs.iloc[i+1:i+6].max():
                resistance_levels.append(highs.iloc[i])
        
        # Remove nearby levels (within 0.5%)
        resistance_levels = sorted(resistance_levels, reverse=True)
        filtered_levels = []
        for level in resistance_levels:
            if not any(abs(level - fl) / fl < 0.005 for fl in filtered_levels):
                filtered_levels.append(level)
        
        return filtered_levels[:5]  # Top 5 levels

    def _find_support_levels(self, df: pd.DataFrame, lookback: int = 100) -> list:
        """Find significant support levels."""
        df_subset = df.tail(lookback)
        lows = df_subset['low']
        
        # Find local minima
        support_levels = []
        for i in range(5, len(lows) - 5):
            if lows.iloc[i] < lows.iloc[i-5:i].min() and lows.iloc[i] < lows.iloc[i+1:i+6].min():
                support_levels.append(lows.iloc[i])
        
        # Remove nearby levels (within 0.5%)
        support_levels = sorted(support_levels)
        filtered_levels = []
        for level in support_levels:
            if not any(abs(level - fl) / fl < 0.005 for fl in filtered_levels):
                filtered_levels.append(level)
        
        return filtered_levels[:5]  # Top 5 levels

    def _check_pullback_setup(self, data_15m: pd.DataFrame, data_4h: pd.DataFrame,
                              current_price: float, support_levels: list,
                              resistance_levels: list, atr: float) -> Optional[Dict]:
        """
        Check for pullback entry setup.
        
        Returns dict with entry details or None.
        """
        if atr <= 0:
            return None
        
        # Check 4h trend alignment
        trend_4h = self._get_trend_direction(data_4h)
        
        # Look for price near a key level (within 0.3% of ATR)
        pullback_threshold = atr * 0.3
        
        # For LONG: look for pullback to support in uptrend
        if trend_4h > 0:
            for support in support_levels:
                distance = abs(current_price - support)
                if distance < pullback_threshold:
                    # Price is near support, check if it's a good pullback
                    if self._is_valid_pullback_long(data_15m, support):
                        entry_price = current_price
                        stop_loss = support - (atr * 0.5)  # Below support
                        risk = entry_price - stop_loss
                        
                        if risk > 0:
                            # Calculate position size (1.5% risk)
                            position_size = self.risk_per_trade / (risk / entry_price)
                            
                            return {
                                'type': 'PULLBACK',
                                'confidence': 0.8,
                                'entry_zone': (support - atr * 0.3, support + atr * 0.3),
                                'reasoning': f'Pullback to 4h-aligned support at {support:.2f}'
                            }
        
        # For SHORT: look for pullback to resistance in downtrend
        elif trend_4h < 0:
            for resistance in resistance_levels:
                distance = abs(current_price - resistance)
                if distance < pullback_threshold:
                    # Price is near resistance, check if it's a good pullback
                    if self._is_valid_pullback_short(data_15m, resistance):
                        entry_price = current_price
                        stop_loss = resistance + (atr * 0.5)  # Above resistance
                        risk = stop_loss - entry_price
                        
                        if risk > 0:
                            # Calculate position size (1.5% risk)
                            position_size = self.risk_per_trade / (risk / entry_price)
                            
                            return {
                                'type': 'PULLBACK',
                                'confidence': 0.8,
                                'entry_zone': (resistance - atr * 0.3, resistance + atr * 0.3),
                                'reasoning': f'Pullback to 4h-aligned resistance at {resistance:.2f}'
                            }
        
        return None

    def _is_valid_pullback_long(self, data_15m: pd.DataFrame, support_level: float) -> bool:
        """Validate pullback for LONG entry."""
        recent_lows = data_15m['low'].tail(10)
        # Check that support held during pullback
        return recent_lows.min() >= support_level * 0.995  # Allow 0.5% tolerance

    def _is_valid_pullback_short(self, data_15m: pd.DataFrame, resistance_level: float) -> bool:
        """Validate pullback for SHORT entry."""
        recent_highs = data_15m['high'].tail(10)
        # Check that resistance held during pullback
        return recent_highs.max() <= resistance_level * 1.005  # Allow 0.5% tolerance

    def _check_breakout_setup(self, data_15m: pd.DataFrame, current_price: float,
                               support_levels: list, resistance_levels: list, atr: float) -> Optional[Dict]:
        """
        Check for breakout entry setup.
        
        Returns dict with entry details or None.
        """
        if atr <= 0:
            return None
        
        # Check for breakout above resistance
        breakout_threshold = atr * 0.2
        
        # Check for LONG breakout
        if resistance_levels:
            closest_resistance = min(resistance_levels, key=lambda x: abs(x - current_price))
            distance_to_resistance = closest_resistance - current_price
            
            if 0 < distance_to_resistance < breakout_threshold:
                # Price is close to breaking resistance
                if self._is_breakout_imminent_long(data_15m, closest_resistance):
                    return {
                        'type': 'BREAKOUT',
                        'confidence': 0.7,
                        'entry_trigger': closest_resistance,
                        'reasoning': f'Breakout above {closest_resistance:.2f} resistance'
                    }
        
        # Check for SHORT breakout
        if support_levels:
            closest_support = min(support_levels, key=lambda x: abs(x - current_price))
            distance_to_support = current_price - closest_support
            
            if 0 < distance_to_support < breakout_threshold:
                # Price is close to breaking support
                if self._is_breakout_imminent_short(data_15m, closest_support):
                    return {
                        'type': 'BREAKOUT',
                        'confidence': 0.7,
                        'entry_trigger': closest_support,
                        'reasoning': f'Breakout below {closest_support:.2f} support'
                    }
        
        return None

    def _is_breakout_imminent_long(self, data_15m: pd.DataFrame, resistance_level: float) -> bool:
        """Check if LONG breakout is imminent."""
        recent_highs = data_15m['high'].tail(5)
        return recent_highs.max() > resistance_level * 0.998  # Within 0.2% of breaking

    def _is_breakout_imminent_short(self, data_15m: pd.DataFrame, support_level: float) -> bool:
        """Check if SHORT breakout is imminent."""
        recent_lows = data_15m['low'].tail(5)
        return recent_lows.min() < support_level * 1.002  # Within 0.2% of breaking

    def _build_signal(self, action: str, setup: Dict, current_price: float, atr: float) -> ScalpSignal:
        """Build a ScalpSignal from setup details."""
        if action == 'LONG':
            entry_price = current_price
            stop_loss = entry_price - (atr * 1.5)
            risk = entry_price - stop_loss
            
            # Take profits at R:R 1:1.5 and 1:3
            take_profit_1 = entry_price + (risk * 1.5)
            take_profit_2 = entry_price + (risk * 3.0)
            
        else:  # SHORT
            entry_price = current_price
            stop_loss = entry_price + (atr * 1.5)
            risk = stop_loss - entry_price
            
            # Take profits at R:R 1:1.5 and 1:3
            take_profit_1 = entry_price - (risk * 1.5)
            take_profit_2 = entry_price - (risk * 3.0)
        
        # Position size based on 1.5% risk
        position_size = self.risk_per_trade / (risk / entry_price) if risk > 0 else 0.0
        
        return ScalpSignal(
            action=action,
            confidence=setup['confidence'],
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit_1=take_profit_1,
            take_profit_2=take_profit_2,
            position_size=min(position_size, 0.1),  # Cap at 10%
            reasoning=setup['reasoning'],
            entry_type=setup['type'],
            entry_zone=setup.get('entry_zone'),
            entry_trigger=setup.get('entry_trigger')
        )
