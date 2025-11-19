"""
Multi-Timeframe Convergence Trading Strategy

Implements DeepSeek's multi-timeframe convergence logic with:
- Regime detection (volatility-based)
- Timeframe alignment scoring
- Liquidity zone detection
- 4-of-6 condition checking for entries
- ATR + liquidity-based stop placement
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, List, Optional
from dataclasses import dataclass
from enum import Enum
from loguru import logger


class MarketRegime(Enum):
    """Market regime classifications based on volatility and trend."""
    LOW_VOL_COMPRESSION = "LOW_VOL_COMPRESSION"
    NORMAL_REGIME = "NORMAL_REGIME"
    TREND_HIGH_VOL = "TREND_HIGH_VOL"
    RANGING_HIGH_VOL = "RANGING_HIGH_VOL"


class AlignmentState(Enum):
    """Timeframe alignment states."""
    STRONG_BULLISH_ALIGNMENT = "STRONG_BULLISH_ALIGNMENT"
    STRONG_BEARISH_ALIGNMENT = "STRONG_BEARISH_ALIGNMENT"
    MIXED_SIGNALS = "MIXED_SIGNALS"


@dataclass
class ConvergenceSignal:
    """Output signal from the convergence strategy."""
    action: str  # 'LONG', 'SHORT', 'HOLD'
    confidence: float  # 0.0 to 1.0
    entry_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    risk_reward_ratio: Optional[float] = None
    alignment_score: Optional[float] = None
    regime: Optional[str] = None
    satisfied_conditions: Optional[List[str]] = None
    liquidity_target: Optional[float] = None


class ConvergenceStrategy:
    """
    Multi-timeframe convergence trading strategy.

    Analyzes multiple timeframes to identify high-probability entries
    based on regime, alignment, and liquidity confluence.
    """

    def __init__(
        self,
        supertrend_period: int = 10,
        supertrend_multiplier: float = 3.0,
        chandelier_period: int = 14,
        atr_period: int = 14,
        liquidity_lookback: int = 50,
        entry_condition_threshold: int = 4,
        min_alignment_votes: int = 3,
        risk_reward_target: float = 3.0
    ):
        """
        Initialize the convergence strategy.

        Args:
            supertrend_period: Period for SuperTrend calculation
            supertrend_multiplier: Multiplier for SuperTrend
            chandelier_period: Period for Chandelier Exit
            atr_period: Period for ATR calculation
            liquidity_lookback: Lookback period for liquidity zone detection
            entry_condition_threshold: Minimum conditions to satisfy for entry (out of 6)
            min_alignment_votes: Minimum timeframe votes for alignment
            risk_reward_target: Target risk-reward ratio
        """
        self.supertrend_period = supertrend_period
        self.supertrend_multiplier = supertrend_multiplier
        self.chandelier_period = chandelier_period
        self.atr_period = atr_period
        self.liquidity_lookback = liquidity_lookback
        self.entry_condition_threshold = entry_condition_threshold
        self.min_alignment_votes = min_alignment_votes
        self.risk_reward_target = risk_reward_target

        logger.info(
            f"ConvergenceStrategy initialized: "
            f"ST({supertrend_period}, {supertrend_multiplier}), "
            f"CE({chandelier_period}), ATR({atr_period}), "
            f"Entry threshold: {entry_condition_threshold}/6"
        )

    def detect_market_regime(
        self,
        volatility: float,
        price_action: str,
        atr: float,
        sma_200_slope: float
    ) -> MarketRegime:
        """
        Detect current market regime based on volatility and price behavior.

        Args:
            volatility: Current volatility (e.g., 0.1 for 10%)
            price_action: 'trending' or 'ranging'
            atr: Average True Range
            sma_200_slope: Slope of 200-period SMA

        Returns:
            MarketRegime enum value
        """
        if volatility > 0.2:  # High volatility
            if price_action == "trending" and abs(sma_200_slope) > 0.01:
                return MarketRegime.TREND_HIGH_VOL
            else:
                return MarketRegime.RANGING_HIGH_VOL
        elif volatility < 0.1:  # Low volatility (compression)
            return MarketRegime.LOW_VOL_COMPRESSION
        else:
            return MarketRegime.NORMAL_REGIME

    def check_timeframe_alignment(
        self,
        mtf_data: Dict[str, Dict[str, Any]]
    ) -> Tuple[AlignmentState, int]:
        """
        Check multi-timeframe alignment for trend direction.

        Args:
            mtf_data: Dict with timeframe keys and trend data values

        Returns:
            Tuple of (AlignmentState, bullish_votes)
        """
        bullish_votes = 0
        bearish_votes = 0

        for tf, data in mtf_data.items():
            trend = data.get('trend', 'NEUTRAL')
            strength = data.get('trend_strength', 0.0)

            # Weight more recent/longer timeframes higher
            weight = self._get_timeframe_weight(tf)

            if trend == 'BULLISH' and strength > 0.5:
                bullish_votes += weight
            elif trend == 'BEARISH' and strength > 0.5:
                bearish_votes += weight

        if bullish_votes >= self.min_alignment_votes:
            return AlignmentState.STRONG_BULLISH_ALIGNMENT, bullish_votes
        elif bearish_votes >= self.min_alignment_votes:
            return AlignmentState.STRONG_BEARISH_ALIGNMENT, bearish_votes
        else:
            return AlignmentState.MIXED_SIGNALS, 0

    def _get_timeframe_weight(self, timeframe: str) -> float:
        """Get weight for timeframe (higher = more important)."""
        weights = {
            '1m': 0.5,
            '5m': 0.7,
            '15m': 1.0,
            '1h': 1.5,
            '4h': 2.0,
            '1d': 3.0
        }
        return weights.get(timeframe, 1.0)

    def identify_liquidity_zones(
        self,
        df: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Identify significant liquidity zones (support/resistance clusters).

        Args:
            df: OHLCV dataframe

        Returns:
            Dict with resistance levels, support levels, and nearest levels
        """
        # Find significant highs and lows
        df['is_high'] = (
            df['high'] == df['high']
            .rolling(window=self.liquidity_lookback, center=True)
            .max()
        )
        df['is_low'] = (
            df['low'] == df['low']
            .rolling(window=self.liquidity_lookback, center=True)
            .min()
        )

        # Get last 5 significant levels
        resistance_levels = df[df['is_high']]['high'].dropna().tail(5).tolist()
        support_levels = df[df['is_low']]['low'].dropna().tail(5).tolist()

        current_price = df['close'].iloc[-1]

        return {
            'resistance_levels': resistance_levels,
            'support_levels': support_levels,
            'current_nearest_support': self._find_nearest_level(
                current_price, support_levels, below=True
            ),
            'current_nearest_resistance': self._find_nearest_level(
                current_price, resistance_levels, below=False
            )
        }

    def _find_nearest_level(
        self,
        price: float,
        levels: List[float],
        below: bool = True
    ) -> Optional[float]:
        """Find nearest level above or below current price."""
        if not levels:
            return None

        if below:
            # Find nearest level below price
            below_levels = [l for l in levels if l < price]
            return max(below_levels) if below_levels else min(levels)
        else:
            # Find nearest level above price
            above_levels = [l for l in levels if l > price]
            return min(above_levels) if above_levels else max(levels)

    def calculate_supertrend(
        self,
        df: pd.DataFrame
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Calculate SuperTrend indicator.

        Args:
            df: OHLCV dataframe

        Returns:
            Tuple of (supertrend_values, direction)
        """
        # Calculate ATR
        df['tr'] = np.maximum(
            df['high'] - df['low'],
            np.maximum(
                abs(df['high'] - df['close'].shift(1)),
                abs(df['low'] - df['close'].shift(1))
            )
        )
        atr = df['tr'].rolling(window=self.atr_period).mean()

        # Calculate SuperTrend
        hl2 = (df['high'] + df['low']) / 2
        basic_upper_band = hl2 + (self.supertrend_multiplier * atr)
        basic_lower_band = hl2 - (self.supertrend_multiplier * atr)

        upper_band = basic_upper_band.copy()
        lower_band = basic_lower_band.copy()

        for i in range(1, len(df)):
            if basic_upper_band.iloc[i] < upper_band.iloc[i-1] or \
               df['close'].iloc[i-1] > upper_band.iloc[i-1]:
                upper_band.iloc[i] = basic_upper_band.iloc[i]
            else:
                upper_band.iloc[i] = upper_band.iloc[i-1]

            if basic_lower_band.iloc[i] > lower_band.iloc[i-1] or \
               df['close'].iloc[i-1] < lower_band.iloc[i-1]:
                lower_band.iloc[i] = basic_lower_band.iloc[i]
            else:
                lower_band.iloc[i] = lower_band.iloc[i-1]

        # Final SuperTrend
        supertrend = pd.Series(index=df.index, dtype=float)
        direction = pd.Series(index=df.index, dtype=float)

        for i in range(len(df)):
            if i == 0:
                supertrend.iloc[i] = lower_band.iloc[i]
                direction.iloc[i] = 1
            else:
                if direction.iloc[i-1] == 1 and lower_band.iloc[i] > supertrend.iloc[i-1]:
                    supertrend.iloc[i] = lower_band.iloc[i]
                    direction.iloc[i] = 1
                elif direction.iloc[i-1] == 1 and lower_band.iloc[i] <= supertrend.iloc[i-1]:
                    supertrend.iloc[i] = supertrend.iloc[i-1]
                    direction.iloc[i] = -1
                elif direction.iloc[i-1] == -1 and upper_band.iloc[i] < supertrend.iloc[i-1]:
                    supertrend.iloc[i] = supertrend.iloc[i-1]
                    direction.iloc[i] = -1
                else:
                    supertrend.iloc[i] = upper_band.iloc[i]
                    direction.iloc[i] = 1

        return supertrend, direction

    def calculate_chandelier_exit(
        self,
        df: pd.DataFrame
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Calculate Chandelier Exit indicator.

        Args:
            df: OHLCV dataframe

        Returns:
            Tuple of (chandelier_long, chandelier_short)
        """
        # Calculate ATR
        df['tr'] = np.maximum(
            df['high'] - df['low'],
            np.maximum(
                abs(df['high'] - df['close'].shift(1)),
                abs(df['low'] - df['close'].shift(1))
            )
        )
        atr = df['tr'].rolling(window=self.atr_period).mean()

        # Chandelier Exit
        chandelier_long = (
            df['high'].rolling(window=self.chandelier_period).max() -
            (atr * self.supertrend_multiplier)
        )
        chandelier_short = (
            df['low'].rolling(window=self.chandelier_period).min() +
            (atr * self.supertrend_multiplier)
        )

        return chandelier_long, chandelier_short

    def long_entry_conditions(
        self,
        current_data: Dict[str, Any],
        liquidity_zones: Dict[str, Any],
        regime: MarketRegime
    ) -> Tuple[bool, List[str]]:
        """
        Check long entry conditions (4 out of 6 must pass).

        Conditions:
        1. SuperTrend bullish
        2. Price above Chandelier Long
        3. Above nearest liquidity support
        4. Multi-timeframe alignment bullish or mixed
        5. Regime appropriate (not high-vol ranging)
        6. Positive order flow

        Args:
            current_data: Current market data with indicators
            liquidity_zones: Liquidity zone information
            regime: Current market regime

        Returns:
            Tuple of (passes, satisfied_conditions)
        """
        conditions = []
        current_price = current_data['close']

        # Condition 1: SuperTrend bullish
        if current_data.get('supertrend_direction', 0) > 0:
            conditions.append('supertrend_bullish')

        # Condition 2: Price above Chandelier Long
        chandelier_long = current_data.get('chandelier_long')
        if chandelier_long and current_price > chandelier_long:
            conditions.append('above_chandelier_long')

        # Condition 3: Above nearest liquidity support
        nearest_support = liquidity_zones.get('current_nearest_support')
        if nearest_support and current_price > nearest_support:
            conditions.append('above_liquidity_support')

        # Condition 4: Alignment is bullish or mixed (not strongly bearish)
        alignment = current_data.get('alignment', AlignmentState.MIXED_SIGNALS)
        if alignment in [AlignmentState.STRONG_BULLISH_ALIGNMENT, AlignmentState.MIXED_SIGNALS]:
            conditions.append('alignment_appropriate')

        # Condition 5: Regime appropriate
        if regime in [MarketRegime.NORMAL_REGIME, MarketRegime.LOW_VOL_COMPRESSION, MarketRegime.TREND_HIGH_VOL]:
            conditions.append('regime_appropriate')

        # Condition 6: Positive order flow
        orderflow_cum = current_data.get('orderflow_cumulative', 0)
        if orderflow_cum > 0:
            conditions.append('positive_orderflow')

        passes = len(conditions) >= self.entry_condition_threshold
        return passes, conditions

    def short_entry_conditions(
        self,
        current_data: Dict[str, Any],
        liquidity_zones: Dict[str, Any],
        regime: MarketRegime
    ) -> Tuple[bool, List[str]]:
        """
        Check short entry conditions (4 out of 6 must pass).

        Conditions:
        1. SuperTrend bearish
        2. Price below Chandelier Short
        3. Below nearest liquidity resistance
        4. Multi-timeframe alignment bearish or mixed
        5. Regime appropriate (not low-vol compression)
        6. Negative order flow

        Args:
            current_data: Current market data with indicators
            liquidity_zones: Liquidity zone information
            regime: Current market regime

        Returns:
            Tuple of (passes, satisfied_conditions)
        """
        conditions = []
        current_price = current_data['close']

        # Condition 1: SuperTrend bearish
        if current_data.get('supertrend_direction', 0) < 0:
            conditions.append('supertrend_bearish')

        # Condition 2: Price below Chandelier Short
        chandelier_short = current_data.get('chandelier_short')
        if chandelier_short and current_price < chandelier_short:
            conditions.append('below_chandelier_short')

        # Condition 3: Below nearest liquidity resistance
        nearest_resistance = liquidity_zones.get('current_nearest_resistance')
        if nearest_resistance and current_price < nearest_resistance:
            conditions.append('below_liquidity_resistance')

        # Condition 4: Alignment is bearish or mixed (not strongly bullish)
        alignment = current_data.get('alignment', AlignmentState.MIXED_SIGNALS)
        if alignment in [AlignmentState.STRONG_BEARISH_ALIGNMENT, AlignmentState.MIXED_SIGNALS]:
            conditions.append('alignment_appropriate')

        # Condition 5: Regime appropriate
        if regime in [MarketRegime.NORMAL_REGIME, MarketRegime.TREND_HIGH_VOL, MarketRegime.RANGING_HIGH_VOL]:
            conditions.append('regime_appropriate')

        # Condition 6: Negative order flow
        orderflow_cum = current_data.get('orderflow_cumulative', 0)
        if orderflow_cum < 0:
            conditions.append('negative_orderflow')

        passes = len(conditions) >= self.entry_condition_threshold
        return passes, conditions

    def calculate_stop_loss(
        self,
        entry_price: float,
        direction: str,
        atr: float,
        liquidity_zones: Dict[str, Any],
        supertrend_value: float
    ) -> float:
        """
        Calculate stop loss using multiple methods and take the most conservative.

        Args:
            entry_price: Entry price
            direction: 'LONG' or 'SHORT'
            atr: Average True Range
            liquidity_zones: Liquidity zone information
            supertrend_value: SuperTrend value

        Returns:
            Stop loss price
        """
        stops = []

        # ATR-based stop (2 ATR)
        atr_stop = entry_price - (2 * atr) if direction == 'LONG' else entry_price + (2 * atr)
        stops.append(atr_stop)

        # Liquidity-based stop (beyond nearest level)
        if direction == 'LONG':
            nearest_support = liquidity_zones.get('current_nearest_support')
            if nearest_support:
                liquidity_stop = nearest_support * 0.995  # 0.5% below support
                stops.append(liquidity_stop)
        else:
            nearest_resistance = liquidity_zones.get('current_nearest_resistance')
            if nearest_resistance:
                liquidity_stop = nearest_resistance * 1.005  # 0.5% above resistance
                stops.append(liquidity_stop)

        # SuperTrend-based stop (more conservative)
        if direction == 'LONG':
            if supertrend_value > entry_price:
                stops.append(supertrend_value)
        else:
            if supertrend_value < entry_price:
                stops.append(supertrend_value)

        # Return most conservative (closest to entry for LONG, farthest for SHORT)
        if direction == 'LONG':
            return max(stops)
        else:
            return min(stops)

    def calculate_position_size(
        self,
        account_balance: float,
        current_volatility: float,
        confidence: float
    ) -> float:
        """
        Calculate position size based on volatility and confidence.

        Args:
            account_balance: Current account balance
            current_volatility: Current volatility
            confidence: Signal confidence (0.0 to 1.0)

        Returns:
            Position size as percentage of balance
        """
        base_risk = 0.02  # 2% base risk
        volatility_adjustment = 1.0 / (1.0 + current_volatility)
        confidence_adjustment = confidence

        final_risk = base_risk * volatility_adjustment * confidence_adjustment

        # Cap between 0.5% and 3% of balance
        return max(min(final_risk, 0.03), 0.005)

    def generate_signal(
        self,
        df: pd.DataFrame,
        mtf_data: Dict[str, Dict[str, Any]]
    ) -> ConvergenceSignal:
        """
        Generate trading signal based on convergence analysis.

        Args:
            df: Current timeframe OHLCV data
            mtf_data: Multi-timeframe data

        Returns:
            ConvergenceSignal object
        """
        if len(df) < max(self.atr_period, self.chandelier_period, self.supertrend_period):
            return ConvergenceSignal(action='HOLD', confidence=0.0)

        # Calculate indicators
        supertrend, st_direction = self.calculate_supertrend(df)
        chandelier_long, chandelier_short = self.calculate_chandelier_exit(df)

        # Current values
        current_price = df['close'].iloc[-1]
        current_supertrend = supertrend.iloc[-1]
        current_st_direction = st_direction.iloc[-1]
        atr = df['tr'].rolling(window=self.atr_period).mean().iloc[-1]
        volatility = atr / current_price

        # Detect market regime
        sma_200 = df['close'].rolling(200).mean()
        sma_200_slope = (sma_200.iloc[-1] - sma_200.iloc[-20]) / sma_200.iloc[-20] if len(df) >= 200 else 0

        # Simple trend detection for regime
        recent_returns = df['close'].pct_change(20).iloc[-1]
        price_action = "trending" if abs(recent_returns) > 0.02 else "ranging"

        regime = self.detect_market_regime(volatility, price_action, atr, sma_200_slope)

        # Check alignment
        alignment, alignment_score = self.check_timeframe_alignment(mtf_data)

        # Identify liquidity zones
        liquidity_zones = self.identify_liquidity_zones(df)

        # Order flow (placeholder - would use actual orderflow data)
        orderflow_cumulative = df['volume'].iloc[-20:].sum() * 0.001  # Simplified

        # Current market data
        current_data = {
            'close': current_price,
            'supertrend_direction': current_st_direction,
            'supertrend_value': current_supertrend,
            'chandelier_long': chandelier_long.iloc[-1],
            'chandelier_short': chandelier_short.iloc[-1],
            'atr': atr,
            'volatility': volatility,
            'regime': regime,
            'alignment': alignment,
            'alignment_score': alignment_score,
            'orderflow_cumulative': orderflow_cumulative
        }

        # Check entry conditions
        long_pass, long_conditions = self.long_entry_conditions(
            current_data, liquidity_zones, regime
        )
        short_pass, short_conditions = self.short_entry_conditions(
            current_data, liquidity_zones, regime
        )

        # Determine action
        if long_pass and not short_pass:
            # LONG signal
            stop_loss = self.calculate_stop_loss(
                current_price, 'LONG', atr, liquidity_zones, current_supertrend
            )
            risk = current_price - stop_loss
            reward = risk * self.risk_reward_target
            take_profit = current_price + reward

            confidence = len(long_conditions) / 6.0
            position_size = self.calculate_position_size(
                10000, volatility, confidence
            )  # Using 10k as base, can be parameterized

            return ConvergenceSignal(
                action='LONG',
                confidence=confidence,
                entry_price=current_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                risk_reward_ratio=self.risk_reward_target,
                alignment_score=alignment_score,
                regime=regime.value,
                satisfied_conditions=long_conditions,
                liquidity_target=liquidity_zones.get('current_nearest_support')
            )

        elif short_pass and not long_pass:
            # SHORT signal
            stop_loss = self.calculate_stop_loss(
                current_price, 'SHORT', atr, liquidity_zones, current_supertrend
            )
            risk = stop_loss - current_price
            reward = risk * self.risk_reward_target
            take_profit = current_price - reward

            confidence = len(short_conditions) / 6.0
            position_size = self.calculate_position_size(
                10000, volatility, confidence
            )

            return ConvergenceSignal(
                action='SHORT',
                confidence=confidence,
                entry_price=current_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                risk_reward_ratio=self.risk_reward_target,
                alignment_score=alignment_score,
                regime=regime.value,
                satisfied_conditions=short_conditions,
                liquidity_target=liquidity_zones.get('current_nearest_resistance')
            )

        else:
            # HOLD
            return ConvergenceSignal(
                action='HOLD',
                confidence=0.0,
                regime=regime.value,
                alignment_score=alignment_score
            )