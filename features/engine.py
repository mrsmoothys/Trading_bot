"""
Feature Engineering Engine
Calculates comprehensive market features for trading decisions.

This module implements all features specified in the DeepSeek Trading System:
- Liquidity Zone Detection
- Order Flow Imbalance
- Enhanced Chandelier Exit
- Advanced Supertrend
- Market Regime Classification
- Multi-Timeframe Convergence
"""

from typing import Dict, Any, Tuple, List
import asyncio
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd
from datetime import datetime
import time
import psutil
from loguru import logger


def calculate_liquidity_zones(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
    lookback_periods: int = 100,
    price_bins: int = 50,
) -> Dict[str, Any]:
    """
    IDENTIFY PRICE LEVELS WHERE LARGE ORDERS RESIDE.

    This function identifies institutional accumulation and distribution zones
    by analyzing volume distribution across price levels.

    Returns:
        Dict containing liquidity zone information including support/resistance levels
        and nearest levels for convergence strategy compatibility
    """
    # Use recent data only
    data_slice = min(lookback_periods, len(close))
    high_slice = high.iloc[-data_slice:]
    low_slice = low.iloc[-data_slice:]
    close_slice = close.iloc[-data_slice:]
    volume_slice = volume.iloc[-data_slice:]

    # Calculate volume-weighted price levels
    typical_price = (high_slice + low_slice + close_slice) / 3
    volume_weighted_price = (typical_price * volume_slice).cumsum() / volume_slice.cumsum()

    # Create price bins
    price_range = (low_slice.min(), high_slice.max())
    price_bins_array = np.linspace(price_range[0], price_range[1], price_bins)

    # Distribute volume across price bins
    volume_at_price = np.zeros(len(price_bins_array) - 1)

    for i in range(len(high_slice)):
        low_price = low_slice.iloc[i]
        high_price = high_slice.iloc[i]
        vol = volume_slice.iloc[i]

        # Distribute volume across price bins this candle touched
        for j in range(len(price_bins_array) - 1):
            bin_low = price_bins_array[j]
            bin_high = price_bins_array[j + 1]

            # Check if there's an overlap
            if high_price < bin_low or low_price > bin_high:
                continue

            # Calculate overlap ratio
            overlap = min(high_price, bin_high) - max(low_price, bin_low)
            overlap_ratio = overlap / (high_price - low_price) if (high_price - low_price) > 0 else 0
            volume_at_price[j] += vol * overlap_ratio

    # Identify significant liquidity levels (top 5%)
    volume_threshold = np.percentile(volume_at_price, 95)
    significant_bins = price_bins_array[:-1][volume_at_price > volume_threshold]

    # Calculate current price relation to liquidity zones
    current_price = close_slice.iloc[-1]
    if len(significant_bins) > 0:
        nearest_zone = significant_bins[np.argmin(np.abs(significant_bins - current_price))]
        distance_to_zone = (current_price - nearest_zone) / current_price
        zone_strength = volume_at_price.max() / volume_at_price.mean() if volume_at_price.mean() > 0 else 0
        above_below = "above" if current_price > nearest_zone else "below"
    else:
        nearest_zone = current_price
        distance_to_zone = 0
        zone_strength = 0
        above_below = "at"

    # Separate into support and resistance levels for convergence strategy
    resistance_levels = [float(z) for z in significant_bins if z > current_price]
    support_levels = [float(z) for z in significant_bins if z < current_price]

    # Find nearest support and resistance
    def find_nearest_level(price, levels, below=True):
        if not levels:
            return None
        if below:
            below_levels = [l for l in levels if l < price]
            return max(below_levels) if below_levels else min(levels)
        else:
            above_levels = [l for l in levels if l > price]
            return min(above_levels) if above_levels else max(levels)

    nearest_support = find_nearest_level(current_price, support_levels, below=True)
    nearest_resistance = find_nearest_level(current_price, resistance_levels, below=False)

    return {
        "liquidity_zones": significant_bins.tolist(),
        "nearest_zone": float(nearest_zone),
        "distance_to_zone_pct": float(distance_to_zone),
        "zone_strength": float(zone_strength),
        "above_below_zone": above_below,
        "total_zones": len(significant_bins),
        # Convergence strategy format
        "resistance_levels": resistance_levels,
        "support_levels": support_levels,
        "current_nearest_support": nearest_support,
        "current_nearest_resistance": nearest_resistance,
    }


def calculate_order_flow_imbalance(
    open_price: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
    period: int = 20,
) -> Dict[str, Any]:
    """
    MEASURE BUYING VS SELLING PRESSURE from OHLCV data.

    Infers order flow from candle structure and volume without tick data.
    """
    # Calculate candle components
    body_size = abs(close - open_price)
    total_range = high - low
    upper_wick = high - np.maximum(open_price, close)
    lower_wick = np.minimum(open_price, close) - low

    # Buying pressure: strong closes with high volume
    buying_pressure = (
        (close > open_price) & (body_size > total_range * 0.6)
    ).astype(int)

    # Selling pressure: strong rejections at highs or bearish closes
    selling_pressure = (
        (upper_wick > body_size * 1.5) | (close < open_price * 0.998)
    ).astype(int)

    # Calculate volume-weighted imbalance
    volume_imbalance = (buying_pressure - selling_pressure) * volume
    imbalance_ratio = volume_imbalance.rolling(period).sum() / volume.rolling(period).sum()

    return {
        "order_flow_imbalance": float(imbalance_ratio.iloc[-1]) if not pd.isna(imbalance_ratio.iloc[-1]) else 0.0,
        "buying_pressure_20ma": float(buying_pressure.rolling(period).mean().iloc[-1]) if not pd.isna(buying_pressure.rolling(period).mean().iloc[-1]) else 0.0,
        "selling_pressure_20ma": float(selling_pressure.rolling(period).mean().iloc[-1]) if not pd.isna(selling_pressure.rolling(period).mean().iloc[-1]) else 0.0,
        "imbalance_trend": float(imbalance_ratio.diff(5).iloc[-1]) if not pd.isna(imbalance_ratio.diff(5).iloc[-1]) else 0.0,
    }


def calculate_enhanced_chandelier_exit(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 22,
    multiplier: float = 3.0,
) -> Dict[str, Any]:
    """
    VOLATILITY-BASED TRAILING STOP WITH ADAPTIVE MULTIPLIER.

    Enhanced version of the Chandelier Exit that adapts to market volatility.
    """
    # Calculate True Range
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()

    # Calculate volatility regime
    volatility_regime = atr / close.rolling(50).std()
    adaptive_multiplier = multiplier * (1 + volatility_regime * 0.5)

    # Calculate Chandelier Exit levels
    long_stop = high.rolling(period).max() - adaptive_multiplier * atr
    short_stop = low.rolling(period).min() + adaptive_multiplier * atr

    # Determine trend
    trend = np.where(
        close > long_stop.shift(1),
        "bullish",
        np.where(close < short_stop.shift(1), "bearish", "neutral"),
    )

    return {
        "chandelier_long_stop": float(long_stop.iloc[-1]) if not pd.isna(long_stop.iloc[-1]) else 0.0,
        "chandelier_short_stop": float(short_stop.iloc[-1]) if not pd.isna(short_stop.iloc[-1]) else 0.0,
        "chandelier_trend": str(trend[-1]) if len(trend) > 0 else "neutral",
        "distance_to_stop_pct": float((close.iloc[-1] - long_stop.iloc[-1]) / close.iloc[-1]) if not pd.isna(long_stop.iloc[-1]) else 0.0,
        "adaptive_multiplier": float(adaptive_multiplier.iloc[-1]) if not pd.isna(adaptive_multiplier.iloc[-1]) else multiplier,
    }


def calculate_advanced_supertrend(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    atr_period: int = 10,
    multiplier: float = 3.0,
) -> Dict[str, Any]:
    """
    ENHANCED SUPERTREND WITH MULTI-TIMEFRAME CONFIRMATION.

    Advanced implementation with confidence scoring and trend consistency.
    """
    # Calculate True Range
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(atr_period).mean()

    # Calculate basic bands
    hl2 = (high + low) / 2
    upper_band = hl2 + (multiplier * atr)
    lower_band = hl2 - (multiplier * atr)

    # Initialize Supertrend
    supertrend = pd.Series(index=close.index, dtype=float)
    trend = pd.Series(index=close.index, dtype=str)

    # Set initial values
    supertrend.iloc[0] = upper_band.iloc[0]
    trend.iloc[0] = "downtrend"

    # Calculate Supertrend
    for i in range(1, len(close)):
        if close.iloc[i] > supertrend.iloc[i - 1]:
            supertrend.iloc[i] = max(lower_band.iloc[i], supertrend.iloc[i - 1])
            trend.iloc[i] = "uptrend"
        else:
            supertrend.iloc[i] = min(upper_band.iloc[i], supertrend.iloc[i - 1])
            trend.iloc[i] = "downtrend"

    # Calculate confidence scoring
    price_distance = abs(close - supertrend) / atr
    trend_strength = price_distance.rolling(5).mean()
    # Convert trend to numeric for consistency calculation
    trend_numeric = pd.Series(np.where(trend == 'uptrend', 1, np.where(trend == 'downtrend', -1, 0)), index=trend.index)
    trend_consistency = trend_numeric.rolling(3).apply(lambda x: len(set(x)) == 1).fillna(0)

    return {
        "supertrend_value": float(supertrend.iloc[-1]) if not pd.isna(supertrend.iloc[-1]) else 0.0,
        "supertrend_trend": str(trend.iloc[-1]) if len(trend) > 0 else "neutral",
        "supertrend_strength": float(trend_strength.iloc[-1]) if not pd.isna(trend_strength.iloc[-1]) else 0.0,
        "trend_consistency": float(trend_consistency.iloc[-1]) if not pd.isna(trend_consistency.iloc[-1]) else 0.0,
        "price_vs_supertrend": float((close.iloc[-1] - supertrend.iloc[-1]) / close.iloc[-1]) if not pd.isna(supertrend.iloc[-1]) else 0.0,
    }


def calculate_market_regime(
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    volume: pd.Series,
    period: int = 50,
) -> Dict[str, Any]:
    """
    IDENTIFY CURRENT MARKET REGIME: TRENDING OR RANGING, HIGH/LOW VOL.

    Classifies market into regimes based on volatility, trend strength, and range.
    """
    # Calculate returns
    returns = close.pct_change()
    realized_vol = returns.rolling(period).std()
    vol_regime = realized_vol.rank(pct=True).iloc[-1] if len(realized_vol) > 0 else 0.5

    # Calculate trend strength
    sma_20 = close.rolling(20).mean()
    sma_50 = close.rolling(50).mean()
    trend_strength = abs(sma_20 - sma_50) / close.rolling(50).std()
    trend_strength_val = float(trend_strength.iloc[-1]) if not pd.isna(trend_strength.iloc[-1]) else 0

    # Calculate range measures
    atr = (high - low).rolling(14).mean()
    range_ratio = atr / close
    range_regime = float(range_ratio.rank(pct=True).iloc[-1]) if len(range_ratio) > 0 else 0.5

    # Volume regime
    volume_zscore = (volume - volume.rolling(50).mean()) / volume.rolling(50).std()
    volume_regime = float(abs(volume_zscore).iloc[-1]) if len(volume_zscore) > 0 else 0

    # Classify regime
    if trend_strength_val > 0.1 and vol_regime > 0.7:
        regime = "TRENDING_HIGH_VOL"
    elif trend_strength_val > 0.1 and vol_regime <= 0.7:
        regime = "TRENDING_LOW_VOL"
    elif trend_strength_val <= 0.1 and range_regime < 0.3:
        regime = "RANGING_COMPRESSION"
    elif trend_strength_val <= 0.1 and range_regime >= 0.3:
        regime = "RANGING_EXPANSION"
    else:
        regime = "TRANSITION"

    return {
        "market_regime": regime,
        "volatility_percentile": float(vol_regime),
        "trend_strength": trend_strength_val,
        "range_percentile": range_regime,
        "volume_anomaly": volume_regime,
        "regime_confidence": float(min(trend_strength_val, vol_regime, range_regime)),
    }


def calculate_multi_timeframe_convergence(
    symbol: str,
    timeframe_data: Dict[str, pd.DataFrame],
    timeframes: List[str] = None,
) -> Dict[str, Any]:
    """
    SCORE HOW ALIGNED DIFFERENT TIMEFRAMES ARE FOR A GIVEN SYMBOL.

    Analyzes multiple timeframes to determine overall market bias.
    """
    if timeframes is None:
        timeframes = ["5m", "15m", "1h", "4h"]

    convergence_scores = {}

    for tf in timeframes:
        if tf not in timeframe_data:
            continue

        data = timeframe_data[tf]
        if len(data) < 50:  # Need sufficient data
            continue

        # Calculate SMAs
        sma_20 = data["close"].rolling(20).mean()
        sma_50 = data["close"].rolling(50).mean()

        # Determine trend
        if sma_20.iloc[-1] > sma_50.iloc[-1]:
            trend = "BULLISH"
        elif sma_20.iloc[-1] < sma_50.iloc[-1]:
            trend = "BEARISH"
        else:
            trend = "NEUTRAL"

        # Calculate RSI
        delta = data["close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        rsi_val = float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else 50

        if rsi_val > 60:
            momentum = "STRONG"
        elif rsi_val < 40:
            momentum = "WEAK"
        else:
            momentum = "NEUTRAL"

        convergence_scores[tf] = {
            "trend": trend,
            "momentum": momentum,
            "rsi": rsi_val,
            "volume_trend": bool(data["volume"].rolling(5).mean().iloc[-1] > data["volume"].rolling(20).mean().iloc[-1]),
        }

    # Calculate alignment score
    if len(convergence_scores) > 0:
        bullish_count = sum(1 for tf in convergence_scores if convergence_scores[tf]["trend"] == "BULLISH")
        bearish_count = sum(1 for tf in convergence_scores if convergence_scores[tf]["trend"] == "BEARISH")
        total_tfs = len(convergence_scores)
        alignment_score = max(bullish_count, bearish_count) / total_tfs if total_tfs > 0 else 0.5
        primary_trend = "BULLISH" if bullish_count > bearish_count else "BEARISH"
    else:
        alignment_score = 0.5
        primary_trend = "NEUTRAL"

    return {
        "timeframe_alignment": float(alignment_score),
        "primary_trend": primary_trend,
        "convergence_details": convergence_scores,
        "trading_timeframe_recommendation": "SWING" if alignment_score > 0.75 else "INTRADAY",
    }


def prepare_convergence_strategy_data(
    symbol: str,
    timeframe_data: Dict[str, pd.DataFrame],
    market_data: Dict[str, pd.Series],
    liquidity_data: Dict[str, Any],
    orderflow_data: Dict[str, Any],
    timeframes: List[str] = None,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
    """
    Prepare multi-timeframe data and current data in the exact format
    expected by the convergence strategy.

    Args:
        symbol: Trading symbol
        timeframe_data: Dict of DataFrames for multiple timeframes
        market_data: Current timeframe market data (OHLCV)
        liquidity_data: Liquidity zone data
        orderflow_data: Order flow data
        timeframes: List of timeframes to analyze

    Returns:
        Tuple of (mtf_data, current_data) ready for convergence strategy
    """
    if timeframes is None:
        timeframes = ["1m", "5m", "15m", "1h", "4h", "1d"]

    # Prepare multi-timeframe data
    mtf_data = {}
    for tf in timeframes:
        if tf not in timeframe_data:
            continue

        data = timeframe_data[tf]
        if len(data) < 50:
            continue

        # Calculate trend using EMA slope
        ema_fast = data["close"].ewm(span=10).mean()
        ema_slow = data["close"].ewm(span=21).mean()

        # Determine trend direction
        if ema_fast.iloc[-1] > ema_slow.iloc[-1]:
            trend = "BULLISH"
        elif ema_fast.iloc[-1] < ema_slow.iloc[-1]:
            trend = "BEARISH"
        else:
            trend = "NEUTRAL"

        # Calculate trend strength (distance between EMAs relative to price)
        trend_strength = abs(ema_fast.iloc[-1] - ema_slow.iloc[-1]) / data["close"].iloc[-1]
        trend_strength = float(trend_strength) if not pd.isna(trend_strength) else 0.0

        mtf_data[tf] = {
            "trend": trend,
            "trend_strength": trend_strength
        }

    # Prepare current market data with ATR
    close = market_data["close"]
    high = market_data["high"]
    low = market_data["low"]

    # Calculate ATR
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14).mean()
    current_atr = float(atr.iloc[-1]) if not pd.isna(atr.iloc[-1]) else 0.0

    # Calculate volatility
    current_price = float(close.iloc[-1])
    volatility = current_atr / current_price if current_price > 0 else 0.0

    # Calculate cumulative order flow
    orderflow_cum = float(orderflow_data.get("order_flow_imbalance", 0.0))

    # Prepare current data dict
    current_data = {
        'close': current_price,
        'atr': current_atr,
        'volatility': volatility,
        'orderflow_cumulative': orderflow_cum
    }

    return mtf_data, current_data


def calculate_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Calculate Relative Strength Index."""
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


class FeatureEngine:
    """
    Main Feature Engineering Engine.
    Orchestrates the calculation of all features in the correct order.
    """

    def __init__(self):
        """Initialize the feature engine."""
        self.feature_cache = {}
        self.cache_ttl = 300  # 5 minutes
        self.performance_metrics: Dict[str, Dict[str, Any]] = {}
        self.last_market_regime: str = "TRANSITION"
        self.executor = ProcessPoolExecutor(max_workers=4)

    def calculate_liquidity_zones(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        volume: pd.Series,
        lookback_periods: int = 100,
        price_bins: int = 50,
    ) -> Dict[str, Any]:
        """
        Wrapper method for calculate_liquidity_zones standalone function.
        For use by signal generators and other components.
        """
        return calculate_liquidity_zones(
            high=high,
            low=low,
            close=close,
            volume=volume,
            lookback_periods=lookback_periods,
            price_bins=price_bins,
        )

    def calculate_order_flow_imbalance(
        self,
        open: pd.Series,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        volume: pd.Series,
        lookback: int = 20,
    ) -> Dict[str, Any]:
        """
        Wrapper method for calculate_order_flow_imbalance standalone function.
        For use by signal generators and other components.
        """
        return calculate_order_flow_imbalance(
            open=open,
            high=high,
            low=low,
            close=close,
            volume=volume,
            lookback=lookback,
        )

    def calculate_enhanced_chandelier_exit(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 22,
        multiplier: float = 3.0,
    ) -> Dict[str, Any]:
        """
        Wrapper method for calculate_enhanced_chandelier_exit standalone function.
        For use by signal generators and other components.
        """
        return calculate_enhanced_chandelier_exit(
            high=high,
            low=low,
            close=close,
            period=period,
            multiplier=multiplier,
        )

    def calculate_advanced_supertrend(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 10,
        multiplier: float = 3.0,
    ) -> Dict[str, Any]:
        """
        Wrapper method for calculate_advanced_supertrend standalone function.
        For use by signal generators and other components.
        """
        return calculate_advanced_supertrend(
            high=high,
            low=low,
            close=close,
            period=period,
            multiplier=multiplier,
        )

    def calculate_market_regime(
        self,
        close: pd.Series,
        short_window: int = 20,
        long_window: int = 60,
    ) -> Dict[str, Any]:
        """
        Wrapper method for calculate_market_regime standalone function.
        For use by signal generators and other components.
        """
        return calculate_market_regime(
            close=close,
            short_window=short_window,
            long_window=long_window,
        )

    def _capture_metrics(self, feature_name: str, func, *args, **kwargs) -> Dict[str, Any]:
        """Execute a feature function while recording latency and memory deltas."""
        process = psutil.Process()
        mem_before = process.memory_info().rss
        start = time.perf_counter()
        result = func(*args, **kwargs)
        latency_ms = (time.perf_counter() - start) * 1000
        mem_after = process.memory_info().rss
        memory_delta_mb = (mem_after - mem_before) / (1024 * 1024)

        self.performance_metrics[feature_name] = {
            "latency_ms": round(latency_ms, 3),
            "memory_delta_mb": round(memory_delta_mb, 3),
            "timestamp": datetime.now().isoformat(),
        }

        return result

    async def compute_all_features(
        self,
        symbol: str,
        market_data: Dict[str, pd.Series],
        timeframe_data: Dict[str, pd.DataFrame] = None,
        system_context=None,
    ) -> Dict[str, Any]:
        """
        Calculate ALL features for a symbol in the prescribed order with conditional recalculation.

        Uses SystemContext to determine which features need recalculation based on:
        - Profile settings (scalp vs swing)
        - Feature type (heavy vs lightweight)
        - Bar closure detection

        Args:
            symbol: Trading symbol (e.g., 'BTCUSDT')
            market_data: Dict with 'open', 'high', 'low', 'close', 'volume' Series
            timeframe_data: Dict of DataFrames for multiple timeframes
            system_context: SystemContext instance for cadence tracking (optional)

        Returns:
            Complete feature set
        """
        features = {}
        timeframe = "15m"  # Default timeframe for single-timeframe features
        current_bar_timestamp = pd.to_datetime(market_data["timestamp"]).iloc[-1] if "timestamp" in market_data else pd.Timestamp.now()
        loop = asyncio.get_running_loop()

        # Helper to check if feature should be recalculated
        def should_calc(feature_name):
            if system_context and hasattr(system_context, 'should_recalculate_feature'):
                return system_context.should_recalculate_feature(
                    feature_name, symbol, timeframe, current_bar_timestamp
                )
            # Default to always recalculating if no context provided
            return True

        # 1. Market Structure Features
        if should_calc("liquidity"):
            print(f"  → Calculating liquidity zones for {symbol}...")
            # Run in executor
            liquidity_result = await loop.run_in_executor(
                self.executor,
                calculate_liquidity_zones,
                market_data["high"],
                market_data["low"],
                market_data["close"],
                market_data["volume"],
                100, # lookback_periods
                50   # price_bins
            )
            features.update(liquidity_result)
        else:
            logger.debug(f"Skipping liquidity calculation for {symbol} (same bar)")

        if should_calc("orderflow"):
            print(f"  → Calculating order flow imbalance for {symbol}...")
            features.update(
                self._capture_metrics(
                    "orderflow",
                    calculate_order_flow_imbalance,
                    market_data["open"],
                    market_data["high"],
                    market_data["low"],
                    market_data["close"],
                    market_data["volume"],
                )
            )
        else:
            logger.debug(f"Skipping orderflow calculation for {symbol} (same bar)")

        # 2. Advanced Technical Indicators
        if should_calc("chandelier"):
            print(f"  → Calculating enhanced chandelier exit for {symbol}...")
            chandelier_result = await loop.run_in_executor(
                self.executor,
                calculate_enhanced_chandelier_exit,
                market_data["high"],
                market_data["low"],
                market_data["close"],
                22,  # period
                3.0  # multiplier
            )
            features.update(chandelier_result)
        else:
            logger.debug(f"Skipping chandelier calculation for {symbol} (same bar)")

        if should_calc("supertrend"):
            print(f"  → Calculating advanced supertrend for {symbol}...")
            supertrend_result = await loop.run_in_executor(
                self.executor,
                calculate_advanced_supertrend,
                market_data["high"],
                market_data["low"],
                market_data["close"],
                10,  # period
                3.0  # multiplier
            )
            features.update(supertrend_result)
        else:
            logger.debug(f"Skipping supertrend calculation for {symbol} (same bar)")

        # 3. Market Regime
        if should_calc("regime"):
            print(f"  → Calculating market regime for {symbol}...")
            regime_result = await loop.run_in_executor(
                self.executor,
                calculate_market_regime,
                market_data["close"],
                market_data["high"],
                market_data["low"],
                market_data["volume"],
                50 # period
            )
            features.update(regime_result)
            regime_value = regime_result.get("market_regime")
            if regime_value and regime_value != "UNKNOWN":
                self.last_market_regime = regime_value
            else:
                features["market_regime"] = self.last_market_regime or "TRANSITION"
        else:
            logger.debug(f"Skipping regime calculation for {symbol} (same bar)")
            features["market_regime"] = self.last_market_regime or "TRANSITION"

        # 4. Multi-timeframe Analysis
        if timeframe_data:
            if should_calc("alignment"):
                print(f"  → Calculating multi-timeframe convergence for {symbol}...")
                features.update(
                    self._capture_metrics(
                        "alignment",
                        calculate_multi_timeframe_convergence,
                        symbol,
                        timeframe_data,
                    )
                )
            else:
                logger.debug(f"Skipping alignment calculation for {symbol} (same bar)")
                # Use cached/default values
                features["timeframe_alignment"] = 0.5
                features["primary_trend"] = "NEUTRAL"
                features["convergence_details"] = {}
        else:
            features["timeframe_alignment"] = 0.5
            features["primary_trend"] = "NEUTRAL"
            features["convergence_details"] = {}
            features["trading_timeframe_recommendation"] = "INTRADAY"
            self.performance_metrics.setdefault("alignment", {
                "latency_ms": 0.0,
                "memory_delta_mb": 0.0,
                "timestamp": datetime.now().isoformat(),
            })

        # Add metadata
        features["symbol"] = symbol
        features["timestamp"] = datetime.now().isoformat()

        return features

    def get_feature_highlights(self, features: Dict[str, Any]) -> List[str]:
        """
        Extract key feature highlights for signal generation.
        """
        highlights = []

        # Check market regime
        regime = features.get("market_regime", "UNKNOWN")
        if "TRENDING" in regime:
            highlights.append(f"Market in {regime} regime")

        # Check liquidity zone proximity
        distance = abs(features.get("distance_to_zone_pct", 0))
        if distance < 0.01:
            highlights.append("Price near liquidity zone")

        # Check order flow
        imbalance = features.get("order_flow_imbalance", 0)
        if imbalance > 0.1:
            highlights.append("Strong buying pressure")
        elif imbalance < -0.1:
            highlights.append("Strong selling pressure")

        # Check trend consistency
        consistency = features.get("trend_consistency", 0)
        if consistency > 0.8:
            trend = features.get("supertrend_trend", "neutral")
            highlights.append(f"Consistent {trend} trend")

        # Check timeframe alignment
        alignment = features.get("timeframe_alignment", 0)
        if alignment > 0.75:
            highlights.append("High timeframe alignment")

        return highlights

    def prepare_convergence_strategy_input(
        self,
        symbol: str,
        timeframe_data: Dict[str, pd.DataFrame],
        market_data: Dict[str, pd.Series],
        timeframes: List[str] = None,
    ) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any], Dict[str, Any]]:
        """
        Prepare all data needed by the convergence strategy in one call.

        Args:
            symbol: Trading symbol
            timeframe_data: Dict of DataFrames for multiple timeframes
            market_data: Current timeframe market data (OHLCV Series)
            timeframes: List of timeframes to analyze

        Returns:
            Tuple of (mtf_data, current_data, liquidity_zones) ready for convergence strategy
        """
        # Calculate all features
        liquidity_data = calculate_liquidity_zones(
            market_data["high"],
            market_data["low"],
            market_data["close"],
            market_data["volume"],
        )

        orderflow_data = calculate_order_flow_imbalance(
            market_data["open"],
            market_data["high"],
            market_data["low"],
            market_data["close"],
            market_data["volume"],
        )

        # Prepare data for convergence strategy
        mtf_data, current_data = prepare_convergence_strategy_data(
            symbol,
            timeframe_data,
            market_data,
            liquidity_data,
            orderflow_data,
            timeframes
        )

        return mtf_data, current_data, liquidity_data
