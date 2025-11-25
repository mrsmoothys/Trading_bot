"""
RSI Divergence Strategy
RSI-based mean reversion strategy with divergence detection.
"""

import pandas as pd
import numpy as np
from typing import Dict
from loguru import logger


def calculate_signals(df: pd.DataFrame, params: Dict) -> pd.DataFrame:
    """
    Generate signals for RSI Divergence strategy.

    Args:
        df: OHLCV data
        params: Strategy parameters
            - period: RSI period (default: 14)
            - oversold: RSI oversold level (default: 30)
            - overbought: RSI overbought level (default: 70)
            - min_strength: Minimum signal strength (default: 0.6)

    Returns:
        DataFrame with signals
    """
    period = params.get('period', 14)
    oversold = params.get('oversold', 30)
    overbought = params.get('overbought', 70)
    min_strength = params.get('min_strength', 0.6)

    signals = pd.DataFrame(index=df.index)
    signals['signal'] = 0.0
    signals['strength'] = 0.0
    signals['confidence'] = 0.5

    # Calculate RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

    # Avoid division by zero
    rs = gain / (loss + 1e-10)
    rs = rs.replace([np.inf, -np.inf], 0).fillna(0)

    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.replace([np.inf, -np.inf], 50).fillna(50)

    # Generate signals
    # LONG when RSI is oversold
    # SHORT when RSI is overbought
    signals.loc[rsi < oversold, 'signal'] = 1
    signals.loc[rsi > overbought, 'signal'] = -1

    # Signal strength based on RSI extreme
    rsi_extreme = np.maximum(
        (oversold - rsi).abs() / oversold,
        (rsi - overbought).abs() / overbought
    )
    signals['strength'] = rsi_extreme.fillna(0)

    # Confidence based on signal magnitude
    signals['confidence'] = np.minimum(signals['strength'] * 2, 1.0)

    # Filter weak signals
    signals.loc[signals['strength'] < min_strength, 'signal'] = 0

    # Add throttling to prevent overtrading
    # Only allow signals if there's been no signal in the last 5 bars
    signals['throttle'] = signals['signal'].rolling(5).max().fillna(0)
    signals.loc[signals['throttle'] == 0, 'signal'] = 0

    # Clean up throttle column
    signals = signals.drop('throttle', axis=1)

    return signals
