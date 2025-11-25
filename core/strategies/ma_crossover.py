"""
MA Crossover Strategy
Simple moving average crossover strategy with trend following.
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple
from loguru import logger


def calculate_signals(df: pd.DataFrame, params: Dict) -> pd.DataFrame:
    """
    Generate signals for MA Crossover strategy.

    Args:
        df: OHLCV data
        params: Strategy parameters
            - fast_period: Fast MA period (default: 10)
            - slow_period: Slow MA period (default: 30)
            - min_strength: Minimum signal strength (default: 0.6)

    Returns:
        DataFrame with signals
    """
    fast_period = params.get('fast_period', 10)
    slow_period = params.get('slow_period', 30)
    min_strength = params.get('min_strength', 0.6)

    signals = pd.DataFrame(index=df.index)
    signals['signal'] = 0.0
    signals['strength'] = 0.0
    signals['confidence'] = 0.5

    # Calculate moving averages
    fast_ma = df['close'].rolling(fast_period).mean()
    slow_ma = df['close'].rolling(slow_period).mean()

    # Generate signals
    # LONG when fast MA crosses above slow MA
    # SHORT when fast MA crosses below slow MA
    signals.loc[fast_ma > slow_ma, 'signal'] = 1
    signals.loc[fast_ma < slow_ma, 'signal'] = -1

    # Signal strength based on MA separation
    ma_diff = (fast_ma - slow_ma) / df['close']
    signals['strength'] = ma_diff.abs().fillna(0)

    # Confidence based on trend consistency
    signal_consistency = signals['signal'].rolling(5).apply(
        lambda x: len(set(x)) == 1 if len(x) == 5 else 0
    ).fillna(0)
    signals['confidence'] = np.maximum(signal_consistency, min_strength)

    # Filter weak signals
    signals.loc[signals['strength'] < min_strength, 'signal'] = 0

    return signals
