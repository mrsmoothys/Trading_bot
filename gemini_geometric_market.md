# Phase 4: Geometric Market Structure Implementation Plan

This plan outlines the implementation of advanced Price Action features (Swing Points, BOS, CHoCH, FVG) to upgrade the system's structural analysis capabilities.

**Prerequisite:** This phase must ONLY be executed after the critical remediation steps in `gemini_plan.md` (Shared Persistence, Hardcoded Paths, DB Persistence) are complete.

## 1. Core Concept: Geometric Price Action
We will introduce a new module `features/structure.py` dedicated to identifying geometric market features. This separates "pure math" indicators (RSI, ATR) from "structural" concepts.

## 2. Implementation Details

### Feature A: Swing Point Identification (Fractals)
**Goal:** Identify local peaks (Swing Highs) and valleys (Swing Lows) to define the market structure.
*   **Logic:** A 5-candle fractal.
    *   **Swing High:** `High[i]` is higher than `High[i-2], High[i-1], High[i+1], High[i+2]`.
    *   **Swing Low:** `Low[i]` is lower than `Low[i-2], Low[i-1], Low[i+1], Low[i+2]`.

### Feature B: Break of Structure (BOS) & Change of Character (CHoCH)
**Goal:** Detect trend continuation and reversal signals.
*   **Trend Definition:** Uptrend = Higher Highs (HH) + Higher Lows (HL).
*   **BOS (Continuation):** Close > Previous Swing High (in Uptrend).
*   **CHoCH (Reversal):** Close < Previous Higher Low (signals Uptrend -> Downtrend).

### Feature C: Fair Value Gaps (FVG)
**Goal:** Identify price inefficiencies.
*   **Bullish FVG:** Low[i] > High[i-2]. The gap is (High[i-2] ... Low[i]).
*   **Bearish FVG:** High[i] < Low[i-2]. The gap is (Low[i-2] ... High[i]).

## 3. Detailed Coding & Integration Guide

### Step 1: Create `features/structure.py`

Create this file to handle the vector logic.

```python
import pandas as pd
import numpy as np
from scipy.signal import argrelextrema

def find_swing_points(highs: pd.Series, lows: pd.Series, order: int = 2) -> pd.DataFrame:
    """
    Identify Swing Highs and Lows using local extrema.
    order=2 means checking 2 candles before and 2 after (5-candle fractal).
    """
    # Find indexes of local maxima and minima
    high_idx = argrelextrema(highs.values, np.greater, order=order)[0]
    low_idx = argrelextrema(lows.values, np.less, order=order)[0]

    swings = pd.DataFrame(index=highs.index)
    swings['swing_high'] = np.nan
    swings['swing_low'] = np.nan

    swings.iloc[high_idx, 0] = highs.iloc[high_idx]
    swings.iloc[low_idx, 1] = lows.iloc[low_idx]
    
    return swings

def calculate_smart_money_structure(df: pd.DataFrame) -> dict:
    """
    Identify BOS, CHoCH and current trend.
    """
    swings = find_swing_points(df['high'], df['low'])
    
    # Get non-NaN swings ordered by time
    last_highs = swings['swing_high'].dropna()
    last_lows = swings['swing_low'].dropna()
    
    if len(last_highs) < 2 or len(last_lows) < 2:
        return {'trend': 'NEUTRAL', 'structure': 'UNDEFINED'}

    current_price = df['close'].iloc[-1]
    prev_high = last_highs.iloc[-1]
    prev_low = last_lows.iloc[-1]
    
    # Simple Trend Logic (Improve with sequence analysis in production)
    trend = 'NEUTRAL'
    structure_event = None
    
    if last_highs.iloc[-1] > last_highs.iloc[-2] and last_lows.iloc[-1] > last_lows.iloc[-2]:
        trend = 'BULLISH'
        if current_price > prev_high:
            structure_event = 'BOS_BULLISH'
    
    elif last_highs.iloc[-1] < last_highs.iloc[-2] and last_lows.iloc[-1] < last_lows.iloc[-2]:
        trend = 'BEARISH'
        if current_price < prev_low:
            structure_event = 'BOS_BEARISH'

    return {
        'trend': trend,
        'last_swing_high': prev_high,
        'last_swing_low': prev_low,
        'event': structure_event
    }

def detect_fair_value_gaps(df: pd.DataFrame) -> list:
    """
    Identify unfilled Fair Value Gaps.
    """
    gaps = []
    # Vectorized approach for speed
    for i in range(2, len(df)):
        # Bullish FVG: Low[i] > High[i-2]
        if df['low'].iloc[i] > df['high'].iloc[i-2]:
            gaps.append({
                'type': 'BULLISH',
                'top': df['low'].iloc[i],
                'bottom': df['high'].iloc[i-2],
                'index': i,
                'timestamp': df.index[i]
            })
        # Bearish FVG: High[i] < Low[i-2]
        elif df['high'].iloc[i] < df['low'].iloc[i-2]:
            gaps.append({
                'type': 'BEARISH',
                'top': df['low'].iloc[i-2],
                'bottom': df['high'].iloc[i],
                'index': i,
                'timestamp': df.index[i]
            })
    
    # Filter only recent/unfilled gaps (optional logic to add here)
    return gaps[-10:] # Return last 10 gaps
```

### Step 2: Integrate into `features/engine.py`

Update the `FeatureEngine` class to use the new module.

```python
# features/engine.py
from features.structure import calculate_smart_money_structure, detect_fair_value_gaps

# Inside FeatureEngine class
def calculate_structure_features(self, df: pd.DataFrame) -> Dict[str, Any]:
    structure = calculate_smart_money_structure(df)
    fvgs = detect_fair_value_gaps(df)
    
    return {
        "market_structure": structure['trend'],
        "structure_event": structure.get('event'),
        "swing_high": structure['last_swing_high'],
        "swing_low": structure['last_swing_low'],
        "nearest_fvg": fvgs[-1] if fvgs else None,
        "active_fvgs": fvgs
    }

# Inside compute_all_features method
# ... after calculating other features ...
structure_features = self.calculate_structure_features(pd.DataFrame(market_data))
features.update(structure_features)
```

### Step 3: Update `SignalGenerator` Logic

Modify `core/signal_generator.py` to respect market structure.

```python
# core/signal_generator.py

def should_trade(self, signal: Dict[str, Any]) -> bool:
    # ... existing checks ...
    
    # STRUCTURAL FILTER
    # Don't short in a Bullish structure unless it's a scalp
    market_structure = signal.get('market_structure', 'NEUTRAL')
    action = signal.get('action', 'HOLD')
    
    if market_structure == 'BULLISH' and 'SHORT' in action:
        # Only allow if we see a Change of Character (reversal signal)
        if signal.get('structure_event') != 'CHOCH_BEARISH':
            logger.info("Trade rejected: Attempting to SHORT against BULLISH structure without CHoCH")
            return False
            
    return True
```

### Step 4: Dashboard Visualization (`ui/dashboard.py`)

Update `create_interactive_chart` to draw the new features.

```python
# ui/dashboard.py

def create_interactive_chart(...):
    # ... existing candle trace ...

    # 1. Plot FVGs (Rectangles)
    if features.get('show_fvg', True):
        fvgs = latest_features.get('active_fvgs', [])
        for gap in fvgs:
            color = 'rgba(0, 255, 0, 0.2)' if gap['type'] == 'BULLISH' else 'rgba(255, 0, 0, 0.2)'
            fig.add_shape(
                type="rect",
                x0=gap['timestamp'],
                y0=gap['bottom'],
                x1=df['timestamp'].iloc[-1], # Extend to current time
                y1=gap['top'],
                fillcolor=color,
                line=dict(width=0),
            )

    # 2. Plot Swing Points
    if features.get('show_structure', True):
        # We need to re-calculate swings for the visualization DF
        from features.structure import find_swing_points
        swings = find_swing_points(df['high'], df['low'])
        
        # Plot Swing Highs
        fig.add_trace(go.Scatter(
            x=swings.index, 
            y=swings['swing_high'],
            mode='markers',
            marker=dict(symbol='triangle-down', size=10, color='red'),
            name='Swing High'
        ))
        
        # Plot Swing Lows
        fig.add_trace(go.Scatter(
            x=swings.index, 
            y=swings['swing_low'],
            mode='markers',
            marker=dict(symbol='triangle-up', size=10, color='green'),
            name='Swing Low'
        ))
```

## 4. Execution Checklist

1.  [ ] **Wait** for Phase 1-3 (Remediation) completion.
2.  [ ] Create `features/structure.py` with the code provided above.
3.  [ ] Update `features/engine.py` to import and call structure functions.
4.  [ ] Update `ui/dashboard.py` to add toggles for "Structure" & "FVG" and render the shapes.
5.  [ ] Run `python -m ui.dashboard` and verify the visual overlays align with price action.