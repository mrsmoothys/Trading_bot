"""
Trading Dashboard - Improved with Backtest Lab Fixes
- View Details modal with equity curve, trade list, heatmaps
- Data integrity - no sample fallback for backtests
- Loading states for selectors
- Data source badges
"""

import asyncio
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output, State, callback, callback_context
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, Tuple, Optional, List
from loguru import logger
import os
import time
import requests
import json
from dotenv import load_dotenv

# Load environment variables immediately
load_dotenv()
IS_TEST_MODE = os.getenv("DASH_TEST_MODE") == "1"

# Import command router for Step 18 - DeepSeek as Manager
from ui.chat_command_router import route_chat_message, set_dashboard_callbacks, get_manager_status
from core.config import config


# Global state for tracking selections
GLOBAL_STATE = {
    'selected_timeframe': '15m',
    'selected_symbol': 'BTCUSDT',
    'active_features': {
        'liquidity': True,
        'supertrend': True,
        'chandelier': True,
        'orderflow': True,
        'regime': True,
        'alignment': False
    }
}

DATA_CACHE: Dict[Tuple[str, str, int], Dict[str, Any]] = {}
CACHE_TTL_SECONDS = int(os.getenv('DASH_DATA_CACHE_TTL', '10'))
USE_SAMPLE_DATA = os.getenv('DASH_USE_SAMPLE_DATA', '0') == '1'

LAST_TOGGLE = {'ts': 0.0}
REGIME_FEATURE_STRATEGY = {
    "TRENDING_HIGH_VOL": ["supertrend", "chandelier"],
    "TRENDING_LOW_VOL": ["supertrend"],
    "RANGING_COMPRESSION": ["liquidity", "orderflow"],
    "RANGING_EXPANSION": ["liquidity", "orderflow", "regime"],
    "TRANSITION": ["orderflow"],
}

CHAT_CLIENT = None
CHAT_SYSTEM_CONTEXT = None
CHAT_INIT_ERROR = None
LIVE_REFRESH_MIN = int(os.getenv('DASH_LIVE_REFRESH_MIN', '1'))
CHAT_RESPONSE_TIMEOUT = int(os.getenv("CHAT_RESPONSE_TIMEOUT", "60"))  # Updated to 60s
# Timeframe minutes helper for freshness/metadata
TF_MINUTES = {'1m': 1, '5m': 5, '15m': 15, '1h': 60, '4h': 240, '1d': 1440}

# Basic feature metrics renderer (used in tests)
def create_feature_metrics_table(
    metrics: Dict[str, Dict[str, Any]],
    current_regime: str = "UNKNOWN",
    recommendations: List[str] = None
) -> html.Div:
    """Render telemetry table for feature latency/memory usage."""
    if not metrics:
        return html.Div(
            "Telemetry pending — run the trading loop to capture feature metrics.",
            style={'color': '#666', 'fontStyle': 'italic', 'padding': '10px'}
        )

    header = html.Tr([
        html.Th("Feature"),
        html.Th("Latency (ms)"),
        html.Th("Memory Δ (MB)"),
        html.Th("Timestamp")
    ])
    rows = []
    for name, data in metrics.items():
        rows.append(html.Tr([
            html.Td(name.title()),
            html.Td(f"{data.get('latency_ms', 0):.2f}"),
            html.Td(f"{data.get('memory_delta_mb', 0):.2f}"),
            html.Td(str(data.get('timestamp', '')).split('T')[0])
        ]))

    table = html.Table([header] + rows, style={
        'width': '100%',
        'borderCollapse': 'collapse',
        'fontSize': '13px'
    })
    overlay_text = ', '.join(recommendations) if recommendations else 'None'
    extra = html.P(
        f"Current regime: {current_regime} | Recommended overlays: {overlay_text}",
        style={'color': '#666', 'fontSize': '12px', 'marginTop': '10px'}
    )
    return html.Div([table, extra])

# Indicator helpers (used by tests)
def calculate_supertrend(df: pd.DataFrame, period: int = 10, multiplier: float = 3.0) -> Dict[str, Any]:
    """Calculate Supertrend indicator."""
    close = df['close']
    high = df['high']
    low = df['low']

    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()

    hl2 = (high + low) / 2
    upper_band = hl2 + (multiplier * atr)
    lower_band = hl2 - (multiplier * atr)

    supertrend = pd.Series(index=df.index, dtype=float)
    direction = pd.Series(index=df.index, dtype=int)  # 1 for uptrend, -1 for downtrend

    supertrend.iloc[0] = upper_band.iloc[0]
    direction.iloc[0] = 1

    for i in range(1, len(df)):
        if close.iloc[i] > supertrend.iloc[i-1]:
            direction.iloc[i] = 1
            supertrend.iloc[i] = max(lower_band.iloc[i], supertrend.iloc[i-1])
        else:
            direction.iloc[i] = -1
            supertrend.iloc[i] = min(upper_band.iloc[i], supertrend.iloc[i-1])

    return {
        'supertrend': supertrend,
        'direction': direction,
        'atr': atr
    }

def calculate_chandelier_exit(df: pd.DataFrame, period: int = 22, multiplier: float = 3.0) -> Dict[str, Any]:
    """Calculate Chandelier Exit indicator."""
    close = df['close']
    high = df['high']
    low = df['low']

    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()

    long_exit = high.rolling(period).max() - (multiplier * atr)
    short_exit = low.rolling(period).min() + (multiplier * atr)

    return {
        'long_exit': long_exit,
        'short_exit': short_exit,
        'atr': atr
    }

def calculate_liquidity_zones(
    df: pd.DataFrame,
    lookback: int = 100,
    price_bins: int = 50,
    min_zone_spacing_pct: float = 0.5,
    percentile_threshold: float = 95.0,
    recent_weight_multiplier: float = 1.5
) -> Dict[str, Any]:
    """
    Wrapper to compute liquidity zones using features.engine.

    Enhanced with support for new parameters:
    - lookback: Number of periods to analyze (default: 100)
    - price_bins: Number of price bins for distribution (default: 50)
    - min_zone_spacing_pct: Minimum spacing between zones in % (default: 0.5)
    - percentile_threshold: Percentile for significant volume (default: 95.0)
    - recent_weight_multiplier: Recent periods weight multiplier (default: 1.5)
    """
    from features.engine import calculate_liquidity_zones as calc_lz

    res = calc_lz(
        high=df['high'],
        low=df['low'],
        close=df['close'],
        volume=df['volume'],
        lookback_periods=lookback,
        price_bins=price_bins,
        min_zone_spacing_pct=min_zone_spacing_pct,
        percentile_threshold=percentile_threshold,
        recent_weight_multiplier=recent_weight_multiplier
    )

    # Normalize key expected by tests
    if 'zones' not in res:
        res['zones'] = res.get('liquidity_zones', [])
    if 'volume_profile' not in res:
        # Enhanced volume profile from algorithm
        if 'volume_stats' in res:
            res['volume_profile'] = {
                'mean': res['volume_stats']['mean'],
                'max': res['volume_stats']['max'],
                'concentration': res['volume_stats']['concentration']
            }
        else:
            # Fallback
            res['volume_profile'] = [
                {'price': float(p), 'volume': float(v)}
                for p, v in zip(df['close'].head(50), df['volume'].head(50))
            ]
    return res

def calculate_market_regime_overlay(df: pd.DataFrame, short_window: int = 20, long_window: int = 60) -> Optional[Dict[str, Any]]:
    """Classify market regime for shading overlays."""
    if len(df) < long_window:
        return None
    returns = df['close'].pct_change()
    realized_vol = returns.rolling(long_window).std().fillna(0)
    vol_threshold = realized_vol.median()
    sma_short = df['close'].rolling(short_window).mean()
    sma_long = df['close'].rolling(long_window).mean()
    trend_strength = (sma_short - sma_long).fillna(0)
    regime_series = pd.Series('RANGING_LOW_VOL', index=df.index)
    regime_series[(trend_strength > 0) & (realized_vol >= vol_threshold)] = 'TRENDING_UP_HIGH_VOL'
    regime_series[(trend_strength > 0) & (realized_vol < vol_threshold)] = 'TRENDING_UP_LOW_VOL'
    regime_series[(trend_strength < 0) & (realized_vol >= vol_threshold)] = 'TRENDING_DOWN_HIGH_VOL'
    regime_series[(trend_strength < 0) & (realized_vol < vol_threshold)] = 'TRENDING_DOWN_LOW_VOL'
    colors = {
        'TRENDING_UP_HIGH_VOL': 'rgba(0, 255, 136, 0.08)',
        'TRENDING_UP_LOW_VOL': 'rgba(0, 136, 255, 0.08)',
        'TRENDING_DOWN_HIGH_VOL': 'rgba(255, 68, 68, 0.12)',
        'TRENDING_DOWN_LOW_VOL': 'rgba(255, 165, 0, 0.1)',
        'RANGING_LOW_VOL': 'rgba(255, 255, 255, 0.03)'
    }
    return {'series': regime_series, 'colors': colors}

def calculate_timeframe_alignment(df: pd.DataFrame) -> Optional[Dict[str, pd.Series]]:
    """Compute multi-EMA alignment signals for overlay markers."""
    if len(df) < 55:
        return None
    ema_fast = df['close'].ewm(span=10, adjust=False).mean()
    ema_mid = df['close'].ewm(span=21, adjust=False).mean()
    ema_slow = df['close'].ewm(span=55, adjust=False).mean()
    bullish = (ema_fast > ema_mid) & (ema_mid > ema_slow)
    bearish = (ema_fast < ema_mid) & (ema_mid < ema_slow)
    return {'bullish': bullish.fillna(False), 'bearish': bearish.fillna(False)}


def set_system_context(context):
    """Set the global system context."""
    global CHAT_SYSTEM_CONTEXT
    CHAT_SYSTEM_CONTEXT = context


def ensure_utc_timestamp(value) -> pd.Timestamp:
    """Convert timestamps to UTC-aware pandas Timestamp."""
    ts = pd.to_datetime(value)
    if ts.tzinfo is None or ts.tzinfo.utcoffset(ts) is None:
        return ts.tz_localize('UTC')
    return ts.tz_convert('UTC')


def run_async_task(async_func, *args, **kwargs):
    """Execute an async callable from sync contexts with graceful fallback."""
    timeout = kwargs.pop("timeout", None)

    async def runner():
        coro = async_func(*args, **kwargs)
        if timeout:
            return await asyncio.wait_for(coro, timeout)
        return await coro

    try:
        return asyncio.run(runner())
    except RuntimeError as exc:
        # Happens if another loop is already running
        if "asyncio.run()" in str(exc):
            loop = asyncio.new_event_loop()
            try:
                asyncio.set_event_loop(loop)
                return loop.run_until_complete(runner())
            finally:
                asyncio.set_event_loop(None)
                loop.close()
        raise


def fetch_market_data(symbol: str, timeframe: str, num_bars: int = 1000, force_refresh: bool = False) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Fetch market data with improved caching and sample data detection.
    Returns (DataFrame, metadata)
    """
    # For pytest/UI smoke runs, avoid real network calls and use synthetic data
    if os.getenv("DASH_TEST_MODE") == "1":
        df = generate_sample_data(symbol, timeframe, num_bars)
        return df, {'used_sample_data': True, 'last_candle_age_min': 0}

    from core.data.data_store import DataStore
    from core.data.binance_client import BinanceClient

    cache_key = f"{symbol}:{timeframe}:{num_bars}"
    current_time = time.time()

    # Check memory cache
    if not force_refresh and cache_key in DATA_CACHE:
        cached_item = DATA_CACHE[cache_key]
        if current_time - cached_item['timestamp'] < CACHE_TTL_SECONDS:
            return cached_item['data'], {'used_sample_data': cached_item.get('used_sample', False), 'last_candle_age_min': 0}

    # Initialize DataStore
    try:
        data_store = DataStore()
    except Exception as e:
        logger.error(f"Failed to initialize DataStore: {e}")
        # Fall back to basic cache
        data_store = None

    # Fetch data with error handling
    ohlcv_data, used_sample_data = get_real_market_data(symbol, timeframe, num_bars)

    # Update memory cache
    DATA_CACHE[cache_key] = {
        'data': ohlcv_data,
        'timestamp': current_time,
        'used_sample': used_sample_data
    }

    # Calculate last candle age
    last_candle_age_min = 0
    if len(ohlcv_data) > 0:
        last_time = pd.to_datetime(ohlcv_data['timestamp'].iloc[-1])
        last_candle_age_min = (datetime.now() - last_time.replace(tzinfo=None)).total_seconds() / 60

    return ohlcv_data, {'used_sample_data': used_sample_data, 'last_candle_age_min': last_candle_age_min}


def get_real_market_data(symbol: str, timeframe: str, num_bars: int) -> Tuple[pd.DataFrame, bool]:
    """
    Fetch real market data. FOR BACKTESTS: This should never fall back to sample data.
    Returns (DataFrame, used_sample_data_bool)
    """
    # Check environment variable for backtest mode
    is_backtest = os.getenv('BACKTEST_MODE', 'false').lower() == 'true'

    try:
        from core.data.binance_client import BinanceClient
        import asyncio

        async def _fetch():
            client = BinanceClient()
            return await client.get_ohlcv(symbol, timeframe, limit=num_bars)

        # Run async fetch
        ohlcv_data = run_async_task(_fetch)

        if ohlcv_data is None or len(ohlcv_data) == 0:
            if is_backtest:
                raise ValueError(f"No real data available for {symbol} {timeframe}. Backtests cannot use sample data.")
            logger.warning("No real data available, generating sample data")
            return generate_sample_data(symbol, timeframe, num_bars), True

        return ohlcv_data, False

    except Exception as e:
        logger.error(f"Error fetching real market data: {e}")

        if is_backtest:
            # HARD FAIL for backtests - no sample data fallback
            raise ValueError(
                f"CRITICAL: Backtest failed - unable to fetch live data for {symbol} {timeframe}. "
                f"Error: {str(e)}. Backtests require live data and cannot use sample data."
            )

        logger.info("Falling back to sample data (non-backtest mode)")
        return generate_sample_data(symbol, timeframe, num_bars), True


def generate_sample_data(symbol: str, timeframe: str, num_bars: int) -> pd.DataFrame:
    """Generate synthetic OHLCV data for demonstration."""
    # Base prices by symbol
    base_prices = {
        'BTCUSDT': 50000,
        'ETHUSDT': 3000,
        'SOLUSDT': 100,
        'ADAUSDT': 0.5,
        'DOTUSDT': 7
    }

    base_price = base_prices.get(symbol, 50000)

    # Create timestamps
    interval_minutes = {
        '1m': 1, '5m': 5, '15m': 15, '30m': 30,
        '1h': 60, '4h': 240, '1d': 1440
    }.get(timeframe, 60)

    end_time = datetime.now()
    start_time = end_time - timedelta(minutes=interval_minutes * num_bars)

    # Generate synthetic price data
    np.random.seed(42)  # For reproducibility
    dates = pd.date_range(start=start_time, end=end_time, freq=f'{interval_minutes}T')

    # Generate random walk with slight upward trend
    returns = np.random.normal(0.0005, 0.02, len(dates))  # 0.05% drift, 2% volatility
    prices = base_price * (1 + returns).cumprod()

    # Create OHLCV data
    df = pd.DataFrame(index=range(len(dates)))
    df['timestamp'] = dates
    df['open'] = prices
    df['high'] = prices * (1 + np.abs(np.random.normal(0, 0.005, len(dates))))
    df['low'] = prices * (1 - np.abs(np.random.normal(0, 0.005, len(dates))))
    df['close'] = prices
    df['volume'] = np.random.uniform(100, 1000, len(dates))

    # Ensure OHLC integrity
    df['high'] = np.maximum(np.maximum(df['open'], df['close']), df['high'])
    df['low'] = np.minimum(np.minimum(df['open'], df['close']), df['low'])

    return df


def create_interactive_chart(df: pd.DataFrame, symbol: str, timeframe: str, features: Dict[str, bool]) -> go.Figure:
    """Create interactive Plotly chart with overlays."""
    # Multi-panel setup
    fig = make_subplots(
        rows=3, cols=1,
        row_heights=[0.70, 0.15, 0.15],
        shared_xaxes=True,
        vertical_spacing=0.02,
        subplot_titles=('Price', 'Volume', 'Order Flow')
    )

    # Price chart (candlestick)
    fig.add_trace(
        go.Candlestick(
            x=df['timestamp'],
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name='Price',
            increasing_line_color='#26a69a',
            decreasing_line_color='#ef5350'
        ),
        row=1, col=1
    )

    # Volume chart
    fig.add_trace(
        go.Bar(
            x=df['timestamp'],
            y=df['volume'],
            name='Volume',
            marker_color='#42a5f5',
            opacity=0.7
        ),
        row=2, col=1
    )

    # Order Flow (if enabled)
    # Order Flow (if enabled)
    if features.get('orderflow', False):
        # Real Order Flow Calculation (based on features/engine.py)
        # Calculate candle components
        body_size = (df['close'] - df['open']).abs()
        total_range = df['high'] - df['low']
        upper_wick = df['high'] - df[['open', 'close']].max(axis=1)
        
        # Buying pressure: strong closes with high volume
        buying_pressure = (
            (df['close'] > df['open']) & (body_size > total_range * 0.6)
        ).astype(int)

        # Selling pressure: strong rejections at highs or bearish closes
        selling_pressure = (
            (upper_wick > body_size * 1.5) | (df['close'] < df['open'] * 0.998)
        ).astype(int)

        # Calculate volume-weighted imbalance
        # We use a rolling sum to smooth it slightly for the chart, similar to the engine's imbalance_ratio but keeping the series
        volume_imbalance = (buying_pressure - selling_pressure) * df['volume']
        
        # Use the raw volume imbalance for the bar chart
        order_flow = volume_imbalance
        
        colors = ['#26a69a' if x >= 0 else '#ef5350' for x in order_flow]

        fig.add_trace(
            go.Bar(
                x=df['timestamp'],
                y=order_flow,
                name='Order Flow',
                marker_color=colors,
                opacity=0.7
            ),
            row=3, col=1
        )

    # Overlays
    shapes = []

    if features.get('liquidity', False):
        zones = calculate_liquidity_zones(
            df,
            lookback=50,  # Reduced from 100 to focus on recent action
            percentile_threshold=90.0  # Reduced from 95.0 to catch more local zones
        )
        for zone in zones.get('zones', zones.get('levels', []))[:6]:
            price = zone['price'] if isinstance(zone, dict) else zone
            shapes.append(dict(
                type="line",
                xref="x1",
                yref="y1",
                x0=df['timestamp'].min(),
                x1=df['timestamp'].max(),
                y0=price,
                y1=price,
                line=dict(color="orange", dash="dash")
            ))

    if features.get('supertrend', False):
        st_data = calculate_supertrend(df)
        fig.add_trace(
            go.Scatter(
                x=df['timestamp'],
                y=st_data['supertrend'],
                mode='lines',
                name='Supertrend'
            ),
            row=1, col=1
        )

    if features.get('chandelier', False):
        ce = calculate_chandelier_exit(df)
        fig.add_trace(go.Scatter(x=df['timestamp'], y=ce['long_exit'], mode='lines', name='Chandelier Long'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df['timestamp'], y=ce['short_exit'], mode='lines', name='Chandelier Short'), row=1, col=1)

    if features.get('regime', False):
        regime = calculate_market_regime_overlay(df)
        if regime and 'series' in regime:
            shapes.append(dict(
                type="rect",
                xref="paper",
                yref="paper",
                x0=0, x1=1, y0=0.0, y1=1.0,
                fillcolor="rgba(255,255,255,0.02)",
                layer="below",
                line=dict(width=0)
            ))

    if features.get('alignment', False):
        fig.add_trace(go.Scatter(x=df['timestamp'], y=df['close'], mode='markers', name='Alignment Bullish', marker=dict(color='green', size=4), visible='legendonly'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df['timestamp'], y=df['close'], mode='markers', name='Alignment Bearish', marker=dict(color='red', size=4), visible='legendonly'), row=1, col=1)

    # Update layout
    fig.update_layout(
        title=f"{symbol} - {timeframe} Timeframe",
        xaxis_title="Time",
        height=800,
        showlegend=True,
        dragmode="pan",
        template="plotly_dark",
        paper_bgcolor="#111",
        plot_bgcolor="#111",
        margin=dict(l=50, r=50, t=80, b=50)
    )
    if shapes:
        fig.update_layout(shapes=shapes)

    fig.update_yaxes(title_text="Price (USDT)", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    fig.update_yaxes(title_text="Order Flow", row=3, col=1)

    return fig


def create_detailed_backtest_view(result, config) -> html.Div:
    """Create detailed backtest analysis with charts and tables."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    # Equity Curve
    if result.equity_curve:
        equity_df = pd.DataFrame(result.equity_curve)
        equity_fig = go.Figure()
        equity_fig.add_trace(
            go.Scatter(
                x=pd.to_datetime(equity_df['timestamp']),
                y=equity_df['equity'],
                mode='lines',
                name='Equity',
                line=dict(color='#26a69a', width=2)
            )
        )
        equity_fig.update_layout(
            title='Equity Curve',
            template='plotly_dark',
            height=300,
            margin=dict(l=20, r=20, t=40, b=20)
        )
    else:
        equity_fig = go.Figure()
        equity_fig.add_annotation(text="No equity curve data available", showarrow=False)
        equity_fig.update_layout(template='plotly_dark', height=300)

    # Trade List Table
    if result.trades:
        trade_rows = []
        for i, trade in enumerate(result.trades[:50], 1):  # Show first 50 trades
            pnl = trade.get('pnl', 0)
            pnl_percent = trade.get('pnl_percent', 0)
            color = "success" if pnl > 0 else "danger"

            trade_rows.append(
                html.Tr([
                    html.Td(str(i)),
                    html.Td(trade.get('entry_time', '')[:19]),
                    html.Td(f"{trade.get('side', '')}"),
                    html.Td(f"${trade.get('entry_price', 0):.2f}"),
                    html.Td(f"${trade.get('exit_price', 0):.2f}"),
                    html.Td(f"${pnl:.2f}", className=f"text-{color}"),
                    html.Td(f"{pnl_percent:.2f}%", className=f"text-{color}"),
                ])
            )

        trade_table = dbc.Table(
            [
                html.Thead(
                    html.Tr([
                        html.Th("#"),
                        html.Th("Entry Time"),
                        html.Th("Side"),
                        html.Th("Entry"),
                        html.Th("Exit"),
                        html.Th("P&L"),
                        html.Th("P&L %")
                    ])
                ),
                html.Tbody(trade_rows)
            ],
            striped=True,
            hover=True,
            size='sm'
        )
    else:
        trade_table = html.P("No trades recorded")

    # Monthly Performance Heatmap (simplified)
    if result.trades:
        # Calculate monthly P&L
        trade_dates = [datetime.fromisoformat(t['entry_time']) for t in result.trades]
        trade_pnls = [t.get('pnl', 0) for t in result.trades]

        monthly_data = {}
        for date, pnl in zip(trade_dates, trade_pnls):
            month_key = date.strftime('%Y-%m')
            monthly_data[month_key] = monthly_data.get(month_key, 0) + pnl

        # Create heatmap data
        months = sorted(monthly_data.keys())
        pnls = [monthly_data[m] for m in months]

        heatmap_fig = go.Figure(
            data=go.Bar(
                x=months,
                y=pnls,
                marker_color=['#26a69a' if x >= 0 else '#ef5350' for x in pnls]
            )
        )
        heatmap_fig.update_layout(
            title='Monthly P&L',
            template='plotly_dark',
            height=200,
            margin=dict(l=20, r=20, t=40, b=20)
        )
    else:
        heatmap_fig = go.Figure()
        heatmap_fig.add_annotation(text="No trade data for monthly analysis", showarrow=False)
        heatmap_fig.update_layout(template='plotly_dark', height=200)

    # Risk Metrics
    risk_stats = dbc.Row([
        dbc.Col([
            html.H6("Risk Metrics"),
            html.P(f"Sharpe Ratio: {result.sharpe_ratio:.2f}"),
            html.P(f"Profit Factor: {result.profit_factor:.2f}"),
            html.P(f"Max Drawdown: {result.max_drawdown:.2f}%"),
            html.P(f"Win Rate: {result.win_rate:.1%}"),
        ], md=3),
        dbc.Col([
            html.H6("Trade Statistics"),
            html.P(f"Total Trades: {result.total_trades}"),
            html.P(f"Winners: {result.winning_trades}"),
            html.P(f"Losers: {result.losing_trades}"),
            html.P(f"Avg Win: ${result.avg_win:.2f}"),
            html.P(f"Avg Loss: ${result.avg_loss:.2f}"),
        ], md=3),
    ])

    # Export buttons
    export_buttons = dbc.ButtonGroup([
        dbc.Button([
            html.I(className="fas fa-download me-2"),
            "Export CSV"
        ], id='export-csv-btn', color='secondary', size='sm'),
        dbc.Button([
            html.I(className="fas fa-file-pdf me-2"),
            "Export PDF"
        ], id='export-pdf-btn', color='secondary', size='sm'),
    ])

    return html.Div([
        dbc.Row([
            dbc.Col([
                html.H5("Backtest Details", className="mb-3"),
                html.P([
                    html.I(className="fas fa-info-circle me-2"),
                    f"Strategy: {config.get('strategy', 'Unknown')} | "
                    f"Symbol: {config.get('symbol', 'Unknown')} | "
                    f"Timeframe: {config.get('timeframe', 'Unknown')} | "
                    f"Period: {config.get('start', '')} → {config.get('end', '')}"
                ], className="text-muted mb-3")
            ])
        ]),
        dbc.Row([
            dbc.Col([dcc.Graph(figure=equity_fig)], md=8),
            dbc.Col([
                html.H6("Summary", className="mb-2"),
                html.H3(f"{result.total_return_pct:+.2f}%", className="text-success" if result.total_return_pct >= 0 else "text-danger"),
                html.P(f"Initial: ${result.initial_capital:,.2f}"),
                html.P(f"Final: ${result.final_capital:,.2f}"),
                html.Hr(),
                risk_stats
            ], md=4)
        ], className='mb-4'),
        dbc.Row([
            dbc.Col([dcc.Graph(figure=heatmap_fig)], md=12)
        ], className='mb-4'),
        dbc.Row([
            dbc.Col([
                html.H6("Trade List (Latest 50)", className="mb-2"),
                trade_table
            ], md=12)
        ], className='mb-3'),
        dbc.Row([
            dbc.Col([export_buttons], md=12, className='d-flex justify-content-end')
        ])
    ])


def create_dashboard_app():
    """Create and configure the Dash app."""
    app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
    app.title = "DeepSeek Trading Dashboard"

    # Main layout
    layout = dbc.Container([
        # Header
        dbc.Row([
            dbc.Col([
                html.H1("DeepSeek Trading System", className="display-4"),
                html.P("Autonomous AI-Powered Trading", className="lead"),
                # Status badges
                html.Div([
                    dbc.Badge("LIVE", color="success", className="me-2", id="live-status-badge"),
                    dbc.Badge("DEMO", color="warning", className="me-2", id="demo-status-badge", style={'display': 'none'}),
                    dbc.Badge("Data Fresh: < 30s", color="info", className="me-2", id="freshness-badge"),
                    dbc.Badge("Age: 0.0m", color="secondary", className="me-2", id="last-age-badge"),
                    dbc.Badge("Sentiment: --", color="dark", id="sentiment-badge"),
                ], className="mb-3")
            ])
        ], className="p-4 bg-primary text-white rounded mb-4"),

        # Tabs
        dbc.Tabs(id='main-tabs', active_tab='tab-market', className='mb-3', children=[
            # Tab 1: Market Analysis
            dbc.Tab(label="Market Analysis", tab_id="tab-market", children=[
                dbc.Card([
                    dbc.CardBody([
                        # Timeframe buttons
                        dbc.Row([
                            dbc.Col([
                                html.H5("Symbol", className="mb-3"),
                                dbc.Select(
                                    id='symbol-selector',
                                    options=[{'label': s, 'value': s} for s in config.symbols],
                                    value=config.symbols[0] if config.symbols else 'BTCUSDT',
                                    className="mb-3"
                                ),
                                html.H5("Timeframe", className="mb-3"),
                                dbc.ButtonGroup([
                                    dbc.Button("1m", id="tf-1m", n_clicks=0, color='outline-primary', size='sm'),
                                    dbc.Button("5m", id="tf-5m", n_clicks=0, color='outline-primary', size='sm'),
                                    dbc.Button("15m", id="tf-15m", n_clicks=0, color='outline-primary', size='sm'),
                                    dbc.Button("1h", id="tf-1h", n_clicks=0, color='outline-primary', size='sm'),
                                    dbc.Button("4h", id="tf-4h", n_clicks=0, color='outline-primary', size='sm'),
                                    dbc.Button("1d", id="tf-1d", n_clicks=0, color='outline-primary', size='sm'),
                                ], size='lg', className='mb-3')
                            ], md=6),
                            dbc.Col([
                                html.H5("Overlays", className="mb-3"),
                                dbc.Checklist(
                                    id='overlay-toggles',
                                    options=[
                                        {"label": "Liquidity", "value": "liquidity"},
                                        {"label": "Order Flow", "value": "orderflow"},
                                        {"label": "Supertrend", "value": "supertrend"},
                                        {"label": "Chandelier", "value": "chandelier"},
                                    ],
                                    value=["liquidity", "orderflow", "supertrend", "chandelier"],
                                    switch=True
                                )
                            ], md=6)
                        ], className='mb-3'),

                        # Refresh button
                        dbc.Row([
                            dbc.Col([
                                dbc.Button([
                                    html.I(className="fas fa-sync-alt me-2"), "Refresh now"
                                ], id='refresh-data-btn', color='primary', n_clicks=0)
                            ], md=12)
                        ], className='mb-3'),

                        # Main chart
                        html.Div([
                            dcc.Graph(
                                id='main-chart',
                                style={'height': '800px'},
                                config={
                                    "scrollZoom": True,
                                    "doubleClick": "reset",
                                    "displaylogo": False,
                                    "modeBarButtonsToRemove": ["select2d", "lasso2d"],
                                    "modeBarButtonsToAdd": ["pan2d"],
                                },
                            )
                        ], className='mb-3'),

                        # Range selectors
                        html.Div([
                            html.P("Range:", className="mb-2"),
                            dbc.ButtonGroup([
                                dbc.Button("50", id='range-50', size='sm'),
                                dbc.Button("100", id='range-100', size='sm'),
                                dbc.Button("200", id='range-200', size='sm'),
                                dbc.Button("All", id='range-all', size='sm'),
                            ])
                        ], className='mb-3'),
                        dcc.RangeSlider(
                            id='chart-range-slider',
                            min=0,
                            max=100,
                            step=1,
                            value=[75, 100],
                            tooltip={"placement": "bottom", "always_visible": False}
                        ),

                        # Convergence Strategy Panel
                        html.Hr(),
                        html.H4("Multi-Timeframe Convergence Strategy", className="mb-3"),
                        dbc.Card([
                            dbc.CardBody([
                                dbc.Row([
                                    dbc.Col([
                                        html.P("Alignment Score:", className="mb-1"),
                                        html.H3(id='alignment-score', children="--"),
                                        html.P("Market Regime:", className="mb-1 mt-3"),
                                        html.H5(id='market-regime', children="--"),
                                    ], md=3),
                                    dbc.Col([
                                        html.P("Entry Level:", className="mb-1"),
                                        html.H5(id='entry-level', children="--"),
                                        html.P("Stop Loss:", className="mb-1 mt-3"),
                                        html.H5(id='stop-loss', children="--"),
                                    ], md=3),
                                    dbc.Col([
                                        html.P("Take Profit:", className="mb-1"),
                                        html.H5(id='take-profit', children="--"),
                                        html.P("Position Size:", className="mb-1 mt-3"),
                                        html.H5(id='position-size', children="--"),
                                    ], md=3),
                                    dbc.Col([
                                        html.Br(),
                                        dbc.Button([
                                            html.I(className="fas fa-play me-2"),
                                            "Run Convergence Strategy"
                                        ], id='run-convergence-btn', color='success', size='lg'),
                                    ], md=3)
                                ])
                            ])
                        ], className='shadow-sm')
                    ])
                ], className='shadow')
            ]),

            # Tab 2: Account & Trading
            dbc.Tab(label="Account & Trading", tab_id="tab-account", children=[
                dbc.Card([
                    dbc.CardHeader(html.H4("Account Status", className="mb-0")),
                    dbc.CardBody([
                        html.H2("$100,000", id="account-portfolio-value"),
                        html.P("Trading account information will appear here."),
                        html.P("This is a demo/placeholder tab.")
                    ])
                ], className='shadow-sm')
            ]),

            # Tab 3: Backtest Lab
            dbc.Tab(label="Backtest Lab", tab_id="tab-backtest", children=[
                dbc.Card([
                    dbc.CardHeader(html.H4("Strategy Backtesting Lab", className="mb-0")),
                    dbc.CardBody([
                        # Data Source Badge Row
                        dbc.Row([
                            dbc.Col([
                                html.Div(id='backtest-data-source-badge', className='mb-3')
                            ], md=12)
                        ]),
                        dbc.Row([
                            dbc.Col([
                                dbc.Label("Trading Symbol"),
                                dbc.Select(
                                    id='backtest-symbol',
                                    options=[{'label': s, 'value': s} for s in config.symbols],
                                    value=config.symbols[0] if config.symbols else 'BTCUSDT'
                                ),
                                # Loading spinner
                                html.Div(id='symbol-loading', children=[], className='mt-1'),
                            ], md=3),
                            dbc.Col([
                                dbc.Label("Timeframe"),
                                dbc.Select(
                                    id='backtest-timeframe',
                                    options=[
                                        {'label': '1 Minute', 'value': '1m'},
                                        {'label': '5 Minutes', 'value': '5m'},
                                        {'label': '15 Minutes', 'value': '15m'},
                                        {'label': '1 Hour', 'value': '1h'},
                                        {'label': '4 Hours', 'value': '4h'},
                                        {'label': '1 Day', 'value': '1d'},
                                    ],
                                    value='1h'
                                ),
                                html.Div(id='timeframe-loading', children=[], className='mt-1'),
                            ], md=3),
                            dbc.Col([
                                dbc.Label("Strategy"),
                                dbc.Select(
                                    id='backtest-strategy',
                                    options=[
                                        {'label': 'Convergence Strategy', 'value': 'convergence'},
                                        {'label': 'Scalp 15m/4h', 'value': 'scalp_15m_4h'},
                                        {'label': 'MA Crossover', 'value': 'ma_crossover'},
                                        {'label': 'RSI Divergence', 'value': 'rsi_divergence'},
                                    ],
                                    value='convergence'
                                ),
                                html.Div(id='strategy-loading', children=[], className='mt-1'),
                            ], md=3),
                            dbc.Col([
                                dbc.Label("Initial Capital ($)"),
                                dbc.Input(id='backtest-capital', type='number', value=10000, min=1000, max=1000000, step=1000),
                            ], md=3),
                        ], className='mb-3'),
                        dbc.Row([
                            dbc.Col([
                                dbc.Label("Start Date"),
                                dcc.DatePickerSingle(
                                    id='backtest-start-date',
                                    date=datetime(2024, 1, 1),
                                    display_format='YYYY-MM-DD'
                                ),
                            ], md=3),
                            dbc.Col([
                                dbc.Label("End Date"),
                                dcc.DatePickerSingle(
                                    id='backtest-end-date',
                                    date=datetime.now(),
                                    display_format='YYYY-MM-DD'
                                ),
                            ], md=3),
                            dbc.Col([
                                html.Br(),
                                dbc.ButtonGroup([
                                    dbc.Button([
                                        html.I(className="fas fa-play me-2"), "Run Backtest"
                                    ], id='run-backtest-btn', color='primary', n_clicks=0),
                                    dbc.Button([
                                        html.I(className="fas fa-trophy me-2"), "Promote to Live"
                                    ], id='promote-strategy-btn', color='success', n_clicks=0, disabled=True),
                                ], size='lg'),
                            ], md=6, className='d-flex align-items-end'),
                        ], className='mb-3'),
                        html.Hr(),
                        dbc.Alert([
                            html.I(className="fas fa-info-circle me-2"),
                            "Backtest results will appear here. Select parameters and click 'Run Backtest' to begin."
                        ], color='info', id='backtest-status'),
                        html.Div(id='chat-backtest-summary', className='mb-3'),
                        html.Div(id='backtest-results-container'),
                    ]),
                ], className='shadow-sm'),
                # Results Modal
                dbc.Modal([
                    dbc.ModalHeader(dbc.ModalTitle("Backtest Results")),
                    dbc.ModalBody(id='backtest-results-modal-body'),
                    dbc.ModalFooter([
                        dbc.Button("Export Report", id='export-backtest-btn', color='secondary', className='me-auto'),
                        dbc.Button("Close", id='backtest-modal-close-btn', color='primary', n_clicks=0),
                    ]),
                ], id='backtest-results-modal', size='xl', is_open=False),
            ])
        ]),

        # Chat Interface - Moved outside Tabs for proper Dash attribute rendering
        dbc.Card([
            dbc.CardHeader(html.H4("DeepSeek AI Assistant", className="mb-0")),
            dbc.CardBody([
                dcc.Textarea(
                    id="chat-history",
                    readOnly=True,
                    style={"width": "100%", "height": "180px"},
                    value=""
                ),
                html.Div([
                    dcc.Input(
                        id="chat-input",
                        type="text",
                        placeholder="Ask about market structure...",
                        style={"width": "70%", "marginRight": "8px"}
                    ),
                    dbc.Button("Send", id="chat-send-btn", color="primary")
                ], className="mt-2")
            ])
        ], className='shadow mt-3 mb-4'),

        # Intervals
        dcc.Interval(id='interval-fast', interval=2000, n_intervals=0, disabled=IS_TEST_MODE),
        dcc.Interval(id='interval-slow', interval=10000, n_intervals=0, disabled=IS_TEST_MODE),

        # Stores
        dcc.Store(id='timeframe-store', data='15m'),
        dcc.Store(id='data-source-store', data={'used_sample_data': False, 'last_update': None}),
        dcc.Store(id='backtest-result-store'),
        dcc.Store(id='backtest-history-store', data=[]),
        dcc.Store(id='chat-history-store', data=[]),
        dcc.Store(id='chat-backtest-store'),
        dcc.Store(id='manager-actions-store', data={'last_command': None, 'last_result': None}),

        # Modals
        dbc.Modal([
            dbc.ModalHeader(dbc.ModalTitle(id='backtest-modal-title')),
            dbc.ModalBody(id='backtest-modal-body'),
            dbc.ModalFooter(dbc.Button("Close", id='backtest-modal-close', className="ms-auto", n_clicks=0))
        ], id="backtest-modal", size="lg", is_open=False),

    ], fluid=True, className="p-4")

    app.layout = layout

    # Status Badge Callback
    @app.callback(
        [Output('live-status-badge', 'style'),
         Output('demo-status-badge', 'style'),
         Output('freshness-badge', 'children'),
         Output('last-age-badge', 'children'),
         Output('sentiment-badge', 'children'),
         Output('sentiment-badge', 'color')],
        [Input('data-source-store', 'data')]
    )
    def update_status_badges(data_source):
        """Update status badges based on data source, testnet setting, and freshness."""
        import os

        # Use actual data source from store, fallback to env var
        used_sample = False
        is_testnet = os.getenv('BINANCE_TESTNET', 'false').lower() == 'true'
        endpoint = None
        last_age = None

        if data_source and isinstance(data_source, dict):
            used_sample = data_source.get('used_sample_data', False)
            is_testnet = data_source.get('is_testnet', is_testnet)
            endpoint = data_source.get('endpoint')
            last_age = data_source.get('last_candle_age_min')
        else:
            used_sample = os.getenv('DASH_USE_SAMPLE_DATA', '0') == '1'

        # Set badge visibility based on sample data and testnet
        if used_sample:
            # Using sample/demo data
            live_style = {'display': 'none'}
            demo_style = {'display': 'inline-block'}
            freshness_text = "⚠️ DEMO DATA - NOT FOR BACKTESTS"
        else:
            # Using real data - show LIVE or TESTNET
            demo_style = {'display': 'none'}
            if is_testnet:
                live_style = {'display': 'inline-block', 'backgroundColor': '#0dcaf0'}
                freshness_text = f"✓ LIVE | Testnet"
            else:
                live_style = {'display': 'inline-block'}
                freshness_text = f"✓ LIVE | Production"

        age_text = f"Last Candle: {last_age:.1f}m ago" if last_age is not None else "Last Candle: --"
        
        # Sentiment Logic
        sentiment_text = "Sentiment: N/A"
        sentiment_color = "secondary"
        
        if 'SYSTEM_CONTEXT' in globals() and SYSTEM_CONTEXT:
            sent_data = SYSTEM_CONTEXT.sentiment
            val = sent_data.get('value', 50)
            cls = sent_data.get('classification', 'Neutral')
            sentiment_text = f"Sentiment: {val} ({cls})"
            
            if val >= 75: sentiment_color = "success"  # Extreme Greed
            elif val >= 55: sentiment_color = "info"   # Greed
            elif val <= 25: sentiment_color = "danger" # Extreme Fear
            elif val <= 45: sentiment_color = "warning" # Fear
            else: sentiment_color = "secondary"        # Neutral

        return live_style, demo_style, freshness_text, age_text, sentiment_text, sentiment_color

    # Backtest Lab Data Source Badge
    @app.callback(
        Output('backtest-data-source-badge', 'children'),
        [Input('backtest-symbol', 'value'),
         Input('backtest-timeframe', 'value')]
    )
    def update_backtest_data_badge(symbol, timeframe):
        """Show data source badge for backtest lab."""
        # Check if live data is available
        try:
            # Set backtest mode to prevent sample data fallback
            os.environ['BACKTEST_MODE'] = 'true'

            # Try to fetch a small amount of data to test availability
            test_df, meta = fetch_market_data(symbol, timeframe, num_bars=10, force_refresh=True)
            used_sample = meta.get('used_sample_data', False)
            last_age = meta.get('last_candle_age_min', 0)

            if used_sample:
                return dbc.Alert([
                    html.I(className="fas fa-exclamation-triangle me-2"),
                    "⚠️ SAMPLE DATA DETECTED - Backtests require live data!"
                ], color='danger', className='mb-0')
            else:
                tf_minutes_map = {'1m': 1, '5m': 5, '15m': 15, '1h': 60, '4h': 240, '1d': 1440}
                tf_minutes = tf_minutes_map.get(timeframe, 60)
                ok_threshold = tf_minutes * 1.5  # allow some buffer past expected bar close
                stale_threshold = tf_minutes * 3

                if last_age <= ok_threshold:
                    badge_color = "success"
                    status_text = "✓ Data OK"
                elif last_age <= stale_threshold:
                    badge_color = "warning"
                    status_text = "⚠ Data Stale"
                else:
                    badge_color = "danger"
                    status_text = "✗ Data Missing/Outdated"

                return dbc.Badge([
                    html.I(className="fas fa-database me-2"),
                    f"{status_text} | Last candle: {last_age:.1f}m ago | Source: Live Binance API"
                ], color=badge_color, className='p-2', id='backtest-data-badge')
        except Exception as e:
            return dbc.Alert([
                html.I(className="fas fa-times-circle me-2"),
                f"✗ DATA ERROR: {str(e)}"
            ], color='danger', className='mb-0')
        finally:
            # Reset backtest mode
            os.environ['BACKTEST_MODE'] = 'false'

    # Loading states for selectors
    @app.callback(
        [Output('symbol-loading', 'children'),
         Output('timeframe-loading', 'children'),
         Output('strategy-loading', 'children')],
        [Input('backtest-symbol', 'value'),
         Input('backtest-timeframe', 'value'),
         Input('backtest-strategy', 'value')]
    )
    def show_loading_states(symbol, timeframe, strategy):
        """Show loading spinners when selectors change."""
        return [
            dbc.Spinner(size="sm", color="primary") if symbol else None,
            dbc.Spinner(size="sm", color="primary") if timeframe else None,
            dbc.Spinner(size="sm", color="primary") if strategy else None,
        ]

    # Main Chart Callback
    @app.callback(
        [Output('main-chart', 'figure'),
         Output('data-source-store', 'data')],
        [Input('interval-fast', 'n_intervals'),
         Input('timeframe-store', 'data'),
         Input('overlay-toggles', 'value'),
         Input('refresh-data-btn', 'n_clicks'),
         Input('symbol-selector', 'value')]
    )
    def update_main_chart(n_intervals, timeframe, overlay_toggles, refresh_clicks, selected_symbol):
        """Update the main-chart figure based on timeframe and overlay selections."""
        # Load state from disk for split-brain synchronization
        if 'CHAT_SYSTEM_CONTEXT' in globals() and CHAT_SYSTEM_CONTEXT:
            CHAT_SYSTEM_CONTEXT.load_from_disk()

        # Use timeframe from store, defaulting to 15m if not set
        timeframe = timeframe or '15m'
        force_refresh = False
        ctx = callback_context
        if ctx and ctx.triggered:
            trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
            if trigger_id == 'refresh-data-btn':
                force_refresh = True

        # Fetch market data
        symbol = selected_symbol or 'BTCUSDT'
        df, fetch_meta = fetch_market_data(symbol, timeframe, num_bars=1000, force_refresh=force_refresh)

        # Prepare overlay features
        overlay_toggles = overlay_toggles or []
        features = {
            'liquidity': 'liquidity' in overlay_toggles,
            'supertrend': 'supertrend' in overlay_toggles,
            'orderflow': 'orderflow' in overlay_toggles,
            'chandelier': 'chandelier' in overlay_toggles,
            'regime': False,
            'alignment': False
        }

        # Create chart
        price_fig = create_interactive_chart(df, symbol, timeframe, features)

        # Prepare data source info for badges
        import os
        is_testnet_env = os.getenv('BINANCE_TESTNET', 'false').lower() == 'true'
        endpoint = os.getenv('BINANCE_FUTURES_URL')
        if not endpoint:
            endpoint = 'https://testnet.binancefuture.com/fapi/v1' if is_testnet_env else 'https://fapi.binance.com/fapi/v1'

        data_source_info = {
            'used_sample_data': fetch_meta.get('used_sample_data', False),
            'last_update': datetime.now().isoformat(),
            'last_candle_age_min': fetch_meta.get('last_candle_age_min', 0),
            'is_testnet': is_testnet_env,
            'endpoint': endpoint
        }

        logger.info(f"Updated main-chart for {symbol} {timeframe} with overlays: {overlay_toggles}, sample_data={data_source_info['used_sample_data']}")

        return price_fig, data_source_info

    # Timeframe Selection Callback
    @app.callback(
        [Output('timeframe-store', 'data'),
         Output('tf-1m', 'className'),
         Output('tf-5m', 'className'),
         Output('tf-15m', 'className'),
         Output('tf-1h', 'className'),
         Output('tf-4h', 'className'),
         Output('tf-1d', 'className')],
        [Input('tf-1m', 'n_clicks'),
         Input('tf-5m', 'n_clicks'),
         Input('tf-15m', 'n_clicks'),
         Input('tf-1h', 'n_clicks'),
         Input('tf-4h', 'n_clicks'),
         Input('tf-1d', 'n_clicks')]
    )
    def update_timeframe_selection(n1m, n5m, n15m, n1h, n4h, n1d):
        """Handle timeframe button clicks and update store."""
        import dash

        ctx = dash.callback_context
        if not ctx.triggered:
            # Initialize all as unselected
            return '15m', 'btn btn-outline-primary btn-sm', 'btn btn-outline-primary btn-sm', \
                   'btn btn-outline-primary btn-sm active', 'btn btn-outline-primary btn-sm', \
                   'btn btn-outline-primary btn-sm', 'btn btn-outline-primary btn-sm'

        # Get clicked button
        clicked = ctx.triggered[0]['prop_id'].split('.')[0]

        # Map clicks to timeframes
        timeframe_map = {
            'tf-1m': '1m',
            'tf-5m': '5m',
            'tf-15m': '15m',
            'tf-1h': '1h',
            'tf-4h': '4h',
            'tf-1d': '1d'
        }

        selected_tf = timeframe_map.get(clicked, '15m')

        # Update button styles
        styles = {}
        for tf_btn, tf_val in timeframe_map.items():
            if tf_val == selected_tf:
                styles[tf_btn] = 'btn btn-primary btn-sm active'
            else:
                styles[tf_btn] = 'btn btn-outline-primary btn-sm'

        return selected_tf, styles['tf-1m'], styles['tf-5m'], styles['tf-15m'], styles['tf-1h'], styles['tf-4h'], styles['tf-1d']

    # Backtest Lab Callbacks
    @app.callback(
        [Output('backtest-status', 'children'),
         Output('backtest-results-container', 'children'),
         Output('promote-strategy-btn', 'disabled'),
         Output('backtest-results-modal', 'is_open'),
         Output('backtest-result-store', 'data'),
         Output('backtest-history-store', 'data')],
        [Input('run-backtest-btn', 'n_clicks')],
        [State('backtest-symbol', 'value'),
         State('backtest-timeframe', 'value'),
         State('backtest-strategy', 'value'),
         State('backtest-capital', 'value'),
         State('backtest-start-date', 'date'),
         State('backtest-end-date', 'date'),
         State('backtest-history-store', 'data')]
    )
    def run_backtest(n_clicks, symbol, timeframe, strategy, capital, start_date, end_date, history):
        """Run backtest with selected parameters."""
        if not n_clicks or n_clicks == 0:
            raise PreventUpdate

        history = history or []

        status_msg = dbc.Alert([
            html.I(className="fas fa-spinner fa-spin me-2"),
            f"Running backtest for {symbol} on {timeframe} timeframe..."
        ], color='info')

        # Set backtest mode to prevent sample data fallback
        os.environ['BACKTEST_MODE'] = 'true'

        try:
            from backtesting.service import BacktestConfig, run_backtest as run_bt

            # FRESHNESS GUARD: Check data quality before running backtest
            tf_minutes_map = {'1m': 1, '5m': 5, '15m': 15, '1h': 60, '4h': 240, '1d': 1440}
            tf_minutes = tf_minutes_map.get(timeframe, 60)
            stale_threshold = tf_minutes * 1.5  # 1.5x timeframe

            # Fetch small sample to check data quality
            test_df, test_meta = fetch_market_data(symbol or "BTCUSDT", timeframe or "15m", num_bars=10, force_refresh=True)

            # Abort if using sample data
            if test_meta.get('used_sample_data', False):
                error_status = dbc.Alert([
                    html.I(className="fas fa-exclamation-triangle me-2"),
                    f"⚠️ BACKTEST ABORTED: Sample data detected for {symbol} {timeframe}. Backtests require live data only."
                ], color='danger')
                return error_status, dash.no_update, True, False, dash.no_update

            # Abort if data is stale
            last_age = test_meta.get('last_candle_age_min', 0)
            if last_age > stale_threshold:
                error_status = dbc.Alert([
                    html.I(className="fas fa-exclamation-triangle me-2"),
                    f"⚠️ BACKTEST ABORTED: Data is stale ({last_age:.1f}m old > {stale_threshold:.1f}m threshold) for {symbol} {timeframe}. Please wait for fresh data or try a different timeframe."
                ], color='danger')
                return error_status, dash.no_update, True, False, dash.no_update

            # Build config
            cfg = BacktestConfig(
                symbol=symbol or "BTCUSDT",
                timeframe=timeframe or "15m",
                start=datetime.fromisoformat(start_date) if start_date else datetime.now() - timedelta(days=365),
                end=datetime.fromisoformat(end_date) if end_date else datetime.now(),
                strategy=strategy or "convergence",
                params={},
                initial_capital=float(capital or 10000),
            )

            result = run_bt(cfg)

            # Build results card from real metrics
            results = dbc.Card([
                dbc.CardBody([
                    html.H5("Backtest Results", className="card-title"),
                    dbc.Row([
                        dbc.Col([html.H6("Total Return"), html.H3(f"{result.total_return_pct:+.2f}%", className="text-success" if result.total_return_pct >= 0 else "text-danger")], md=3),
                        dbc.Col([html.H6("Total Trades"), html.H3(result.total_trades, className="text-info")], md=3),
                        dbc.Col([html.H6("Win Rate"), html.H3(f"{result.win_rate*100:.1f}%", className="text-success")], md=3),
                        dbc.Col([html.H6("Max Drawdown"), html.H3(f"{result.max_drawdown:.2f}%", className="text-warning")], md=3),
                    ], className='mb-3'),
                    dbc.Row([
                        dbc.Col([html.H6("Sharpe Ratio"), html.H4(f"{result.sharpe_ratio:.2f}")], md=3),
                        dbc.Col([html.H6("Profit Factor"), html.H4(f"{result.profit_factor:.2f}")], md=3),
                        dbc.Col([html.H6("Final Capital"), html.H4(f"${result.final_capital:,.2f}")], md=3),
                        dbc.Col([html.H6("Initial Capital"), html.H4(f"${result.initial_capital:,.2f}")], md=3),
                    ]),
                    html.P([
                        html.I(className="fas fa-info-circle me-2"),
                        f"{symbol} | {timeframe} | {strategy} | {cfg.start.date()} → {cfg.end.date()}"
                    ], className="text-muted"),
                    dbc.Button([
                        html.I(className="fas fa-chart-line me-2"), "View Details"
                    ], id='view-details-btn', color='info', size='sm', n_clicks=0),
                ])
            ], className='shadow-sm')

            success_status = dbc.Alert([
                html.I(className="fas fa-check-circle me-2"),
                "Backtest completed successfully. Click 'View Details' for more metrics."
            ], color='success')

            result_dict = result.to_dict()
            if result_dict.get("trades"):
                result_dict["trades"] = result_dict["trades"][:50]
            if result_dict.get("equity_curve"):
                result_dict["equity_curve"] = result_dict["equity_curve"][:500]

            run_meta = {
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "symbol": symbol,
                "timeframe": timeframe,
                "strategy": strategy,
                "start": cfg.start.isoformat(),
                "end": cfg.end.isoformat(),
                "total_return_pct": result.total_return_pct,
                "total_trades": result.total_trades,
                "win_rate": result.win_rate,
                "max_drawdown": result.max_drawdown,
                "sharpe_ratio": result.sharpe_ratio,
                "profit_factor": result.profit_factor,
                "final_capital": result.final_capital,
                "initial_capital": result.initial_capital,
            }

            # Append to history (keep last 50)
            new_history = (history + [run_meta])[-50:]

            # Render a compact history table
            if new_history:
                hist_rows = []
                for row in reversed(new_history[-10:]):  # show last 10
                    hist_rows.append(html.Tr([
                        html.Td(row["timestamp"].split("T")[0]),
                        html.Td(row["symbol"]),
                        html.Td(row["timeframe"]),
                        html.Td(row["strategy"]),
                        html.Td(f"{row['total_return_pct']:+.2f}%"),
                        html.Td(row["total_trades"]),
                        html.Td(f"{row['win_rate']*100:.1f}%"),
                    ]))
                history_table = dbc.Table([
                    html.Thead(html.Tr([
                        html.Th("Date"), html.Th("Symbol"), html.Th("TF"), html.Th("Strategy"),
                        html.Th("Return"), html.Th("Trades"), html.Th("Win%")
                    ])),
                    html.Tbody(hist_rows)
                ], bordered=True, striped=True, hover=True, size="sm")
            else:
                history_table = html.Div("No historical runs yet.", className="text-muted")

            history_card = dbc.Card([
                dbc.CardHeader("Recent Backtests"),
                dbc.CardBody(history_table),
                dbc.CardFooter(dbc.Button("Download History (CSV)", id="download-history-btn", color="secondary", size="sm"))
            ], className="mt-3")

            results_block = html.Div([results, history_card])

            return success_status, results_block, False, False, result_dict, new_history

        except Exception as exc:
            logger.exception("Backtest failed")
            error_status = dbc.Alert([
                html.I(className="fas fa-exclamation-triangle me-2"),
                f"Backtest failed: {exc}"
            ], color='danger')

            return error_status, dash.no_update, True, False, dash.no_update, history
        finally:
            # Reset backtest mode
            os.environ['BACKTEST_MODE'] = 'false'

    @app.callback(
        [Output('backtest-results-modal-body', 'children'),
         Output('backtest-results-modal', 'is_open', allow_duplicate=True)],
        [Input('view-details-btn', 'n_clicks')],
        [State('backtest-results-modal', 'is_open'),
         State('backtest-result-store', 'data')],
        prevent_initial_call=True
    )
    def toggle_backtest_modal(n_clicks, is_open, result_data):
        """Show detailed backtest results in modal."""
        if not n_clicks:
            raise PreventUpdate

        new_state = not is_open

        if not result_data:
            details_content = dbc.Container([
                html.H5("Detailed Performance Metrics", className="mb-3"),
                dbc.Alert([
                    html.I(className="fas fa-exclamation-triangle me-2"),
                    "No backtest result available. Please run a backtest first."
                ], color="warning"),
            ])
            return details_content, new_state

        equity_fig = None
        if result_data.get("equity_curve"):
            import plotly.graph_objs as go
            eq_df = pd.DataFrame(result_data["equity_curve"])
            if not eq_df.empty:
                equity_fig = go.Figure()
                equity_fig.add_trace(go.Scatter(
                    x=pd.to_datetime(eq_df['timestamp']) if 'timestamp' in eq_df else eq_df.index,
                    y=eq_df['equity'],
                    mode='lines',
                    name='Equity'
                ))
                equity_fig.update_layout(
                    height=300,
                    margin=dict(l=20, r=20, t=30, b=30),
                    template='plotly_dark',
                    title="Equity Curve"
                )

        trades_table = None
        if result_data.get("trades"):
            trades_df = pd.DataFrame(result_data["trades"])
            if not trades_df.empty:
                cols = ['entry_time', 'exit_time', 'side', 'entry_price', 'exit_price', 'pnl', 'pnl_percent']
                trades_table = dbc.Table.from_dataframe(
                    trades_df[cols].head(50),
                    striped=True,
                    bordered=True,
                    hover=True,
                    size='sm'
                )

        summary_rows = []
        summary_fields = [
            ("Total Return", f"{result_data.get('total_return_pct', 0):+.2f}%"),
            ("Trades", result_data.get('total_trades', 0)),
            ("Win Rate", f"{result_data.get('win_rate', 0)*100:.1f}%"),
            ("Max Drawdown", f"{result_data.get('max_drawdown', 0):.2f}%"),
            ("Sharpe", result_data.get('sharpe_ratio', 0)),
            ("Profit Factor", result_data.get('profit_factor', 0)),
            ("Final Capital", f"${result_data.get('final_capital', 0):,.2f}"),
            ("Initial Capital", f"${result_data.get('initial_capital', 0):,.2f}")
        ]
        for label, val in summary_fields:
            summary_rows.append(html.Div([html.Strong(f"{label}: "), html.Span(val)], className="mb-1"))

        details_content = dbc.Container([
            html.H5("Detailed Performance Metrics", className="mb-3"),
            dbc.Row([
                dbc.Col(summary_rows, md=4),
                dbc.Col(dcc.Graph(figure=equity_fig) if equity_fig else html.Div("No equity curve available"), md=8),
            ], className="mb-3"),
            html.H6("Recent Trades", className="mt-2"),
            trades_table or html.Div("No trades available for this run."),
            html.Div(className="mt-3", children=[
                dbc.Button("Export CSV", id="export-csv-btn", color="secondary", size="sm", className="me-2"),
                dbc.Button("Export PDF", id="export-pdf-btn", color="secondary", size="sm")
            ])
        ])

        return details_content, new_state

    # Chat callback
    logger.info("Registering chat callback...")
    @app.callback(
        [Output("chat-history", "value"),
         Output("chat-history-store", "data"),
         Output("chat-input", "value")],
        [Input("chat-send-btn", "n_clicks"),
         Input("chat-input", "n_submit")],
        [State("chat-input", "value"),
         State("chat-history-store", "data")],
        prevent_initial_call=True,
    )
    def handle_chat(n_clicks, n_submit, message, history):
        logger.info(f"Chat callback triggered: n_clicks={n_clicks}, n_submit={n_submit}, message={message}")
        if (not n_clicks and not n_submit) or not message:
            raise PreventUpdate

        history = history or []
        history.append({"role": "user", "content": message, "ts": datetime.utcnow().isoformat() + "Z"})

        reply = "AI service unavailable (demo response)."
        try:
            resp = route_chat_message(message)
            if resp:
                # route_chat_message returns Tuple[bool, str, Dict]
                # Format the response appropriately
                success, response_text, metadata = resp
                reply = response_text
        except Exception as exc:
            logger.error(f"Chat error: {exc}")
            reply = f"AI error: {exc}"

        history.append({"role": "assistant", "content": reply, "ts": datetime.utcnow().isoformat() + "Z"})
        # Render text area content
        rendered = []
        for h in history[-20:]:
            rendered.append(f"[{h.get('ts','')}] {h['role']}: {h['content']}")
        logger.info(f"Chat response: {reply[:100]}...")
        return "\n".join(rendered), history, ""

    return app


# Initialize real SystemContext before creating app
SYSTEM_CONTEXT = None
try:
    from core.system_context import SystemContext
    SYSTEM_CONTEXT = SystemContext()
    set_system_context(SYSTEM_CONTEXT)
    print("✓ Real SystemContext initialized")

    # Initialize command router for Step 18 - DeepSeek as Manager
    from ui.chat_command_router import set_system_context as set_router_context
    from ui.chat_command_router import set_dashboard_callbacks as set_router_callbacks
    set_router_context(SYSTEM_CONTEXT)
    print("✓ Command Router initialized")

except Exception as e:
    print(f"⚠ Warning: Could not initialize SystemContext: {e}")
    print("  Dashboard will run with limited functionality")

# Expose a module-level Dash app instance for dash.testing and gunicorn
app = create_dashboard_app()
server = app.server


if __name__ == "__main__":
    import sys
    import os

    # Check for demo/live mode
    use_sample = os.getenv('DASH_USE_SAMPLE_DATA', '0') == '1'
    is_testnet = os.getenv('BINANCE_TESTNET', 'false').lower() == 'true'
    futures_url = os.getenv('BINANCE_FUTURES_URL')
    if not futures_url:
        futures_url = 'https://testnet.binancefuture.com/fapi/v1' if is_testnet else 'https://fapi.binance.com/fapi/v1'
    data_label = "Demo/Sample Data" if use_sample else f"Binance {'Testnet' if is_testnet else 'Live'} API ({futures_url})"
    mode = "DEMO MODE" if use_sample else ("TESTNET MODE" if is_testnet else "LIVE MODE")

    # Use the existing app instance (created at module level)
    print(f"\n{'='*60}")
    print(f"  DeepSeek Trading Dashboard - {mode}")
    print(f"{'='*60}")
    print(f"  URL: http://127.0.0.1:8050")
    print(f"  Data: {data_label}")
    print(f"{'='*60}\n")

    app.run(host='0.0.0.0', port=8050, debug=False, dev_tools_ui=False)
