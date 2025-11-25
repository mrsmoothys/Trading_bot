"""
Backtest Service API
Reusable service for running backtests with configurable strategies.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
from loguru import logger
import json
import hashlib
import os

from models.backtester import Backtester, BacktestResults, Trade


@dataclass
class BacktestConfig:
    """Configuration for a backtest run."""
    symbol: str
    timeframe: str
    start: datetime
    end: datetime
    strategy: str
    params: Dict[str, Any]
    initial_capital: float = 10000.0
    commission_rate: float = 0.001
    slippage: float = 0.0005

    def to_json(self) -> str:
        """Convert config to JSON string."""
        config_dict = asdict(self)
        config_dict['start'] = self.start.isoformat()
        config_dict['end'] = self.end.isoformat()
        return json.dumps(config_dict, indent=2)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BacktestConfig':
        """Create config from dictionary."""
        # Parse dates
        if isinstance(data['start'], str):
            data['start'] = datetime.fromisoformat(data['start'])
        if isinstance(data['end'], str):
            data['end'] = datetime.fromisoformat(data['end'])

        return cls(**data)

    def get_config_hash(self) -> str:
        """Generate hash for config to detect duplicates."""
        config_str = self.to_json()
        return hashlib.md5(config_str.encode()).hexdigest()


@dataclass
class BacktestResult:
    """Result from a backtest run."""
    config_hash: str
    symbol: str
    timeframe: str
    strategy: str
    start: datetime
    end: datetime
    initial_capital: float
    final_capital: float
    total_return: float
    total_return_pct: float
    annualized_return: float
    max_drawdown: float
    sharpe_ratio: float
    win_rate: float
    profit_factor: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    avg_win: float
    avg_loss: float
    equity_curve: List[Dict[str, Any]]
    trades: List[Dict[str, Any]]
    timestamp: datetime

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        result_dict = asdict(self)
        result_dict['start'] = self.start.isoformat()
        result_dict['end'] = self.end.isoformat()
        result_dict['timestamp'] = self.timestamp.isoformat()
        return result_dict

    def to_json(self) -> str:
        """Convert result to JSON string."""
        return json.dumps(self.to_dict(), indent=2, default=str)

    @classmethod
    def from_backtest_results(
        cls,
        config: BacktestConfig,
        results: BacktestResults
    ) -> 'BacktestResult':
        """Create BacktestResult from BacktestResults and config."""
        config_hash = config.get_config_hash()

        # Extract equity curve
        equity_curve = []
        if hasattr(results, 'equity_curve'):
            equity_curve = results.equity_curve

        # Extract trades
        trades = []
        if results.trades:
            for trade in results.trades:
                trades.append({
                    'entry_time': trade.entry_time.isoformat(),
                    'exit_time': trade.exit_time.isoformat() if trade.exit_time else None,
                    'entry_price': trade.entry_price,
                    'exit_price': trade.exit_price,
                    'side': trade.side,
                    'quantity': trade.quantity,
                    'pnl': trade.pnl,
                    'pnl_percent': trade.pnl_percent,
                    'commission': trade.commission,
                    'exit_reason': trade.exit_reason
                })

        return cls(
            config_hash=config_hash,
            symbol=config.symbol,
            timeframe=config.timeframe,
            strategy=config.strategy,
            start=config.start,
            end=config.end,
            initial_capital=config.initial_capital,
            final_capital=results.total_return * config.initial_capital + config.initial_capital,
            total_return=results.total_return,
            total_return_pct=results.total_return * 100,
            annualized_return=results.annualized_return,
            max_drawdown=results.max_drawdown.item() if isinstance(results.max_drawdown, (pd.Series, pd.DataFrame)) else float(results.max_drawdown),
            sharpe_ratio=results.sharpe_ratio,
            win_rate=results.win_rate,
            profit_factor=results.profit_factor,
            total_trades=results.total_trades,
            winning_trades=results.winning_trades,
            losing_trades=results.losing_trades,
            avg_win=results.avg_win,
            avg_loss=results.avg_loss,
            equity_curve=equity_curve,
            trades=trades,
            timestamp=datetime.now()
        )


def generate_signals(df: pd.DataFrame, strategy: str, params: Dict[str, Any]) -> pd.DataFrame:
    """
    Generate trading signals for a given strategy.

    Args:
        df: Market data with OHLCV
        strategy: Strategy name ('sma', 'rsi', 'macd', 'convergence')
        params: Strategy parameters

    Returns:
        DataFrame with columns ['signal', 'strength', 'confidence']
    """
    signals = pd.DataFrame(index=df.index)
    signals['signal'] = 0.0
    signals['strength'] = 0.0
    signals['confidence'] = 0.5

    if strategy == 'sma':
        # Simple Moving Average crossover
        fast_period = params.get('fast_period', 20)
        slow_period = params.get('slow_period', 50)

        sma_fast = df['close'].rolling(fast_period).mean()
        sma_slow = df['close'].rolling(slow_period).mean()

        # Generate signals
        signals.loc[df['close'] > sma_fast, 'signal'] = 1
        signals.loc[df['close'] < sma_fast, 'signal'] = -1

        # Signal strength based on distance between SMAs
        sma_diff = (sma_fast - sma_slow) / df['close']
        signals['strength'] = sma_diff.abs().fillna(0)

        # Confidence based on signal consistency
        signal_consistency = signals['signal'].rolling(5).apply(lambda x: len(set(x)) == 1).fillna(0)
        signals['confidence'] = signal_consistency

    elif strategy == 'rsi':
        # RSI mean reversion
        period = params.get('period', 14)
        oversold = params.get('oversold', 30)
        overbought = params.get('overbought', 70)

        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

        # Fix: Avoid division by zero and handle small values
        rs = gain / (loss + 1e-10)  # Add small epsilon to prevent division by zero
        rs = rs.replace([np.inf, -np.inf], 0).fillna(0)  # Replace inf and NaN with 0

        rsi = 100 - (100 / (1 + rs))
        rsi = rsi.replace([np.inf, -np.inf], 50).fillna(50)  # Replace inf/NaN with neutral 50

        # Generate signals
        signals.loc[rsi < oversold, 'signal'] = 1  # Oversold - buy
        signals.loc[rsi > overbought, 'signal'] = -1  # Overbought - sell

        # Signal strength based on RSI extreme
        rsi_extreme = np.maximum(
            (oversold - rsi).abs() / oversold,
            (rsi - overbought).abs() / overbought
        )
        signals['strength'] = rsi_extreme.fillna(0)

        # Confidence based on signal magnitude
        signals['confidence'] = np.minimum(signals['strength'] * 2, 1.0)

    elif strategy == 'macd':
        # MACD trend following
        fast_period = params.get('fast_period', 12)
        slow_period = params.get('slow_period', 26)
        signal_period = params.get('signal_period', 9)

        exp1 = df['close'].ewm(span=fast_period).mean()
        exp2 = df['close'].ewm(span=slow_period).mean()
        macd = exp1 - exp2
        signal_line = macd.ewm(span=signal_period).mean()
        histogram = macd - signal_line

        # Generate signals
        signals.loc[macd > signal_line, 'signal'] = 1
        signals.loc[macd < signal_line, 'signal'] = -1

        # Signal strength based on histogram magnitude
        signals['strength'] = histogram.abs().fillna(0) / df['close']

        # Confidence based on histogram trend
        histogram_trend = histogram.rolling(3).apply(lambda x: len(set(np.sign(x))) == 1).fillna(0)
        signals['confidence'] = histogram_trend

    elif strategy in ['convergence', 'scalp_15m_4h']:
        # Multi-timeframe convergence strategy (simplified)
        # In production, this would call the actual convergence system

        # Calculate trend using EMAs
        ema_fast = df['close'].ewm(span=10).mean()
        ema_slow = df['close'].ewm(span=21).mean()
        trend = np.where(ema_fast > ema_slow, 1, -1)

        # Calculate volatility with bounds checking
        returns = df['close'].pct_change(fill_method=None).fillna(0)
        returns = returns.replace([np.inf, -np.inf], 0)  # Replace inf with 0
        volatility = returns.rolling(20).std().fillna(0)

        # Generate signals based on trend with volatility filter
        signals['signal'] = trend
        signals['strength'] = volatility.fillna(0)
        signals['confidence'] = 0.7  # High confidence for convergence signals

    elif strategy == 'ma_crossover':
        # Moving Average Crossover
        try:
            from core.strategies.ma_crossover import calculate_signals as ma_calc
            signals = ma_calc(df, params)
        except Exception as e:
            logger.warning(f"Could not use ma_crossover strategy, using fallback: {e}")
            # Fallback to simple SMA crossover
            fast_period = params.get('fast_period', 20)
            slow_period = params.get('slow_period', 50)

            sma_fast = df['close'].rolling(fast_period).mean()
            sma_slow = df['close'].rolling(slow_period).mean()

            signals = pd.DataFrame(index=df.index)
            signals['signal'] = 0.0
            signals['strength'] = 0.0
            signals['confidence'] = 0.5

            signals.loc[sma_fast > sma_slow, 'signal'] = 1
            signals.loc[sma_fast < sma_slow, 'signal'] = -1

    elif strategy == 'rsi_divergence':
        # RSI Divergence
        try:
            from core.strategies.rsi_divergence import calculate_signals as rsi_calc
            signals = rsi_calc(df, params)
        except Exception as e:
            logger.warning(f"Could not use rsi_divergence strategy, using fallback: {e}")
            # Fallback to simple RSI
            period = params.get('period', 14)
            oversold = params.get('oversold', 30)
            overbought = params.get('overbought', 70)

            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

            rs = gain / (loss + 1e-10)
            rs = rs.replace([np.inf, -np.inf], 0).fillna(0)

            rsi = 100 - (100 / (1 + rs))
            rsi = rsi.replace([np.inf, -np.inf], 50).fillna(50)

            signals = pd.DataFrame(index=df.index)
            signals['signal'] = 0.0
            signals['strength'] = 0.0
            signals['confidence'] = 0.5

            signals.loc[rsi < oversold, 'signal'] = 1
            signals.loc[rsi > overbought, 'signal'] = -1

    else:
        logger.warning(f"Unknown strategy: {strategy}, using passive strategy")
        # Passive strategy - no trading
        signals['signal'] = 0
        signals['strength'] = 0
        signals['confidence'] = 0

    return signals.fillna(0)


def load_market_data(symbol: str, timeframe: str, start: datetime, end: datetime) -> pd.DataFrame:
    """
    Load market data for backtesting.

    Args:
        symbol: Trading symbol (e.g., 'BTCUSDT')
        timeframe: Data timeframe (e.g., '1h', '1d')
        start: Start datetime
        end: End datetime

    Returns:
        DataFrame with OHLCV data
    """
    # Cap bars per timeframe to keep sweeps responsive
    MAX_BARS = {
        '1m': 1500,
        '5m': 2500,
        '15m': 4000,
        '30m': 5000,
        '1h': 6000,
        '4h': 7000,
        '1d': 3000,
        '1w': 2000
    }

    try:
        # Try to load from cache/parquet first
        from ui.dashboard import fetch_market_data

        # Calculate number of bars needed
        timeframe_minutes = {
            '1m': 1, '5m': 5, '15m': 15, '30m': 30,
            '1h': 60, '2h': 120, '4h': 240, '6h': 360,
            '8h': 480, '12h': 720, '1d': 1440, '1w': 10080
        }

        minutes = timeframe_minutes.get(timeframe, 60)
        total_minutes = (end - start).total_seconds() / 60
        num_bars = int(total_minutes / minutes) + 100  # Extra bars for warmup
        max_bars = MAX_BARS.get(timeframe, 8000)
        num_bars = max(500, min(num_bars, max_bars))

        # Fetch data
        df, meta = fetch_market_data(symbol, timeframe, num_bars=num_bars, force_refresh=True)

        if df is None or len(df) == 0:
            raise ValueError(f"No data fetched for {symbol} {timeframe}")

        # Filter by date range
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp')

        # Ensure timezone compatibility
        if df.index.tz is not None:
            # DataFrame has timezone, localize start/end if they're naive
            if start.tzinfo is None:
                start = start.tz_localize('UTC')
            if end.tzinfo is None:
                end = end.tz_localize('UTC')

        # Filter data
        mask = (df.index >= start) & (df.index <= end)
        df = df[mask].copy()

        if len(df) == 0:
            raise ValueError(f"No data in date range {start} to {end}")

        logger.info(f"Loaded {len(df)} bars for {symbol} {timeframe} from {df.index[0]} to {df.index[-1]}")
        return df

    except (ImportError, SyntaxError, Exception) as e:
        # Fallback: generate synthetic data for testing
        logger.warning(f"Could not load real market data ({type(e).__name__}), generating synthetic data for testing")

        # Calculate number of bars needed
        timeframe_minutes = {
            '1m': 1, '5m': 5, '15m': 15, '30m': 30,
            '1h': 60, '2h': 120, '4h': 240, '6h': 360,
            '8h': 480, '12h': 720, '1d': 1440, '1w': 10080
        }

        minutes = timeframe_minutes.get(timeframe, 60)
        total_minutes = (end - start).total_seconds() / 60
        num_bars = int(total_minutes / minutes) + 100

        # Generate synthetic price data
        np.random.seed(42)  # For reproducible results
        dates = pd.date_range(start=start, periods=num_bars, freq=f'{minutes}min')

        # Generate random walk with slight upward trend
        price = 50000  # Starting price (like BTC)
        # Clamp returns to prevent extreme outliers
        returns = np.random.normal(0.001, 0.02, num_bars)
        returns = np.clip(returns, -0.1, 0.1)  # Cap at ±10% per bar
        prices = [price]

        # Fix: Add bounds checking to prevent infinity from compounding
        for ret in returns[1:]:
            new_price = prices[-1] * (1 + ret)
            # Cap price to prevent overflow to infinity
            if not np.isfinite(new_price) or new_price <= 0:
                new_price = prices[-1]  # Use previous price
            # Cap at much tighter bounds (5x to 0.2x of initial price to prevent extreme values)
            new_price = np.clip(new_price, price * 0.2, price * 5)
            prices.append(new_price)

        # Create OHLCV data
        df = pd.DataFrame(index=dates)
        df['close'] = prices

        # Generate OHLC from close prices
        df['open'] = df['close'].shift(1)
        df['high'] = df[['open', 'close']].max(axis=1) * (1 + np.abs(np.random.normal(0, 0.005, num_bars)))
        df['low'] = df[['open', 'close']].min(axis=1) * (1 - np.abs(np.random.normal(0, 0.005, num_bars)))
        df['volume'] = np.random.uniform(100, 1000, num_bars) * 1000000

        # Fill first row
        df.loc[df.index[0], 'open'] = df.loc[df.index[0], 'close']

        # Filter by date range
        mask = (df.index >= start) & (df.index <= end)
        df = df[mask].copy()

        # Fix: Validate all prices are finite before returning
        for col in ['open', 'high', 'low', 'close']:
            inf_mask = ~np.isfinite(df[col])
            if inf_mask.any():
                logger.warning(f"Found {inf_mask.sum()} non-finite values in {col}, replacing with previous value")
                df[col] = df[col].replace([np.inf, -np.inf], np.nan).ffill().fillna(price)

        # Ensure we have enough data
        if len(df) < 50:
            raise ValueError(f"Insufficient data generated: only {len(df)} bars (need at least 50)")

        logger.info(f"Generated {len(df)} synthetic bars for {symbol} {timeframe}")
        return df


def run_backtest(config: BacktestConfig) -> BacktestResult:
    """
    Run a backtest with the given configuration.

    Args:
        config: BacktestConfig with all parameters

    Returns:
        BacktestResult with performance metrics

    Raises:
        ValueError: If config is invalid or data cannot be loaded
    """
    logger.info(f"Starting backtest: {config.symbol} {config.timeframe} {config.strategy}")
    logger.info(f"Period: {config.start} to {config.end}")
    logger.info(f"Params: {config.params}")

    # Load market data
    df = load_market_data(config.symbol, config.timeframe, config.start, config.end)

    if len(df) < 50:
        raise ValueError(f"Insufficient data: only {len(df)} bars available")

    # Generate signals
    signals = generate_signals(df, config.strategy, config.params)

    # Initialize backtester
    backtester = Backtester(
        initial_capital=config.initial_capital,
        commission_rate=config.commission_rate,
        slippage=config.slippage
    )

    # Run backtest
    results = backtester.run_backtest(df, signals, f"{config.strategy}_{config.symbol}")

    # Convert to BacktestResult
    backtest_result = BacktestResult.from_backtest_results(config, results)

    logger.info(
        f"Backtest complete: {backtest_result.total_trades} trades, "
        f"{backtest_result.win_rate:.1%} win rate, "
        f"{backtest_result.total_return_pct:.2f}% return, "
        f"Sharpe: {backtest_result.sharpe_ratio:.2f}"
    )

    return backtest_result


def run_backtest_batch(configs: List[BacktestConfig]) -> List[Tuple[BacktestConfig, Optional[BacktestResult], Optional[str]]]:
    """
    Run multiple backtests in batch.

    Args:
        configs: List of BacktestConfig objects

    Returns:
        List of tuples (config, result or None, error_message or None)
    """
    logger.info(f"Running batch backtest: {len(configs)} configurations")

    results = []
    for i, config in enumerate(configs, 1):
        logger.info(f"Running backtest {i}/{len(configs)}: {config.strategy} {config.symbol}")
        try:
            result = run_backtest(config)
            results.append((config, result, None))
        except Exception as e:
            logger.error(f"Backtest failed for config {i}: {e}")
            results.append((config, None, str(e)))

    logger.info(f"Batch backtest complete: {len([r for r in results if r[1] is not None])} successful")
    return results
