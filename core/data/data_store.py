"""
Data Store
In-memory caching and persistence for market data.
Optimized for M1 MacBook with 4GB memory limit.
"""

import os
import pickle
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
from pathlib import Path
from loguru import logger


class DataStore:
    """
    In-memory data store with persistence.
    Manages cached market data with TTL.
    """

    def __init__(self, cache_dir: str = "data/cache"):
        """
        Initialize data store.

        Args:
            cache_dir: Directory for persistent cache
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Memory cache
        self.cache: Dict[str, Any] = {}
        self.cache_ttl: Dict[str, datetime] = {}

        # Default TTL settings (in seconds)
        self.default_ttl = 300  # 5 minutes for price data
        self.price_data_ttl = 60  # 1 minute for real-time prices
        self.historical_ttl = 3600  # 1 hour for historical data

        # Cache size management
        self.max_cache_size = 100  # Maximum number of cached items
        self.max_memory_mb = 100  # Maximum memory usage in MB

        logger.info(f"DataStore initialized with cache dir: {self.cache_dir}")

    def _schedule_background_task(self, coro):
        """Run async tasks whether or not an event loop is active."""
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(coro)
        except RuntimeError:
            asyncio.run(coro)

    def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None
    ):
        """
        Store data in cache with TTL.

        Args:
            key: Cache key
            value: Data to store
            ttl: Time to live in seconds
        """
        if ttl is None:
            ttl = self.default_ttl

        # Check cache size
        if len(self.cache) >= self.max_cache_size:
            self._evict_oldest()

        # Store in memory
        self.cache[key] = value
        self.cache_ttl[key] = datetime.now() + timedelta(seconds=ttl)

        # Save to disk asynchronously (with safe fallback if no loop)
        self._schedule_background_task(self._save_to_disk(key, value))

        logger.debug(f"Cached data: {key} (TTL: {ttl}s)")

    def get(self, key: str) -> Optional[Any]:
        """
        Retrieve data from cache.

        Args:
            key: Cache key

        Returns:
            Cached data or None if not found/expired
        """
        # Check if exists
        if key not in self.cache:
            return None

        # Check if expired
        if datetime.now() > self.cache_ttl[key]:
            del self.cache[key]
            del self.cache_ttl[key]
            return None

        return self.cache[key]

    async def get_or_fetch(
        self,
        key: str,
        fetch_func,
        ttl: Optional[int] = None
    ) -> Any:
        """
        Get from cache or fetch if not available.

        Args:
            key: Cache key
            fetch_func: Async function to fetch data
            ttl: Time to live

        Returns:
            Data
        """
        cached = self.get(key)
        if cached is not None:
            return cached

        # Fetch data asynchronously
        data = await fetch_func()
        if data is not None:
            self.set(key, data, ttl)
        return data

    async def store_ohlcv(
        self,
        symbol: str,
        interval: str,
        data: pd.DataFrame
    ):
        """Store OHLCV data."""
        key = f"ohlcv:{symbol}:{interval}"
        self.set(key, data, self.historical_ttl)

    def get_ohlcv(
        self,
        symbol: str,
        interval: str
    ) -> Optional[pd.DataFrame]:
        """Get OHLCV data."""
        key = f"ohlcv:{symbol}:{interval}"
        return self.get(key)

    async def store_price(self, symbol: str, price: float):
        """Store current price."""
        key = f"price:{symbol}"
        self.set(key, price, self.price_data_ttl)

    def get_price(self, symbol: str) -> Optional[float]:
        """Get current price."""
        key = f"price:{symbol}"
        return self.get(key)

    async def store_features(
        self,
        symbol: str,
        features: Dict[str, Any]
    ):
        """Store calculated features."""
        key = f"features:{symbol}"
        self.set(key, features, self.historical_ttl)

    def get_features(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get calculated features."""
        key = f"features:{symbol}"
        return self.get(key)

    def _evict_oldest(self):
        """Evict oldest cached items to manage memory."""
        # Sort by expiry time
        sorted_items = sorted(
            self.cache_ttl.items(),
            key=lambda x: x[1]
        )

        # Remove 10% of oldest items
        num_to_remove = max(1, len(sorted_items) // 10)

        for key, _ in sorted_items[:num_to_remove]:
            del self.cache[key]
            del self.cache_ttl[key]

        logger.info(f"Evicted {num_to_remove} items from cache")

    async def _save_to_disk(self, key: str, value: Any):
        """Save cache item to disk."""
        try:
            file_path = self.cache_dir / f"{key.replace(':', '_')}.pkl"

            # For DataFrames, save as parquet
            if isinstance(value, pd.DataFrame):
                file_path = file_path.with_suffix('.parquet')
                value.to_parquet(file_path, index=False)
            else:
                with open(file_path, 'wb') as f:
                    pickle.dump(value, f)

        except Exception as e:
            logger.error(f"Error saving cache to disk: {e}")

    def _load_from_disk(self, key: str) -> Optional[Any]:
        """Load cache item from disk."""
        try:
            file_path = self.cache_dir / f"{key.replace(':', '_')}"

            # Try parquet first (for DataFrames)
            parquet_path = file_path.with_suffix('.parquet')
            if parquet_path.exists():
                return pd.read_parquet(parquet_path)

            # Try pickle
            pickle_path = file_path.with_suffix('.pkl')
            if pickle_path.exists():
                with open(pickle_path, 'rb') as f:
                    return pickle.load(f)

            return None

        except Exception as e:
            logger.error(f"Error loading cache from disk: {e}")
            return None

    def clear_cache(self):
        """Clear all cached data."""
        self.cache.clear()
        self.cache_ttl.clear()
        logger.info("Cache cleared")

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_items = len(self.cache)
        expired_items = sum(
            1 for expiry in self.cache_ttl.values()
            if datetime.now() > expiry
        )

        # Calculate approximate memory usage
        memory_bytes = 0
        for key, value in self.cache.items():
            try:
                if isinstance(value, pd.DataFrame):
                    memory_bytes += value.memory_usage(deep=True).sum()
                elif isinstance(value, dict):
                    memory_bytes += len(str(value))
                else:
                    memory_bytes += len(str(value))
            except:
                pass

        memory_mb = memory_bytes / (1024 * 1024)

        return {
            "total_items": total_items,
            "active_items": total_items - expired_items,
            "expired_items": expired_items,
            "memory_usage_mb": memory_mb,
            "max_memory_mb": self.max_memory_mb,
            "cache_hit_rate": self._calculate_hit_rate()
        }

    def _calculate_hit_rate(self) -> float:
        """Calculate cache hit rate."""
        # This is a simplified implementation
        # In production, you'd track hits/misses
        return 0.75  # Placeholder

    def cleanup_expired(self):
        """Remove expired items from cache."""
        now = datetime.now()
        expired_keys = [
            key for key, expiry in self.cache_ttl.items()
            if now > expiry
        ]

        for key in expired_keys:
            del self.cache[key]
            del self.cache_ttl[key]

        if expired_keys:
            logger.info(f"Cleaned up {len(expired_keys)} expired items")

    def preload_common_symbols(
        self,
        symbols: List[str],
        intervals: List[str] = ['1h', '4h']
    ):
        """
        Preload data for common symbols.

        Args:
            symbols: List of symbols
            intervals: List of timeframes
        """
        logger.info(f"Preloading data for {len(symbols)} symbols")

        # This would typically fetch data in the background
        # For now, just log the intention
        for symbol in symbols:
            for interval in intervals:
                key = f"ohlcv:{symbol}:{interval}"
                # Check if already cached
                if not self.get(key):
                    logger.debug(f"Will preload {symbol} {interval}")

    def _get_parquet_path(self, symbol: str, interval: str) -> Path:
        """
        Get Parquet file path for symbol/interval.

        Args:
            symbol: Trading symbol (e.g., 'BTCUSDT')
            interval: Timeframe (e.g., '1m', '1h')

        Returns:
            Path to Parquet file
        """
        return self.cache_dir / f"{symbol}_{interval}_history.parquet"

    def _kline_to_dataframe(self, kline_data: Dict[str, Any]) -> pd.DataFrame:
        """
        Convert Binance kline data to DataFrame.

        Args:
            kline_data: Kline data from Binance WebSocket

        Returns:
            DataFrame with OHLCV data
        """
        return pd.DataFrame([{
            'timestamp': pd.to_datetime(kline_data['t'], unit='ms'),
            'open': float(kline_data['o']),
            'high': float(kline_data['h']),
            'low': float(kline_data['l']),
            'close': float(kline_data['c']),
            'volume': float(kline_data['v']),
            'close_time': pd.to_datetime(kline_data['T'], unit='ms'),
            'quote_asset_volume': float(kline_data['q']),
            'number_of_trades': int(kline_data['n']),
            'taker_buy_base_asset_volume': float(kline_data['V']),
            'taker_buy_quote_asset_volume': float(kline_data['Q']),
            'is_closed': bool(kline_data['x'])
        }])

    def update_from_websocket(
        self,
        symbol: str,
        interval: str,
        kline_data: Dict[str, Any]
    ) -> Optional[pd.DataFrame]:
        """
        Update cache from WebSocket kline data.

        Args:
            symbol: Trading symbol
            interval: Timeframe
            kline_data: Kline data from WebSocket

        Returns:
            Updated DataFrame or None
        """
        try:
            # Convert kline to DataFrame
            new_row = self._kline_to_dataframe(kline_data)

            # Get cache key
            cache_key = (symbol, interval)

            # Get existing DataFrame from cache
            if cache_key in self.cache:
                df = self.cache[cache_key].copy()

                # Check if this kline already exists (shouldn't for closed klines, but handle anyway)
                last_timestamp = df['timestamp'].iloc[-1] if not df.empty else None

                if df.empty or new_row['timestamp'].iloc[0] > last_timestamp:
                    # Append new data
                    df = pd.concat([df, new_row], ignore_index=True)
                else:
                    # Update existing row
                    mask = df['timestamp'] == new_row['timestamp'].iloc[0]
                    df.loc[mask] = new_row.iloc[0]

            else:
                # Create new DataFrame
                df = new_row

            # Keep only last 1000 candles in memory cache (optimized for M1)
            if len(df) > 1000:
                df = df.tail(1000).reset_index(drop=True)

            # Update cache
            self.cache[cache_key] = df

            # Persist to Parquet asynchronously (30-day history)
            self._schedule_background_task(
                self._save_historical_data(symbol, interval, df)
            )

            logger.debug(f"Updated cache: {symbol} {interval} - {len(df)} candles")

            return df

        except Exception as e:
            logger.error(f"Error updating from WebSocket: {e}")
            return None

    async def _save_historical_data(
        self,
        symbol: str,
        interval: str,
        df: pd.DataFrame
    ):
        """
        Save historical data to Parquet (30-day history).

        Args:
            symbol: Trading symbol
            interval: Timeframe
            df: DataFrame to save
        """
        try:
            parquet_path = self._get_parquet_path(symbol, interval)

            # Load existing data if available
            if parquet_path.exists():
                existing_df = pd.read_parquet(parquet_path)

                # Merge and remove duplicates
                combined_df = pd.concat([existing_df, df], ignore_index=True)
                combined_df = combined_df.drop_duplicates(subset=['timestamp'], keep='last')
                combined_df = combined_df.sort_values('timestamp').reset_index(drop=True)
            else:
                combined_df = df

            # Filter to keep only last 30 days
            cutoff_date = datetime.now() - timedelta(days=30)
            combined_df = combined_df[combined_df['timestamp'] >= cutoff_date]

            # Save to Parquet
            combined_df.to_parquet(parquet_path, index=False)
            logger.debug(f"Saved {len(combined_df)} candles to {parquet_path}")

        except Exception as e:
            logger.error(f"Error saving historical data: {e}")

    def get_historical_data(
        self,
        symbol: str,
        interval: str,
        force_refresh: bool = False
    ) -> Optional[pd.DataFrame]:
        """
        Get historical data from cache or Parquet.

        Args:
            symbol: Trading symbol
            interval: Timeframe
            force_refresh: Force load from Parquet

        Returns:
            DataFrame with historical data
        """
        cache_key = (symbol, interval)

        # Try memory cache first (unless force_refresh)
        if not force_refresh and cache_key in self.cache:
            return self.cache[cache_key]

        # Load from Parquet
        try:
            parquet_path = self._get_parquet_path(symbol, interval)

            if parquet_path.exists():
                df = pd.read_parquet(parquet_path)

                # Update cache
                if len(df) > 1000:
                    df = df.tail(1000).reset_index(drop=True)
                self.cache[cache_key] = df

                logger.debug(f"Loaded {len(df)} candles from Parquet: {symbol} {interval}")
                return df

        except Exception as e:
            logger.error(f"Error loading historical data: {e}")

        return None

    async def get_market_data(
        self,
        symbol: str,
        interval: str,
        limit: int = 100
    ) -> Optional[pd.DataFrame]:
        """
        Get market data (from cache or fetch).

        Args:
            symbol: Trading symbol
            interval: Timeframe
            limit: Maximum number of candles

        Returns:
            DataFrame with market data
        """
        df = self.get_historical_data(symbol, interval)

        if df is not None and len(df) > 0:
            return df.tail(limit).reset_index(drop=True)

        logger.warning(f"No data available for {symbol} {interval}")
        return None

    def clear_symbol_data(self, symbol: str, interval: str):
        """
        Clear cached data for specific symbol/interval.

        Args:
            symbol: Trading symbol
            interval: Timeframe
        """
        cache_key = (symbol, interval)
        if cache_key in self.cache:
            del self.cache[cache_key]
            logger.info(f"Cleared cache for {symbol} {interval}")

    def get_cached_symbols(self) -> List[Tuple[str, str]]:
        """
        Get list of cached symbol/interval pairs.

        Returns:
            List of (symbol, interval) tuples
        """
        return list(self.cache.keys())

    async def cleanup_old_historical_data(self, days_to_keep: int = 30):
        """
        Clean up old historical data from disk.

        Args:
            days_to_keep: Number of days to keep
        """
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)

        cleaned_count = 0
        for parquet_file in self.cache_dir.glob("*_history.parquet"):
            try:
                df = pd.read_parquet(parquet_file)
                if not df.empty:
                    # Filter to keep only recent data
                    recent_df = df[df['timestamp'] >= cutoff_date]

                    if len(recent_df) < len(df):
                        # Save cleaned data
                        recent_df.to_parquet(parquet_file, index=False)
                        cleaned_count += len(df) - len(recent_df)

            except Exception as e:
                logger.error(f"Error cleaning {parquet_file}: {e}")

        if cleaned_count > 0:
            logger.info(f"Cleaned up {cleaned_count} old candles from disk")
