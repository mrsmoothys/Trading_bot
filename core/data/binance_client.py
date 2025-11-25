"""
Binance Client
Market data fetching and management for Binance Futures Demo API.
Handles OHLCV data, account info, and real-time price feeds.
"""

import os
import time
import asyncio
import json
import websockets
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Set
import pandas as pd
import requests
from loguru import logger


class BinanceClient:
    """
    Binance API client for futures trading.
    Fetches market data and provides real-time price feeds.
    """

    def __init__(self):
        """Initialize Binance client with URL configuration."""
        self.api_key = os.getenv('BINANCE_API_KEY')
        self.api_secret = os.getenv('BINANCE_API_SECRET')
        self.testnet = os.getenv('BINANCE_TESTNET', 'false').lower() == 'true'

        # Determine futures URL - honor BINANCE_FUTURES_URL if provided
        futures_url = os.getenv('BINANCE_FUTURES_URL')
        if futures_url:
            self.base_url = futures_url
        elif self.testnet:
            self.base_url = 'https://testnet.binancefuture.com/fapi/v1'
        else:
            self.base_url = 'https://fapi.binance.com/fapi/v1'

        # Determine spot URL for reference
        spot_url = os.getenv('BINANCE_SPOT_URL')
        if spot_url:
            self.spot_base_url = spot_url
        elif self.testnet:
            self.spot_base_url = 'https://testnet.binance.vision/api/v3'
        else:
            self.spot_base_url = 'https://api.binance.com/api/v3'

        # Log endpoint selection
        endpoint_type = "TESTNET" if self.testnet else "LIVE"
        safe_key = self.api_key or ""
        key_preview = f"{safe_key[:4]}...{safe_key[-4:]}" if safe_key else "None"
        logger.info(f"Initialized Binance Client - Mode: {endpoint_type}")
        logger.info(f"  Futures URL: {self.base_url}")
        logger.info(f"  Spot URL: {self.spot_base_url}")
        logger.info(f"  API Key: {key_preview}")

        self.session = requests.Session()
        self.rate_limit = 1200  # requests per minute
        self._last_request_time = 0

    def _rate_limit(self):
        """Implement rate limiting."""
        now = time.time()
        elapsed = now - self._last_request_time
        min_interval = 60.0 / self.rate_limit
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)
        self._last_request_time = time.time()

    def _make_request(self, method: str, endpoint: str, params: Dict = None) -> Dict:
        """
        Make authenticated API request.

        Args:
            method: HTTP method
            endpoint: API endpoint
            params: Request parameters

        Returns:
            API response data
        """
        self._rate_limit()

        url = f"{self.base_url}{endpoint}"

        if method == "GET":
            response = self.session.get(url, params=params, timeout=10)
        elif method == "POST":
            response = self.session.post(url, params=params, timeout=10)
        else:
            raise ValueError(f"Unsupported method: {method}")

        if response.status_code != 200:
            logger.error(f"API request failed: {response.status_code} - {response.text}")
            raise Exception(f"API error: {response.status_code}")

        return response.json()

    async def get_ohlcv(
        self,
        symbol: str,
        interval: str,
        limit: int = 500,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Get OHLCV (candlestick) data.

        Args:
            symbol: Trading symbol (e.g., 'BTCUSDT')
            interval: Time interval (1m, 5m, 15m, 1h, 4h, 1d)
            limit: Number of candles (max 1500)
            start_time: Start time in milliseconds
            end_time: End time in milliseconds

        Returns:
            DataFrame with OHLCV data
        """
        try:
            params = {
                'symbol': symbol,
                'interval': interval,
                'limit': limit
            }

            if start_time:
                params['startTime'] = start_time
            if end_time:
                params['endTime'] = end_time

            data = self._make_request('GET', '/klines', params)

            # Convert to DataFrame
            df = pd.DataFrame(data, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume',
                'ignore'
            ])

            # Convert types
            # Normalize to UTC to avoid local timezone drift in plots
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
            df['open'] = df['open'].astype(float)
            df['high'] = df['high'].astype(float)
            df['low'] = df['low'].astype(float)
            df['close'] = df['close'].astype(float)
            df['volume'] = df['volume'].astype(float)

            # Select only needed columns
            df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]

            logger.info(f"Fetched {len(df)} candles for {symbol} {interval}")

            return df

        except Exception as e:
            logger.error(f"Error fetching OHLCV data: {e}")
            raise

    async def get_current_price(self, symbol: str) -> float:
        """
        Get current price for a symbol.

        Args:
            symbol: Trading symbol

        Returns:
            Current price
        """
        try:
            data = self._make_request('GET', '/ticker/price', {'symbol': symbol})
            return float(data['price'])
        except Exception as e:
            logger.error(f"Error fetching price: {e}")
            raise

    async def get_24hr_ticker(self, symbol: str) -> Dict[str, Any]:
        """
        Get 24hr ticker statistics.

        Args:
            symbol: Trading symbol

        Returns:
            24hr statistics
        """
        try:
            data = self._make_request('GET', '/ticker/24hr', {'symbol': symbol})

            # Convert numeric fields
            for key in [
                'lastPrice', 'lastQty', 'bidPrice', 'bidQty',
                'askPrice', 'askQty', 'openPrice', 'highPrice',
                'lowPrice', 'volume', 'quoteVolume', 'priceChange',
                'priceChangePercent', 'weightedAvgPrice'
            ]:
                if key in data:
                    data[key] = float(data[key])

            return data
        except Exception as e:
            logger.error(f"Error fetching 24hr ticker: {e}")
            raise

    async def get_order_book(self, symbol: str, limit: int = 100) -> Dict[str, Any]:
        """
        Get order book depth.

        Args:
            symbol: Trading symbol
            limit: Depth limit (5, 10, 20, 50, 100, 500, 1000)

        Returns:
            Order book data
        """
        try:
            data = self._make_request('GET', '/depth', {
                'symbol': symbol,
                'limit': limit
            })

            # Convert to float
            data['bids'] = [[float(bid[0]), float(bid[1])] for bid in data['bids']]
            data['asks'] = [[float(ask[0]), float(ask[1])] for ask in data['asks']]

            return data
        except Exception as e:
            logger.error(f"Error fetching order book: {e}")
            raise

    async def get_exchange_info(self) -> Dict[str, Any]:
        """
        Get exchange trading rules and symbol information.

        Returns:
            Exchange information
        """
        try:
            return self._make_request('GET', '/exchangeInfo')
        except Exception as e:
            logger.error(f"Error fetching exchange info: {e}")
            raise

    async def get_account_info(self) -> Dict[str, Any]:
        """
        Get futures account information.

        Returns:
            Account information
        """
        try:
            # Note: This requires authentication and may not work in testnet
            # without proper setup
            params = {'timestamp': int(time.time() * 1000)}
            # Would need to add signature here for authenticated requests
            logger.warning("Account info requires authentication - not fully implemented")
            return {}
        except Exception as e:
            logger.error(f"Error fetching account info: {e}")
            return {}

    async def get_historical_data(
        self,
        symbol: str,
        interval: str,
        days: int = 30
    ) -> pd.DataFrame:
        """
        Get historical data for multiple days.

        Args:
            symbol: Trading symbol
            interval: Time interval
            days: Number of days of history

        Returns:
            DataFrame with historical OHLCV data
        """
        try:
            # Calculate start and end times
            end_time = int(datetime.now().timestamp() * 1000)
            start_time = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)

            # Fetch data in chunks if needed
            all_data = []
            current_start = start_time

            while current_start < end_time:
                # Binance max limit is 1500 candles
                chunk_days = min(30, days)
                chunk_end = int((datetime.fromtimestamp(current_start / 1000) + timedelta(days=chunk_days)).timestamp() * 1000)

                df = await self.get_ohlcv(
                    symbol=symbol,
                    interval=interval,
                    limit=1500,
                    start_time=current_start,
                    end_time=chunk_end
                )

                if len(df) == 0:
                    break

                all_data.append(df)

                current_start = int(df['timestamp'].iloc[-1].timestamp() * 1000)

                # Avoid rate limiting
                await asyncio.sleep(0.1)

            if not all_data:
                logger.warning(f"No data fetched for {symbol}")
                return pd.DataFrame()

            # Combine all chunks
            result = pd.concat(all_data, ignore_index=True)
            result = result.drop_duplicates(subset=['timestamp']).sort_values('timestamp')

            logger.info(f"Fetched {len(result)} historical candles for {symbol}")

            return result

        except Exception as e:
            logger.error(f"Error fetching historical data: {e}")
            raise

    async def get_multiple_timeframes(
        self,
        symbol: str,
        intervals: List[str] = ['5m', '15m', '1h', '4h']
    ) -> Dict[str, pd.DataFrame]:
        """
        Get data for multiple timeframes.

        Args:
            symbol: Trading symbol
            intervals: List of timeframes

        Returns:
            Dict mapping timeframe to DataFrame
        """
        try:
            result = {}

            for interval in intervals:
                df = await self.get_ohlcv(symbol, interval, limit=500)
                result[interval] = df

                # Rate limiting
                await asyncio.sleep(0.05)

            logger.info(f"Fetched data for {len(intervals)} timeframes for {symbol}")

            return result

        except Exception as e:
            logger.error(f"Error fetching multi-timeframe data: {e}")
            raise

    def format_symbol(self, symbol: str) -> str:
        """Format symbol for API (uppercase)."""
        return symbol.upper()

    def interval_to_minutes(self, interval: str) -> int:
        """Convert interval string to minutes."""
        mapping = {
            '1m': 1, '3m': 3, '5m': 5, '15m': 15, '30m': 30,
            '1h': 60, '2h': 120, '4h': 240, '6h': 360, '8h': 480, '12h': 720,
            '1d': 1440, '3d': 4320, '1w': 10080, '1M': 43200
        }
        return mapping.get(interval, 60)


class WebSocketStreamManager:
    """
    WebSocket stream manager for real-time market data from Binance.
    Handles multiple streams with automatic reconnection.
    """

    def __init__(self, testnet: bool = True):
        """
        Initialize WebSocket stream manager.

        Args:
            testnet: Use testnet or mainnet
        """
        self.testnet = testnet
        self.base_ws_url = 'wss://stream.binancefuture.com/ws' if not testnet else 'wss://stream.binancefuture.com/ws'

        # Active streams
        self.active_streams: Dict[str, websockets.WebSocketServerProtocol] = {}
        self.stream_subscribers: Dict[str, Set[Callable]] = {}

        # Connection management
        self.max_reconnect_attempts = 5
        self.reconnect_delay = 5  # seconds

        # Track subscriptions
        self.price_streams: Set[str] = set()
        self.kline_streams: Dict[str, str] = {}  # {symbol_interval: symbol}
        self.depth_streams: Set[str] = set()

        logger.info(f"WebSocketStreamManager initialized (testnet={testnet})")

    async def start_price_stream(
        self,
        symbol: str,
        callback: Callable[[Dict[str, Any]], None]
    ) -> str:
        """
        Start streaming price updates for a symbol.

        Args:
            symbol: Trading symbol (e.g., 'btcusdt')
            callback: Function to call with price data

        Returns:
            Stream ID
        """
        stream_id = f"price_{symbol.lower()}"

        # Register callback
        if stream_id not in self.stream_subscribers:
            self.stream_subscribers[stream_id] = set()
        self.stream_subscribers[stream_id].add(callback)

        # Start stream if not already running
        if stream_id not in self.active_streams:
            self.price_streams.add(symbol.lower())
            await self._connect_price_stream(stream_id)

        logger.info(f"Started price stream for {symbol}")
        return stream_id

    async def start_kline_stream(
        self,
        symbol: str,
        interval: str,
        callback: Callable[[Dict[str, Any]], None]
    ) -> str:
        """
        Start streaming kline/candlestick updates.

        Args:
            symbol: Trading symbol
            interval: Time interval (1m, 5m, etc.)
            callback: Function to call with kline data

        Returns:
            Stream ID
        """
        stream_id = f"kline_{symbol.lower()}_{interval}"

        # Register callback
        if stream_id not in self.stream_subscribers:
            self.stream_subscribers[stream_id] = set()
        self.stream_subscribers[stream_id].add(callback)

        # Start stream if not already running
        if stream_id not in self.active_streams:
            key = f"{symbol.lower()}@{interval}"
            self.kline_streams[key] = symbol.lower()
            await self._connect_kline_stream(stream_id)

        logger.info(f"Started kline stream for {symbol} {interval}")
        return stream_id

    async def start_depth_stream(
        self,
        symbol: str,
        callback: Callable[[Dict[str, Any]], None],
        depth: str = '20'
    ) -> str:
        """
        Start streaming order book depth updates.

        Args:
            symbol: Trading symbol
            callback: Function to call with depth data
            depth: Depth level (5, 10, 20, 50, 100, 500, 1000)

        Returns:
            Stream ID
        """
        stream_id = f"depth_{symbol.lower()}_{depth}"

        # Register callback
        if stream_id not in self.stream_subscribers:
            self.stream_subscribers[stream_id] = set()
        self.stream_subscribers[stream_id].add(callback)

        # Start stream if not already running
        if stream_id not in self.active_streams:
            key = f"{symbol.lower()}@{depth}depth"
            self.depth_streams.add(key)
            await self._connect_depth_stream(stream_id)

        logger.info(f"Started depth stream for {symbol} {depth}")
        return stream_id

    async def _connect_price_stream(self, stream_id: str):
        """Connect to price stream."""
        try:
            symbols_str = '/'.join([f"{s}@ticker" for s in self.price_streams])
            stream_name = f"{symbols_str}"

            async with websockets.connect(f"{self.base_ws_url}/{stream_name}") as websocket:
                self.active_streams[stream_id] = websocket
                logger.info(f"Connected to price stream: {stream_id}")

                async for message in websocket:
                    data = json.loads(message)

                    # Notify all subscribers
                    for callback in self.stream_subscribers.get(stream_id, set()):
                        try:
                            callback(data)
                        except Exception as e:
                            logger.error(f"Error in price callback: {e}")

        except Exception as e:
            logger.error(f"Error in price stream {stream_id}: {e}")
            await self._reconnect(stream_id, self._connect_price_stream)

    async def _connect_kline_stream(self, stream_id: str):
        """Connect to kline stream."""
        try:
            streams_str = '/'.join([f"{s}@kline" for s in self.kline_streams.keys()])
            stream_name = f"{streams_str}"

            async with websockets.connect(f"{self.base_ws_url}/{stream_name}") as websocket:
                self.active_streams[stream_id] = websocket
                logger.info(f"Connected to kline stream: {stream_id}")

                async for message in websocket:
                    data = json.loads(message)

                    # Extract kline data
                    if 'kline' in data:
                        kline = data['kline']

                        # Notify all subscribers
                        for callback in self.stream_subscribers.get(stream_id, set()):
                            try:
                                callback(kline)
                            except Exception as e:
                                logger.error(f"Error in kline callback: {e}")

        except Exception as e:
            logger.error(f"Error in kline stream {stream_id}: {e}")
            await self._reconnect(stream_id, self._connect_kline_stream)

    async def _connect_depth_stream(self, stream_id: str):
        """Connect to depth stream."""
        try:
            streams_str = '/'.join(self.depth_streams)
            stream_name = f"{streams_str}"

            async with websockets.connect(f"{self.base_ws_url}/{stream_name}") as websocket:
                self.active_streams[stream_id] = websocket
                logger.info(f"Connected to depth stream: {stream_id}")

                async for message in websocket:
                    data = json.loads(message)

                    # Notify all subscribers
                    for callback in self.stream_subscribers.get(stream_id, set()):
                        try:
                            callback(data)
                        except Exception as e:
                            logger.error(f"Error in depth callback: {e}")

        except Exception as e:
            logger.error(f"Error in depth stream {stream_id}: {e}")
            await self._reconnect(stream_id, self._connect_depth_stream)

    async def _reconnect(self, stream_id: str, connect_func):
        """Attempt to reconnect a stream."""
        for attempt in range(self.max_reconnect_attempts):
            logger.info(f"Reconnecting {stream_id} (attempt {attempt + 1}/{self.max_reconnect_attempts})")
            await asyncio.sleep(self.reconnect_delay)

            try:
                await connect_func(stream_id)
                return  # Success
            except Exception as e:
                logger.error(f"Reconnection failed: {e}")

        logger.error(f"Max reconnection attempts reached for {stream_id}")

    async def stop_stream(self, stream_id: str):
        """Stop a specific stream."""
        if stream_id in self.active_streams:
            websocket = self.active_streams.pop(stream_id)
            await websocket.close()
            logger.info(f"Stopped stream: {stream_id}")

    async def stop_all_streams(self):
        """Stop all active streams."""
        for stream_id in list(self.active_streams.keys()):
            await self.stop_stream(stream_id)

        self.price_streams.clear()
        self.kline_streams.clear()
        self.depth_streams.clear()
        self.stream_subscribers.clear()

        logger.info("All streams stopped")

    def get_stream_status(self) -> Dict[str, Any]:
        """Get status of all streams."""
        return {
            "active_streams": len(self.active_streams),
            "price_streams": list(self.price_streams),
            "kline_streams": list(self.kline_streams.keys()),
            "depth_streams": list(self.depth_streams),
            "total_subscribers": sum(len(subs) for subs in self.stream_subscribers.values())
        }


class UnifiedBinanceClient:
    """
    Unified client combining REST API and WebSocket streaming.
    Provides seamless access to market data.
    """

    def __init__(self, testnet: bool = True):
        """
        Initialize unified client.

        Args:
            testnet: Use testnet or mainnet
        """
        self.rest_client = BinanceClient()
        self.ws_manager = WebSocketStreamManager(testnet=testnet)
        self.testnet = testnet

        logger.info("UnifiedBinanceClient initialized")

    async def get_ohlcv(self, *args, **kwargs) -> pd.DataFrame:
        """Get OHLCV data via REST API."""
        return await self.rest_client.get_ohlcv(*args, **kwargs)

    async def get_price(self, symbol: str) -> float:
        """Get current price via REST API."""
        return await self.rest_client.get_current_price(symbol)

    async def start_price_stream(self, symbol: str, callback: Callable[[Dict], None]) -> str:
        """Start real-time price stream."""
        return await self.ws_manager.start_price_stream(symbol, callback)

    async def start_kline_stream(
        self,
        symbol: str,
        interval: str,
        callback: Callable[[Dict], None]
    ) -> str:
        """Start real-time kline stream."""
        return await self.ws_manager.start_kline_stream(symbol, interval, callback)

    async def start_depth_stream(
        self,
        symbol: str,
        callback: Callable[[Dict], None],
        depth: str = '20'
    ) -> str:
        """Start real-time depth stream."""
        return await self.ws_manager.start_depth_stream(symbol, callback, depth)

    async def close(self):
        """Close all connections."""
        await self.ws_manager.stop_all_streams()
        logger.info("UnifiedBinanceClient closed")
