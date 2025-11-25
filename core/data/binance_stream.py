"""
Binance WebSocket stream client with reconnect logic.
Handles real-time kline data from Binance futures API.
"""

import asyncio
import json
import logging
from typing import Callable, Optional, Dict, Any
from datetime import datetime, timezone
import websockets
from loguru import logger

logger = logging.getLogger(__name__)


class BinanceStream:
    """
    Binance WebSocket stream client for real-time market data.

    Features:
    - Automatic reconnection on disconnect
    - Configurable ping interval
    - Callback-based event handling
    - Support for multiple symbols and timeframes
    """

    def __init__(
        self,
        ping_interval: int = 20,
        ping_timeout: int = 10,
        reconnect_delay: int = 5
    ):
        """
        Initialize Binance stream client.

        Args:
            ping_interval: WebSocket ping interval in seconds
            ping_timeout: WebSocket ping timeout in seconds
            reconnect_delay: Delay between reconnection attempts
        """
        self.ping_interval = ping_interval
        self.ping_timeout = ping_timeout
        self.reconnect_delay = reconnect_delay
        self.running = False
        self.tasks = []

    async def stream_klines(
        self,
        symbol: str,
        interval: str,
        handler: Callable[[str, str, Dict[str, Any]], None]
    ):
        """
        Stream kline data for a symbol.

        Args:
            symbol: Trading symbol (e.g., 'BTCUSDT')
            interval: Kline interval (e.g., '1m', '5m', '1h')
            handler: Async callback function to handle kline data
        """
        self.running = True
        stream_name = f"{symbol.lower()}@kline_{interval}"
        url = f"wss://fstream.binance.com/ws/{stream_name}"

        logger.info(f"Starting WebSocket stream for {symbol} {interval}")

        while self.running:
            try:
                async with websockets.connect(
                    url,
                    ping_interval=self.ping_interval,
                    ping_timeout=self.ping_timeout
                ) as ws:
                    logger.info(f"Connected to {url}")

                    async for message in ws:
                        try:
                            data = json.loads(message)
                            kline = data.get('k', {})

                            # Only process closed klines
                            if kline.get('x', False):
                                await handler(symbol, interval, kline)

                        except json.JSONDecodeError as e:
                            logger.error(f"Failed to decode message: {e}")
                        except Exception as e:
                            logger.error(f"Error processing message: {e}")

            except websockets.exceptions.ConnectionClosedError:
                logger.warning(f"Connection closed for {symbol} {interval}, reconnecting...")
            except websockets.exceptions.InvalidURI:
                logger.error(f"Invalid WebSocket URI: {url}")
                break
            except Exception as e:
                logger.error(f"WebSocket error for {symbol} {interval}: {e}")
                logger.info(f"Reconnecting in {self.reconnect_delay} seconds...")

            if self.running:
                await asyncio.sleep(self.reconnect_delay)

        logger.info(f"Stopped WebSocket stream for {symbol} {interval}")

    async def stream_multiple_symbols(
        self,
        symbols: list,
        interval: str,
        handler: Callable[[str, str, Dict[str, Any]], None]
    ):
        """
        Stream kline data for multiple symbols.

        Args:
            symbols: List of trading symbols
            interval: Kline interval
            handler: Async callback function
        """
        tasks = []
        for symbol in symbols:
            task = asyncio.create_task(
                self.stream_klines(symbol, interval, handler)
            )
            tasks.append(task)

        await asyncio.gather(*tasks)

    def stop(self):
        """Stop all WebSocket streams."""
        logger.info("Stopping all WebSocket streams...")
        self.running = False

        for task in self.tasks:
            task.cancel()

    async def close(self):
        """Close WebSocket streams and cleanup."""
        self.stop()
        await asyncio.gather(*self.tasks, return_exceptions=True)
        logger.info("WebSocket streams closed")


# Example usage and testing
async def example_handler(symbol: str, interval: str, kline_data: Dict[str, Any]):
    """
    Example handler for kline data.

    Args:
        symbol: Trading symbol
        interval: Kline interval
        kline_data: Kline data from Binance
    """
    kline = kline_data
    event_time = datetime.fromtimestamp(kline['t'] / 1000, tz=timezone.utc)

    logger.info(
        f"[{event_time}] {symbol} {interval} - "
        f"O: {kline['o']}, H: {kline['h']}, L: {kline['l']}, C: {kline['c']}, "
        f"V: {kline['v']}, Status: {kline['x']}"
    )


async def main():
    """Main function to test the stream."""
    stream = BinanceStream()

    # Start streaming for multiple symbols
    symbols = ['btcusdt', 'ethusdt']
    interval = '1m'

    try:
        await stream.stream_multiple_symbols(symbols, interval, example_handler)
    except KeyboardInterrupt:
        logger.info("Received interrupt signal")
    finally:
        await stream.close()


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Run the stream
    asyncio.run(main())
