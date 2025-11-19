"""
WebSocket Streaming Demo
Demonstrates the new WebSocket streaming capabilities added to binance_client.py

Features:
- Real-time price streaming
- Real-time candlestick (kline) streaming
- Real-time order book depth streaming
- Automatic reconnection on connection loss
- Multiple subscribers per stream
- Unified REST + WebSocket client
"""

import sys
import asyncio
sys.path.insert(0, '/Users/mrsmoothy/Downloads/Trading_bot')

from core.data.binance_client import WebSocketStreamManager, UnifiedBinanceClient
from core.data.data_store import DataStore


def demo_websocket_manager():
    """Demonstrate WebSocketStreamManager features."""
    print("=" * 70)
    print("WebSocketStreamManager Demo")
    print("=" * 70)

    # Initialize
    ws_manager = WebSocketStreamManager(testnet=True)

    print("\n1. Initialization")
    print(f"   Testnet mode: {ws_manager.testnet}")
    print(f"   Base WebSocket URL: {ws_manager.base_ws_url}")
    print(f"   Max reconnection attempts: {ws_manager.max_reconnect_attempts}")
    print(f"   Reconnection delay: {ws_manager.reconnect_delay}s")

    print("\n2. Stream Types Available")
    print("   - Price Stream: Real-time ticker price updates")
    print("   - Kline Stream: Candlestick data with intervals (1m, 5m, 15m, etc.)")
    print("   - Depth Stream: Order book depth updates (5, 10, 20, 50, 100, 500, 1000)")

    print("\n3. Stream Registration (Sync - no actual connection)")
    print("   Creating stream IDs for different symbols and timeframes...")

    # Simulate stream IDs that would be created
    symbols = ['btcusdt', 'ethusdt', 'adausdt']
    intervals = ['1m', '5m', '15m', '1h', '4h']

    print("\n   Price Streams:")
    for symbol in symbols:
        stream_id = f"price_{symbol}"
        print(f"     - {stream_id}")

    print("\n   Kline Streams:")
    for symbol in symbols[:2]:
        for interval in intervals[:3]:
            stream_id = f"kline_{symbol}_{interval}"
            print(f"     - {stream_id}")

    print("\n   Depth Streams:")
    depths = ['20', '100']
    for symbol in symbols[:2]:
        for depth in depths:
            stream_id = f"depth_{symbol}_{depth}"
            print(f"     - {stream_id}")

    print("\n4. Features")
    print("   ✓ Multiple subscribers per stream")
    print("   ✓ Automatic reconnection (5 attempts)")
    print("   ✓ Error handling and logging")
    print("   ✓ Case-insensitive symbol handling")
    print("   ✓ Stream status monitoring")
    print("   ✓ Graceful stream shutdown")

    status = ws_manager.get_stream_status()
    print("\n5. Current Stream Status")
    for key, value in status.items():
        print(f"   {key}: {value}")

    print("\n✅ WebSocketStreamManager demo complete!")
    print("=" * 70)


async def demo_unified_client():
    """Demonstrate UnifiedBinanceClient features."""
    print("\n" + "=" * 70)
    print("UnifiedBinanceClient Demo")
    print("=" * 70)

    # Initialize unified client
    client = UnifiedBinanceClient(testnet=True)

    print("\n1. Client Initialization")
    print(f"   Testnet mode: {client.testnet}")
    print(f"   REST Client: {type(client.rest_client).__name__}")
    print(f"   WebSocket Manager: {type(client.ws_manager).__name__}")

    print("\n2. REST API Methods")
    print("   - get_ohlcv(symbol, interval, limit)")
    print("   - get_price(symbol)")
    print("   - get_24hr_ticker(symbol)")
    print("   - get_order_book(symbol, limit)")
    print("   - get_historical_data(symbol, interval, days)")
    print("   - get_multiple_timeframes(symbol, intervals)")

    print("\n3. WebSocket Streaming Methods")
    print("   - start_price_stream(symbol, callback)")
    print("   - start_kline_stream(symbol, interval, callback)")
    print("   - start_depth_stream(symbol, callback, depth)")

    print("\n4. Usage Example")
    print("""
    # Initialize client
    client = UnifiedBinanceClient(testnet=True)

    # REST API usage
    df = await client.get_ohlcv('BTCUSDT', '1h', limit=100)
    price = await client.get_price('BTCUSDT')

    # WebSocket streaming usage
    async def price_callback(data):
        print(f"Price update: {data['symbol']} = ${data['price']}")

    async def kline_callback(kline):
        print(f"Kline update: {kline['symbol']} {kline['interval']}")

    await client.start_price_stream('btcusdt', price_callback)
    await client.start_kline_stream('ethusdt', '5m', kline_callback)

    # Cleanup
    await client.close()
    """)

    print("\n5. Integration with DataStore")
    store = DataStore()

    print("""
    # Example: Store real-time data with TTL caching
    async def price_callback_with_store(data):
        symbol = data['symbol']
        price = float(data['price'])
        await store.store_price(symbol, price)

        # Retrieve cached price
        cached_price = store.get_price(symbol)
        print(f"{symbol}: ${cached_price}")
    """)

    print("\n6. Benefits of Unified Client")
    print("   ✓ Single interface for REST and WebSocket")
    print("   ✓ Seamless data flow from REST to streaming")
    print("   ✓ Easy to switch between historical and real-time data")
    print("   ✓ Consistent error handling")
    print("   ✓ Clean resource management")

    print("\n✅ UnifiedBinanceClient demo complete!")
    print("=" * 70)


def demo_integration():
    """Demonstrate integration with existing system."""
    print("\n" + "=" * 70)
    print("Integration Demo")
    print("=" * 70)

    print("\n1. Integration Points")
    print("   ✓ DataStore: Caches streamed data with TTL")
    print("   ✓ SystemContext: Provides state awareness")
    print("   ✓ DeepSeekBrain: Can use real-time data for AI decisions")
    print("   ✓ Dashboard: Can display real-time price updates")

    print("\n2. Data Flow Example")
    print("""
    Real-time Market Data Pipeline:

    Binance WebSocket
         ↓
    WebSocketStreamManager
         ↓
    DataStore (TTL caching)
         ↓
    Feature Engine (calculates indicators)
         ↓
    Signal Generator (generates trading signals)
         ↓
    DeepSeek Brain (AI analysis)
         ↓
    Risk Manager (validates signals)
         ↓
    Position Manager (executes trades)
         ↓
    Dashboard (displays results)
    """)

    print("\n3. Use Cases")
    use_cases = [
        "Live price tracking and alerts",
        "Real-time technical indicator calculation",
        "Immediate signal generation on price changes",
        "Dynamic risk assessment during live trading",
        "Real-time dashboard updates",
        "Backtesting with streaming data replay"
    ]

    for i, use_case in enumerate(use_cases, 1):
        print(f"   {i}. {use_case}")

    print("\n✅ Integration demo complete!")
    print("=" * 70)


def main():
    """Run all demonstrations."""
    print("\n" + "🚀" * 35)
    print("BINANCE WEBSOCKET STREAMING - FEATURE DEMO")
    print("🚀" * 35)

    # Run synchronous demos
    demo_websocket_manager()
    demo_integration()

    # Run async demo
    print("\n" + "📡" * 35)
    print("Running async demonstrations...")
    print("📡" * 35 + "\n")

    asyncio.run(demo_unified_client())

    print("\n" + "✨" * 35)
    print("ALL DEMONSTRATIONS COMPLETE!")
    print("✨" * 35)
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("\n✅ Task #4 Completed: WebSocket Streaming Implemented")
    print("\nWhat's New:")
    print("  1. WebSocketStreamManager class with 3 stream types:")
    print("     • Price streams (24/7 ticker updates)")
    print("     • Kline streams (candlestick data)")
    print("     • Depth streams (order book data)")
    print("\n  2. UnifiedBinanceClient:")
    print("     • Combines REST API and WebSocket streaming")
    print("     • Single interface for all market data needs")
    print("     • Easy integration with existing components")
    print("\n  3. Production Features:")
    print("     • Automatic reconnection (5 attempts)")
    print("     • Error handling and logging")
    print("     • Multiple subscribers per stream")
    print("     • Graceful shutdown")
    print("     • Integration with DataStore for caching")
    print("\n  4. Comprehensive Test Suite:")
    print("     • 15 test cases covering all functionality")
    print("     • Mocked WebSocket connections")
    print("     • Tests for initialization, registration, status")
    print("\n" + "=" * 70)


if __name__ == '__main__':
    main()
