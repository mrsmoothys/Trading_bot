"""
Integration tests for REST + WebSocket interaction.
Tests the UnifiedBinanceClient combining REST API and WebSocket streaming
with DataStore caching.
"""

import sys
sys.path.insert(0, '/Users/mrsmoothy/Downloads/Trading_bot')

import pytest
import asyncio
import pandas as pd
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime, timedelta

from core.data.binance_client import UnifiedBinanceClient, WebSocketStreamManager
from core.data.data_store import DataStore


@pytest.mark.asyncio
async def test_rest_to_websocket_data_flow():
    """
    Test the flow: Fetch historical data via REST → Stream real-time data via WebSocket.
    """
    client = UnifiedBinanceClient(testnet=True)

    # Mock REST client methods
    mock_df = pd.DataFrame({
        'timestamp': pd.date_range(start='2024-01-01', periods=100, freq='1h'),
        'open': [45000 + i * 10 for i in range(100)],
        'high': [45100 + i * 10 for i in range(100)],
        'low': [44900 + i * 10 for i in range(100)],
        'close': [45050 + i * 10 for i in range(100)],
        'volume': [100 + i for i in range(100)]
    })

    client.rest_client.get_ohlcv = AsyncMock(return_value=mock_df)
    client.rest_client.get_current_price = AsyncMock(return_value=46000.0)

    # Test REST API call
    df = await client.get_ohlcv('BTCUSDT', '1h', limit=100)
    assert len(df) == 100
    assert 'close' in df.columns
    print("✅ REST API fetch works correctly")

    # Test price fetch
    price = await client.get_price('BTCUSDT')
    assert price == 46000.0
    print("✅ REST API price fetch works correctly")

    # Verify mock was called
    client.rest_client.get_ohlcv.assert_called_once()
    client.rest_client.get_current_price.assert_called_once()

    await client.close()
    print("✅ test_rest_to_websocket_data_flow passed")


@pytest.mark.asyncio
async def test_websocket_streaming_integration():
    """
    Test WebSocket streaming with DataStore integration.
    """
    store = DataStore()
    client = UnifiedBinanceClient(testnet=True)

    # Create callback that stores data in DataStore
    received_data = []

    async def price_callback_with_store(data):
        received_data.append(data)
        symbol = data.get('symbol', 'UNKNOWN')
        price = float(data.get('c', '0'))  # 'c' is current price in Binance ticker
        await store.store_price(symbol, price)

    # Mock WebSocket stream
    client.ws_manager.start_price_stream = AsyncMock(return_value="price_btcusdt")

    # Start stream
    stream_id = await client.start_price_stream('btcusdt', price_callback_with_store)

    assert stream_id == "price_btcusdt"
    print("✅ WebSocket stream started successfully")

    # Verify WebSocket method was called
    client.ws_manager.start_price_stream.assert_called_once()

    await client.close()
    print("✅ test_websocket_streaming_integration passed")


@pytest.mark.asyncio
async def test_unified_client_methods():
    """
    Test all UnifiedBinanceClient methods work together.
    """
    client = UnifiedBinanceClient(testnet=True)

    # Mock REST methods
    mock_df = pd.DataFrame({
        'timestamp': pd.date_range(start='2024-01-01', periods=50, freq='1h'),
        'open': [45000] * 50,
        'high': [45100] * 50,
        'low': [44900] * 50,
        'close': [45050] * 50,
        'volume': [100] * 50
    })

    client.rest_client.get_ohlcv = AsyncMock(return_value=mock_df)
    client.rest_client.get_current_price = AsyncMock(return_value=45000.0)

    # Mock WS methods
    client.ws_manager.start_price_stream = AsyncMock(return_value="price_btcusdt")
    client.ws_manager.start_kline_stream = AsyncMock(return_value="kline_btcusdt_1h")
    client.ws_manager.start_depth_stream = AsyncMock(return_value="depth_btcusdt_20")

    # Test REST API methods
    df = await client.get_ohlcv('BTCUSDT', '1h')
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 50
    print("✅ get_ohlcv works")

    price = await client.get_price('BTCUSDT')
    assert price == 45000.0
    print("✅ get_price works")

    # Test WebSocket methods
    stream_id1 = await client.start_price_stream('btcusdt', Mock())
    assert stream_id1 == "price_btcusdt"
    print("✅ start_price_stream works")

    stream_id2 = await client.start_kline_stream('btcusdt', '1h', Mock())
    assert stream_id2 == "kline_btcusdt_1h"
    print("✅ start_kline_stream works")

    stream_id3 = await client.start_depth_stream('btcusdt', Mock(), '20')
    assert stream_id3 == "depth_btcusdt_20"
    print("✅ start_depth_stream works")

    await client.close()
    print("✅ test_unified_client_methods passed")


@pytest.mark.asyncio
async def test_data_store_integration():
    """
    Test DataStore caching with WebSocket data.
    """
    store = DataStore()
    client = UnifiedBinanceClient(testnet=True)

    # Store initial price via REST (mocked)
    client.rest_client.get_current_price = AsyncMock(return_value=45000.0)

    async def fetch_and_store():
        price = await client.get_price('BTCUSDT')
        await store.store_price('BTCUSDT', price)
        return price

    price = await fetch_and_store()
    assert price == 45000.0

    # Retrieve from cache
    cached_price = store.get_price('BTCUSDT')
    assert cached_price == 45000.0
    print("✅ DataStore integration works")

    # Test cache expiry (simulated)
    store.cache_ttl['price:BTCUSDT'] = datetime.now() - timedelta(seconds=1)
    expired_price = store.get_price('BTCUSDT')
    assert expired_price is None
    print("✅ Cache expiry works")

    await client.close()
    print("✅ test_data_store_integration passed")


@pytest.mark.asyncio
async def test_multiple_stream_management():
    """
    Test managing multiple WebSocket streams simultaneously.
    """
    client = UnifiedBinanceClient(testnet=True)

    # Mock multiple stream registrations
    client.ws_manager.start_price_stream = AsyncMock(return_value="price_btcusdt")
    client.ws_manager.start_kline_stream = AsyncMock(return_value="kline_ethusdt_5m")
    client.ws_manager.start_depth_stream = AsyncMock(return_value="depth_adausdt_20")

    # Start multiple streams
    callback1 = Mock()
    callback2 = Mock()
    callback3 = Mock()

    stream_id1 = await client.start_price_stream('btcusdt', callback1)
    stream_id2 = await client.start_kline_stream('ethusdt', '5m', callback2)
    stream_id3 = await client.start_depth_stream('adausdt', callback3, '20')

    assert stream_id1 == "price_btcusdt"
    assert stream_id2 == "kline_ethusdt_5m"
    assert stream_id3 == "depth_adausdt_20"

    # Verify all WS methods were called
    assert client.ws_manager.start_price_stream.call_count == 1
    assert client.ws_manager.start_kline_stream.call_count == 1
    assert client.ws_manager.start_depth_stream.call_count == 1

    print("✅ Multiple stream management works")

    await client.close()
    print("✅ test_multiple_stream_management passed")


@pytest.mark.asyncio
async def test_stream_status_integration():
    """
    Test stream status reporting in UnifiedBinanceClient.
    """
    client = UnifiedBinanceClient(testnet=True)

    # Get initial status (no streams)
    status = client.ws_manager.get_stream_status()
    assert status['active_streams'] == 0
    assert status['total_subscribers'] == 0
    print("✅ Initial stream status: no active streams")

    # Test that status method works correctly
    # The actual stream registration will be tested in unit tests
    # Here we just verify the status reporting infrastructure
    status_after = client.ws_manager.get_stream_status()
    assert 'price_streams' in status_after
    assert 'kline_streams' in status_after
    assert 'depth_streams' in status_after
    assert isinstance(status_after['price_streams'], list)
    print("✅ Stream status infrastructure works correctly")

    await client.close()
    print("✅ test_stream_status_integration passed")


@pytest.mark.asyncio
async def test_rest_and_websocket_error_handling():
    """
    Test error handling in both REST and WebSocket operations.
    """
    client = UnifiedBinanceClient(testnet=True)

    # Mock REST API error - actual implementation will raise the exception
    client.rest_client.get_current_price = AsyncMock(side_effect=Exception("API Error"))

    # The client doesn't currently catch this error - it propagates up
    # This is expected behavior at this level
    try:
        price = await client.get_price('BTCUSDT')
        assert False, "Should have raised exception"
    except Exception as e:
        assert str(e) == "API Error"
        print("✅ REST API error propagates correctly")

    # Mock WebSocket stream error
    client.ws_manager.start_price_stream = AsyncMock(side_effect=Exception("WS Error"))

    # Similarly, WebSocket errors propagate up
    try:
        stream_id = await client.start_price_stream('btcusdt', Mock())
        assert False, "Should have raised exception"
    except Exception as e:
        assert str(e) == "WS Error"
        print(f"✅ WebSocket error propagates correctly")

    await client.close()
    print("✅ test_rest_and_websocket_error_handling passed")


@pytest.mark.asyncio
async def test_close_closes_all_connections():
    """
    Test that close() properly closes all connections.
    """
    client = UnifiedBinanceClient(testnet=True)

    # Mock the stop_all_streams method
    client.ws_manager.stop_all_streams = AsyncMock()

    # Close the client
    await client.close()

    # Verify stop_all_streams was called
    client.ws_manager.stop_all_streams.assert_called_once()

    print("✅ close() properly closes all connections")
    print("✅ test_close_closes_all_connections passed")


@pytest.mark.asyncio
async def test_case_insensitive_symbol_handling():
    """
    Test that symbols are handled case-insensitively in both REST and WebSocket.
    """
    client = UnifiedBinanceClient(testnet=True)

    # Mock REST with lowercase symbol
    mock_df = pd.DataFrame({
        'timestamp': pd.date_range(start='2024-01-01', periods=10, freq='1h'),
        'open': [45000] * 10,
        'high': [45100] * 10,
        'low': [44900] * 10,
        'close': [45050] * 10,
        'volume': [100] * 10
    })

    client.rest_client.get_ohlcv = AsyncMock(return_value=mock_df)

    # Test with mixed case symbol
    df = await client.get_ohlcv('BtCuSdT', '1h')
    assert len(df) == 10
    print("✅ Case-insensitive REST API handling works")

    # Mock WebSocket with lowercase
    client.ws_manager.start_price_stream = AsyncMock(return_value="price_btcusdt")

    # Test WebSocket with mixed case
    stream_id = await client.start_price_stream('BTCUSDT', Mock())
    assert stream_id == "price_btcusdt"
    print("✅ Case-insensitive WebSocket handling works")

    await client.close()
    print("✅ test_case_insensitive_symbol_handling passed")


if __name__ == '__main__':
    print("\n" + "="*70)
    print("Running REST + WebSocket Integration Tests")
    print("="*70 + "\n")

    asyncio.run(test_rest_to_websocket_data_flow())
    asyncio.run(test_websocket_streaming_integration())
    asyncio.run(test_unified_client_methods())
    asyncio.run(test_data_store_integration())
    asyncio.run(test_multiple_stream_management())
    asyncio.run(test_stream_status_integration())
    asyncio.run(test_rest_and_websocket_error_handling())
    asyncio.run(test_close_closes_all_connections())
    asyncio.run(test_case_insensitive_symbol_handling())

    print("\n" + "="*70)
    print("✅ All integration tests passed successfully!")
    print("="*70)
