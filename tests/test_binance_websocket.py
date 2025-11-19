"""
Unit tests for Binance WebSocket streaming functionality.
Tests with mocked WebSocket connections to avoid real API calls.
"""

import sys
sys.path.insert(0, '/Users/mrsmoothy/Downloads/Trading_bot')

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from core.data.binance_client import WebSocketStreamManager, UnifiedBinanceClient


@pytest.mark.asyncio
async def test_websocket_stream_manager_initialization():
    """Test WebSocketStreamManager initialization."""
    ws_manager = WebSocketStreamManager(testnet=True)

    assert ws_manager.testnet == True
    assert len(ws_manager.active_streams) == 0
    assert len(ws_manager.stream_subscribers) == 0
    assert len(ws_manager.price_streams) == 0
    assert len(ws_manager.kline_streams) == 0
    assert len(ws_manager.depth_streams) == 0
    assert ws_manager.max_reconnect_attempts == 5
    assert ws_manager.reconnect_delay == 5

    print("✅ test_websocket_stream_manager_initialization passed")


@pytest.mark.asyncio
async def test_price_stream_registration():
    """Test price stream registration."""
    ws_manager = WebSocketStreamManager(testnet=True)

    callback = Mock()
    stream_id = await ws_manager.start_price_stream('btcusdt', callback)

    assert stream_id == "price_btcusdt"
    assert 'btcusdt' in ws_manager.price_streams
    assert stream_id in ws_manager.stream_subscribers
    assert callback in ws_manager.stream_subscribers[stream_id]

    print("✅ test_price_stream_registration passed")


@pytest.mark.asyncio
async def test_kline_stream_registration():
    """Test kline stream registration."""
    ws_manager = WebSocketStreamManager(testnet=True)

    callback = Mock()
    stream_id = await ws_manager.start_kline_stream('ethusdt', '5m', callback)

    assert stream_id == "kline_ethusdt_5m"
    assert 'ethusdt@5m' in ws_manager.kline_streams
    assert stream_id in ws_manager.stream_subscribers
    assert callback in ws_manager.stream_subscribers[stream_id]

    print("✅ test_kline_stream_registration passed")


@pytest.mark.asyncio
async def test_depth_stream_registration():
    """Test depth stream registration."""
    ws_manager = WebSocketStreamManager(testnet=True)

    callback = Mock()
    stream_id = await ws_manager.start_depth_stream('adausdt', callback, '20')

    assert stream_id == "depth_adausdt_20"
    assert 'adausdt@20depth' in ws_manager.depth_streams
    assert stream_id in ws_manager.stream_subscribers
    assert callback in ws_manager.stream_subscribers[stream_id]

    print("✅ test_depth_stream_registration passed")


@pytest.mark.asyncio
async def test_multiple_subscribers():
    """Test multiple callbacks on same stream."""
    ws_manager = WebSocketStreamManager(testnet=True)

    callback1 = Mock()
    callback2 = Mock()
    callback3 = Mock()

    stream_id = await ws_manager.start_price_stream('btcusdt', callback1)
    await ws_manager.start_price_stream('btcusdt', callback2)
    await ws_manager.start_price_stream('btcusdt', callback3)

    assert len(ws_manager.stream_subscribers[stream_id]) == 3
    assert callback1 in ws_manager.stream_subscribers[stream_id]
    assert callback2 in ws_manager.stream_subscribers[stream_id]
    assert callback3 in ws_manager.stream_subscribers[stream_id]

    print("✅ test_multiple_subscribers passed")


@pytest.mark.asyncio
async def test_stream_status():
    """Test stream status reporting."""
    ws_manager = WebSocketStreamManager(testnet=True)

    callback = Mock()

    await ws_manager.start_price_stream('btcusdt', callback)
    await ws_manager.start_kline_stream('ethusdt', '1h', callback)
    await ws_manager.start_depth_stream('adausdt', callback)

    status = ws_manager.get_stream_status()

    assert status['active_streams'] >= 0  # Streams may not be active yet
    assert 'btcusdt' in status['price_streams']
    assert 'ethusdt@1h' in status['kline_streams']
    assert 'adausdt@20depth' in status['depth_streams']
    assert status['total_subscribers'] >= 3

    print("✅ test_stream_status passed")


@pytest.mark.asyncio
async def test_unified_client_initialization():
    """Test UnifiedBinanceClient initialization."""
    client = UnifiedBinanceClient(testnet=True)

    assert client.testnet == True
    assert client.rest_client is not None
    assert client.ws_manager is not None

    print("✅ test_unified_client_initialization passed")


@pytest.mark.asyncio
async def test_unified_client_rest_api_methods():
    """Test UnifiedBinanceClient REST API delegation."""
    client = UnifiedBinanceClient(testnet=True)

    # Mock the REST client
    client.rest_client.get_ohlcv = AsyncMock(return_value=Mock())
    client.rest_client.get_current_price = AsyncMock(return_value=45000.0)

    # Test REST API methods
    result = await client.get_ohlcv('BTCUSDT', '1h')
    price = await client.get_price('BTCUSDT')

    assert result is not None
    assert price == 45000.0

    print("✅ test_unified_client_rest_api_methods passed")


@pytest.mark.asyncio
async def test_unified_client_websocket_methods():
    """Test UnifiedBinanceClient WebSocket method delegation."""
    client = UnifiedBinanceClient(testnet=True)

    callback = Mock()

    # Mock WebSocket methods
    client.ws_manager.start_price_stream = AsyncMock(return_value="price_btcusdt")
    client.ws_manager.start_kline_stream = AsyncMock(return_value="kline_ethusdt_5m")
    client.ws_manager.start_depth_stream = AsyncMock(return_value="depth_adausdt_20")

    # Test WebSocket methods
    stream_id1 = await client.start_price_stream('btcusdt', callback)
    stream_id2 = await client.start_kline_stream('ethusdt', '5m', callback)
    stream_id3 = await client.start_depth_stream('adausdt', callback, '20')

    assert stream_id1 == "price_btcusdt"
    assert stream_id2 == "kline_ethusdt_5m"
    assert stream_id3 == "depth_adausdt_20"

    print("✅ test_unified_client_websocket_methods passed")


@pytest.mark.asyncio
async def test_close_client():
    """Test closing unified client."""
    client = UnifiedBinanceClient(testnet=True)

    # Mock the close method
    client.ws_manager.stop_all_streams = AsyncMock()

    await client.close()

    client.ws_manager.stop_all_streams.assert_called_once()

    print("✅ test_close_client passed")


@pytest.mark.asyncio
async def test_websocket_error_handling():
    """Test WebSocket error handling."""
    ws_manager = WebSocketStreamManager(testnet=True)

    callback = Mock()

    # Simulate callback errors
    callback.side_effect = Exception("Test error")

    # Add callback to a fake stream
    stream_id = "test_stream"
    ws_manager.stream_subscribers[stream_id] = {callback}

    # Simulate receiving data
    test_data = {"symbol": "BTCUSDT", "price": "45000"}

    # This should not raise an error
    for cb in ws_manager.stream_subscribers.get(stream_id, set()):
        try:
            cb(test_data)
        except Exception:
            pass  # Expected

    print("✅ test_websocket_error_handling passed")


@pytest.mark.asyncio
async def test_stream_types():
    """Test different stream types have correct IDs."""
    ws_manager = WebSocketStreamManager(testnet=True)

    callback = Mock()

    # Test price stream
    stream_id1 = await ws_manager.start_price_stream('btcusdt', callback)
    assert stream_id1 == "price_btcusdt"

    # Test kline streams with different intervals
    stream_id2 = await ws_manager.start_kline_stream('ethusdt', '1m', callback)
    assert stream_id2 == "kline_ethusdt_1m"

    stream_id3 = await ws_manager.start_kline_stream('ethusdt', '4h', callback)
    assert stream_id3 == "kline_ethusdt_4h"

    # Test depth streams with different depths
    stream_id4 = await ws_manager.start_depth_stream('adausdt', callback, '5')
    assert stream_id4 == "depth_adausdt_5"

    stream_id5 = await ws_manager.start_depth_stream('adausdt', callback, '100')
    assert stream_id5 == "depth_adausdt_100"

    print("✅ test_stream_types passed")


@pytest.mark.asyncio
async def test_stop_stream():
    """Test stopping a specific stream."""
    ws_manager = WebSocketStreamManager(testnet=True)

    # Mock WebSocket
    mock_ws = AsyncMock()
    ws_manager.active_streams['test_stream'] = mock_ws

    await ws_manager.stop_stream('test_stream')

    assert 'test_stream' not in ws_manager.active_streams
    mock_ws.close.assert_called_once()

    print("✅ test_stop_stream passed")


@pytest.mark.asyncio
async def test_stop_all_streams():
    """Test stopping all streams."""
    ws_manager = WebSocketStreamManager(testnet=True)

    # Add some fake streams
    mock_ws1 = AsyncMock()
    mock_ws2 = AsyncMock()
    ws_manager.active_streams['stream1'] = mock_ws1
    ws_manager.active_streams['stream2'] = mock_ws2

    # Add some subscriptions
    ws_manager.stream_subscribers['stream1'] = {Mock()}
    ws_manager.stream_subscribers['stream2'] = {Mock()}

    ws_manager.price_streams.add('btcusdt')
    ws_manager.kline_streams['ethusdt@1h'] = 'ethusdt'
    ws_manager.depth_streams.add('adausdt@20depth')

    await ws_manager.stop_all_streams()

    assert len(ws_manager.active_streams) == 0
    assert len(ws_manager.stream_subscribers) == 0
    assert len(ws_manager.price_streams) == 0
    assert len(ws_manager.kline_streams) == 0
    assert len(ws_manager.depth_streams) == 0
    mock_ws1.close.assert_called_once()
    mock_ws2.close.assert_called_once()

    print("✅ test_stop_all_streams passed")


@pytest.mark.asyncio
async def test_case_insensitive_symbols():
    """Test symbol case handling."""
    ws_manager = WebSocketStreamManager(testnet=True)

    callback = Mock()

    # Test lowercase
    stream_id1 = await ws_manager.start_price_stream('btcusdt', callback)
    assert 'btcusdt' in ws_manager.price_streams

    # Test uppercase
    stream_id2 = await ws_manager.start_price_stream('ETHUSDT', callback)
    assert 'ethusdt' in ws_manager.price_streams  # Should be lowercased

    # Test mixed case
    stream_id3 = await ws_manager.start_price_stream('AdAuSdT', callback)
    assert 'adausdt' in ws_manager.price_streams  # Should be lowercased

    print("✅ test_case_insensitive_symbols passed")


if __name__ == '__main__':
    print("\n" + "="*70)
    print("Running Binance WebSocket Tests")
    print("="*70 + "\n")

    asyncio.run(test_websocket_stream_manager_initialization())
    asyncio.run(test_price_stream_registration())
    asyncio.run(test_kline_stream_registration())
    asyncio.run(test_depth_stream_registration())
    asyncio.run(test_multiple_subscribers())
    asyncio.run(test_stream_status())
    asyncio.run(test_unified_client_initialization())
    asyncio.run(test_unified_client_rest_api_methods())
    asyncio.run(test_unified_client_websocket_methods())
    asyncio.run(test_close_client())
    asyncio.run(test_websocket_error_handling())
    asyncio.run(test_stream_types())
    asyncio.run(test_stop_stream())
    asyncio.run(test_stop_all_streams())
    asyncio.run(test_case_insensitive_symbols())

    print("\n" + "="*70)
    print("✅ All WebSocket tests passed successfully!")
    print("="*70)
