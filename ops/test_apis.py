#!/usr/bin/env python3
"""
API Testing Script
Tests all APIs with actual API keys to verify functionality.
"""

import os
import sys
import asyncio
from datetime import datetime
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_binance_spot_api():
    """Test Binance Spot API."""
    print("\n" + "=" * 70)
    print("BINANCE SPOT API TEST")
    print("=" * 70)

    try:
        # Test 1: Get server time
        print("\n📡 Testing server time...")
        response = requests.get('https://api.binance.com/api/v3/time', timeout=5)
        if response.status_code == 200:
            server_time = response.json()['serverTime']
            dt = datetime.fromtimestamp(server_time / 1000)
            print(f"   ✅ Server time: {dt.strftime('%Y-%m-%d %H:%M:%S')}")
        else:
            print(f"   ❌ Failed: {response.status_code}")
            return False

        # Test 2: Get BTCUSDT price
        print("\n📡 Testing price data...")
        response = requests.get('https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT', timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ BTC Price: ${float(data['price']):,.2f}")
        else:
            print(f"   ❌ Failed: {response.status_code}")
            return False

        # Test 3: Get 24hr statistics
        print("\n📡 Testing 24hr statistics...")
        response = requests.get('https://api.binance.com/api/v3/ticker/24hr?symbol=BTCUSDT', timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ 24h Change: {data['priceChangePercent']}%")
            print(f"   ✅ 24h High: ${float(data['highPrice']):,.2f}")
            print(f"   ✅ 24h Low: ${float(data['lowPrice']):,.2f}")
            print(f"   ✅ Volume: {float(data['volume']):,.2f} BTC")
        else:
            print(f"   ❌ Failed: {response.status_code}")
            return False

        print("\n✅ Binance Spot API: ALL TESTS PASSED")
        return True

    except Exception as e:
        print(f"\n❌ Binance Spot API Error: {e}")
        return False


def test_binance_futures_api():
    """Test Binance Futures API."""
    print("\n" + "=" * 70)
    print("BINANCE FUTURES API TEST")
    print("=" * 70)

    try:
        # Test 1: Get server time
        print("\n📡 Testing server time...")
        response = requests.get('https://fapi.binance.com/fapi/v1/time', timeout=5)
        if response.status_code == 200:
            server_time = response.json()['serverTime']
            dt = datetime.fromtimestamp(server_time / 1000)
            print(f"   ✅ Server time: {dt.strftime('%Y-%m-%d %H:%M:%S')}")
        else:
            print(f"   ❌ Failed: {response.status_code}")
            return False

        # Test 2: Get BTCUSDT perpetual price
        print("\n📡 Testing perpetual price...")
        response = requests.get('https://fapi.binance.com/fapi/v1/ticker/price?symbol=BTCUSDT', timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ BTCUSDT Perpetual Price: ${float(data['price']):,.2f}")
        else:
            print(f"   ❌ Failed: {response.status_code}")
            return False

        # Test 3: Get exchange info
        print("\n📡 Testing exchange info...")
        response = requests.get('https://fapi.binance.com/fapi/v1/exchangeInfo', timeout=5)
        if response.status_code == 200:
            data = response.json()
            futures_count = len([s for s in data['symbols'] if s['contractType'] == 'PERPETUAL'])
            print(f"   ✅ Futures symbols available: {futures_count}")
            sample = [s['symbol'] for s in data['symbols'] if s['contractType'] == 'PERPETUAL'][:5]
            print(f"   ✅ Sample symbols: {sample}")
        else:
            print(f"   ❌ Failed: {response.status_code}")
            return False

        print("\n✅ Binance Futures API: ALL TESTS PASSED")
        return True

    except Exception as e:
        print(f"\n❌ Binance Futures API Error: {e}")
        return False


async def test_deepseek_api():
    """Test DeepSeek API with actual key."""
    print("\n" + "=" * 70)
    print("DEEPSEEK API TEST")
    print("=" * 70)

    api_key = os.getenv('DEEPSEEK_API_KEY')
    if not api_key or api_key == 'your_deepseek_api_key_here':
        print("\n❌ DEEPSEEK_API_KEY not configured in .env file")
        print("   Please add your API key to the .env file")
        return False

    try:
        # Import openai (installed via pip)
        from openai import AsyncOpenAI

        print("\n📡 Initializing DeepSeek client...")
        client = AsyncOpenAI(
            api_key=api_key,
            base_url='https://api.deepseek.com/v1'
        )
        print("   ✅ Client initialized")

        # Test 1: List models
        print("\n📡 Testing models endpoint...")
        models = await client.models.list()
        model_ids = [m.id for m in models.data]
        print(f"   ✅ Available models: {model_ids[:5]}")

        # Test 2: Simple chat completion
        print("\n📡 Testing chat completion...")
        response = await client.chat.completions.create(
            model='deepseek-chat',
            messages=[
                {'role': 'user', 'content': 'Hello! Please respond with exactly: API TEST SUCCESSFUL'}
            ],
            max_tokens=50,
            temperature=0
        )

        message = response.choices[0].message.content
        print(f"   ✅ Response: {message}")

        if 'SUCCESSFUL' in message.upper():
            print("   ✅ DeepSeek API is working correctly!")
        else:
            print("   ⚠️  Response received but may not be as expected")

        print("\n✅ DeepSeek API: ALL TESTS PASSED")
        return True

    except Exception as e:
        print(f"\n❌ DeepSeek API Error: {e}")
        return False


def test_authenticated_binance():
    """Test authenticated Binance API endpoints."""
    print("\n" + "=" * 70)
    print("BINANCE AUTHENTICATED API TEST")
    print("=" * 70)

    api_key = os.getenv('BINANCE_API_KEY')
    api_secret = os.getenv('BINANCE_API_SECRET')

    if not api_key or api_key == 'your_binance_api_key_here':
        print("\n⚠️  BINANCE_API_KEY not configured")
        print("   Skipping authenticated tests (use testnet for safety)")
        return None

    try:
        import hmac
        import hashlib

        print("\n📡 Testing authenticated endpoint (account info)...")

        # Create signature
        timestamp = int(datetime.now().timestamp() * 1000)
        query_string = f'timestamp={timestamp}'
        signature = hmac.new(
            api_secret.encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()

        url = f'https://api.binance.com/api/v3/account?{query_string}&signature={signature}'
        headers = {'X-MBX-APIKEY': api_key}

        response = requests.get(url, headers=headers, timeout=5)

        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ Account connected")
            print(f"   ✅ Can trade: {data.get('canTrade', False)}")
            print(f"   ✅ Account Type: {data.get('accountType', 'N/A')}")
            print("\n✅ Authenticated Binance API: WORKING")
            return True
        else:
            print(f"   ❌ Failed: {response.status_code}")
            print(f"   Response: {response.text[:200]}")
            return False

    except Exception as e:
        print(f"\n❌ Authenticated Binance API Error: {e}")
        return False


def main():
    """Run all API tests."""
    print("\n" + "=" * 70)
    print("DEEPSEEK TRADING SYSTEM - API TEST SUITE")
    print("=" * 70)
    print(f"\nStarted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Check if .env exists
    if not os.path.exists('.env'):
        print("\n⚠️  WARNING: .env file not found!")
        print("   Copy .env.example to .env and add your API keys")
        print("   The test will use .env.example values which are placeholders")

    # Run tests
    results = {}

    # Test public APIs
    results['binance_spot'] = test_binance_spot_api()
    results['binance_futures'] = test_binance_futures_api()

    # Test authenticated Binance
    results['binance_auth'] = test_authenticated_binance()

    # Test DeepSeek
    results['deepseek'] = asyncio.run(test_deepseek_api())

    # Summary
    print("\n" + "=" * 70)
    print("API TEST SUMMARY")
    print("=" * 70)

    for api_name, result in results.items():
        if result is None:
            status = "⏭️  SKIPPED"
        elif result:
            status = "✅ PASSED"
        else:
            status = "❌ FAILED"
        print(f"{api_name:20s}: {status}")

    # Overall result
    passed = sum(1 for r in results.values() if r is True)
    total = len([r for r in results.values() if r is not None])

    print(f"\nOverall: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 ALL TESTS PASSED! APIs are ready to use.")
        return 0
    elif passed > 0:
        print(f"\n⚠️  {passed}/{total} tests passed. Some APIs need attention.")
        return 1
    else:
        print("\n❌ ALL TESTS FAILED. Check your API configuration.")
        return 2


if __name__ == "__main__":
    sys.exit(main())
