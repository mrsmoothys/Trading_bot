#!/usr/bin/env python
"""
Data Refresher Microservice (P1.3)

Keeps the DataStore cache warm by continuously fetching OHLCV data for all timeframes.
This ensures the dashboard always has fresh data even when the main trading loop is idle.

Usage:
    python scripts/data_refresher.py --symbol BTCUSDT --interval 60
"""
import asyncio
import time
import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.data.binance_client import BinanceClient
from core.data.data_store import DataStore

# Timeframes to refresh
TIMEFRAMES = ['1m', '5m', '15m', '1h', '4h', '1d']

async def refresh_symbol_timeframes(symbol: str, binance_client: BinanceClient, datastore: DataStore):
    """Refresh all timeframes for a symbol."""
    results = {}
    
    for timeframe in TIMEFRAMES:
        try:
            # Fetch fresh data from Binance
            ohlcv_data = await binance_client.get_ohlcv(symbol, timeframe, limit=500)
            
            if ohlcv_data is not None and len(ohlcv_data) > 0:
                # Store in DataStore
                await datastore.store_ohlcv(symbol, timeframe, ohlcv_data)
                
                last_timestamp = ohlcv_data['timestamp'].iloc[-1]
                last_time = pd.to_datetime(last_timestamp).strftime('%Y-%m-%d %H:%M:%S')
                
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Refreshed {symbol} {timeframe} - "
                      f"{len(ohlcv_data)} candles, Last: {last_time}")
                results[timeframe] = True
            else:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] ⚠ Failed to fetch {symbol} {timeframe}")
                results[timeframe] = False
                
        except Exception as e:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] ✗ Error refreshing {symbol} {timeframe}: {e}")
            results[timeframe] = False
            
    return results

async def main():
    """Main loop to continuously refresh data."""
    parser = argparse.ArgumentParser(description='Keep DataStore cache warm')
    parser.add_argument('--symbol', default='BTCUSDT', help='Trading symbol (default: BTCUSDT)')
    parser.add_argument('--interval', type=int, default=60, help='Refresh interval in seconds (default: 60)')
    parser.add_argument('--once', action='store_true', help='Run once instead of continuously')
    args = parser.parse_args()
    
    print("=" * 70)
    print("Data Refresher Microservice - P1.3")
    print("=" * 70)
    print(f"Symbol: {args.symbol}")
    print(f"Timeframes: {', '.join(TIMEFRAMES)}")
    print(f"Refresh Interval: {args.interval}s")
    print(f"Mode: {'One-time refresh' if args.once else 'Continuous'}")
    print("=" * 70)
    
    binance_client = BinanceClient()
    datastore = DataStore(cache_dir='data/cache')
    
    try:
        if args.once:
            # Run once
            print(f"\nRunning one-time refresh for {args.symbol}...")
            results = await refresh_symbol_timeframes(args.symbol, binance_client, datastore)
            success_count = sum(1 for v in results.values() if v)
            print(f"\n✓ Refreshed {success_count}/{len(TIMEFRAMES)} timeframes successfully")
        else:
            # Continuous loop
            print(f"\nStarting continuous refresh loop (Ctrl+C to stop)...")
            iteration = 0
            while True:
                iteration += 1
                print(f"\n--- Iteration {iteration} ---")
                
                results = await refresh_symbol_timeframes(args.symbol, binance_client, datastore)
                success_count = sum(1 for v in results.values() if v)
                
                print(f"✓ Success: {success_count}/{len(TIMEFRAMES)} timeframes")
                
                # Wait for next interval
                if iteration % 10 == 0:
                    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Waiting {args.interval}s until next refresh...")
                await asyncio.sleep(args.interval)
                
    except KeyboardInterrupt:
        print("\n\n🛑 Data refresher stopped by user")
    except Exception as e:
        print(f"\n\n✗ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        print("\n✓ Data refresher shutting down gracefully")

if __name__ == '__main__':
    import pandas as pd
    asyncio.run(main())
