"""Quick health check for Binance data pipeline."""
import asyncio
from datetime import datetime

from core.data.binance_client import BinanceClient
from core.data.data_store import DataStore


async def main(symbol: str = "BTCUSDT", interval: str = "15m", limit: int = 100):
    client = BinanceClient()
    store = DataStore()

    print(f"[Check] Fetching {limit} candles for {symbol} {interval}...")
    df = await client.get_ohlcv(symbol, interval, limit=limit)
    print(f"[Check] Retrieved {len(df)} rows spanning {df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}")

    await store.store_ohlcv(symbol, interval, df)
    cached = store.get_ohlcv(symbol, interval)
    if cached is not None:
        print(f"[Check] Cache contains {len(cached)} rows (latest close: {cached['close'].iloc[-1]:.2f})")
    else:
        print("[WARN] Cache retrieval failed - verify DataStore configuration")

    price = await client.get_current_price(symbol)
    print(f"[Check] Live price for {symbol}: {price:.2f} (UTC {datetime.utcnow().isoformat()})")


if __name__ == "__main__":
    asyncio.run(main())
