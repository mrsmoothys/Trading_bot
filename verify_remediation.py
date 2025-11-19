
import asyncio
import time
import pandas as pd
import numpy as np
from features.engine import FeatureEngine

async def test_feature_engine_performance():
    print("Testing FeatureEngine performance...")
    engine = FeatureEngine()
    
    # Create dummy data
    dates = pd.date_range(start='2023-01-01', periods=200, freq='1h')
    data = {
        'timestamp': pd.Series(dates),
        'open': pd.Series(np.random.randn(200) + 100, index=dates),
        'high': pd.Series(np.random.randn(200) + 105, index=dates),
        'low': pd.Series(np.random.randn(200) + 95, index=dates),
        'close': pd.Series(np.random.randn(200) + 100, index=dates),
        'volume': pd.Series(np.random.randn(200) * 1000 + 10000, index=dates)
    }
    
    start_time = time.time()
    
    # Simulate parallel calls for multiple symbols
    symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'ADAUSDT']
    tasks = []
    
    for symbol in symbols:
        tasks.append(engine.compute_all_features(symbol, data))
        
    await asyncio.gather(*tasks)
    
    end_time = time.time()
    duration = end_time - start_time
    
    print(f"Processed {len(symbols)} symbols in {duration:.4f} seconds")
    print("If this was sequential and blocking, it would likely take longer.")
    print("Test passed if no errors occurred and execution was successful.")

if __name__ == "__main__":
    try:
        asyncio.run(test_feature_engine_performance())
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
