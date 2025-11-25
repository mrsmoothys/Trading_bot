"""
Entry point for running the dashboard via `python -m ui.dashboard`
"""

if __name__ == "__main__":
    from ui.dashboard import create_dashboard_app, set_system_context
    import sys
    import os
    from datetime import datetime

    # Initialize real SystemContext
    try:
        from core.system_context import SystemContext
        system_context = SystemContext()
        set_system_context(system_context)
        print("✓ Real SystemContext initialized")
    except Exception as e:
        print(f"⚠ Warning: Could not initialize SystemContext: {e}")
        print("  Dashboard will run with limited functionality")
        # Create minimal fallback context
        class MinimalContext:
            def __init__(self):
                self.overlay_state = {
                    "liquidity": True,
                    "supertrend": True,
                    "orderflow": True,
                    "chandelier": False,
                    "regime": False,
                    "alignment": False
                }
                self.system_health = {
                    "memory_usage": 0,
                    "cpu_usage": 0,
                    "errors": [],
                    "last_check": datetime.now(),
                }
                self.risk_metrics = {
                    "portfolio_value": 100000.0,
                    "unrealized_pnl": 0.0,
                    "total_exposure": 0.0,
                    "max_drawdown": 0.0,
                }
        set_system_context(MinimalContext())

    # Check for demo/live mode
    use_sample = os.getenv('DASH_USE_SAMPLE_DATA', '0') == '1'
    is_testnet = os.getenv('BINANCE_TESTNET', 'false').lower() == 'true'
    futures_url = os.getenv('BINANCE_FUTURES_URL')
    if not futures_url:
        futures_url = 'https://testnet.binancefuture.com/fapi/v1' if is_testnet else 'https://fapi.binance.com/fapi/v1'
    data_label = "Demo/Sample Data" if use_sample else f"Binance {'Testnet' if is_testnet else 'Live'} API ({futures_url})"
    mode = "DEMO MODE" if use_sample else ("TESTNET MODE" if is_testnet else "LIVE MODE")

    # Create and run the app
    print(f"\n{'='*60}")
    print(f"  DeepSeek Trading Dashboard - {mode}")
    print(f"{'='*60}")
    print(f"  URL: http://127.0.0.1:8050")
    print(f"  Data: {data_label}")
    print(f"{'='*60}\n")

    app = create_dashboard_app()
    app.run(host='0.0.0.0', port=8050, debug=False, dev_tools_ui=False)
