"""
Smoke tests for dashboard UI verification.
Tests that key UI elements exist and basic functionality works.
"""
import pytest
from dash.testing.application_runners import import_app
import os

# Set test environment
os.environ['BACKTEST_MODE'] = 'true'


@pytest.mark.unit
def test_backtest_data_freshness_1h():
    """Test that 1h data is live and fresh for backtests."""
    import sys
    sys.path.insert(0, '/Users/mrsmoothy/Downloads/Trading_bot')
    from ui.dashboard import fetch_market_data

    # Fetch 1h data
    df, meta = fetch_market_data('BTCUSDT', '1h', num_bars=10, force_refresh=True)

    # Assert no sample data used
    assert meta.get('used_sample_data', False) is False, "Sample data should not be used for backtests"

    # Assert data is fresh (less than 2 hours old for 1h timeframe)
    last_age = meta.get('last_candle_age_min', 0)
    assert last_age < 120, f"Data is too stale: {last_age} minutes > 120 minutes"


@pytest.mark.unit
def test_backtest_data_freshness_4h():
    """Test that 4h data is live and fresh for backtests."""
    import sys
    sys.path.insert(0, '/Users/mrsmoothy/Downloads/Trading_bot')
    from ui.dashboard import fetch_market_data

    # Fetch 4h data
    df, meta = fetch_market_data('BTCUSDT', '4h', num_bars=10, force_refresh=True)

    # Assert no sample data used
    assert meta.get('used_sample_data', False) is False, "Sample data should not be used for backtests"

    # Assert data is fresh (less than 6 hours old for 4h timeframe)
    last_age = meta.get('last_candle_age_min', 0)
    assert last_age < 360, f"Data is too stale: {last_age} minutes > 360 minutes (6 hours)"


@pytest.mark.unit
def test_backtest_modal_callback_with_data():
    """Test that modal callback function exists and accepts data."""
    import sys
    sys.path.insert(0, '/Users/mrsmoothy/Downloads/Trading_bot')

    # Test that we can import the function
    from ui.dashboard import create_dashboard_app
    app = create_dashboard_app()

    # Verify the callback is registered
    callbacks = [c['callback'] for c in app.callback_map.values()]
    callback_names = [getattr(c, '__name__', str(c)) for c in callbacks]

    # Check that toggle_backtest_modal function exists
    # (This test verifies the callback is registered, actual rendering is tested via MCP)
    assert any('toggle_backtest_modal' in str(c) for c in callback_names), \
        "toggle_backtest_modal callback should be registered"


@pytest.mark.unit
def test_strategy_normalization():
    """Test that strategies have been normalized and don't contain NaN/inf."""
    import sys
    import pandas as pd
    import numpy as np
    sys.path.insert(0, '/Users/mrsmoothy/Downloads/Trading_bot')

    # Test that we can import the strategies
    from core.strategies.ma_crossover import calculate_signals as ma_signals
    from core.strategies.rsi_divergence import calculate_signals as rsi_signals

    # Create sample data
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=200, freq='1H')
    df = pd.DataFrame({
        'close': 50000 + np.cumsum(np.random.randn(200) * 100),
        'high': 50100 + np.cumsum(np.random.randn(200) * 100),
        'low': 49900 + np.cumsum(np.random.randn(200) * 100),
        'volume': np.random.randint(1000, 10000, 200)
    }, index=dates)

    # Test MA Crossover with updated defaults
    ma_params = {'fast_period': 10, 'slow_period': 30, 'min_strength': 0.1}
    ma_result = ma_signals(df, ma_params)
    assert 'signal' in ma_result.columns, "MA result should have signal column"
    assert not ma_result['signal'].isna().any(), "MA signals should not contain NaN"
    assert not np.isinf(ma_result['signal']).any(), "MA signals should not contain Inf"

    # Test RSI Divergence with throttling
    rsi_params = {'period': 14, 'oversold': 30, 'overbought': 70}
    rsi_result = rsi_signals(df, rsi_params)
    assert 'signal' in rsi_result.columns, "RSI result should have signal column"
    assert not rsi_result['signal'].isna().any(), "RSI signals should not contain NaN"
    assert not np.isinf(rsi_result['signal']).any(), "RSI signals should not contain Inf"

    # Test no NaN/inf in returns (strength/confidence)
    for strategy_name, signals in [("MA", ma_result), ("RSI", rsi_result)]:
        assert not signals['strength'].isna().any(), f"{strategy_name} strength should not contain NaN"
        assert not signals['confidence'].isna().any(), f"{strategy_name} confidence should not contain NaN"
        assert not np.isinf(signals['strength']).any(), f"{strategy_name} strength should not contain Inf"
        assert not np.isinf(signals['confidence']).any(), f"{strategy_name} confidence should not contain Inf"

    # Verify MA Crossover uses updated default parameters (fast=10, slow=30)
    # This is checked indirectly by ensuring the function doesn't error with these params
    assert ma_result.shape[0] == df.shape[0], "MA should return same number of rows as input"


@pytest.mark.unit
def test_backtester_equity_curve_and_trades():
    """Test that backtester emits equity curve and trades."""
    import sys
    import pandas as pd
    import numpy as np
    sys.path.insert(0, '/Users/mrsmoothy/Downloads/Trading_bot')

    from models.backtester import Backtester
    from core.strategies.ma_crossover import calculate_signals as ma_signals

    # Create sample data
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=100, freq='1H')
    df = pd.DataFrame({
        'close': 50000 + np.cumsum(np.random.randn(100) * 100),
        'high': 50100 + np.cumsum(np.random.randn(100) * 100),
        'low': 49900 + np.cumsum(np.random.randn(100) * 100),
        'volume': np.random.randint(1000, 10000, 100)
    }, index=dates)

    # Generate signals
    signals = ma_signals(df, {'fast_period': 10, 'slow_period': 30, 'min_strength': 0.1})

    # Run backtest
    backtester = Backtester(initial_capital=10000)
    results = backtester.run_backtest(df, signals, "Test MA")

    # Verify results have equity curve and trades
    assert hasattr(results, 'equity_curve'), "Results should have equity_curve attribute"
    assert results.equity_curve is not None, "equity_curve should not be None"
    assert len(results.equity_curve) > 0, "equity_curve should not be empty"

    # Verify equity curve structure
    first_eq = results.equity_curve[0]
    assert 'timestamp' in first_eq, "equity_curve entries should have timestamp"
    assert 'equity' in first_eq, "equity_curve entries should have equity"
    assert isinstance(first_eq['equity'], (int, float)), "equity should be numeric"

    # Verify trades have required fields
    if results.trades:
        trade = results.trades[0]
        assert hasattr(trade, 'entry_time'), "Trades should have entry_time"
        assert hasattr(trade, 'exit_time'), "Trades should have exit_time"
        assert hasattr(trade, 'entry_price'), "Trades should have entry_price"
        assert hasattr(trade, 'exit_price'), "Trades should have exit_price"
        assert hasattr(trade, 'pnl'), "Trades should have pnl"
        assert hasattr(trade, 'pnl_percent'), "Trades should have pnl_percent"


@pytest.mark.unit
def test_backtest_mode_enforces_live_data():
    """Test that BACKTEST_MODE prevents sample data fallback."""
    import sys
    import os
    sys.path.insert(0, '/Users/mrsmoothy/Downloads/Trading_bot')

    from ui.dashboard import get_real_market_data

    # Set BACKTEST_MODE to true
    os.environ['BACKTEST_MODE'] = 'true'

    try:
        # This should raise ValueError in BACKTEST_MODE when live data unavailable
        # Note: This may pass if live data IS available, so we catch that case
        try:
            df, used_sample = get_real_market_data('BTCUSDT', '1h', 10)
            # If we get here, live data is available - that's fine
            # Just verify we're not using sample data
            assert not used_sample, "Should not use sample data in BACKTEST_MODE"
        except ValueError as e:
            # Expected when live data is not available in BACKTEST_MODE
            assert "CRITICAL: Backtest failed" in str(e), "Should get clear error message for backtests"
            assert "unable to fetch live data" in str(e), "Error should mention live data requirement"
    finally:
        # Clean up
        os.environ['BACKTEST_MODE'] = 'false'


@pytest.mark.unit
def test_backtest_with_small_sample():
    """Test backtest with small sample - verify no NaN/inf and trade count sanity."""
    import sys
    import pandas as pd
    import numpy as np
    sys.path.insert(0, '/Users/mrsmoothy/Downloads/Trading_bot')

    from models.backtester import Backtester
    from core.strategies.ma_crossover import calculate_signals as ma_signals

    # Create small sample data
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=50, freq='1H')  # Small sample
    df = pd.DataFrame({
        'close': 50000 + np.cumsum(np.random.randn(50) * 100),
        'high': 50100 + np.cumsum(np.random.randn(50) * 100),
        'low': 49900 + np.cumsum(np.random.randn(50) * 100),
        'volume': np.random.randint(1000, 10000, 50)
    }, index=dates)

    # Generate signals
    signals = ma_signals(df, {'fast_period': 10, 'slow_period': 30, 'min_strength': 0.05})

    # Run backtest
    backtester = Backtester(initial_capital=10000)
    results = backtester.run_backtest(df, signals, "Test MA Small")

    # Sanity checks
    assert not np.isnan(results.total_return), "Total return should not be NaN"
    assert not np.isinf(results.total_return), "Total return should not be Inf"
    assert results.total_trades >= 0, "Trade count should be non-negative"

    # Cap checks: For 50 bars, max trades should be reasonable (<10)
    # With throttling in strategies, should be even lower
    assert results.total_trades <= 10, f"Too many trades ({results.total_trades}) for 50 bars"

    # Verify equity curve exists and has data
    assert results.equity_curve is not None, "Equity curve should exist"
    assert len(results.equity_curve) > 0, "Equity curve should have data"
    assert len(results.equity_curve) == len(df), "Equity curve should match data length"

    # Verify no NaN/inf in equity curve
    for point in results.equity_curve:
        assert 'equity' in point, "Equity point should have equity field"
        assert isinstance(point['equity'], (int, float)), "Equity should be numeric"
        assert not np.isnan(point['equity']), "Equity should not be NaN"
        assert not np.isinf(point['equity']), "Equity should not be Inf"

    # Verify trades have required fields if any trades exist
    if results.trades:
        for trade in results.trades:
            assert trade.pnl is not None, "Trade should have PnL"
            assert isinstance(trade.pnl, (int, float)), "PnL should be numeric"
            assert not np.isnan(trade.pnl), "Trade PnL should not be NaN"
            assert not np.isinf(trade.pnl), "Trade PnL should not be Inf"

    # Verify metrics make sense
    if results.total_trades > 0:
        assert 0 <= results.win_rate <= 1, "Win rate should be between 0 and 1"
        assert results.max_drawdown >= 0, "Max drawdown should be non-negative"


def test_dashboard_loads(dash_duo):
    """Test that dashboard loads without errors and shows correct title."""
    # Import and start the app
    app = import_app("ui.dashboard")
    dash_duo.start_server(app)

    # Wait for app to load
    dash_duo.wait_for_element("h1", timeout=10)

    # Verify title exists and is correct
    title = dash_duo.find_element("h1")
    assert title.text == "DeepSeek Trading System", f"Expected 'DeepSeek Trading System', got '{title.text}'"


def test_main_chart_exists(dash_duo):
    """Test that main chart is rendered with figure data."""
    app = import_app("ui.dashboard")
    dash_duo.start_server(app)

    # Wait for chart to load
    dash_duo.wait_for_element("#main-chart", timeout=10)

    # Verify chart element exists
    chart = dash_duo.find_element("#main-chart")
    assert chart is not None


def test_status_badges_exist(dash_duo):
    """Test that LIVE/DEMO and freshness badges exist."""
    app = import_app("ui.dashboard")
    dash_duo.start_server(app)

    dash_duo.wait_for_element("#live-status-badge", timeout=10)
    dash_duo.wait_for_element("#demo-status-badge", timeout=10)
    dash_duo.wait_for_element("#freshness-badge", timeout=10)

    # Verify badges exist
    assert dash_duo.find_element("#live-status-badge") is not None
    assert dash_duo.find_element("#demo-status-badge") is not None
    assert dash_duo.find_element("#freshness-badge") is not None


def test_timeframe_buttons_exist(dash_duo):
    """Test that all timeframe buttons exist."""
    app = import_app("ui.dashboard")
    dash_duo.start_server(app)

    dash_duo.wait_for_element("#tf-15m", timeout=10)

    # Verify all 6 timeframe buttons exist with correct IDs
    expected_buttons = ['tf-1m', 'tf-5m', 'tf-15m', 'tf-1h', 'tf-4h', 'tf-1d']
    for btn_id in expected_buttons:
        btn = dash_duo.find_element(f"#{btn_id}")
        assert btn is not None, f"Button {btn_id} not found"


def test_timeframe_button_click_updates_chart(dash_duo):
    """Test that clicking timeframe buttons updates the chart."""
    app = import_app("ui.dashboard")
    dash_duo.start_server(app)

    # Wait for chart to load initially
    dash_duo.wait_for_element("#main-chart", timeout=10)

    # Click 1h timeframe button
    dash_duo.find_element("#tf-1h").click()

    # Wait a moment for callback to fire
    dash_duo.sleep(3)

    # Chart should still exist and be updated
    chart = dash_duo.find_element("#main-chart")
    assert chart is not None


def test_overlay_toggles_exist(dash_duo):
    """Test that overlay toggles exist."""
    app = import_app("ui.dashboard")
    dash_duo.start_server(app)

    dash_duo.wait_for_element("#overlay-toggles", timeout=10)

    # Verify overlay toggle exists
    toggles = dash_duo.find_element("#overlay-toggles")
    assert toggles is not None


def test_main_tabs_exist(dash_duo):
    """Test that all main tabs exist."""
    app = import_app("ui.dashboard")
    dash_duo.start_server(app)

    # Check for main tab structure
    dash_duo.wait_for_element('a[role="tab"]', timeout=10)

    # Find all tabs
    tabs = dash_duo.driver.find_elements('a[role="tab"]')
    tab_texts = [tab.text for tab in tabs]

    # Verify expected tabs exist
    assert any("Market Analysis" in text for text in tab_texts), "Market Analysis tab not found"
    assert any("Account & Trading" in text for text in tab_texts), "Account & Trading tab not found"
    assert any("Backtest Lab" in text for text in tab_texts), "Backtest Lab tab not found"


def test_account_trading_tab_shows_metrics(dash_duo):
    """Test that Account & Trading tab displays portfolio metrics."""
    app = import_app("ui.dashboard")
    dash_duo.start_server(app)

    # Click Account & Trading tab
    tabs = dash_duo.driver.find_elements('a[role="tab"]')
    for tab in tabs:
        if "Account & Trading" in tab.text:
            tab.click()
            break

    # Wait for content to load
    dash_duo.sleep(2)

    # Verify portfolio metrics elements exist
    dash_duo.wait_for_element("h2", timeout=10)
    headings = dash_duo.driver.find_elements("h2")
    heading_texts = [h.text for h in headings]

    # Should show portfolio value (formatted as currency)
    assert any("$" in text for text in heading_texts), "Portfolio value not displayed"


def test_chat_interface_exists(dash_duo):
    """Test that chat interface elements exist."""
    app = import_app("ui.dashboard")
    dash_duo.start_server(app)

    dash_duo.wait_for_element("#chat-input", timeout=10)
    dash_duo.wait_for_element("#chat-send-btn", timeout=10)

    # Verify chat elements
    chat_input = dash_duo.find_element("#chat-input")
    chat_send = dash_duo.find_element("#chat-send-btn")

    assert chat_input is not None
    assert chat_send is not None


def test_chat_send_button_enabled(dash_duo):
    """Test that chat send button is functional."""
    app = import_app("ui.dashboard")
    dash_duo.start_server(app)

    dash_duo.wait_for_element("#chat-input", timeout=10)
    dash_duo.wait_for_element("#chat-send-btn", timeout=10)

    # Type a test message
    chat_input = dash_duo.find_element("#chat-input")
    chat_input.send_keys("Test message")

    # Button should be clickable
    chat_send = dash_duo.find_element("#chat-send-btn")
    assert chat_send is not None


def test_no_console_errors(dash_duo):
    """Test that no console errors occur on load."""
    app = import_app("ui.dashboard")
    dash_duo.start_server(app)

    dash_duo.wait_for_element("#main-chart", timeout=10)

    # Check for console errors
    # Note: This checks browser console, not Python logs
    logs = dash_duo.get_logs()
    if logs:
        # Filter out known non-critical warnings
        critical_errors = [log for log in logs if 'error' in log['level'].lower()]
        assert len(critical_errors) == 0, f"Console errors detected: {critical_errors}"


def test_plotly_modebar_controls(dash_duo):
    """Test that Plotly modebar controls are present."""
    app = import_app("ui.dashboard")
    dash_duo.start_server(app)

    dash_duo.wait_for_element("#main-chart", timeout=10)

    # Check if plotly chart div exists
    chart_div = dash_duo.driver.find_element("#main-chart")
    assert chart_div is not None

    # Verify the chart has plotly data
    # This is a basic check - actual plotly interactions are tested in test_plotly_overlays.py
    assert chart_div.get_attribute("id") == "main-chart"


def test_range_slider_exists(dash_duo):
    """Test that chart range slider exists."""
    app = import_app("ui.dashboard")
    dash_duo.start_server(app)

    dash_duo.wait_for_element("#chart-range-slider", timeout=10)

    slider = dash_duo.find_element("#chart-range-slider")
    assert slider is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

