"""
Smoke test to verify dashboard layout integrity.
Tests that all callback outputs have corresponding DOM elements.
"""

import dash
from dash import html
import sys
import os

# Add project root to path
sys.path.insert(0, '/Users/mrsmoothy/Downloads/Trading_bot')

def test_layout_imports():
    """Test that dashboard module imports successfully."""
    try:
        from ui.dashboard import layout, create_dashboard_app
        return True, "✓ Dashboard imports successfully"
    except Exception as e:
        return False, f"✗ Import failed: {e}"

def test_layout_structure():
    """Test that layout() returns a valid Dash component."""
    try:
        from ui.dashboard import layout

        layout_component = layout()

        # Check it's a valid Dash component
        assert isinstance(layout_component, html.Div), "Layout must be html.Div"

        # Check it has children property (don't access directly to avoid initialization issues)
        assert hasattr(layout_component, 'children'), "Layout must have children"

        return True, f"✓ Layout structure valid (html.Div with children)"
    except Exception as e:
        return False, f"✗ Layout structure test failed: {e}"

def test_required_dcc_components():
    """Test that all required dcc.Store components exist."""
    try:
        from ui.dashboard import layout
        layout_component = layout()

        # Convert to string and check for required components
        layout_str = layout_component.to_plotly_json() if hasattr(layout_component, 'to_plotly_json') else str(layout_component)

        required_components = [
            'timeframe-store',
            'chart-view-store',
            'market-data-store',
            'chat-store',
            'trading-status-store',
            'backtest-result-store',
            'interval-component',
            'price-chart',
            'chat-history',
            'feature-toggles'
        ]

        missing = [comp for comp in required_components if comp not in layout_str]

        if missing:
            return False, f"✗ Missing components: {missing}"
        return True, f"✓ All {len(required_components)} required components present"
    except Exception as e:
        return False, f"✗ Component test failed: {e}"

def test_callback_outputs():
    """Test that critical callback output IDs exist in layout."""
    try:
        from ui.dashboard import layout
        layout_component = layout()

        # Check for critical callback output IDs
        critical_outputs = [
            'live-price',
            'price-change',
            'system-health',
            'feature-metrics',
            'convergence-status',
            'scalp-status',
            'chat-status',
            'perf-fps',
            'perf-memory'
        ]

        # Get layout as dict for inspection
        import json
        layout_json = layout_component.to_plotly_json() if hasattr(layout_component, 'to_plotly_json') else {}
        layout_str = json.dumps(layout_json) if layout_json else str(layout_component)
        missing = [output for output in critical_outputs if output not in layout_str]

        if missing:
            return False, f"✗ Missing callback outputs: {missing}"
        return True, f"✓ All {len(critical_outputs)} critical callback outputs present"
    except Exception as e:
        return False, f"✗ Callback output test failed: {e}"

def test_dashboard_app_creation():
    """Test that create_dashboard_app() works."""
    try:
        from ui.dashboard import create_dashboard_app

        app = create_dashboard_app()

        # Verify it's a Dash app
        assert isinstance(app, dash.Dash), "Must return Dash app instance"

        # Verify layout is set
        assert app.layout is not None, "App must have layout"

        return True, "✓ Dashboard app created successfully"
    except Exception as e:
        return False, f"✗ App creation failed: {e}"

def run_all_tests():
    """Run all smoke tests and report results."""
    print("\n" + "="*70)
    print("DASHBOARD LAYOUT SMOKE TEST")
    print("="*70 + "\n")

    tests = [
        ("Import Test", test_layout_imports),
        ("Layout Structure", test_layout_structure),
        ("DCC Components", test_required_dcc_components),
        ("Callback Outputs", test_callback_outputs),
        ("App Creation", test_dashboard_app_creation)
    ]

    results = []
    for test_name, test_func in tests:
        try:
            success, message = test_func()
            results.append((test_name, success, message))
            print(f"{'✓' if success else '✗'} {test_name}: {message}")
        except Exception as e:
            results.append((test_name, False, f"Exception: {e}"))
            print(f"✗ {test_name}: Exception - {e}")

    print("\n" + "="*70)
    passed = sum(1 for _, success, _ in results if success)
    total = len(results)
    print(f"RESULTS: {passed}/{total} tests passed")
    print("="*70 + "\n")

    return all(success for _, success, _ in results)

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
