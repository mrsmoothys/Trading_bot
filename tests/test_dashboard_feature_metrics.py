import sys
from pathlib import Path
from dash import html

sys.path.append(str(Path(__file__).resolve().parents[1]))

from ui.dashboard import create_feature_metrics_table


def test_feature_metrics_table_with_data():
    metrics = {
        'liquidity': {'latency_ms': 12.34, 'memory_delta_mb': 0.5, 'timestamp': '2025-11-15T10:00:00'},
        'orderflow': {'latency_ms': 8.0, 'memory_delta_mb': 0.2, 'timestamp': '2025-11-15T10:00:01'}
    }
    component = create_feature_metrics_table(metrics, current_regime="TRENDING_HIGH_VOL", recommendations=["supertrend"])
    assert isinstance(component, html.Div)
    # Ensure table rows created
    table = component.children[0]
    assert len(table.children) == 3  # header + 2 rows


def test_feature_metrics_table_no_data():
    component = create_feature_metrics_table({}, current_regime="UNKNOWN", recommendations=None)
    assert isinstance(component, html.Div)
    assert "Telemetry pending" in component.children
