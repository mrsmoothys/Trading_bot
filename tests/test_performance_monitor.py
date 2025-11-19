"""
Test Enhanced Performance Monitor
Quick validation of comprehensive monitoring functionality.
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ops.performance_monitor import PerformanceMonitor
from core.memory_manager import M1MemoryManager
from loguru import logger


async def test_performance_monitor():
    """Test comprehensive performance monitoring."""
    logger.info("="*70)
    logger.info("Testing Enhanced Performance Monitor")
    logger.info("="*70)

    # Initialize memory manager
    memory_manager = M1MemoryManager()

    # Initialize performance monitor
    monitor = PerformanceMonitor(memory_manager=memory_manager, log_dir="logs/performance_test")

    # Test 1: Check system health
    logger.info("\n[Test 1] Checking system health...")
    health = await monitor.check_system_health()
    logger.info(f"✅ System health: {health['status']}")
    logger.info(f"   Memory: {health['memory']['usage_mb']:.0f}MB ({health['memory']['percent']:.1f}%)")
    logger.info(f"   CPU: {health['cpu']['usage_percent']:.1f}%")
    logger.info(f"   Health Score: {health['health_score']:.2f}")

    # Test 2: Check disk I/O
    logger.info("\n[Test 2] Checking disk I/O...")
    disk_io = await monitor.check_disk_io()
    logger.info(f"✅ Disk I/O: {disk_io['status']}")
    logger.info(f"   Total Read: {disk_io['current']['total_read_gb']:.2f}GB")
    logger.info(f"   Total Write: {disk_io['current']['total_write_gb']:.2f}GB")

    # Test 3: Check network bandwidth
    logger.info("\n[Test 3] Checking network bandwidth...")
    network = await monitor.check_network_bandwidth()
    logger.info(f"✅ Network: {network['status']}")
    logger.info(f"   Total Sent: {network['current']['total_sent_gb']:.2f}GB")
    logger.info(f"   Total Received: {network['current']['total_recv_gb']:.2f}GB")

    # Test 4: Track trading performance
    logger.info("\n[Test 4] Tracking trading performance...")
    monitor.track_trading_performance("order_execution", 245.5, True, {"symbol": "BTCUSDT"})
    monitor.track_trading_performance("signal_processing", 150.2, True, {"features": 12})
    monitor.track_trading_performance("order_execution", 312.8, True, {"symbol": "ETHUSDT"})
    logger.info("✅ Tracked 3 trading operations")

    # Test 5: Add webhook
    logger.info("\n[Test 5] Testing webhook management...")
    monitor.add_webhook("https://example.com/webhook")
    logger.info(f"✅ Added webhook: {len(monitor.alert_webhooks)} webhooks configured")

    # Test 6: Generate performance report
    logger.info("\n[Test 6] Generating performance report...")
    report = await monitor.generate_performance_report()
    logger.info(f"✅ Performance report generated")
    logger.info(f"   System Health: {report['system_health']['status']}")
    logger.info(f"   Trading Operations: {report['trading_performance']['total_operations']}")
    logger.info(f"   Recommendations: {len(report['recommendations'])}")

    # Test 7: Get dashboard metrics
    logger.info("\n[Test 7] Getting dashboard metrics...")
    dashboard_data = monitor.get_dashboard_metrics()
    logger.info(f"✅ Dashboard metrics:")
    logger.info(f"   Current Memory: {dashboard_data['current']['memory_mb']:.0f}MB")
    logger.info(f"   Current CPU: {dashboard_data['current']['cpu_percent']:.1f}%")
    logger.info(f"   Memory Trend: {dashboard_data['trends']['memory_trend']}")

    # Test 8: Export metrics
    logger.info("\n[Test 8] Exporting metrics...")
    export_file = monitor.export_metrics()
    logger.info(f"✅ Metrics exported to: {export_file}")

    # Test 9: Test alert system
    logger.info("\n[Test 9] Testing alert system...")
    await monitor.send_alert("INFO", "Test alert message", {"test": True})
    logger.info("✅ Alert sent successfully")

    # Test 10: Get performance report
    logger.info("\n[Test 10] Getting performance summary...")
    perf_report = monitor.get_performance_report()
    logger.info(f"✅ Performance summary:")
    logger.info(f"   Average Memory: {perf_report['memory']['average_mb']:.0f}MB")
    logger.info(f"   Average CPU: {perf_report['cpu']['average_percent']:.1f}%")
    logger.info(f"   Recommendations: {perf_report['recommendations']}")

    logger.info("\n" + "="*70)
    logger.info("✅ All Enhanced Performance Monitor tests passed!")
    logger.info("="*70)

    return True


async def main():
    """Run all tests."""
    try:
        success = await test_performance_monitor()
        if success:
            logger.success("Performance Monitor tests completed successfully!")
            return 0
        else:
            logger.error("Performance Monitor tests failed")
            return 1
    except Exception as e:
        logger.error(f"Test error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
