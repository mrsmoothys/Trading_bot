"""
Performance Monitor - Enhanced with Comprehensive Monitoring
Monitors system performance, API latency, health metrics, disk I/O, network bandwidth,
trading performance, and provides automated alerts and dashboards.
"""

import asyncio
import time
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Callable
import psutil
import aiohttp
from loguru import logger

from core.memory_manager import M1MemoryManager


class PerformanceMonitor:
    """
    Enhanced Performance Monitor with comprehensive system and trading metrics.
    Tracks CPU, memory, disk I/O, network bandwidth, API latency, error rates,
    trading performance, and provides automated alerts and dashboards.
    """

    def __init__(self, memory_manager: M1MemoryManager, log_dir: str = "logs/performance"):
        """
        Initialize enhanced performance monitor.

        Args:
            memory_manager: M1MemoryManager instance
            log_dir: Directory for performance logs
        """
        self.memory_manager = memory_manager
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Metrics storage
        self.metrics_history: List[Dict[str, Any]] = []
        self.api_latency_history: Dict[str, List[float]] = {}
        self.error_history: List[Dict[str, Any]] = []

        # Enhanced tracking
        self.disk_io_history: List[Dict[str, Any]] = []
        self.network_bandwidth_history: List[Dict[str, Any]] = []
        self.trading_metrics_history: List[Dict[str, Any]] = []

        # Baseline measurements for delta calculation
        self.baseline_disk_io: Optional[Dict[str, int]] = None
        self.baseline_network: Optional[Dict[str, int]] = None
        self.last_measurement_time: Optional[datetime] = None

        # Configuration
        self.check_interval = 60  # Check every 60 seconds
        self.warning_thresholds = {
            'memory_mb': 3500,
            'cpu_percent': 80,
            'api_latency_ms': 2000,
            'error_rate': 0.05,  # 5% error rate
            'disk_read_mb_s': 100,  # 100 MB/s read
            'disk_write_mb_s': 100,  # 100 MB/s write
            'network_bandwidth_mbps': 50,  # 50 Mbps
            'order_execution_ms': 500  # 500ms for order execution
        }

        # Alert callbacks and notification config
        self.alert_callbacks: List[Callable] = []
        self.alert_webhooks: List[str] = []
        self.alert_emails: List[str] = []

        # Performance report configuration
        self.report_interval = 3600  # Generate reports every hour
        self.last_report_time: Optional[datetime] = None

        logger.info("Enhanced PerformanceMonitor initialized with comprehensive metrics tracking")

    async def check_system_health(self) -> Dict[str, Any]:
        """
        Check overall system health.

        Returns:
            Health status dictionary
        """
        try:
            # Get memory metrics
            memory = self.memory_manager.get_current_memory_usage()

            # Get CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_count = psutil.cpu_count()

            # Get process info
            process = psutil.Process()
            process_info = {
                'cpu_percent': process.cpu_percent(),
                'memory_mb': process.memory_info().rss / 1024 / 1024,
                'threads': process.num_threads(),
                'open_files': len(process.open_files()),
                'connections': len(process.connections())
            }

            # Get disk usage
            disk = psutil.disk_usage('/')
            disk_usage = {
                'total_gb': disk.total / 1024 / 1024 / 1024,
                'used_gb': disk.used / 1024 / 1024 / 1024,
                'free_gb': disk.free / 1024 / 1024 / 1024,
                'percent': disk.percent
            }

            # Get network stats
            net_io = psutil.net_io_counters()
            network = {
                'bytes_sent': net_io.bytes_sent,
                'bytes_recv': net_io.bytes_recv,
                'packets_sent': net_io.packets_sent,
                'packets_recv': net_io.packets_recv
            }

            # Calculate health status
            health_status = "HEALTHY"
            warnings = []
            critical_issues = []

            # Check memory
            if memory['rss_mb'] > self.warning_thresholds['memory_mb']:
                warnings.append(f"High memory usage: {memory['rss_mb']:.0f}MB")
                if memory['rss_mb'] > 3800:
                    health_status = "CRITICAL"
                    critical_issues.append(f"Critical memory usage: {memory['rss_mb']:.0f}MB")

            # Check CPU
            if cpu_percent > self.warning_thresholds['cpu_percent']:
                warnings.append(f"High CPU usage: {cpu_percent:.1f}%")

            # Check disk
            if disk_usage['percent'] > 90:
                critical_issues.append(f"Disk usage critical: {disk_usage['percent']:.1f}%")
                health_status = "CRITICAL"

            health_report = {
                'timestamp': datetime.now().isoformat(),
                'status': health_status,
                'memory': {
                    'usage_mb': memory['rss_mb'],
                    'percent': memory['percent'],
                    'available_mb': memory['available_mb']
                },
                'cpu': {
                    'usage_percent': cpu_percent,
                    'count': cpu_count,
                    'process_percent': process_info['cpu_percent']
                },
                'process': process_info,
                'disk': disk_usage,
                'network': network,
                'warnings': warnings,
                'critical_issues': critical_issues,
                'health_score': self._calculate_health_score(memory, cpu_percent, disk_usage)
            }

            # Store in history
            self.metrics_history.append(health_report)
            if len(self.metrics_history) > 1000:
                self.metrics_history = self.metrics_history[-1000:]

            return health_report

        except Exception as e:
            logger.error(f"Error checking system health: {e}")
            return {
                'timestamp': datetime.now().isoformat(),
                'status': 'ERROR',
                'error': str(e)
            }

    async def check_disk_io(self) -> Dict[str, Any]:
        """
        Check disk I/O performance with read/write rates.

        Returns:
            Disk I/O metrics
        """
        try:
            current_time = datetime.now()
            disk_io = psutil.disk_io_counters()

            current_stats = {
                'read_bytes': disk_io.read_bytes,
                'write_bytes': disk_io.write_bytes,
                'read_count': disk_io.read_count,
                'write_count': disk_io.write_count,
                'read_time': disk_io.read_time,
                'write_time': disk_io.write_time
            }

            # Calculate rates if we have baseline
            rates = {}
            if self.baseline_disk_io and self.last_measurement_time:
                time_delta = (current_time - self.last_measurement_time).total_seconds()
                if time_delta > 0:
                    # Calculate MB/s rates
                    read_delta = (current_stats['read_bytes'] - self.baseline_disk_io['read_bytes']) / 1024 / 1024
                    write_delta = (current_stats['write_bytes'] - self.baseline_disk_io['write_bytes']) / 1024 / 1024

                    rates = {
                        'read_mb_per_sec': read_delta / time_delta,
                        'write_mb_per_sec': write_delta / time_delta,
                        'read_ops_per_sec': (current_stats['read_count'] - self.baseline_disk_io['read_count']) / time_delta,
                        'write_ops_per_sec': (current_stats['write_count'] - self.baseline_disk_io['write_count']) / time_delta,
                        'total_mb_per_sec': (read_delta + write_delta) / time_delta
                    }

            # Update baseline
            self.baseline_disk_io = current_stats
            self.last_measurement_time = current_time

            disk_io_report = {
                'timestamp': current_time.isoformat(),
                'current': {
                    'total_read_gb': current_stats['read_bytes'] / 1024 / 1024 / 1024,
                    'total_write_gb': current_stats['write_bytes'] / 1024 / 1024 / 1024,
                    'read_operations': current_stats['read_count'],
                    'write_operations': current_stats['write_count']
                },
                'rates': rates,
                'status': 'HEALTHY'
            }

            # Check for high I/O
            if rates:
                if rates['read_mb_per_sec'] > self.warning_thresholds['disk_read_mb_s']:
                    disk_io_report['status'] = 'WARNING'
                    logger.warning(f"High disk read rate: {rates['read_mb_per_sec']:.2f} MB/s")
                if rates['write_mb_per_sec'] > self.warning_thresholds['disk_write_mb_s']:
                    disk_io_report['status'] = 'WARNING'
                    logger.warning(f"High disk write rate: {rates['write_mb_per_sec']:.2f} MB/s")

            # Store in history
            self.disk_io_history.append(disk_io_report)
            if len(self.disk_io_history) > 1000:
                self.disk_io_history = self.disk_io_history[-1000:]

            return disk_io_report

        except Exception as e:
            logger.error(f"Error checking disk I/O: {e}")
            return {'timestamp': datetime.now().isoformat(), 'status': 'ERROR', 'error': str(e)}

    async def check_network_bandwidth(self) -> Dict[str, Any]:
        """
        Check network bandwidth and throughput.

        Returns:
            Network bandwidth metrics
        """
        try:
            current_time = datetime.now()
            net_io = psutil.net_io_counters()

            current_stats = {
                'bytes_sent': net_io.bytes_sent,
                'bytes_recv': net_io.bytes_recv,
                'packets_sent': net_io.packets_sent,
                'packets_recv': net_io.packets_recv,
                'errin': net_io.errin,
                'errout': net_io.errout,
                'dropin': net_io.dropin,
                'dropout': net_io.dropout
            }

            # Calculate bandwidth if we have baseline
            bandwidth = {}
            if self.baseline_network and self.last_measurement_time:
                time_delta = (current_time - self.last_measurement_time).total_seconds()
                if time_delta > 0:
                    # Calculate Mbps
                    sent_delta = (current_stats['bytes_sent'] - self.baseline_network['bytes_sent']) * 8 / 1024 / 1024
                    recv_delta = (current_stats['bytes_recv'] - self.baseline_network['bytes_recv']) * 8 / 1024 / 1024

                    bandwidth = {
                        'upload_mbps': sent_delta / time_delta,
                        'download_mbps': recv_delta / time_delta,
                        'total_mbps': (sent_delta + recv_delta) / time_delta,
                        'packets_sent_per_sec': (current_stats['packets_sent'] - self.baseline_network['packets_sent']) / time_delta,
                        'packets_recv_per_sec': (current_stats['packets_recv'] - self.baseline_network['packets_recv']) / time_delta,
                        'error_rate': ((current_stats['errin'] + current_stats['errout']) -
                                     (self.baseline_network['errin'] + self.baseline_network['errout'])) / time_delta if time_delta > 0 else 0
                    }

            # Update baseline
            self.baseline_network = current_stats

            network_report = {
                'timestamp': current_time.isoformat(),
                'current': {
                    'total_sent_gb': current_stats['bytes_sent'] / 1024 / 1024 / 1024,
                    'total_recv_gb': current_stats['bytes_recv'] / 1024 / 1024 / 1024,
                    'packets_sent': current_stats['packets_sent'],
                    'packets_recv': current_stats['packets_recv'],
                    'errors': current_stats['errin'] + current_stats['errout'],
                    'drops': current_stats['dropin'] + current_stats['dropout']
                },
                'bandwidth': bandwidth,
                'status': 'HEALTHY'
            }

            # Check for high bandwidth usage
            if bandwidth:
                if bandwidth['total_mbps'] > self.warning_thresholds['network_bandwidth_mbps']:
                    network_report['status'] = 'WARNING'
                    logger.warning(f"High network bandwidth: {bandwidth['total_mbps']:.2f} Mbps")

            # Store in history
            self.network_bandwidth_history.append(network_report)
            if len(self.network_bandwidth_history) > 1000:
                self.network_bandwidth_history = self.network_bandwidth_history[-1000:]

            return network_report

        except Exception as e:
            logger.error(f"Error checking network bandwidth: {e}")
            return {'timestamp': datetime.now().isoformat(), 'status': 'ERROR', 'error': str(e)}

    def track_trading_performance(self, operation_type: str, duration_ms: float,
                                  success: bool, details: Optional[Dict[str, Any]] = None):
        """
        Track trading operation performance.

        Args:
            operation_type: Type of operation (e.g., 'order_execution', 'signal_processing')
            duration_ms: Operation duration in milliseconds
            success: Whether the operation was successful
            details: Additional operation details
        """
        try:
            trading_metric = {
                'timestamp': datetime.now().isoformat(),
                'operation_type': operation_type,
                'duration_ms': duration_ms,
                'success': success,
                'details': details or {}
            }

            self.trading_metrics_history.append(trading_metric)
            if len(self.trading_metrics_history) > 1000:
                self.trading_metrics_history = self.trading_metrics_history[-1000:]

            # Check for slow operations
            if operation_type == 'order_execution' and duration_ms > self.warning_thresholds['order_execution_ms']:
                logger.warning(f"Slow order execution: {duration_ms:.0f}ms (threshold: {self.warning_thresholds['order_execution_ms']}ms)")

        except Exception as e:
            logger.error(f"Error tracking trading performance: {e}")

    async def send_alert(self, level: str, message: str, details: Optional[Dict[str, Any]] = None):
        """
        Send alert via configured channels (webhooks, email, file).

        Args:
            level: Alert level (CRITICAL, WARNING, INFO)
            message: Alert message
            details: Additional alert details
        """
        alert_data = {
            'timestamp': datetime.now().isoformat(),
            'level': level,
            'message': message,
            'details': details or {}
        }

        # Log to file
        alert_file = self.log_dir / f"alerts_{datetime.now().strftime('%Y%m%d')}.json"
        try:
            with open(alert_file, 'a') as f:
                f.write(json.dumps(alert_data) + '\n')
        except Exception as e:
            logger.error(f"Failed to write alert to file: {e}")

        # Send to webhooks
        if self.alert_webhooks:
            async with aiohttp.ClientSession() as session:
                for webhook_url in self.alert_webhooks:
                    try:
                        async with session.post(webhook_url, json=alert_data, timeout=aiohttp.ClientTimeout(total=5)) as response:
                            if response.status != 200:
                                logger.warning(f"Webhook alert failed: {webhook_url} returned {response.status}")
                    except Exception as e:
                        logger.error(f"Error sending webhook alert to {webhook_url}: {e}")

        # Call alert callbacks
        for callback in self.alert_callbacks:
            try:
                await callback(level, message, details)
            except Exception as e:
                logger.error(f"Error in alert callback: {e}")

    async def generate_performance_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive performance report with trends and recommendations.

        Returns:
            Performance report
        """
        try:
            current_time = datetime.now()

            # Get system health
            system_health = await self.check_system_health()

            # Calculate trading performance statistics
            recent_trades = [t for t in self.trading_metrics_history
                           if (current_time - datetime.fromisoformat(t['timestamp'])).total_seconds() < 3600]

            trading_stats = {
                'total_operations': len(recent_trades),
                'successful_operations': len([t for t in recent_trades if t['success']]),
                'failed_operations': len([t for t in recent_trades if not t['success']]),
                'success_rate': len([t for t in recent_trades if t['success']]) / len(recent_trades) if recent_trades else 0
            }

            # Calculate average execution times
            order_executions = [t for t in recent_trades if t['operation_type'] == 'order_execution']
            if order_executions:
                trading_stats['avg_order_execution_ms'] = sum(t['duration_ms'] for t in order_executions) / len(order_executions)
                trading_stats['max_order_execution_ms'] = max(t['duration_ms'] for t in order_executions)
                trading_stats['min_order_execution_ms'] = min(t['duration_ms'] for t in order_executions)

            # Get disk I/O stats
            disk_io_stats = {}
            if self.disk_io_history:
                recent_disk = self.disk_io_history[-60:]
                if recent_disk and recent_disk[-1].get('rates'):
                    avg_read = sum(d['rates'].get('read_mb_per_sec', 0) for d in recent_disk if d.get('rates')) / len([d for d in recent_disk if d.get('rates')])
                    avg_write = sum(d['rates'].get('write_mb_per_sec', 0) for d in recent_disk if d.get('rates')) / len([d for d in recent_disk if d.get('rates')])
                    disk_io_stats = {
                        'avg_read_mb_per_sec': avg_read,
                        'avg_write_mb_per_sec': avg_write,
                        'total_mb_per_sec': avg_read + avg_write
                    }

            # Get network stats
            network_stats = {}
            if self.network_bandwidth_history:
                recent_network = self.network_bandwidth_history[-60:]
                if recent_network and recent_network[-1].get('bandwidth'):
                    avg_upload = sum(n['bandwidth'].get('upload_mbps', 0) for n in recent_network if n.get('bandwidth')) / len([n for n in recent_network if n.get('bandwidth')])
                    avg_download = sum(n['bandwidth'].get('download_mbps', 0) for n in recent_network if n.get('bandwidth')) / len([n for n in recent_network if n.get('bandwidth')])
                    network_stats = {
                        'avg_upload_mbps': avg_upload,
                        'avg_download_mbps': avg_download,
                        'total_mbps': avg_upload + avg_download
                    }

            report = {
                'timestamp': current_time.isoformat(),
                'system_health': system_health,
                'trading_performance': trading_stats,
                'disk_io': disk_io_stats,
                'network': network_stats,
                'api_latency': {
                    api: {
                        'avg_ms': sum(latencies[-60:]) / len(latencies[-60:]) if latencies else 0,
                        'max_ms': max(latencies[-60:]) if latencies else 0,
                        'samples': len(latencies[-60:]) if latencies else 0
                    }
                    for api, latencies in self.api_latency_history.items()
                },
                'recommendations': self._get_comprehensive_recommendations(
                    system_health, trading_stats, disk_io_stats, network_stats
                )
            }

            # Save report to file
            report_file = self.log_dir / f"performance_report_{current_time.strftime('%Y%m%d_%H%M%S')}.json"
            try:
                with open(report_file, 'w') as f:
                    json.dump(report, f, indent=2)
                logger.info(f"Performance report saved to: {report_file}")
            except Exception as e:
                logger.error(f"Failed to save performance report: {e}")

            return report

        except Exception as e:
            logger.error(f"Error generating performance report: {e}")
            return {'timestamp': datetime.now().isoformat(), 'status': 'ERROR', 'error': str(e)}

    def _get_comprehensive_recommendations(self, system_health: Dict, trading_stats: Dict,
                                          disk_io_stats: Dict, network_stats: Dict) -> List[str]:
        """
        Get comprehensive performance recommendations based on all metrics.

        Args:
            system_health: System health metrics
            trading_stats: Trading performance stats
            disk_io_stats: Disk I/O statistics
            network_stats: Network statistics

        Returns:
            List of recommendations
        """
        recommendations = []

        # System health recommendations
        if system_health.get('memory', {}).get('usage_mb', 0) > 3500:
            recommendations.append("🔴 CRITICAL: Memory usage is high - consider reducing cache size or restarting")
        elif system_health.get('memory', {}).get('usage_mb', 0) > 3000:
            recommendations.append("⚠️ WARNING: Memory usage approaching limit - monitor closely")

        if system_health.get('cpu', {}).get('usage_percent', 0) > 80:
            recommendations.append("⚠️ WARNING: High CPU usage - review intensive operations")

        # Trading performance recommendations
        if trading_stats.get('success_rate', 1.0) < 0.95:
            recommendations.append(f"⚠️ WARNING: Trading success rate is low ({trading_stats['success_rate']*100:.1f}%) - investigate failures")

        if trading_stats.get('avg_order_execution_ms', 0) > self.warning_thresholds['order_execution_ms']:
            recommendations.append(f"⚠️ WARNING: Slow order execution ({trading_stats['avg_order_execution_ms']:.0f}ms) - check API latency")

        # Disk I/O recommendations
        if disk_io_stats.get('total_mb_per_sec', 0) > 150:
            recommendations.append("⚠️ WARNING: High disk I/O - consider optimizing data access patterns")

        # Network recommendations
        if network_stats.get('total_mbps', 0) > self.warning_thresholds['network_bandwidth_mbps']:
            recommendations.append("⚠️ WARNING: High network bandwidth usage - review data transfer optimization")

        # General recommendations
        if not recommendations:
            recommendations.append("✅ All systems operating within normal parameters")
        else:
            recommendations.insert(0, f"Found {len(recommendations)} items requiring attention")

        return recommendations

    async def check_api_health(self, api_endpoints: Dict[str, str]) -> Dict[str, Any]:
        """
        Check API endpoint health and latency.

        Args:
            api_endpoints: Dict of name -> URL

        Returns:
            API health report
        """
        results = {}

        async with aiohttp.ClientSession() as session:
            for name, url in api_endpoints.items():
                try:
                    start_time = time.time()
                    async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as response:
                        latency = (time.time() - start_time) * 1000  # Convert to ms

                        # Store latency
                        if name not in self.api_latency_history:
                            self.api_latency_history[name] = []
                        self.api_latency_history[name].append(latency)

                        # Keep only last 100 latencies
                        if len(self.api_latency_history[name]) > 100:
                            self.api_latency_history[name] = self.api_latency_history[name][-100:]

                        # Calculate status
                        if response.status == 200:
                            status = "HEALTHY"
                            error = None
                        else:
                            status = "ERROR"
                            error = f"HTTP {response.status}"

                        results[name] = {
                            'status': status,
                            'latency_ms': latency,
                            'http_status': response.status,
                            'error': error,
                            'timestamp': datetime.now().isoformat()
                        }

                        # Log slow APIs
                        if latency > self.warning_thresholds['api_latency_ms']:
                            logger.warning(f"Slow API response: {name} took {latency:.0f}ms")

                except asyncio.TimeoutError:
                    logger.warning(f"API timeout: {name}")
                    results[name] = {
                        'status': 'TIMEOUT',
                        'latency_ms': self.warning_thresholds['api_latency_ms'],
                        'error': 'Timeout',
                        'timestamp': datetime.now().isoformat()
                    }

                except Exception as e:
                    logger.error(f"API check error for {name}: {e}")
                    results[name] = {
                        'status': 'ERROR',
                        'latency_ms': 0,
                        'error': str(e),
                        'timestamp': datetime.now().isoformat()
                    }

        return results

    async def check_error_rate(self, window_minutes: int = 60) -> Dict[str, Any]:
        """
        Check error rate over a time window.

        Args:
            window_minutes: Time window in minutes

        Returns:
            Error rate report
        """
        cutoff_time = datetime.now() - timedelta(minutes=window_minutes)

        recent_errors = [
            e for e in self.error_history
            if datetime.fromisoformat(e['timestamp']) > cutoff_time
        ]

        total_operations = len(recent_errors) + len([e for e in self.error_history
                                                    if datetime.fromisoformat(e['timestamp']) > cutoff_time
                                                    and e.get('success', False)])

        error_rate = len(recent_errors) / total_operations if total_operations > 0 else 0

        return {
            'error_rate': error_rate,
            'error_count': len(recent_errors),
            'total_operations': total_operations,
            'window_minutes': window_minutes,
            'status': 'HEALTHY' if error_rate < 0.05 else 'WARNING',
            'recent_errors': recent_errors[-10:]  # Last 10 errors
        }

    def _calculate_health_score(self, memory: Dict, cpu: float, disk: Dict) -> float:
        """
        Calculate overall health score (0-1).

        Args:
            memory: Memory metrics
            cpu: CPU usage percentage
            disk: Disk metrics

        Returns:
            Health score (0-1, higher is better)
        """
        score = 1.0

        # Deduct for high memory usage
        if memory['rss_mb'] > 3000:
            score -= 0.3
        elif memory['rss_mb'] > 2500:
            score -= 0.2

        # Deduct for high CPU usage
        if cpu > 80:
            score -= 0.2
        elif cpu > 60:
            score -= 0.1

        # Deduct for high disk usage
        if disk['percent'] > 90:
            score -= 0.3
        elif disk['percent'] > 80:
            score -= 0.1

        return max(0, score)

    async def monitoring_loop(self):
        """Enhanced monitoring loop with comprehensive system, disk, network, and trading metrics."""
        while True:
            try:
                # Check system health
                health = await self.check_system_health()

                # Check disk I/O
                disk_io = await self.check_disk_io()

                # Check network bandwidth
                network = await self.check_network_bandwidth()

                # Log critical issues
                if health['status'] == 'CRITICAL':
                    logger.critical(f"System health critical: {health.get('critical_issues', [])}")
                    await self.send_alert('CRITICAL', 'System health critical', health)

                    # Trigger alert callbacks
                    for callback in self.alert_callbacks:
                        try:
                            await callback('CRITICAL', health)
                        except Exception as e:
                            logger.error(f"Error in alert callback: {e}")

                elif health['status'] == 'WARNING':
                    logger.warning(f"System health warning: {health.get('warnings', [])}")

                # Check disk I/O warnings
                if disk_io['status'] == 'WARNING':
                    await self.send_alert('WARNING', 'High disk I/O detected', disk_io)

                # Check network warnings
                if network['status'] == 'WARNING':
                    await self.send_alert('WARNING', 'High network bandwidth detected', network)

                # Generate performance report if interval elapsed
                if (not self.last_report_time or
                    (datetime.now() - self.last_report_time).total_seconds() >= self.report_interval):
                    report = await self.generate_performance_report()
                    self.last_report_time = datetime.now()
                    logger.info("Performance report generated")

                # Check APIs (if configured)
                # In production, this would be configured with actual API endpoints
                # api_endpoints = {
                #     'binance': 'https://testnet.binancefuture.com/fapi/v1/ping',
                #     'deepseek': 'https://api.deepseek.com/v1/models'
                # }
                # api_health = await self.check_api_health(api_endpoints)

                await asyncio.sleep(self.check_interval)

            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(self.check_interval)

    def add_alert_callback(self, callback: Callable):
        """
        Add alert callback function.

        Args:
            callback: Async function that takes (level, message, details)
        """
        self.alert_callbacks.append(callback)

    def add_webhook(self, webhook_url: str):
        """
        Add webhook URL for alert notifications.

        Args:
            webhook_url: Webhook URL
        """
        if webhook_url not in self.alert_webhooks:
            self.alert_webhooks.append(webhook_url)
            logger.info(f"Added alert webhook: {webhook_url}")

    def remove_webhook(self, webhook_url: str):
        """
        Remove webhook URL from alert notifications.

        Args:
            webhook_url: Webhook URL to remove
        """
        if webhook_url in self.alert_webhooks:
            self.alert_webhooks.remove(webhook_url)
            logger.info(f"Removed alert webhook: {webhook_url}")

    def get_dashboard_metrics(self) -> Dict[str, Any]:
        """
        Get metrics formatted for dashboard display.

        Returns:
            Dashboard-ready metrics
        """
        if not self.metrics_history:
            return {'status': 'NO_DATA', 'message': 'No metrics available yet'}

        latest = self.metrics_history[-1] if self.metrics_history else {}
        recent_hour = self.metrics_history[-60:] if len(self.metrics_history) >= 60 else self.metrics_history

        dashboard_data = {
            'current': {
                'memory_mb': latest.get('memory', {}).get('usage_mb', 0),
                'cpu_percent': latest.get('cpu', {}).get('usage_percent', 0),
                'health_score': latest.get('health_score', 0),
                'status': latest.get('status', 'UNKNOWN')
            },
            'trends': {
                'memory_trend': self._calculate_trend([m['memory']['usage_mb'] for m in recent_hour if 'memory' in m]),
                'cpu_trend': self._calculate_trend([m['cpu']['usage_percent'] for m in recent_hour if 'cpu' in m])
            },
            'disk_io': {},
            'network': {},
            'trading': {}
        }

        # Add disk I/O if available
        if self.disk_io_history:
            latest_disk = self.disk_io_history[-1]
            dashboard_data['disk_io'] = {
                'read_mb_per_sec': latest_disk.get('rates', {}).get('read_mb_per_sec', 0),
                'write_mb_per_sec': latest_disk.get('rates', {}).get('write_mb_per_sec', 0),
                'status': latest_disk.get('status', 'UNKNOWN')
            }

        # Add network if available
        if self.network_bandwidth_history:
            latest_network = self.network_bandwidth_history[-1]
            dashboard_data['network'] = {
                'upload_mbps': latest_network.get('bandwidth', {}).get('upload_mbps', 0),
                'download_mbps': latest_network.get('bandwidth', {}).get('download_mbps', 0),
                'status': latest_network.get('status', 'UNKNOWN')
            }

        # Add trading metrics if available
        if self.trading_metrics_history:
            recent_trades = self.trading_metrics_history[-100:]
            successful = len([t for t in recent_trades if t['success']])
            dashboard_data['trading'] = {
                'total_operations': len(recent_trades),
                'success_rate': successful / len(recent_trades) if recent_trades else 0,
                'avg_execution_ms': sum(t['duration_ms'] for t in recent_trades) / len(recent_trades) if recent_trades else 0
            }

        return dashboard_data

    def _calculate_trend(self, values: List[float]) -> str:
        """
        Calculate trend direction from values.

        Args:
            values: List of numeric values

        Returns:
            Trend direction: 'INCREASING', 'DECREASING', 'STABLE'
        """
        if not values or len(values) < 2:
            return 'STABLE'

        # Calculate simple linear trend
        first_half = sum(values[:len(values)//2]) / (len(values)//2)
        second_half = sum(values[len(values)//2:]) / (len(values) - len(values)//2)

        diff_percent = ((second_half - first_half) / first_half * 100) if first_half > 0 else 0

        if diff_percent > 5:
            return 'INCREASING'
        elif diff_percent < -5:
            return 'DECREASING'
        else:
            return 'STABLE'

    def export_metrics(self, output_file: Optional[str] = None) -> str:
        """
        Export all metrics to JSON file for analysis.

        Args:
            output_file: Output file path (optional)

        Returns:
            Path to exported file
        """
        if not output_file:
            output_file = self.log_dir / f"metrics_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        try:
            export_data = {
                'export_timestamp': datetime.now().isoformat(),
                'metrics_history': self.metrics_history,
                'api_latency_history': self.api_latency_history,
                'error_history': self.error_history,
                'disk_io_history': self.disk_io_history,
                'network_bandwidth_history': self.network_bandwidth_history,
                'trading_metrics_history': self.trading_metrics_history,
                'thresholds': self.warning_thresholds
            }

            with open(output_file, 'w') as f:
                json.dump(export_data, f, indent=2)

            logger.info(f"Metrics exported to: {output_file}")
            return str(output_file)

        except Exception as e:
            logger.error(f"Failed to export metrics: {e}")
            return ""

    def clear_history(self, keep_recent_hours: int = 24):
        """
        Clear old metrics history, keeping only recent data.

        Args:
            keep_recent_hours: Number of hours to keep
        """
        cutoff_time = datetime.now() - timedelta(hours=keep_recent_hours)

        # Clear old metrics
        self.metrics_history = [
            m for m in self.metrics_history
            if datetime.fromisoformat(m['timestamp']) > cutoff_time
        ]

        self.disk_io_history = [
            d for d in self.disk_io_history
            if datetime.fromisoformat(d['timestamp']) > cutoff_time
        ]

        self.network_bandwidth_history = [
            n for n in self.network_bandwidth_history
            if datetime.fromisoformat(n['timestamp']) > cutoff_time
        ]

        self.trading_metrics_history = [
            t for t in self.trading_metrics_history
            if datetime.fromisoformat(t['timestamp']) > cutoff_time
        ]

        self.error_history = [
            e for e in self.error_history
            if datetime.fromisoformat(e['timestamp']) > cutoff_time
        ]

        logger.info(f"Cleared metrics history older than {keep_recent_hours} hours")

    def log_error(self, error_message: str, error_type: str = "GENERAL"):
        """
        Log an error for tracking.

        Args:
            error_message: Error message
            error_type: Type of error
        """
        error_record = {
            'timestamp': datetime.now().isoformat(),
            'message': error_message,
            'type': error_type
        }

        self.error_history.append(error_record)

        # Keep only last 1000 errors
        if len(self.error_history) > 1000:
            self.error_history = self.error_history[-1000:]

        logger.error(f"[{error_type}] {error_message}")

    def log_success(self, operation: str, details: str = ""):
        """
        Log a successful operation.

        Args:
            operation: Operation name
            details: Additional details
        """
        error_record = {
            'timestamp': datetime.now().isoformat(),
            'operation': operation,
            'details': details,
            'success': True
        }

        self.error_history.append(error_record)

    def get_performance_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive performance report.

        Returns:
            Performance report
        """
        if not self.metrics_history:
            return {'message': 'No metrics available yet'}

        # Calculate statistics
        recent = self.metrics_history[-60:]  # Last hour

        avg_memory = sum(m['memory']['usage_mb'] for m in recent) / len(recent)
        max_memory = max(m['memory']['usage_mb'] for m in recent)
        avg_cpu = sum(m['cpu']['usage_percent'] for m in recent) / len(recent)
        avg_health_score = sum(m['health_score'] for m in recent) / len(recent)

        # API latency statistics
        api_stats = {}
        for api, latencies in self.api_latency_history.items():
            if latencies:
                api_stats[api] = {
                    'avg_ms': sum(latencies) / len(latencies),
                    'max_ms': max(latencies),
                    'min_ms': min(latencies),
                    'samples': len(latencies)
                }

        return {
            'summary': {
                'average_memory_mb': avg_memory,
                'max_memory_mb': max_memory,
                'average_cpu_percent': avg_cpu,
                'average_health_score': avg_health_score,
                'samples': len(recent)
            },
            'memory': {
                'current_mb': recent[-1]['memory']['usage_mb'] if recent else 0,
                'average_mb': avg_memory,
                'max_mb': max_memory,
                'limit_mb': 4000
            },
            'cpu': {
                'current_percent': recent[-1]['cpu']['usage_percent'] if recent else 0,
                'average_percent': avg_cpu,
                'cores': recent[-1]['cpu']['count'] if recent else 0
            },
            'apis': api_stats,
            'errors': {
                'total': len(self.error_history),
                'recent': len([e for e in self.error_history
                             if (datetime.now() - datetime.fromisoformat(e['timestamp'])).total_seconds() < 3600])
            },
            'recommendations': self._get_recommendations(avg_memory, avg_cpu, avg_health_score)
        }

    def _get_recommendations(self, avg_memory: float, avg_cpu: float, health_score: float) -> List[str]:
        """Get performance recommendations."""
        recommendations = []

        if avg_memory > 3500:
            recommendations.append("Consider reducing data cache size to lower memory usage")
            recommendations.append("Implement more aggressive garbage collection")

        if avg_cpu > 70:
            recommendations.append("Review CPU-intensive operations for optimization")
            recommendations.append("Consider reducing calculation frequency")

        if health_score < 0.7:
            recommendations.append("System health is suboptimal - review all metrics")
            recommendations.append("Check for memory leaks or inefficient code")

        if not recommendations:
            recommendations.append("System performance is healthy")

        return recommendations

    async def run_health_check(self) -> Dict[str, Any]:
        """
        Run comprehensive health check.

        Returns:
            Complete health check report
        """
        health = await self.check_system_health()
        error_rate = await self.check_error_rate()

        # Determine overall status
        overall_status = "HEALTHY"
        if health['status'] == 'CRITICAL' or error_rate['error_rate'] > 0.1:
            overall_status = "CRITICAL"
        elif health['status'] == 'WARNING' or error_rate['error_rate'] > 0.05:
            overall_status = "WARNING"

        return {
            'overall_status': overall_status,
            'timestamp': datetime.now().isoformat(),
            'system_health': health,
            'error_rate': error_rate,
            'recommendations': self._get_recommendations(
                health['memory']['usage_mb'],
                health['cpu']['usage_percent'],
                health['health_score']
            )
        }
