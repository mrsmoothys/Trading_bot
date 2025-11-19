"""
Memory Manager
Manages system memory for M1 MacBook with 4GB limit.
Optimizes memory usage and prevents OOM errors.
"""

import gc
import psutil
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from loguru import logger
import pandas as pd


class M1MemoryManager:
    """
    Memory manager optimized for M1 MacBook Air 8GB RAM.
    Keeps memory usage under 4GB for the application.
    """

    def __init__(self, memory_limit_mb: int = 4000):
        """
        Initialize memory manager.

        Args:
            memory_limit_mb: Memory limit in MB (default: 4GB)
        """
        self.memory_limit = memory_limit_mb * 1024 * 1024  # Convert to bytes
        self.warning_threshold = self.memory_limit * 0.85  # 85% warning
        self.critical_threshold = self.memory_limit * 0.95  # 95% critical

        # Track memory usage over time
        self.memory_history: List[Dict[str, Any]] = []

        # Cache for cleanup tracking
        self.last_cleanup = datetime.now()

        logger.info(f"M1MemoryManager initialized with {memory_limit_mb}MB limit")

    def get_current_memory_usage(self) -> Dict[str, float]:
        """
        Get current memory usage.

        Returns:
            Dictionary with memory metrics
        """
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            memory_percent = process.memory_percent()

            return {
                'rss_mb': memory_info.rss / 1024 / 1024,  # Resident Set Size
                'vms_mb': memory_info.vms / 1024 / 1024,  # Virtual Memory Size
                'percent': memory_percent,
                'available_mb': psutil.virtual_memory().available / 1024 / 1024,
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error getting memory usage: {e}")
            return {
                'rss_mb': 0,
                'vms_mb': 0,
                'percent': 0,
                'available_mb': 0,
                'timestamp': datetime.now().isoformat()
            }

    def optimize_context_for_deepseek(self, system_context) -> Dict[str, Any]:
        """
        Optimize system context for DeepSeek within memory limits.

        Args:
            system_context: SystemContext instance

        Returns:
            Optimized context dictionary
        """
        try:
            # Keep only essential data
            optimized = {
                "timestamp": datetime.now().isoformat(),
                "market_regime": system_context.market_regime,
                "feature_performance": self._compress_features(system_context.feature_performance),
                "active_positions": {
                    symbol: {
                        "side": pos.get("side"),
                        "quantity": pos.get("quantity"),
                        "unrealized_pnl": pos.get("unrealized_pnl"),
                        "entry_price": pos.get("entry_price"),
                        "confidence": pos.get("confidence", 0.5)
                    }
                    for symbol, pos in system_context.active_positions.items()
                },
                "risk_exposure": system_context.risk_metrics.get("total_exposure", 0),
                "current_drawdown": system_context.risk_metrics.get("current_drawdown", 0),
                "system_health": {
                    "memory_usage_mb": system_context.system_health.get("memory_usage", 0),
                    "recent_errors": system_context.system_health.get("errors", [])[-5:]
                },
                "recent_trades": [
                    {
                        "symbol": t.symbol,
                        "pnl_percent": t.pnl_percent,
                        "exit_reason": t.exit_reason,
                        "confidence": t.confidence,
                    }
                    for t in system_context.trade_history[-10:]
                ],
                "current_features": self._compress_features_dict(system_context.feature_calculations)
            }

            return optimized

        except Exception as e:
            logger.error(f"Error optimizing context for DeepSeek: {e}")
            return {}

    def _compress_features(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Compress feature data to essential info."""
        compressed = {}
        for name, data in features.items():
            if hasattr(data, '__dict__'):
                # FeatureMetrics object
                compressed[name] = {
                    "accuracy": getattr(data, 'accuracy', 0),
                    "trend": getattr(data, 'trend', 'neutral'),
                    "total_signals": getattr(data, 'total_signals', 0)
                }
            else:
                # Dictionary
                compressed[name] = {
                    "accuracy": data.get("accuracy", 0),
                    "trend": data.get("trend", "neutral"),
                    "total_signals": data.get("total_signals", 0)
                }
        return compressed

    def _compress_features_dict(self, features_dict: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Compress feature calculations."""
        compressed = {}
        for symbol, features in features_dict.items():
            # Keep only numeric values and short strings
            compressed[symbol] = {
                k: v for k, v in features.items()
                if isinstance(v, (int, float, bool)) or
                   (isinstance(v, str) and len(v) < 100)
            }
        return compressed

    async def cleanup_cache(self, data_store):
        """
        Clean up expired and old cache entries.

        Args:
            data_store: DataStore instance
        """
        try:
            logger.info("Starting cache cleanup...")

            # Clean expired items
            data_store.cleanup_expired()

            # Get cache stats
            stats = data_store.get_cache_stats()

            # If memory usage is high, clear more aggressively
            memory_usage = self.get_current_memory_usage()
            if memory_usage['rss_mb'] > self.memory_limit / 1024 / 1024 * 0.8:
                logger.warning("High memory usage - aggressive cleanup")

                # Clear all cache
                data_store.clear_cache()
                logger.info("Cache cleared due to high memory usage")

            # Force garbage collection
            gc.collect()

            self.last_cleanup = datetime.now()

            logger.info(f"Cache cleanup completed. Stats: {stats}")

        except Exception as e:
            logger.error(f"Error during cache cleanup: {e}")

    def optimize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Optimize DataFrame memory usage.

        Args:
            df: Input DataFrame

        Returns:
            Optimized DataFrame
        """
        try:
            # Optimize numeric types
            for col in df.select_dtypes(include=['int64']).columns:
                df[col] = pd.to_numeric(df[col], downcast='integer')

            for col in df.select_dtypes(include=['float64']).columns:
                df[col] = pd.to_numeric(df[col], downcast='float')

            # Optimize object types
            for col in df.select_dtypes(include=['object']).columns:
                if df[col].nunique() / len(df) < 0.5:  # Less than 50% unique
                    df[col] = df[col].astype('category')

            return df

        except Exception as e:
            logger.error(f"Error optimizing DataFrame: {e}")
            return df

    def should_trigger_cleanup(self) -> bool:
        """
        Check if cleanup should be triggered.

        Returns:
            True if cleanup should run
        """
        # Check memory usage
        memory = self.get_current_memory_usage()

        if memory['rss_mb'] > self.memory_limit / 1024 / 1024 * 0.8:
            return True

        # Check if last cleanup was too long ago
        if (datetime.now() - self.last_cleanup).total_seconds() > 300:  # 5 minutes
            return True

        return False

    def get_memory_alerts(self) -> List[Dict[str, Any]]:
        """
        Get memory-related alerts.

        Returns:
            List of alert dictionaries
        """
        alerts = []
        memory = self.get_current_memory_usage()

        # Warning level
        if memory['rss_mb'] * 1024 * 1024 > self.warning_threshold:
            alerts.append({
                'level': 'WARNING',
                'message': f"Memory usage at {memory['rss_mb']:.0f}MB",
                'timestamp': datetime.now().isoformat()
            })

        # Critical level
        if memory['rss_mb'] * 1024 * 1024 > self.critical_threshold:
            alerts.append({
                'level': 'CRITICAL',
                'message': f"Critical memory usage at {memory['rss_mb']:.0f}MB",
                'timestamp': datetime.now().isoformat()
            })

        return alerts

    async def monitor_memory_loop(self, interval: int = 60):
        """
        Continuous memory monitoring loop.

        Args:
            interval: Check interval in seconds
        """
        while True:
            try:
                memory = self.get_current_memory_usage()

                # Log memory usage
                if memory['percent'] > 50:  # Log if over 50%
                    logger.info(f"Memory usage: {memory['rss_mb']:.0f}MB ({memory['percent']:.1f}%)")

                # Store in history
                self.memory_history.append(memory)

                # Keep only last 100 entries
                if len(self.memory_history) > 100:
                    self.memory_history = self.memory_history[-100:]

                # Get alerts
                alerts = self.get_memory_alerts()
                for alert in alerts:
                    if alert['level'] == 'CRITICAL':
                        logger.critical(alert['message'])
                    elif alert['level'] == 'WARNING':
                        logger.warning(alert['message'])

                await asyncio.sleep(interval)

            except Exception as e:
                logger.error(f"Error in memory monitoring loop: {e}")
                await asyncio.sleep(interval)

    def get_memory_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive memory report.

        Returns:
            Memory report dictionary
        """
        current = self.get_current_memory_usage()
        history = self.memory_history[-60:] if self.memory_history else []

        if history:
            avg_usage = sum(h['rss_mb'] for h in history) / len(history)
            max_usage = max(h['rss_mb'] for h in history)
            min_usage = min(h['rss_mb'] for h in history)
        else:
            avg_usage = max_usage = min_usage = current['rss_mb']

        return {
            'current': current,
            'statistics': {
                'average_mb': avg_usage,
                'max_mb': max_usage,
                'min_mb': min_usage,
                'samples': len(history)
            },
            'limits': {
                'configured_mb': self.memory_limit / 1024 / 1024,
                'warning_mb': self.warning_threshold / 1024 / 1024,
                'critical_mb': self.critical_threshold / 1024 / 1024
            },
            'status': 'HEALTHY' if current['rss_mb'] * 1024 * 1024 < self.warning_threshold else 'WARNING',
            'alerts': self.get_memory_alerts()
        }

    def force_cleanup(self):
        """Force aggressive cleanup."""
        logger.info("Forcing aggressive cleanup...")

        # Clear all Python caches
        gc.collect()

        # Clear module caches
        import sys
        modules_to_clear = [m for m in sys.modules.keys() if 'cache' in m.lower()]
        for module in modules_to_clear:
            if module in sys.modules:
                del sys.modules[module]

        logger.info("Aggressive cleanup completed")
