"""
Alert System
Handles browser notifications, logs, and threshold-based alerts for trading signals.
"""
import json
import os
from datetime import datetime
from typing import Dict, Any, Optional, List, Callable
from loguru import logger


class AlertLevel:
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    SUCCESS = "success"


class AlertManager:
    """
    Manages alerts for entry, take profit, and stop loss thresholds.
    Supports browser notifications and file logging.
    """

    def __init__(self, log_dir: str = "logs/alerts"):
        """
        Initialize AlertManager.

        Args:
            log_dir: Directory to store alert logs
        """
        self.log_dir = log_dir
        self.alert_history: List[Dict[str, Any]] = []
        self.notification_callbacks: List[Callable] = []
        self.active_thresholds: Dict[str, Dict[str, Any]] = {}

        # Ensure log directory exists
        os.makedirs(log_dir, exist_ok=True)

        logger.info(f"AlertManager initialized with log directory: {log_dir}")

    def register_notification_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """
        Register a callback for browser notifications.

        Args:
            callback: Function that accepts alert dict and shows notification
        """
        self.notification_callbacks.append(callback)
        logger.debug(f"Registered notification callback (total: {len(self.notification_callbacks)})")

    def add_threshold(
        self,
        symbol: str,
        threshold_type: str,  # 'entry', 'stop_loss', 'take_profit_1', 'take_profit_2'
        target_price: float,
        current_price: float,
        direction: str,  # 'above', 'below'
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Add a price threshold to monitor.

        Args:
            symbol: Trading symbol (e.g., 'BTCUSDT')
            threshold_type: Type of threshold
            target_price: Price level to monitor
            current_price: Current market price
            direction: 'above' or 'below'
            metadata: Additional info (entry_price, confidence, etc.)

        Returns:
            threshold_id: Unique ID for this threshold
        """
        threshold_id = f"{symbol}_{threshold_type}_{int(datetime.now().timestamp())}"

        threshold = {
            "id": threshold_id,
            "symbol": symbol,
            "type": threshold_type,
            "target_price": target_price,
            "direction": direction,
            "created_at": datetime.now().isoformat(),
            "triggered": False,
            "metadata": metadata or {},
            "alert_level": self._determine_alert_level(threshold_type)
        }

        self.active_thresholds[threshold_id] = threshold

        logger.info(
            f"Added threshold: {symbol} {threshold_type} {direction} {target_price:.2f} "
            f"(current: {current_price:.2f})"
        )

        return threshold_id

    def check_thresholds(self, symbol: str, current_price: float) -> List[Dict[str, Any]]:
        """
        Check all active thresholds for a symbol.

        Args:
            symbol: Trading symbol
            current_price: Current market price

        Returns:
            List of triggered alerts
        """
        triggered = []

        for threshold_id, threshold in list(self.active_thresholds.items()):
            # Skip if already triggered or different symbol
            if threshold["triggered"] or threshold["symbol"] != symbol:
                continue

            target = threshold["target_price"]
            direction = threshold["direction"]

            # Check if threshold is hit
            hit = False
            if direction == "above" and current_price >= target:
                hit = True
            elif direction == "below" and current_price <= target:
                hit = True

            if hit:
                threshold["triggered"] = True
                threshold["triggered_at"] = datetime.now().isoformat()
                threshold["trigger_price"] = current_price

                alert = self._create_alert(threshold, current_price)
                triggered.append(alert)

                # Send notification
                self._send_notification(alert)

                # Log alert
                self._log_alert(alert)

                # Add to history
                self.alert_history.append(alert)

                logger.warning(
                    f"Threshold triggered: {symbol} {threshold['type']} at {current_price:.2f} "
                    f"(target: {target:.2f})"
                )

        return triggered

    def remove_threshold(self, threshold_id: str) -> bool:
        """
        Remove a threshold by ID.

        Args:
            threshold_id: Threshold ID to remove

        Returns:
            True if removed, False if not found
        """
        if threshold_id in self.active_thresholds:
            del self.active_thresholds[threshold_id]
            logger.debug(f"Removed threshold: {threshold_id}")
            return True
        return False

    def clear_thresholds(self, symbol: Optional[str] = None):
        """
        Clear all thresholds, optionally for a specific symbol.

        Args:
            symbol: If provided, only clear thresholds for this symbol
        """
        if symbol:
            to_remove = [
                tid for tid, t in self.active_thresholds.items()
                if t["symbol"] == symbol
            ]
            for tid in to_remove:
                del self.active_thresholds[tid]
            logger.info(f"Cleared {len(to_remove)} thresholds for {symbol}")
        else:
            count = len(self.active_thresholds)
            self.active_thresholds.clear()
            logger.info(f"Cleared all {count} thresholds")

    def get_active_thresholds(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get list of active thresholds.

        Args:
            symbol: Optional symbol filter

        Returns:
            List of threshold dicts
        """
        if symbol:
            return [
                t for t in self.active_thresholds.values()
                if t["symbol"] == symbol and not t["triggered"]
            ]
        return [t for t in self.active_thresholds.values() if not t["triggered"]]

    def get_alert_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get recent alert history.

        Args:
            limit: Maximum number of alerts to return

        Returns:
            List of alerts (most recent first)
        """
        return self.alert_history[-limit:]

    def _determine_alert_level(self, threshold_type: str) -> str:
        """Determine alert level based on threshold type."""
        if threshold_type in ["stop_loss"]:
            return AlertLevel.CRITICAL
        elif threshold_type in ["take_profit_1", "take_profit_2"]:
            return AlertLevel.SUCCESS
        elif threshold_type in ["entry"]:
            return AlertLevel.INFO
        return AlertLevel.WARNING

    def _create_alert(self, threshold: Dict[str, Any], current_price: float) -> Dict[str, Any]:
        """Create an alert dict from a triggered threshold."""
        return {
            "id": threshold["id"],
            "symbol": threshold["symbol"],
            "type": threshold["type"],
            "level": threshold["alert_level"],
            "message": self._format_message(threshold, current_price),
            "timestamp": datetime.now().isoformat(),
            "target_price": threshold["target_price"],
            "current_price": current_price,
            "direction": threshold["direction"],
            "metadata": threshold["metadata"]
        }

    def _format_message(self, threshold: Dict[str, Any], current_price: float) -> str:
        """Format alert message."""
        symbol = threshold["symbol"]
        ttype = threshold["type"].replace("_", " ").title()
        target = threshold["target_price"]
        direction = threshold["direction"]

        metadata = threshold.get("metadata", {})
        confidence = metadata.get("confidence", 0)
        entry_type = metadata.get("entry_type", "")

        base_msg = f"{symbol} {ttype} {direction} {target:.2f}"
        extra = []

        if confidence > 0:
            extra.append(f"confidence: {confidence:.1%}")

        if entry_type:
            extra.append(f"type: {entry_type}")

        if extra:
            return f"{base_msg} ({', '.join(extra)})"

        return base_msg

    def _send_notification(self, alert: Dict[str, Any]):
        """Send notification via registered callbacks."""
        for callback in self.notification_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Failed to send notification: {e}")

    def _log_alert(self, alert: Dict[str, Any]):
        """Log alert to file."""
        try:
            log_file = os.path.join(self.log_dir, f"{alert['symbol'].lower()}_alerts.log")

            log_entry = {
                "timestamp": alert["timestamp"],
                "level": alert["level"],
                "type": alert["type"],
                "symbol": alert["symbol"],
                "message": alert["message"],
                "target": alert["target_price"],
                "current": alert["current_price"],
                "metadata": alert.get("metadata", {})
            }

            with open(log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry) + "\n")

        except Exception as e:
            logger.error(f"Failed to log alert: {e}")

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get alert statistics.

        Returns:
            Dict with statistics
        """
        total_alerts = len(self.alert_history)
        by_level = {}
        by_type = {}

        for alert in self.alert_history:
            level = alert["level"]
            ttype = alert["type"]

            by_level[level] = by_level.get(level, 0) + 1
            by_type[ttype] = by_type.get(ttype, 0) + 1

        return {
            "total_alerts": total_alerts,
            "active_thresholds": len(self.get_active_thresholds()),
            "by_level": by_level,
            "by_type": by_type,
            "last_alert": self.alert_history[-1] if self.alert_history else None
        }


# Global alert manager instance
_alert_manager: Optional[AlertManager] = None


def get_alert_manager() -> AlertManager:
    """Get global AlertManager instance."""
    global _alert_manager
    if _alert_manager is None:
        _alert_manager = AlertManager()
    return _alert_manager


def add_entry_alert(
    symbol: str,
    entry_price: float,
    current_price: float,
    direction: str,
    metadata: Optional[Dict[str, Any]] = None
) -> str:
    """
    Convenience function to add entry price alert.

    Args:
        symbol: Trading symbol
        entry_price: Target entry price
        current_price: Current price
        direction: 'above' or 'below'
        metadata: Additional info

    Returns:
        threshold_id
    """
    return get_alert_manager().add_threshold(
        symbol, "entry", entry_price, current_price, direction, metadata
    )


def add_stop_loss_alert(
    symbol: str,
    stop_loss: float,
    current_price: float,
    direction: str,
    metadata: Optional[Dict[str, Any]] = None
) -> str:
    """
    Convenience function to add stop loss alert.

    Args:
        symbol: Trading symbol
        stop_loss: Stop loss price
        current_price: Current price
        direction: 'above' or 'below'
        metadata: Additional info

    Returns:
        threshold_id
    """
    return get_alert_manager().add_threshold(
        symbol, "stop_loss", stop_loss, current_price, direction, metadata
    )


def add_take_profit_alert(
    symbol: str,
    tp_number: int,  # 1 or 2
    take_profit_price: float,
    current_price: float,
    direction: str,
    metadata: Optional[Dict[str, Any]] = None
) -> str:
    """
    Convenience function to add take profit alert.

    Args:
        symbol: Trading symbol
        tp_number: Take profit level (1 or 2)
        take_profit_price: Take profit price
        current_price: Current price
        direction: 'above' or 'below'
        metadata: Additional info

    Returns:
        threshold_id
    """
    return get_alert_manager().add_threshold(
        symbol, f"take_profit_{tp_number}", take_profit_price, current_price, direction, metadata
    )


def check_all_alerts(symbol: str, current_price: float) -> List[Dict[str, Any]]:
    """
    Convenience function to check all alerts for a symbol.

    Args:
        symbol: Trading symbol
        current_price: Current market price

    Returns:
        List of triggered alerts
    """
    return get_alert_manager().check_thresholds(symbol, current_price)


def clear_all_alerts(symbol: Optional[str] = None):
    """
    Convenience function to clear all alerts.

    Args:
        symbol: If provided, only clear for this symbol
    """
    get_alert_manager().clear_thresholds(symbol)
