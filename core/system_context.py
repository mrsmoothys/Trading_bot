"""
System Context Module
Provides comprehensive system state management for DeepSeek integration.
Handles feature performance tracking, position management, risk metrics, and more.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
import json
import yaml
import os
from pathlib import Path
from loguru import logger
from core.data.sentiment import fetch_fear_and_greed_index


@dataclass
class TradeRecord:
    """Record of a completed trade."""
    symbol: str
    entry_time: datetime
    exit_time: datetime
    side: str  # 'LONG' or 'SHORT'
    entry_price: float
    exit_price: float
    quantity: float
    pnl: float
    pnl_percent: float
    confidence: float
    reasoning: str
    exit_reason: str


@dataclass
class FeatureMetrics:
    """Performance metrics for a feature."""
    name: str
    accuracy: float
    total_signals: int = 0
    successful_signals: int = 0
    trend: str = "neutral"  # 'improving', 'stable', 'declining'
    last_updated: datetime = field(default_factory=datetime.now)


class SystemContext:
    """
    Central system context provider for DeepSeek AI integration.
    Maintains comprehensive system awareness including:
    - Feature performance metrics
    - Active positions and P&L
    - Portfolio risk exposure
    - Market regime
    - System health
    - Trade history
    - Feature calculations
    """

    def __init__(self, config_path: Optional[str] = None):
        """Initialize SystemContext with configuration."""
        self.config_path = config_path or "config/system_config.yaml"
        self._load_config()

        # Core state tracking
        self.feature_performance: Dict[str, FeatureMetrics] = {}
        self.feature_resource_usage: Dict[str, Dict[str, Any]] = {}
        self.active_positions: Dict[str, Dict[str, Any]] = {}
        self.risk_metrics: Dict[str, Any] = {
            "total_exposure": 0.0,
            "current_drawdown": 0.0,
            "max_drawdown": 0.0,
            "portfolio_value": 100000.0,  # Starting value
            "unrealized_pnl": 0.0,
        }
        self.market_regime: str = "UNKNOWN"
        self.system_health: Dict[str, Any] = {
            "memory_usage": 0,
            "cpu_usage": 0,
            "errors": [],
            "last_check": datetime.now(),
        }
        self.trade_history: List[TradeRecord] = []
        self.feature_calculations: Dict[str, Dict[str, Any]] = {}

        # External Sentiment
        self.sentiment: Dict[str, Any] = fetch_fear_and_greed_index()

        # Overlay state tracking - tracks which chart overlays are currently active
        self.overlay_state: Dict[str, bool] = {
            "liquidity": False,
            "supertrend": False,
            "chandelier": False,
            "orderflow": True,  # Default enabled
            "regime": False,
            "alignment": False
        }
        self.overlay_history: List[Dict[str, Any]] = []

        # Feature profile tracking - tracks which strategy mode is active
        self.selected_feature_profile: str = "custom"  # 'scalp', 'swing', or 'custom'
        self.profile_history: List[Dict[str, Any]] = []

        # Track last bar timestamp for each symbol:timeframe to enable conditional feature recalculation
        # Format: {symbol: {timeframe: datetime}}
        self.last_bar_timestamps: Dict[str, Dict[str, datetime]] = {}

        # Feature cadence definitions - heavy features only recalc on bar close, lightweight on every update
        # Tied to profile selection to optimize performance
        self.feature_cadence = {
            "scalp": {
                # Heavy features (bar-close only)
                "heavy": ["liquidity", "alignment"],
                # Lightweight features (every update)
                "lightweight": ["orderflow", "regime", "chandelier", "supertrend"]
            },
            "swing": {
                # Heavy features (bar-close only)
                "heavy": ["liquidity", "supertrend", "alignment"],
                # Lightweight features (every update)
                "lightweight": ["orderflow", "regime", "chandelier"]
            },
            "custom": {
                # Heavy features (bar-close only)
                "heavy": ["liquidity", "supertrend", "alignment"],
                # Lightweight features (every update)
                "lightweight": ["orderflow", "regime", "chandelier"]
            }
        }

        # Memory management
        self.max_trade_history = self.config.get("memory", {}).get("max_trade_history", 1000)
        self.max_conversation_memory = self.config.get("memory", {}).get("max_conversation_memory", 100)
        self.conversation_memory: List[Dict[str, Any]] = []

        # Reconcile with database on startup (Step 3)
        self._reconcile_with_database()

    def _reconcile_with_database(self):
        """
        Reconcile active positions with database on startup.
        """
        try:
            # Check if database persistence is enabled (feature flag)
            if not self.config.get('database', {}).get('persist_positions', False):
                logger.debug("Database position persistence disabled (feature flag off)")
                return

            import asyncio
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Schedule async task
                asyncio.create_task(self._reconcile_async())
            else:
                loop.run_until_complete(self._reconcile_async())
        except Exception as e:
            logger.warning(f"Could not reconcile with database (non-critical): {e}")

    async def _reconcile_async(self):
        """Async reconciliation with database."""
        try:
            from ops.db import get_database
            db = get_database()

            # Load active positions from database
            db_positions = await db.get_active_positions()

            # Reconcile with in-memory state
            if db_positions:
                logger.info(f"Reconciling {len(db_positions)} active positions from database")
                for pos in db_positions:
                    symbol = pos['symbol']
                    # Convert database record to SystemContext format
                    self.active_positions[symbol] = {
                        'id': pos['id'],
                        'symbol': symbol,
                        'side': pos['side'],
                        'entry_price': pos['entry_price'],
                        'quantity': pos['quantity'],
                        'leverage': pos['leverage'],
                        'unrealized_pnl': pos['unrealized_pnl'],
                        'value': pos['value'],
                        'entry_time': pos['entry_time'],
                        'confidence': pos['confidence'],
                        'reasoning': pos['reasoning'],
                        'strategy': pos['strategy'],
                        'stop_loss': pos['stop_loss'],
                        'take_profit': pos['take_profit'],
                        'source': pos['source'],
                        'last_updated': pos['last_updated'],
                    }
                logger.info(f"Loaded {len(db_positions)} positions from database into SystemContext")
            else:
                logger.debug("No active positions found in database")

        except Exception as e:
            logger.error(f"Failed to reconcile with database: {e}")

    def _load_config(self):
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        except FileNotFoundError:
            print(f"⚠️  Warning: Config file {self.config_path} not found. Using defaults.")
            self.config = {}

    def update_config(self, category: str, key: str, value: Any):
        """
        Allow the AI to update system configuration safely.
        
        Args:
            category: Config category (e.g., 'risk', 'trading')
            key: Config key (e.g., 'max_drawdown', 'position_size')
            value: New value to set
            
        Raises:
            ValueError: If the update violates safety guardrails
        """
        # 1. Safety Guardrails
        if category == "risk":
            if key == "max_drawdown" and (not isinstance(value, (int, float)) or value > 0.20):
                raise ValueError(f"Safety Guardrail: Max drawdown cannot exceed 20% (tried setting {value})")
            if key == "max_position_size" and (not isinstance(value, (int, float)) or value > 0.10):
                raise ValueError(f"Safety Guardrail: Max position size cannot exceed 10% (tried setting {value})")
                
        if category == "trading":
            if key == "leverage" and (not isinstance(value, (int, float)) or value > 5):
                raise ValueError(f"Safety Guardrail: Max leverage cannot exceed 5x (tried setting {value})")

        # 2. Update the setting
        if category not in self.config:
            self.config[category] = {}
            
        old_value = self.config[category].get(key, "N/A")
        self.config[category][key] = value
        
        # 3. Log the change
        logger.info(f"🤖 AI updated config: {category}.{key} changed from {old_value} to {value}")
        
        # Persist changes if needed (optional, for now we keep it runtime only)
        # self.save_config() 

    def update_feature_performance(self, feature_name: str, signal_correct: bool):
        """Update feature performance metrics."""
        if feature_name not in self.feature_performance:
            self.feature_performance[feature_name] = FeatureMetrics(
                name=feature_name,
                accuracy=0.0
            )

        metrics = self.feature_performance[feature_name]
        metrics.total_signals += 1
        if signal_correct:
            metrics.successful_signals += 1

        # Update accuracy
        metrics.accuracy = metrics.successful_signals / metrics.total_signals
        metrics.last_updated = datetime.now()

        # Update trend
        if metrics.total_signals >= 10:
            recent_signals = min(10, metrics.total_signals)
            recent_correct = sum(1 for _ in range(recent_signals) if _ < metrics.successful_signals)
            recent_accuracy = recent_correct / recent_signals

            if recent_accuracy > metrics.accuracy + 0.05:
                metrics.trend = "improving"
            elif recent_accuracy < metrics.accuracy - 0.05:
                metrics.trend = "declining"
            else:
                metrics.trend = "stable"

    def update_position(self, symbol: str, position_data: Dict[str, Any]):
        """Update active position data."""
        self.active_positions[symbol] = {
            **position_data,
            "last_updated": datetime.now().isoformat(),
        }
        self._update_risk_metrics()
        # Save to disk for split-brain synchronization
        self.save_to_disk()
        # Save to database for persistence (Step 3)
        self._save_position_to_db(symbol, self.active_positions[symbol])

    def _save_position_to_db(self, symbol: str, position_data: Dict[str, Any]):
        """Save position to database asynchronously."""
        try:
            import asyncio
            from ops.db import get_database

            # Check if we should save to DB (feature flag)
            if not self.config.get('database', {}).get('persist_positions', False):
                return

            # Run async database operation in event loop
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If we're already in an async context, schedule the task
                asyncio.create_task(self._save_position_async(position_data))
            else:
                # If no event loop, create one temporarily
                loop.run_until_complete(self._save_position_async(position_data))
        except Exception as e:
            logger.warning(f"Could not save position to database (non-critical): {e}")

    async def _save_position_async(self, position_data: Dict[str, Any]):
        """Async wrapper for saving position to database."""
        try:
            from ops.db import get_database
            db = get_database()
            await db.save_active_position(position_data)
        except Exception as e:
            logger.error(f"Failed to save position to database: {e}")

    def close_position(self, symbol: str, exit_data: Dict[str, Any]):
        """Close a position and record the trade."""
        if symbol not in self.active_positions:
            return

        entry_data = self.active_positions[symbol]
        trade = TradeRecord(
            symbol=symbol,
            entry_time=datetime.fromisoformat(entry_data["entry_time"]),
            exit_time=datetime.now(),
            side=entry_data["side"],
            entry_price=entry_data["entry_price"],
            exit_price=exit_data["exit_price"],
            quantity=entry_data["quantity"],
            pnl=exit_data["pnl"],
            pnl_percent=exit_data["pnl_percent"],
            confidence=entry_data.get("confidence", 0.0),
            reasoning=entry_data.get("reasoning", ""),
            exit_reason=exit_data.get("reason", "MANUAL"),
        )

        self.trade_history.append(trade)

        # Keep trade history within limits
        if len(self.trade_history) > self.max_trade_history:
            self.trade_history = self.trade_history[-self.max_trade_history:]

        # Remove from active positions
        del self.active_positions[symbol]
        self._update_risk_metrics()
        # Save to disk for split-brain synchronization
        self.save_to_disk()
        # Remove from database (Step 3)
        self._remove_position_from_db(symbol)

    def _remove_position_from_db(self, symbol: str):
        """Remove position from database asynchronously."""
        try:
            import asyncio
            from ops.db import get_database

            # Check if we should save to DB (feature flag)
            if not self.config.get('database', {}).get('persist_positions', False):
                return

            # Run async database operation
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.create_task(self._remove_position_async(symbol))
            else:
                loop.run_until_complete(self._remove_position_async(symbol))
        except Exception as e:
            logger.warning(f"Could not remove position from database (non-critical): {e}")

    async def _remove_position_async(self, symbol: str):
        """Async wrapper for removing position from database."""
        try:
            from ops.db import get_database
            db = get_database()
            # For now, just log - actual removal would be in trade history
            logger.info(f"Position {symbol} closed and removed from active tracking")
        except Exception as e:
            logger.error(f"Failed to remove position from database: {e}")

    def _update_risk_metrics(self):
        """Update portfolio risk metrics."""
        total_unrealized_pnl = sum(
            pos.get("unrealized_pnl", 0.0) for pos in self.active_positions.values()
        )
        self.risk_metrics["unrealized_pnl"] = total_unrealized_pnl

        # Calculate drawdown
        current_value = self.risk_metrics["portfolio_value"] + total_unrealized_pnl
        peak_value = max(
            current_value,
            max([t.entry_price * t.quantity for t in self.trade_history], default=current_value)
        )
        self.risk_metrics["current_drawdown"] = (peak_value - current_value) / peak_value
        self.risk_metrics["max_drawdown"] = max(
            self.risk_metrics["max_drawdown"],
            self.risk_metrics["current_drawdown"]
        )

        # Calculate total exposure
        total_exposure = sum(
            abs(pos.get("value", 0.0)) for pos in self.active_positions.values()
        )
        self.risk_metrics["total_exposure"] = total_exposure / current_value

    def update_market_regime(self, regime: str):
        """Update current market regime."""
        self.market_regime = regime

    def update_system_health(self, memory_mb: float, cpu_percent: float, errors: List[str]):
        """Update system health metrics."""
        self.system_health.update({
            "memory_usage": memory_mb,
            "cpu_usage": cpu_percent,
            "errors": errors,
            "last_check": datetime.now(),
        })
        # Save to disk for split-brain synchronization
        self.save_to_disk()

    def update_feature_calculations(self, symbol: str, features: Dict[str, Any]):
        """Update calculated features for a symbol."""
        self.feature_calculations[symbol] = {
            **features,
            "timestamp": datetime.now().isoformat(),
        }

    def update_feature_resource_metrics(self, metrics: Dict[str, Dict[str, Any]]):
        """Store latest feature resource usage metrics."""
        if not metrics:
            return
        self.feature_resource_usage.update(metrics)

    def update_overlay_state(self, overlay_changes: Dict[str, bool]):
        """
        Update the state of chart overlays.

        Args:
            overlay_changes: Dictionary mapping overlay names to their new state
                           e.g., {'supertrend': True, 'liquidity': False}
        """
        # Track what changed
        changes_made = []
        for overlay_name, new_state in overlay_changes.items():
            if overlay_name in self.overlay_state:
                old_state = self.overlay_state[overlay_name]
                if old_state != new_state:
                    self.overlay_state[overlay_name] = new_state
                    changes_made.append({
                        "overlay": overlay_name,
                        "old_state": old_state,
                        "new_state": new_state,
                        "timestamp": datetime.now().isoformat()
                    })

        # Add to history if there were changes
        if changes_made:
            self.overlay_history.extend(changes_made)
            # Keep only last 100 changes
            if len(self.overlay_history) > 100:
                self.overlay_history = self.overlay_history[-100:]

    def update_feature_profile(self, profile: str):
        """
        Update the selected feature profile.

        Args:
            profile: One of 'scalp', 'swing', or 'custom'
        """
        if profile not in ["scalp", "swing", "custom"]:
            raise ValueError(f"Invalid profile '{profile}'. Must be 'scalp', 'swing', or 'custom'")

        old_profile = self.selected_feature_profile
        self.selected_feature_profile = profile

        # Track in history
        self.profile_history.append({
            "old_profile": old_profile,
            "new_profile": profile,
            "timestamp": datetime.now().isoformat()
        })

        # Keep only last 50 changes
        if len(self.profile_history) > 50:
            self.profile_history = self.profile_history[-50:]

    def should_recalculate_feature(self, feature_name: str, symbol: str, timeframe: str, current_bar_timestamp: datetime) -> bool:
        """
        Determine if a feature should be recalculated based on feature type and bar closure.

        Uses the current profile's cadence settings to determine if a feature should be recalculated.
        Heavy features only recalc on bar close, lightweight features recalc every update.

        Args:
            feature_name: Name of the feature to check
            symbol: Trading symbol (e.g., 'BTCUSDT')
            timeframe: Timeframe (e.g., '15m', '1h')
            current_bar_timestamp: Timestamp of the current/latest bar

        Returns:
            True if feature should be recalculated, False if it can use cached values
        """
        # Get cadence settings based on current profile
        profile = self.selected_feature_profile
        cadence = self.feature_cadence.get(profile, self.feature_cadence["custom"])

        heavy_features = set(cadence.get("heavy", []))
        lightweight_features = set(cadence.get("lightweight", []))

        if feature_name in heavy_features:
            # Check if new bar has closed for this symbol:timeframe
            symbol_tf_key = f"{symbol}:{timeframe}"
            last_timestamp = self.last_bar_timestamps.get(symbol_tf_key)

            if last_timestamp is None:
                # First time calculating this feature - always recalculate
                self.last_bar_timestamps[symbol_tf_key] = current_bar_timestamp
                logger.debug(f"First calculation of {feature_name} for {symbol_tf_key} - recalculating")
                return True

            # Recalculate if timestamp has changed (new bar)
            if current_bar_timestamp > last_timestamp:
                self.last_bar_timestamps[symbol_tf_key] = current_bar_timestamp
                logger.debug(f"New bar detected for {symbol_tf_key} - recalculating {feature_name}")
                return True

            # Same bar - skip recalculation for heavy features
            logger.debug(f"Same bar for {symbol_tf_key} - skipping heavy feature {feature_name}")
            return False

        elif feature_name in lightweight_features:
            # Always recalculate lightweight features
            logger.debug(f"Lightweight feature {feature_name} - always recalculating")
            return True

        # Unknown feature - default to recalculating
        logger.debug(f"Unknown feature {feature_name} - defaulting to recalculate")
        return True

    def add_conversation_message(self, user_message: str, ai_response: str, message_type: str = "strategy"):
        """Add a message to conversation memory."""
        self.conversation_memory.append({
            "timestamp": datetime.now().isoformat(),
            "user": user_message,
            "ai": ai_response,
            "type": message_type,
        })

        # Keep memory within limits
        if len(self.conversation_memory) > self.max_conversation_memory:
            self.conversation_memory = self.conversation_memory[-self.max_conversation_memory:]

    def get_context_for_deepseek(self) -> Dict[str, Any]:
        """
        Prepare comprehensive context for DeepSeek AI analysis.
        Returns optimized context that respects memory constraints.
        """
        return {
            "timestamp": datetime.now().isoformat(),
            "market_regime": self.market_regime,
            "sentiment": self.sentiment,  # Add sentiment to context
            "selected_feature_profile": self.selected_feature_profile,
            "overlay_state": self.overlay_state.copy(),  # Current active chart overlays
            "recent_overlay_changes": [
                change for change in self.overlay_history[-10:]  # Last 10 overlay changes
            ],
            "recent_profile_changes": [
                change for change in self.profile_history[-5:]  # Last 5 profile changes
            ],
            "feature_performance": {
                name: {
                    "accuracy": metrics.accuracy,
                    "trend": metrics.trend,
                    "total_signals": metrics.total_signals,
                }
                for name, metrics in self.feature_performance.items()
            },
            "feature_resource_usage": self.feature_resource_usage.copy(),
            "active_positions": {
                symbol: {
                    "side": pos.get("side"),
                    "quantity": pos.get("quantity"),
                    "unrealized_pnl": pos.get("unrealized_pnl"),
                    "entry_price": pos.get("entry_price"),
                }
                for symbol, pos in self.active_positions.items()
            },
            "risk_exposure": self.risk_metrics.get("total_exposure", 0),
            "current_drawdown": self.risk_metrics.get("current_drawdown", 0),
            "system_health": {
                "memory_usage_mb": self.system_health.get("memory_usage", 0),
                "cpu_usage_percent": self.system_health.get("cpu_usage", 0),
                "recent_errors": self.system_health.get("errors", [])[-5:],  # Last 5 errors
            },
            "recent_trades": [
                {
                    "symbol": t.symbol,
                    "pnl_percent": t.pnl_percent,
                    "exit_reason": t.exit_reason,
                    "confidence": t.confidence,
                }
                for t in self.trade_history[-10:]  # Last 10 trades
            ],
            "current_features": {
                symbol: {
                    k: v for k, v in features.items()
                    if not isinstance(v, (list, dict)) or len(str(v)) < 100
                }
                for symbol, features in self.feature_calculations.items()
            },
            "conversation_memory": [
                {
                    "timestamp": msg["timestamp"],
                    "user": msg["user"],
                    "ai": msg["ai"],
                    "type": msg.get("type", "strategy")
                }
                for msg in self.conversation_memory[-10:]  # Last 10 conversation turns
            ],
        }

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        if not self.trade_history:
            return {
                "total_trades": 0,
                "win_rate": 0,
                "total_pnl": 0,
                "sharpe_ratio": 0,
            }

        # Calculate metrics
        closed_trades = [t for t in self.trade_history if t.exit_time]
        winning_trades = [t for t in closed_trades if t.pnl > 0]
        losing_trades = [t for t in closed_trades if t.pnl < 0]

        win_rate = len(winning_trades) / len(closed_trades) if closed_trades else 0
        total_pnl = sum(t.pnl for t in closed_trades)

        # Calculate Sharpe ratio (simplified)
        if len(closed_trades) > 1:
            returns = [t.pnl_percent for t in closed_trades]
            avg_return = sum(returns) / len(returns)
            return_std = (sum((r - avg_return) ** 2 for r in returns) / len(returns)) ** 0.5
            sharpe_ratio = avg_return / return_std if return_std > 0 else 0
        else:
            sharpe_ratio = 0

        return {
            "total_trades": len(closed_trades),
            "winning_trades": len(winning_trades),
            "losing_trades": len(losing_trades),
            "win_rate": win_rate,
            "total_pnl": total_pnl,
            "avg_win": sum(t.pnl for t in winning_trades) / len(winning_trades) if winning_trades else 0,
            "avg_loss": sum(t.pnl for t in losing_trades) / len(losing_trades) if losing_trades else 0,
            "max_drawdown": self.risk_metrics.get("max_drawdown", 0),
            "sharpe_ratio": sharpe_ratio,
        }

    def get_feature_ranking(self) -> List[Dict[str, Any]]:
        """Rank features by performance."""
        return sorted(
            [
                {
                    "name": name,
                    "accuracy": metrics.accuracy,
                    "trend": metrics.trend,
                    "total_signals": metrics.total_signals,
                }
                for name, metrics in self.feature_performance.items()
            ],
            key=lambda x: x["accuracy"],
            reverse=True,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize system context to dictionary."""
        return {
            "feature_performance": {
                name: {
                    "accuracy": metrics.accuracy,
                    "trend": metrics.trend,
                    "total_signals": metrics.total_signals,
                    "successful_signals": metrics.successful_signals,
                    "last_updated": metrics.last_updated.isoformat(),
                }
                for name, metrics in self.feature_performance.items()
            },
            "active_positions": self.active_positions,
            "risk_metrics": self.risk_metrics,
            "market_regime": self.market_regime,
            "system_health": self.system_health,
            "trade_count": len(self.trade_history),
        }

    def save_state(self, filepath: str):
        """Save system state to file."""
        state = self.to_dict()
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2, default=str)

    def load_state(self, filepath: str):
        """Load system state from file."""
        try:
            with open(filepath, 'r') as f:
                state = json.load(f)

            # Restore state (simplified - in production, would need full deserialization)
            self.risk_metrics = state.get("risk_metrics", {})
            self.market_regime = state.get("market_regime", "UNKNOWN")
            print(f"✅ System state loaded from {filepath}")
        except FileNotFoundError:
            print(f"⚠️  State file {filepath} not found, starting fresh")

    # Split-brain fix: File-based synchronization for dashboard/trading engine
    STATE_FILE = "data/system_state.json"

    def save_to_disk(self):
        """
        Save critical system state to disk for split-brain synchronization.
        Atomic write pattern to prevent corruption.
        """
        # Ensure data directory exists
        os.makedirs(os.path.dirname(self.STATE_FILE), exist_ok=True)

        state = {
            "active_positions": self.active_positions,
            "risk_metrics": self.risk_metrics,
            "trade_history": [
                {
                    "symbol": t.symbol,
                    "entry_time": t.entry_time.isoformat(),
                    "exit_time": t.exit_time.isoformat(),
                    "side": t.side,
                    "entry_price": t.entry_price,
                    "exit_price": t.exit_price,
                    "quantity": t.quantity,
                    "pnl": t.pnl,
                    "pnl_percent": t.pnl_percent,
                    "confidence": t.confidence,
                    "reasoning": t.reasoning,
                    "exit_reason": t.exit_reason,
                }
                for t in self.trade_history
            ],
            "system_health": self.system_health,
            "market_regime": self.market_regime,
            "last_updated": datetime.now().isoformat(),
        }

        # Atomic write pattern to prevent corruption
        temp_file = self.STATE_FILE + ".tmp"
        with open(temp_file, 'w') as f:
            json.dump(state, f, indent=2, default=str)
        os.replace(temp_file, self.STATE_FILE)

        logger.debug(f"System state saved to {self.STATE_FILE}")

    def load_from_disk(self):
        """
        Load critical system state from disk for split-brain synchronization.
        Returns True if state was loaded, False if no state file exists.
        """
        if not os.path.exists(self.STATE_FILE):
            return False

        try:
            with open(self.STATE_FILE, 'r') as f:
                state = json.load(f)

            # Restore critical state
            self.active_positions = state.get("active_positions", {})
            self.risk_metrics = state.get("risk_metrics", {})
            self.system_health = state.get("system_health", {})
            self.market_regime = state.get("market_regime", "UNKNOWN")

            # Restore trade history
            trade_data = state.get("trade_history", [])
            self.trade_history = [
                TradeRecord(
                    symbol=t["symbol"],
                    entry_time=datetime.fromisoformat(t["entry_time"]),
                    exit_time=datetime.fromisoformat(t["exit_time"]),
                    side=t["side"],
                    entry_price=t["entry_price"],
                    exit_price=t["exit_price"],
                    quantity=t["quantity"],
                    pnl=t["pnl"],
                    pnl_percent=t["pnl_percent"],
                    confidence=t["confidence"],
                    reasoning=t["reasoning"],
                    exit_reason=t["exit_reason"],
                )
                for t in trade_data
            ]

            logger.debug(f"System state loaded from {self.STATE_FILE}")
            return True
        except Exception as e:
            logger.error(f"Failed to load system state from {self.STATE_FILE}: {e}")
            return False
