"""
Main Application Entry Point
DeepSeek Integrated Trading System (DITS)

This is the main application that orchestrates all components:
- Market data fetching
- Feature engineering
- DeepSeek AI analysis
- Signal generation
- Position management
- Risk management
- Dashboard and chat interface
"""

import asyncio
import os
import signal
import sys
from datetime import datetime
from loguru import logger
import yaml

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.system_context import SystemContext
from core.memory_manager import M1MemoryManager
from core.data.binance_client import BinanceClient
from core.data.data_store import DataStore
from features.engine import FeatureEngine
from deepseek.client import DeepSeekBrain
from core.signal_generator import SignalGenerator
from core.autonomous_optimizer import AutonomousOptimizer
from execution.position_manager import PositionManager
from execution.risk_manager import RiskManager
from ops.performance_monitor import PerformanceMonitor
from ui.dashboard import create_dashboard_app
from ui.chat_interface import create_chat_app


class DeepSeekTradingSystem:
    """
    Main trading system orchestrator.
    Coordinates all components and manages the trading loop.
    """

    def __init__(self, config_path: str = "config/system_config.yaml"):
        """
        Initialize the trading system.

        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path
        self.config = self._load_config()

        # Initialize components
        self.system_context = SystemContext(config_path)
        self.memory_manager = M1MemoryManager(
            self.config.get('memory', {}).get('limit_mb', 4000)
        )
        self.data_store = DataStore()
        self.binance_client = BinanceClient()
        self.deepseek_brain = DeepSeekBrain(self.system_context)
        self.feature_engine = FeatureEngine()
        self.position_manager = PositionManager(
            self.system_context,
            self.deepseek_brain,
            self.binance_client
        )
        self.risk_manager = RiskManager(self.system_context)
        self.performance_monitor = PerformanceMonitor(self.memory_manager)

        self.signal_generator = SignalGenerator(
            self.system_context,
            self.deepseek, # Changed from deepseek_brain
            self.feature_engine
        )
        
        # Initialize Autonomous Optimizer
        self.optimizer = AutonomousOptimizer(
            self.system_context,
            self.deepseek
        )

        # Trading symbols
        self.symbols = self.config.get('trading', {}).get('symbols', ['BTCUSDT'])
        self.timeframes = self.config.get('trading', {}).get('timeframes', ['5m', '15m', '1h', '4h'])

        # State
        self.running = False
        self.trading_paused = False

        logger.info("DeepSeekTradingSystem initialized")

    def _load_config(self) -> dict:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return {}

    async def start(self):
        """Start the trading system."""
        logger.info("=" * 70)
        logger.info("Starting DeepSeek Integrated Trading System")
        logger.info("=" * 70)

        try:
            # Setup signal handlers
            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGTERM, self._signal_handler)

            # Start monitoring tasks
            monitoring_tasks = [
                asyncio.create_task(self.memory_manager.monitor_memory_loop()),
                asyncio.create_task(self.performance_monitor.monitoring_loop())
            ]

            # Start autonomous optimizer
            asyncio.create_task(self.optimizer.start())

            # Start main trading loop
            trading_task = asyncio.create_task(self.trading_loop())

            # Wait for trading task to complete
            await trading_task

            # Cancel monitoring tasks
            for task in monitoring_tasks:
                task.cancel()

            logger.info("Trading system stopped")

        except Exception as e:
            logger.error(f"Error in main loop: {e}")
            raise

    async def trading_loop(self):
        """Main trading loop."""
        logger.info("Starting trading loop...")

        while self.running:
            try:
                if not self.trading_paused:
                    # Process symbols in parallel
                    tasks = [self.process_symbol(symbol) for symbol in self.symbols]
                    await asyncio.gather(*tasks)
                    
                    # Autonomous Mode Switching
                    # Check market regime from the first symbol (assuming correlated market)
                    # In a real scenario, we might aggregate regimes
                    if self.symbols:
                        market_data = await self.binance_client.get_current_price(self.symbols[0])
                        # We use a simplified check here, ideally we'd use the FeatureEngine's regime
                        # But for now, let's ask DeepSeek periodically or use the SystemContext's last regime
                        current_regime = self.system_context.market_regime
                        
                        if current_regime == "VOLATILE" and self.system_context.selected_feature_profile != "scalp":
                            logger.info("🤖 AI switching to SCALP mode due to VOLATILE regime")
                            self.system_context.update_feature_profile("scalp")
                        elif current_regime == "TRENDING" and self.system_context.selected_feature_profile != "swing":
                            logger.info("🤖 AI switching to SWING mode due to TRENDING regime")
                            self.system_context.update_feature_profile("swing")

                    # Update positions
                    await self.position_manager.update_positions()

                    # Check emergency stop
                    if self.risk_manager.emergency_stop():
                        logger.critical("Emergency stop triggered!")
                        self.trading_paused = True

                # Wait before next iteration
                await asyncio.sleep(60)  # Check every minute

            except Exception as e:
                logger.error(f"Error in trading loop: {e}")
                await asyncio.sleep(10)

    async def process_symbol(self, symbol: str):
        """
        Process a single symbol through the trading pipeline.

        Args:
            symbol: Trading symbol
        """
        try:
            logger.debug(f"Processing {symbol}")

            # Check if we can open new positions
            if not self.position_manager.can_open_new_position():
                return

            # Fetch market data for multiple timeframes
            timeframe_data = await self.binance_client.get_multiple_timeframes(
                symbol, self.timeframes
            )

            # Use 1h data as primary
            primary_data = timeframe_data.get('1h', None)
            if primary_data is None or len(primary_data) < 50:
                logger.warning(f"Insufficient data for {symbol}")
                return

            # Convert to dict format
            market_data = {
                'timestamp': primary_data['timestamp'],
                'open': primary_data['open'],
                'high': primary_data['high'],
                'low': primary_data['low'],
                'close': primary_data['close'],
                'volume': primary_data['volume']
            }

            # Generate signal
            signal = await self.signal_generator.generate_signal(
                symbol, market_data, timeframe_data
            )

            # Check if should trade
            if self.signal_generator.should_trade(signal):
                # Validate trade
                validation = self.risk_manager.validate_trade(
                    signal, signal['position_size']
                )

                if validation['approved']:
                    # Get position evaluation
                    evaluation = await self.position_manager.evaluate_signal(
                        signal,
                        self.system_context.risk_metrics.get('total_exposure', 0)
                    )

                    if evaluation['approved']:
                        # Place order
                        order_result = await self.position_manager.place_order(
                            signal,
                            validation['adjusted_size']
                        )

                        if order_result['success']:
                            logger.info(f"Trade executed: {symbol} {signal['action']}")

            # Cache data
            await self.data_store.store_ohlcv(symbol, '1h', primary_data)
            await self.data_store.store_price(symbol, primary_data['close'].iloc[-1])

        except Exception as e:
            logger.error(f"Error processing {symbol}: {e}")
            self.performance_monitor.log_error(f"Error processing {symbol}: {e}")

    def start_dashboard(self, port: int = 8050, debug: bool = False):
        """
        Start the web dashboard.

        Args:
            port: Dashboard port
            debug: Enable debug mode
        """
        logger.info(f"Starting dashboard on port {port}...")

        # Create Dash app with multiple pages
        app = create_dashboard_app()

        # Note: In a full implementation, you would:
        # 1. Register all pages
        # 2. Connect callbacks to real data
        # 3. Start the server

        logger.info(f"Dashboard ready at http://localhost:{port}")
        # app.run_server(host='0.0.0.0', port=port, debug=debug)

    def start_chat(self, port: int = 8051, debug: bool = False):
        """
        Start the chat interface.

        Args:
            port: Chat port
            debug: Enable debug mode
        """
        logger.info(f"Starting chat interface on port {port}...")

        # Create Dash app for chat
        app = create_chat_app()

        logger.info(f"Chat interface ready at http://localhost:{port}")
        # app.run_server(host='0.0.0.0', port=port, debug=debug)

    def pause_trading(self):
        """Pause trading operations."""
        self.trading_paused = True
        logger.warning("Trading paused")

    def resume_trading(self):
        """Resume trading operations."""
        self.trading_paused = False
        logger.info("Trading resumed")

    def stop(self):
        """Stop the trading system."""
        logger.info("Stopping trading system...")
        self.running = False

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        logger.info(f"Received signal {signum}, shutting down...")
        self.stop()

    def get_status(self) -> dict:
        """
        Get current system status.

        Returns:
            Status dictionary
        """
        return {
            'running': self.running,
            'trading_paused': self.trading_paused,
            'symbols': self.symbols,
            'active_positions': len(self.system_context.active_positions),
            'memory_usage_mb': self.memory_manager.get_current_memory_usage()['rss_mb'],
            'performance_report': self.performance_monitor.get_performance_report()
        }


async def main():
    """Main entry point."""
    # Initialize system
    system = DeepSeekTradingSystem()

    # Start the system
    system.running = True
    await system.start()


if __name__ == "__main__":
    try:
        # Configure logging
        logger.add(
            "logs/trading_system_{time}.log",
            rotation="1 day",
            retention="7 days",
            level="INFO",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}"
        )

        # Create logs directory
        os.makedirs("logs", exist_ok=True)

        # Run main
        asyncio.run(main())

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.critical(f"Fatal error: {e}")
        sys.exit(1)
