import os
import yaml
from typing import List, Dict, Any, Optional
from loguru import logger

class TradingConfig:
    """
    Centralized configuration manager for the trading system.
    Loads settings from config/system_config.yaml.
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(TradingConfig, cls).__new__(cls)
            cls._instance._load_config()
        return cls._instance

    def _load_config(self):
        """Load configuration from YAML file."""
        # Determine path relative to this file
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        config_path = os.path.join(base_dir, "config", "system_config.yaml")
        
        self.config = {}
        try:
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    self.config = yaml.safe_load(f)
                logger.info(f"Loaded configuration from {config_path}")
            else:
                logger.warning(f"Config file not found at {config_path}, using defaults")
        except Exception as e:
            logger.error(f"Error loading config: {e}")

    @property
    def symbols(self) -> List[str]:
        """Get list of trading symbols."""
        return self.config.get('trading', {}).get('symbols', ['BTCUSDT'])

    @property
    def timeframes(self) -> List[str]:
        """Get list of supported timeframes."""
        return self.config.get('trading', {}).get('timeframes', ['5m', '15m', '1h', '4h', '1d'])

    @property
    def dashboard_port(self) -> int:
        """Get dashboard port."""
        return self.config.get('dashboard', {}).get('port', 8050)

    @property
    def dashboard_debug(self) -> bool:
        """Get dashboard debug mode."""
        return self.config.get('dashboard', {}).get('debug', False)

    def get(self, key: str, default: Any = None) -> Any:
        """Get arbitrary config value using dot notation (e.g. 'trading.base_position_size')."""
        keys = key.split('.')
        value = self.config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
            else:
                return default
        return value if value is not None else default

# Global instance
config = TradingConfig()
