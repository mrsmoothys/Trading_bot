"""
Configuration Generators
Generate backtest configurations for grid search and random sampling.
"""

from typing import Dict, List, Any, Iterable, Optional, Tuple
from datetime import datetime
import itertools
import random
import numpy as np
from loguru import logger

from backtesting.service import BacktestConfig


def generate_configs(
    symbol: str,
    timeframe: str,
    start: datetime,
    end: datetime,
    strategies: List[str],
    param_grids: Dict[str, Dict[str, List[Any]]],
    initial_capital: float = 10000.0,
    search_type: str = "grid",
    n_samples: Optional[int] = None,
    commission_rate: float = 0.001,
    slippage: float = 0.0005
) -> List[BacktestConfig]:
    """
    Generate BacktestConfig objects for different search strategies.

    Args:
        symbol: Trading symbol
        timeframe: Data timeframe
        start: Start datetime
        end: End datetime
        strategies: List of strategy names
        param_grids: Dict mapping strategy to parameter grids
        initial_capital: Initial capital for backtests
        search_type: Type of search ('grid', 'random')
        n_samples: Number of samples for random search
        commission_rate: Commission rate
        slippage: Slippage

    Returns:
        List of BacktestConfig objects
    """
    configs = []

    for strategy in strategies:
        if strategy not in param_grids:
            logger.warning(f"Strategy {strategy} not in param_grids, skipping")
            continue

        strategy_params = param_grids[strategy]

        if search_type == "grid":
            strategy_configs = _generate_grid_configs(
                symbol=symbol,
                timeframe=timeframe,
                start=start,
                end=end,
                strategy=strategy,
                param_grid=strategy_params,
                initial_capital=initial_capital,
                commission_rate=commission_rate,
                slippage=slippage
            )
        elif search_type == "random":
            if n_samples is None:
                raise ValueError("n_samples required for random search")

            strategy_configs = _generate_random_configs(
                symbol=symbol,
                timeframe=timeframe,
                start=start,
                end=end,
                strategy=strategy,
                param_space=strategy_params,
                n_samples=n_samples,
                initial_capital=initial_capital,
                commission_rate=commission_rate,
                slippage=slippage
            )
        else:
            raise ValueError(f"Unknown search_type: {search_type}")

        configs.extend(strategy_configs)
        logger.info(f"Generated {len(strategy_configs)} configs for {strategy}")

    logger.info(f"Total generated configurations: {len(configs)}")
    return configs


def _generate_grid_configs(
    symbol: str,
    timeframe: str,
    start: datetime,
    end: datetime,
    strategy: str,
    param_grid: Dict[str, List[Any]],
    initial_capital: float,
    commission_rate: float,
    slippage: float
) -> List[BacktestConfig]:
    """Generate configs using grid search."""
    # Get all parameter combinations
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())

    configs = []

    for combination in itertools.product(*param_values):
        params = dict(zip(param_names, combination))

        config = BacktestConfig(
            symbol=symbol,
            timeframe=timeframe,
            start=start,
            end=end,
            strategy=strategy,
            params=params,
            initial_capital=initial_capital,
            commission_rate=commission_rate,
            slippage=slippage
        )

        configs.append(config)

    return configs


def _generate_random_configs(
    symbol: str,
    timeframe: str,
    start: datetime,
    end: datetime,
    strategy: str,
    param_space: Dict[str, Any],
    n_samples: int,
    initial_capital: float,
    commission_rate: float,
    slippage: float
) -> List[BacktestConfig]:
    """Generate configs using random sampling."""
    configs = []

    for _ in range(n_samples):
        params = {}

        for param_name, param_config in param_space.items():
            param_type = param_config.get('type', 'int')
            param_value = None

            if param_type == 'int':
                min_val = param_config.get('min', 1)
                max_val = param_config.get('max', 100)
                param_value = random.randint(min_val, max_val)

            elif param_type == 'float':
                min_val = param_config.get('min', 0.0)
                max_val = param_config.get('max', 1.0)
                param_value = random.uniform(min_val, max_val)

            elif param_type == 'choice':
                choices = param_config.get('choices', [1, 2, 3])
                param_value = random.choice(choices)

            elif param_type == 'loguniform':
                # Log-uniform distribution for parameters spanning multiple orders of magnitude
                min_val = param_config.get('min', 0.001)
                max_val = param_config.get('max', 1.0)
                # Sample from log space
                log_min = np.log10(min_val)
                log_max = np.log10(max_val)
                param_value = 10 ** random.uniform(log_min, log_max)

            else:
                raise ValueError(f"Unknown parameter type: {param_type}")

            params[param_name] = param_value

        config = BacktestConfig(
            symbol=symbol,
            timeframe=timeframe,
            start=start,
            end=end,
            strategy=strategy,
            params=params,
            initial_capital=initial_capital,
            commission_rate=commission_rate,
            slippage=slippage
        )

        configs.append(config)

    return configs


# Predefined parameter grids for common strategies
SMA_PARAM_GRID = {
    'fast_period': [5, 10, 20, 30],
    'slow_period': [50, 100, 150, 200]
}

RSI_PARAM_GRID = {
    'period': [10, 14, 21, 30],
    'oversold': [20, 25, 30, 35],
    'overbought': [65, 70, 75, 80]
}

MACD_PARAM_GRID = {
    'fast_period': [8, 12, 16],
    'slow_period': [21, 26, 30],
    'signal_period': [6, 9, 12]
}

CONVERGENCE_PARAM_GRID = {
    # Convergence strategy typically doesn't need parameter tuning
    # but we can add thresholds or weights if needed
}


# Predefined random parameter spaces for random search
SMA_PARAM_SPACE = {
    'fast_period': {'type': 'int', 'min': 5, 'max': 50},
    'slow_period': {'type': 'int', 'min': 20, 'max': 200}
}

RSI_PARAM_SPACE = {
    'period': {'type': 'int', 'min': 5, 'max': 30},
    'oversold': {'type': 'int', 'min': 10, 'max': 40},
    'overbought': {'type': 'int', 'min': 60, 'max': 90}
}

MACD_PARAM_SPACE = {
    'fast_period': {'type': 'int', 'min': 5, 'max': 20},
    'slow_period': {'type': 'int', 'min': 15, 'max': 40},
    'signal_period': {'type': 'int', 'min': 5, 'max': 15}
}

CONVERGENCE_PARAM_SPACE = {
    # Convergence strategy
}


def get_strategy_param_grid(strategy: str) -> Dict[str, List[Any]]:
    """Get predefined grid for a strategy."""
    grids = {
        'sma': SMA_PARAM_GRID,
        'rsi': RSI_PARAM_GRID,
        'macd': MACD_PARAM_GRID,
        'convergence': CONVERGENCE_PARAM_GRID
    }

    return grids.get(strategy, {})


def get_strategy_param_space(strategy: str) -> Dict[str, Any]:
    """Get predefined parameter space for a strategy."""
    spaces = {
        'sma': SMA_PARAM_SPACE,
        'rsi': RSI_PARAM_SPACE,
        'macd': MACD_PARAM_SPACE,
        'convergence': CONVERGENCE_PARAM_SPACE
    }

    return spaces.get(strategy, {})


def create_custom_param_grid(
    strategy: str,
    custom_params: Dict[str, List[Any]]
) -> Dict[str, List[Any]]:
    """
    Create a custom parameter grid by extending the base grid.

    Args:
        strategy: Strategy name
        custom_params: Custom parameters to add/override

    Returns:
        Combined parameter grid
    """
    base_grid = get_strategy_param_grid(strategy)

    # Create a copy and update with custom params
    combined = base_grid.copy()
    combined.update(custom_params)

    return combined


def create_custom_param_space(
    strategy: str,
    custom_params: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Create a custom parameter space by extending the base space.

    Args:
        strategy: Strategy name
        custom_params: Custom parameters to add/override

    Returns:
        Combined parameter space
    """
    base_space = get_strategy_param_space(strategy)

    # Create a copy and update with custom params
    combined = base_space.copy()
    combined.update(custom_params)

    return combined


def estimate_grid_size(param_grids: Dict[str, Dict[str, List[Any]]]) -> int:
    """
    Estimate the total number of configurations for a grid.

    Args:
        param_grids: Dict mapping strategy to parameter grids

    Returns:
        Estimated total number of configurations
    """
    total = 0

    for strategy, param_grid in param_grids.items():
        strategy_total = 1
        for param_name, param_values in param_grid.items():
            strategy_total *= len(param_values)
        total += strategy_total

    return total


def validate_param_grid(strategy: str, param_grid: Dict[str, List[Any]]) -> Tuple[bool, List[str]]:
    """
    Validate a parameter grid for a strategy.

    Args:
        strategy: Strategy name
        param_grid: Parameter grid to validate

    Returns:
        Tuple of (is_valid, error_messages)
    """
    errors = []

    # Check if strategy is known
    known_strategies = ['sma', 'rsi', 'macd', 'convergence']
    if strategy not in known_strategies:
        errors.append(f"Unknown strategy: {strategy}")

    # Check parameter structure
    for param_name, param_values in param_grid.items():
        if not isinstance(param_values, list):
            errors.append(f"Parameter {param_name} must be a list")
        elif len(param_values) == 0:
            errors.append(f"Parameter {param_name} cannot be empty")

        # Check for valid values based on parameter type
        for value in param_values:
            if not isinstance(value, (int, float)):
                errors.append(f"Parameter {param_name} contains non-numeric value: {value}")

    return len(errors) == 0, errors


def validate_param_space(strategy: str, param_space: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validate a parameter space for a strategy.

    Args:
        strategy: Strategy name
        param_space: Parameter space to validate

    Returns:
        Tuple of (is_valid, error_messages)
    """
    errors = []

    known_strategies = ['sma', 'rsi', 'macd', 'convergence']
    if strategy not in known_strategies:
        errors.append(f"Unknown strategy: {strategy}")

    for param_name, param_config in param_space.items():
        if not isinstance(param_config, dict):
            errors.append(f"Parameter {param_name} must be a dict")
            continue

        param_type = param_config.get('type')
        if param_type not in ['int', 'float', 'choice', 'loguniform']:
            errors.append(f"Parameter {param_name} has unknown type: {param_type}")

        if param_type == 'choice':
            if 'choices' not in param_config:
                errors.append(f"Parameter {param_name} missing 'choices'")
            elif not isinstance(param_config['choices'], list):
                errors.append(f"Parameter {param_name} 'choices' must be a list")

        elif param_type in ['int', 'float', 'loguniform']:
            if 'min' not in param_config or 'max' not in param_config:
                errors.append(f"Parameter {param_name} missing 'min' or 'max'")
            elif param_config['min'] >= param_config['max']:
                errors.append(f"Parameter {param_name} min must be < max")

    return len(errors) == 0, errors
