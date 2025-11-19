"""
Experiment Orchestrator
Automates config sweeps and runs backtests with logging.
"""

from typing import Iterable, List, Dict, Any, Optional, Tuple
from datetime import datetime
from loguru import logger
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from backtesting.service import BacktestConfig, BacktestResult, run_backtest
from research.experiment_store import ExperimentStore
from research.config_generators import generate_configs


class ExperimentRunner:
    """
    Orchestrates experiment runs with automatic logging.

    Features:
    - Grid search and random sampling of configurations
    - Parallel execution of backtests
    - Automatic logging to SQLite database
    - Duplicate detection and skipping
    - Progress tracking and reporting
    """

    def __init__(
        self,
        db_path: str = "data/experiments.sqlite",
        max_workers: int = 4,
        enable_logging: bool = True
    ):
        """
        Initialize experiment runner.

        Args:
            db_path: Path to SQLite database for logging results
            max_workers: Maximum number of parallel backtest workers
            enable_logging: Whether to log results to database
        """
        self.max_workers = max_workers
        self.enable_logging = enable_logging

        if enable_logging:
            self.store = ExperimentStore(db_path)
        else:
            self.store = None

        logger.info(f"ExperimentRunner initialized: workers={max_workers}, logging={enable_logging}")

    def run(
        self,
        configs: Iterable[BacktestConfig],
        search_type: str = "grid",
        search_params: Optional[Dict[str, Any]] = None,
        force_rerun: bool = False
    ) -> List[Tuple[BacktestConfig, Optional[BacktestResult], Optional[str]]]:
        """
        Run experiments for a collection of configurations.

        Args:
            configs: Iterable of BacktestConfig objects
            search_type: Type of search ('grid', 'random', 'bayesian')
            search_params: Additional parameters for the search
            force_rerun: Whether to rerun experiments even if they exist

        Returns:
            List of tuples (config, result or None, error message or None)
        """
        configs = list(configs)
        total_configs = len(configs)

        logger.info(f"Starting experiment run: {total_configs} configs, search={search_type}")

        if self.enable_logging and not force_rerun:
            # Filter out configs that have already been run
            original_count = len(configs)
            configs = [c for c in configs if not self.store.config_exists(c.get_config_hash())]
            skipped = original_count - len(configs)

            if skipped > 0:
                logger.info(f"Skipped {skipped} duplicate configurations")

        if not configs:
            logger.warning("No configurations to run")
            return []

        start_time = time.time()
        results = []

        # Run with parallel execution
        if self.max_workers > 1:
            results = self._run_parallel(configs)
        else:
            results = self._run_sequential(configs)

        elapsed_time = time.time() - start_time
        successful = sum(1 for _, r, _ in results if r is not None)
        failed = total_configs - successful

        logger.info(
            f"Experiment run complete: {successful}/{total_configs} successful, "
            f"{failed} failed in {elapsed_time:.1f}s"
        )

        return results

    def _run_sequential(
        self,
        configs: List[BacktestConfig]
    ) -> List[Tuple[BacktestConfig, Optional[BacktestResult], Optional[str]]]:
        """Run experiments sequentially."""
        results = []

        for i, config in enumerate(configs, 1):
            logger.info(f"Running experiment {i}/{len(configs)}: {config.symbol} {config.strategy}")
            result, error = self._run_single(config)
            results.append((config, result, error))

        return results

    def _run_parallel(
        self,
        configs: List[BacktestConfig]
    ) -> List[Tuple[BacktestConfig, Optional[BacktestResult], Optional[str]]]:
        """Run experiments in parallel using ThreadPoolExecutor."""
        results = [None] * len(configs)
        futures = {}

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            for i, config in enumerate(configs):
                future = executor.submit(self._run_single_with_index, config, i)
                futures[future] = i

            # Collect results as they complete
            for future in as_completed(futures):
                i = futures[future]
                try:
                    result, error = future.result()
                    results[i] = (configs[i], result, error)

                    # Log progress
                    completed = sum(1 for r in results if r is not None)
                    logger.info(f"Progress: {completed}/{len(configs)} completed")

                except Exception as e:
                    logger.error(f"Error in parallel execution: {e}")
                    results[i] = (configs[i], None, str(e))

        return results

    def _run_single_with_index(
        self,
        config: BacktestConfig,
        index: int
    ) -> Tuple[Optional[BacktestResult], Optional[str]]:
        """Run single experiment with index (for parallel execution)."""
        return self._run_single(config)

    def _run_single(
        self,
        config: BacktestConfig
    ) -> Tuple[Optional[BacktestResult], Optional[str]]:
        """Run single experiment."""
        try:
            # Run backtest
            result = run_backtest(config)

            # Log to database if enabled
            if self.enable_logging and self.store:
                self.store.log_experiment(config, result)

            return result, None

        except Exception as e:
            error_msg = str(e)
            logger.error(f"Experiment failed: {error_msg}")

            # Log error to database if enabled
            if self.enable_logging and self.store:
                self.store.log_error(config, error_msg)

            return None, error_msg

    def grid_search(
        self,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: datetime,
        strategies: List[str],
        param_grids: Dict[str, Dict[str, List[Any]]],
        initial_capital: float = 10000.0,
        **kwargs
    ) -> List[Tuple[BacktestConfig, Optional[BacktestResult], Optional[str]]]:
        """
        Run grid search over parameter space.

        Args:
            symbol: Trading symbol
            timeframe: Data timeframe
            start: Start date
            end: End date
            strategies: List of strategy names
            param_grids: Dict mapping strategy to parameter grid
            initial_capital: Initial capital for backtests
            **kwargs: Additional arguments for run()

        Returns:
            List of experiment results
        """
        logger.info(f"Starting grid search: {len(strategies)} strategies")

        # Generate configurations
        configs = generate_configs(
            symbol=symbol,
            timeframe=timeframe,
            start=start,
            end=end,
            strategies=strategies,
            param_grids=param_grids,
            initial_capital=initial_capital,
            search_type="grid"
        )

        logger.info(f"Generated {len(configs)} configurations for grid search")

        # Run experiments
        return self.run(configs, search_type="grid", **kwargs)

    def random_search(
        self,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: datetime,
        strategies: List[str],
        param_spaces: Dict[str, Dict[str, Any]],
        n_samples: int,
        initial_capital: float = 10000.0,
        **kwargs
    ) -> List[Tuple[BacktestConfig, Optional[BacktestResult], Optional[str]]]:
        """
        Run random search over parameter space.

        Args:
            symbol: Trading symbol
            timeframe: Data timeframe
            start: Start date
            end: End date
            strategies: List of strategy names
            param_spaces: Dict mapping strategy to parameter space
            n_samples: Number of random samples to generate
            initial_capital: Initial capital for backtests
            **kwargs: Additional arguments for run()

        Returns:
            List of experiment results
        """
        logger.info(f"Starting random search: {n_samples} samples")

        # Generate configurations
        configs = generate_configs(
            symbol=symbol,
            timeframe=timeframe,
            start=start,
            end=end,
            strategies=strategies,
            param_grids=param_spaces,  # Same format, just interpreted differently
            n_samples=n_samples,
            initial_capital=initial_capital,
            search_type="random"
        )

        logger.info(f"Generated {len(configs)} configurations for random search")

        # Run experiments
        return self.run(configs, search_type="random", **kwargs)

    def get_results(
        self,
        metric: str = "sharpe_ratio",
        top_n: Optional[int] = None,
        filter criteria: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve experiment results from database.

        Args:
            metric: Metric to sort by
            top_n: Number of top results to return
            filter_criteria: Optional filters (symbol, strategy, etc.)

        Returns:
            List of result dictionaries
        """
        if not self.enable_logging or not self.store:
            logger.warning("Logging not enabled, cannot retrieve results")
            return []

        return self.store.get_experiments(
            metric=metric,
            top_n=top_n,
            filter_criteria=filter_criteria
        )

    def get_best_config(self, metric: str = "sharpe_ratio") -> Optional[Dict[str, Any]]:
        """
        Get the best performing configuration.

        Args:
            metric: Metric to optimize

        Returns:
            Best configuration dict or None
        """
        if not self.enable_logging or not self.store:
            logger.warning("Logging not enabled, cannot retrieve best config")
            return None

        results = self.store.get_experiments(metric=metric, top_n=1)
        return results[0] if results else None

    def get_experiment_summary(self) -> Dict[str, Any]:
        """Get summary of all experiments."""
        if not self.enable_logging or not self.store:
            return {"error": "Logging not enabled"}

        return self.store.get_summary()


class ExperimentBatch:
    """
    Helper class for running batches of experiments.
    """

    def __init__(self, runner: ExperimentRunner):
        self.runner = runner
        self.configs = []

    def add_config(self, config: BacktestConfig):
        """Add a configuration to the batch."""
        self.configs.append(config)

    def add_grid(
        self,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: datetime,
        strategies: List[str],
        param_grids: Dict[str, Dict[str, List[Any]]],
        initial_capital: float = 10000.0
    ):
        """Add a grid search to the batch."""
        configs = generate_configs(
            symbol=symbol,
            timeframe=timeframe,
            start=start,
            end=end,
            strategies=strategies,
            param_grids=param_grids,
            initial_capital=initial_capital,
            search_type="grid"
        )
        self.configs.extend(configs)

    def run(self, **kwargs) -> List[Tuple[BacktestConfig, Optional[BacktestResult], Optional[str]]]:
        """Run all configurations in the batch."""
        if not self.configs:
            logger.warning("No configurations in batch")
            return []

        logger.info(f"Running batch with {len(self.configs)} configurations")
        return self.runner.run(self.configs, **kwargs)


# Helper function for easy experiment setup
def create_experiment_runner(
    db_path: str = "data/experiments.sqlite",
    max_workers: int = 4
) -> ExperimentRunner:
    """Create and return an ExperimentRunner with sensible defaults."""
    return ExperimentRunner(
        db_path=db_path,
        max_workers=max_workers,
        enable_logging=True
    )
