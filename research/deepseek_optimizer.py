"""
DeepSeek Optimization Loop
Let DeepSeek AI propose new configurations based on experiment history.
"""

from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import json
import asyncio
from loguru import logger

from backtesting.service import BacktestConfig, BacktestResult
from research.experiment_store import ExperimentStore


class DeepSeekOptimizer:
    """
    AI-powered configuration optimizer using DeepSeek.

    Features:
    - Analyzes experiment history
    - Proposes new configurations
    - Validates AI suggestions
    - Iterative optimization loop
    """

    def __init__(
        self,
        experiment_store: ExperimentStore,
        deepseek_client: Optional[Any] = None
    ):
        """
        Initialize optimizer.

        Args:
            experiment_store: Store with experiment history
            deepseek_client: Optional DeepSeek client instance
        """
        self.store = experiment_store
        self.deepseek_client = deepseek_client
        self.optimization_history = []

        logger.info("DeepSeekOptimizer initialized")

    async def propose_config(
        self,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: datetime,
        strategy: str,
        target_metric: str = "sharpe_ratio",
        num_experiments: int = 10,
        constraints: Optional[Dict[str, Any]] = None
    ) -> Optional[BacktestConfig]:
        """
        Propose a new configuration based on experiment history.

        Args:
            symbol: Trading symbol
            timeframe: Data timeframe
            start: Start datetime
            end: End datetime
            strategy: Strategy to optimize
            target_metric: Metric to optimize
            num_experiments: Number of recent experiments to analyze
            constraints: Optimization constraints

        Returns:
            BacktestConfig or None if proposal fails
        """
        logger.info(f"Generating proposal for {strategy} on {symbol}")

        # Get experiment history
        history = self.store.get_experiments(
            filter_criteria={'symbol': symbol, 'strategy': strategy},
            metric=target_metric,
            top_n=num_experiments
        )

        if not history:
            logger.warning(f"No experiment history for {strategy} on {symbol}")
            return await self._propose_initial_config(
                symbol, timeframe, start, end, strategy, constraints
            )

        # Build prompt
        prompt = self._build_optimization_prompt(
            history=history,
            symbol=symbol,
            timeframe=timeframe,
            strategy=strategy,
            target_metric=target_metric,
            constraints=constraints
        )

        # Ask DeepSeek
        try:
            if self.deepseek_client:
                response = await self.deepseek_client.chat(prompt, "optimization")
            else:
                response = self._generate_demo_proposal(history, constraints)

            # Parse and validate
            config = self._parse_proposal(response, symbol, timeframe, start, end, strategy)

            if config:
                logger.info(f"Generated proposal: {config.params}")
                return config
            else:
                logger.error("Failed to parse AI proposal")
                return None

        except Exception as e:
            logger.error(f"Error generating proposal: {e}")
            return None

    async def _propose_initial_config(
        self,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: datetime,
        strategy: str,
        constraints: Optional[Dict[str, Any]] = None
    ) -> Optional[BacktestConfig]:
        """Propose initial configuration when no history exists."""
        logger.info(f"No history for {strategy}, proposing initial config")

        # Use default parameters based on strategy
        default_params = self._get_default_params(strategy)

        return BacktestConfig(
            symbol=symbol,
            timeframe=timeframe,
            start=start,
            end=end,
            strategy=strategy,
            params=default_params
        )

    def _build_optimization_prompt(
        self,
        history: List[Dict[str, Any]],
        symbol: str,
        timeframe: str,
        strategy: str,
        target_metric: str,
        constraints: Optional[Dict[str, Any]] = None
    ) -> str:
        """Build prompt for DeepSeek with experiment history."""
        # Get top 5 experiments
        top_experiments = history[:5]
        worst_experiments = history[-5:] if len(history) >= 5 else []

        # Build experiment summary
        summary_lines = []
        summary_lines.append(f"Optimizing {strategy} strategy for {symbol} on {timeframe} timeframe")
        summary_lines.append(f"Target metric: {target_metric}\n")

        summary_lines.append("TOP PERFORMING EXPERIMENTS:")
        for exp in top_experiments:
            summary_lines.append(
                f"  - {exp['params']}: {target_metric}={exp[target_metric]:.4f}, "
                f"return={exp['total_return_pct']:.2f}%, "
                f"drawdown={exp['max_drawdown']:.2f}%"
            )

        summary_lines.append("\nPOOR PERFORMING EXPERIMENTS:")
        for exp in worst_experiments:
            summary_lines.append(
                f"  - {exp['params']}: {target_metric}={exp[target_metric]:.4f}, "
                f"return={exp['total_return_pct']:.2f}%, "
                f"drawdown={exp['max_drawdown']:.2f}%"
            )

        # Add constraints
        constraints_text = ""
        if constraints:
            constraints_text = "\nCONSTRAINTS:\n"
            for key, value in constraints.items():
                constraints_text += f"  - {key}: {value}\n"

        # Build full prompt
        prompt = f"""
{''.join(summary_lines)}

{constraints_text}

Based on this data, analyze what parameter combinations work well and what doesn't.
Then propose a new configuration that is likely to improve the {target_metric}.

Your response MUST be a valid JSON object with the following structure:
{{
    "rationale": "Brief explanation of your reasoning",
    "params": {{
        "parameter_name": value,
        ...
    }},
    "expected_improvement": "Brief description of expected improvement"
}}

For {strategy} strategy, use these parameter ranges:
{self._get_parameter_hints(strategy)}
        """.strip()

        return prompt

    def _parse_proposal(
        self,
        response: str,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: datetime,
        strategy: str
    ) -> Optional[BacktestConfig]:
        """Parse and validate DeepSeek's proposal."""
        try:
            # Extract JSON from response
            # Handle both direct JSON and wrapped responses
            json_start = response.find('{')
            json_end = response.rfind('}') + 1

            if json_start == -1 or json_end == 0:
                logger.error("No JSON found in response")
                return None

            json_str = response[json_start:json_end]
            proposal = json.loads(json_str)

            # Validate structure
            if 'params' not in proposal:
                logger.error("Proposal missing 'params' field")
                return None

            params = proposal['params']

            # Validate parameters for strategy
            if not self._validate_params(strategy, params):
                logger.error("Invalid parameters for strategy")
                return None

            # Create config
            config = BacktestConfig(
                symbol=symbol,
                timeframe=timeframe,
                start=start,
                end=end,
                strategy=strategy,
                params=params
            )

            # Store optimization attempt
            self.optimization_history.append({
                'timestamp': datetime.now(),
                'proposal': proposal,
                'config_hash': config.get_config_hash()
            })

            logger.info(f"Parsed proposal: {proposal.get('rationale', 'No rationale')}")
            return config

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON from response: {e}")
            logger.error(f"Response: {response}")
            return None
        except Exception as e:
            logger.error(f"Error parsing proposal: {e}")
            return None

    def _validate_params(self, strategy: str, params: Dict[str, Any]) -> bool:
        """Validate parameters for a strategy."""
        if strategy == 'sma':
            return ('fast_period' in params and 'slow_period' in params and
                    params['fast_period'] < params['slow_period'])

        elif strategy == 'rsi':
            return ('period' in params and
                    'oversold' in params and
                    'overbought' in params and
                    params['oversold'] < params['overbought'])

        elif strategy == 'macd':
            return ('fast_period' in params and
                    'slow_period' in params and
                    'signal_period' in params and
                    params['fast_period'] < params['slow_period'])

        elif strategy == 'convergence':
            return True  # No parameters required

        return False

    def _get_parameter_hints(self, strategy: str) -> str:
        """Get parameter hints for DeepSeek."""
        hints = {
            'sma': """
  - fast_period: 5-50 (period for fast moving average)
  - slow_period: 20-200 (period for slow moving average, must be > fast_period)
            """,
            'rsi': """
  - period: 5-30 (RSI calculation period)
  - oversold: 10-40 (oversold threshold, must be < overbought)
  - overbought: 60-90 (overbought threshold, must be > oversold)
            """,
            'macd': """
  - fast_period: 5-20 (fast EMA period)
  - slow_period: 15-40 (slow EMA period, must be > fast_period)
  - signal_period: 5-15 (signal line period)
            """,
            'convergence': """
  - No parameters required for convergence strategy
            """
        }

        return hints.get(strategy, "")

    def _get_default_params(self, strategy: str) -> Dict[str, Any]:
        """Get default parameters for a strategy."""
        defaults = {
            'sma': {'fast_period': 20, 'slow_period': 50},
            'rsi': {'period': 14, 'oversold': 30, 'overbought': 70},
            'macd': {'fast_period': 12, 'slow_period': 26, 'signal_period': 9},
            'convergence': {}
        }

        return defaults.get(strategy, {})

    def _generate_demo_proposal(
        self,
        history: List[Dict[str, Any]],
        constraints: Optional[Dict[str, Any]] = None
    ) -> str:
        """Generate a demo proposal when DeepSeek is not available."""
        # Find best performing experiment
        if not history:
            return json.dumps({
                "rationale": "No history available",
                "params": {},
                "expected_improvement": "Baseline configuration"
            })

        best = history[0]
        current_best_sharpe = best.get('sharpe_ratio', 0)

        # Make a small adjustment to the best config
        params = best.get('params', {}).copy()

        # Simple heuristic: if return is good but drawdown is high, reduce parameters
        # If return is low, increase them
        if params.get('fast_period'):
            if best.get('max_drawdown', 0) > 0.1:
                params['fast_period'] = max(5, int(params['fast_period'] * 0.9))
            else:
                params['fast_period'] = min(50, int(params['fast_period'] * 1.1))

        if params.get('slow_period'):
            if best.get('max_drawdown', 0) > 0.1:
                params['slow_period'] = max(params.get('fast_period', 20) + 1, int(params['slow_period'] * 0.9))
            else:
                params['slow_period'] = min(200, int(params['slow_period'] * 1.1))

        return json.dumps({
            "rationale": f"Adjusted parameters based on best experiment (Sharpe: {current_best_sharpe:.3f})",
            "params": params,
            "expected_improvement": "Reduced drawdown while maintaining returns"
        })


class OptimizationLoop:
    """
    Continuous optimization loop using DeepSeek + backtesting.
    """

    def __init__(
        self,
        optimizer: DeepSeekOptimizer,
        experiment_runner: Any,
        max_iterations: int = 10,
        convergence_threshold: Optional[float] = None
    ):
        """
        Initialize optimization loop.

        Args:
            optimizer: DeepSeekOptimizer instance
            experiment_runner: ExperimentRunner instance
            max_iterations: Maximum number of iterations
            convergence_threshold: Stop when improvement is below this threshold
        """
        self.optimizer = optimizer
        self.runner = experiment_runner
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        self.history = []

    async def run(
        self,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: datetime,
        strategy: str,
        target_metric: str = "sharpe_ratio",
        constraints: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[BacktestConfig, Optional[BacktestResult], Optional[str]]]:
        """
        Run optimization loop.

        Args:
            symbol: Trading symbol
            timeframe: Data timeframe
            start: Start datetime
            end: End datetime
            strategy: Strategy to optimize
            target_metric: Metric to optimize
            constraints: Optimization constraints

        Returns:
            List of optimization attempts with results
        """
        logger.info(
            f"Starting optimization loop: {strategy} on {symbol}, "
            f"target={target_metric}, max_iter={self.max_iterations}"
        )

        results = []
        best_metric_value = float('-inf')

        for i in range(self.max_iterations):
            logger.info(f"Optimization iteration {i+1}/{self.max_iterations}")

            # Propose new config
            config = await self.optimizer.propose_config(
                symbol=symbol,
                timeframe=timeframe,
                start=start,
                end=end,
                strategy=strategy,
                target_metric=target_metric,
                num_experiments=min(10 + i * 5, 50),
                constraints=constraints
            )

            if not config:
                logger.error("Failed to generate proposal, stopping")
                break

            # Run backtest
            try:
                from backtesting.service import run_backtest
                result = run_backtest(config)
                error = None
            except Exception as e:
                result = None
                error = str(e)

            results.append((config, result, error))

            # Log result
            if result:
                metric_value = getattr(result, target_metric, 0)
                logger.info(
                    f"Iteration {i+1}: {target_metric}={metric_value:.4f}, "
                    f"return={result.total_return_pct:.2f}%"
                )

                # Check convergence
                if self.convergence_threshold and i > 0:
                    improvement = metric_value - best_metric_value
                    if improvement < self.convergence_threshold:
                        logger.info(f"Converged: improvement={improvement:.4f} < threshold")
                        break

                # Update best
                if metric_value > best_metric_value:
                    best_metric_value = metric_value
            else:
                logger.error(f"Iteration {i+1} failed: {error}")

            # Add small delay to avoid overwhelming the system
            await asyncio.sleep(0.1)

        logger.info(f"Optimization complete: {len(results)} iterations")
        return results

    def save_optimization_log(self, filepath: str):
        """Save optimization log to JSON file."""
        log_data = {
            'timestamp': datetime.now().isoformat(),
            'max_iterations': self.max_iterations,
            'history': self.history
        }

        with open(filepath, 'w') as f:
            json.dump(log_data, f, indent=2, default=str)

        logger.info(f"Optimization log saved to {filepath}")
