# Experiment Pipeline Documentation

## Overview

The Experiment Pipeline provides a complete workflow for autonomous backtest optimization, experiment tracking, and configuration promotion. This system enables systematic exploration of trading strategy parameters, AI-driven optimization, and safe deployment of winning configurations to production.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Experiment Pipeline                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    │
│  │ Backtest API │───▶│   SQLite     │───▶│  DeepSeek    │    │
│  │ (Service)    │    │    Store     │    │  Optimizer   │    │
│  └──────────────┘    └──────────────┘    └──────────────┘    │
│         │                   │                   │              │
│         ▼                   ▼                   ▼              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    │
│  │   CLI/HTTP   │    │   CLI View   │    │  Promotion   │    │
│  │  Interface   │    │    Tool      │    │   Script     │    │
│  └──────────────┘    └──────────────┘    └──────────────┘    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Components

### 1. Backtest Service (`backtesting/service.py`)

Core API for running backtests with deterministic metrics.

**Key Features:**
- `BacktestConfig`: Configuration dataclass
- `BacktestResult`: Results with comprehensive metrics
- `run_backtest()`: Main execution function
- Support for SMA, RSI, MACD, and Convergence strategies

**Metrics Calculated:**
- Total Return & Annualized Return
- Sharpe Ratio
- Max Drawdown
- Win Rate & Profit Factor
- Trade Statistics (wins, losses, avg win/loss)

### 2. Configuration Generators (`research/config_generators.py`)

Generate parameter combinations for systematic testing.

**Search Types:**
- **Grid Search**: Cartesian product of parameter ranges
- **Random Search**: Random sampling with distribution support

**Supported Parameter Types:**
- `int`: Integer ranges
- `float`: Float ranges
- `choice`: Discrete choices
- `loguniform`: Log-uniform distribution

**Predefined Strategies:**
```python
SMA_PARAM_GRID = {
    'fast_period': [5, 10, 20, 30],
    'slow_period': [50, 100, 150, 200]
}

RSI_PARAM_GRID = {
    'period': [10, 14, 21, 30],
    'oversold': [20, 25, 30, 35],
    'overbought': [65, 70, 75, 80]
}
```

### 3. Experiment Store (`research/experiment_store.py`)

SQLite-based tracking for all experiments and results.

**Database Schema:**
```sql
experiments (
    id, config_hash, timestamp, symbol, timeframe, strategy,
    start, end, initial_capital, final_capital,
    total_return, total_return_pct, annualized_return,
    max_drawdown, sharpe_ratio, win_rate, profit_factor,
    total_trades, winning_trades, losing_trades,
    avg_win, avg_loss, config_json, result_json
)

errors (
    id, config_hash, timestamp, symbol, timeframe, strategy,
    error_message, config_json
)
```

**Key Features:**
- Duplicate detection via config hashing
- Filtered retrieval with sorting
- Summary statistics
- Export to JSON

### 4. Experiment Orchestrator (`research/orchestrator.py`)

Coordinate large-scale experiment execution.

**Features:**
- Parallel execution with ThreadPoolExecutor
- Sequential execution option
- Automatic logging to SQLite
- Progress tracking
- Error handling

**Usage:**
```python
runner = ExperimentRunner(
    store=ExperimentStore("data/experiments.sqlite"),
    max_workers=4
)

# Run grid search
results = runner.grid_search(configs, resume_on_error=True)

# Run random search
results = runner.random_search(configs, n_samples=100)
```

### 5. DeepSeek Optimizer (`research/deepseek_optimizer.py`)

AI-driven configuration optimization using DeepSeek.

**Features:**
- Analyzes experiment history
- Proposes new configurations
- Iterative optimization loop
- Validation of AI suggestions
- Demo mode when AI unavailable

**Optimization Loop:**
```python
optimizer = DeepSeekOptimizer(experiment_store, deepseek_client)
loop = OptimizationLoop(optimizer, runner, max_iterations=10)

results = await loop.run(
    symbol="BTCUSDT",
    timeframe="15m",
    start=datetime(2024, 1, 1),
    end=datetime(2024, 12, 31),
    strategy="sma",
    target_metric="sharpe_ratio"
)
```

### 6. CLI Viewer (`scripts/show_experiments.py`)

Rich terminal interface for viewing experiment results.

**Commands:**
```bash
# Show top 10 experiments by Sharpe ratio
python scripts/show_experiments.py top sharpe_ratio --n 10

# Show summary statistics
python scripts/show_experiments.py summary

# List all experiments with filters
python scripts/show_experiments.py list --symbol BTCUSDT

# Show detailed view of experiment
python scripts/show_experiments.py show 42

# Show recent errors
python scripts/show_experiments.py errors

# Export to JSON
python scripts/show_experiments.py export results.json
```

### 7. Strategy Promotion (`scripts/promote_strategy.py`)

Safely promote winning configurations to production.

**Validation Criteria:**
- Sharpe Ratio ≥ threshold (default: 1.0)
- Max Drawdown ≤ threshold (default: 10%)
- Minimum trades (default: 5)
- Positive total return
- Clean git working directory

**Usage:**
```bash
# Promote experiment 42 to 'scalp' profile
python scripts/promote_strategy.py 42 scalp

# With custom criteria
python scripts/promote_strategy.py 42 scalp \
    --min-sharpe 1.5 \
    --max-drawdown 0.08 \
    --min-trades 10

# Auto-commit to git
python scripts/promote_strategy.py 42 scalp --auto-commit

# List all profiles
python scripts/promote_strategy.py list

# Show profile details
python scripts/promote_strategy.py show scalp
```

## Workflow

### Complete Experiment Pipeline

1. **Generate Configurations**
```python
from research.config_generators import generate_configs
from datetime import datetime

configs = generate_configs(
    symbol="BTCUSDT",
    timeframe="15m",
    start=datetime(2024, 1, 1),
    end=datetime(2024, 12, 31),
    strategies=["sma", "rsi"],
    param_grids={
        "sma": {"fast_period": [10, 20], "slow_period": [50, 100]},
        "rsi": {"period": [14, 21], "oversold": [30, 35], "overbought": [70, 75]}
    },
    search_type="grid"
)
```

2. **Run Experiments**
```python
from research.orchestrator import ExperimentRunner
from research.experiment_store import ExperimentStore

runner = ExperimentRunner(
    store=ExperimentStore("data/experiments.sqlite"),
    max_workers=4
)

results = runner.run_experiments(configs)
```

3. **View Results**
```bash
python scripts/show_experiments.py top sharpe_ratio --n 20
```

4. **AI Optimization (Optional)**
```python
import asyncio
from research.deepseek_optimizer import DeepSeekOptimizer, OptimizationLoop

async def optimize():
    optimizer = DeepSeekOptimizer(experiment_store)
    loop = OptimizationLoop(optimizer, runner, max_iterations=10)

    results = await loop.run(
        symbol="BTCUSDT",
        timeframe="15m",
        start=datetime(2024, 1, 1),
        end=datetime(2024, 12, 31),
        strategy="sma",
        target_metric="sharpe_ratio"
    )

asyncio.run(optimize())
```

5. **Promote Winning Config**
```bash
# Get experiment ID from results
python scripts/show_experiments.py top sharpe_ratio --n 1

# Promote to production
python scripts/promote_strategy.py 42 production_profile \
    --min-sharpe 1.5 \
    --auto-commit
```

## Configuration Files

### Experiment Database
Default location: `data/experiments.sqlite`

### Strategy Profiles
Location: `config/strategy_profiles.yaml`

Example profile:
```yaml
production_scalp:
  promoted_at: "2024-01-15T10:30:00"
  experiment_id: 42
  strategy: "sma"
  symbol: "BTCUSDT"
  timeframe: "15m"
  params:
    fast_period: 20
    slow_period: 50
  performance:
    total_return: 0.1534
    sharpe_ratio: 1.842
    max_drawdown: 0.087
    win_rate: 0.623
    total_trades: 47
  validation_criteria:
    min_sharpe: 1.5
    max_drawdown: 0.1
    min_trades: 10
```

## API Reference

### Backtest Service API

**Run Single Backtest:**
```python
from backtesting.service import BacktestConfig, run_backtest
from datetime import datetime

config = BacktestConfig(
    symbol="BTCUSDT",
    timeframe="15m",
    start=datetime(2024, 1, 1),
    end=datetime(2024, 12, 31),
    strategy="sma",
    params={"fast_period": 20, "slow_period": 50},
    initial_capital=10000.0
)

result = run_backtest(config)

print(f"Sharpe Ratio: {result.sharpe_ratio}")
print(f"Total Return: {result.total_return_pct:.2%}")
```

**Run Batch Backtests:**
```python
from research.orchestrator import ExperimentRunner

configs = [...]  # List of BacktestConfig
runner = ExperimentRunner(store=ExperimentStore("data/experiments.sqlite"))

results = runner.run_experiments(configs, parallel=True)
```

### FastAPI Endpoints

Start server:
```bash
python -m uvicorn backtesting.api:app --reload --port 8000
```

**Endpoints:**

```http
POST /backtest
Content-Type: application/json

{
    "symbol": "BTCUSDT",
    "timeframe": "15m",
    "strategy": "sma",
    "params": {"fast_period": 20, "slow_period": 50},
    "start": "2024-01-01",
    "end": "2024-12-31",
    "initial_capital": 10000.0
}
```

```http
POST /batch-backtest
Content-Type: application/json

{
    "configs": [...]
}
```

```http
GET /strategies
```

### CLI Commands

**Backtest CLI:**
```bash
# Run single backtest
python -m backtesting.cli run \
    --symbol BTCUSDT \
    --timeframe 15m \
    --strategy sma \
    --params '{"fast_period": 20, "slow_period": 50}' \
    --start 2024-01-01 \
    --end 2024-12-31

# Run batch from file
python -m backtesting.cli batch configs.json

# List strategies
python -m backtesting.cli strategies
```

**Show Experiments CLI:**
```bash
python scripts/show_experiments.py top sharpe_ratio --n 20 --json
python scripts/show_experiments.py summary
python scripts/show_experiments.py list --strategy sma
python scripts/show_experiments.py show 42
python scripts/show_experiments.py errors --limit 10
python scripts/show_experiments.py export results.json
```

**Promote Strategy CLI:**
```bash
python scripts/promote_strategy.py promote 42 profile_name
python scripts/promote_strategy.py list
python scripts/promote_strategy.py show profile_name
```

## Environment Variables

**Optional DeepSeek Integration:**
```bash
export DEEPSEEK_API_KEY="your-api-key"
export DEEPSEEK_BASE_URL="https://api.deepseek.com"
```

Without API key, the optimizer runs in demo mode with heuristic proposals.

## Best Practices

### 1. Experiment Design

- Start with grid search to understand parameter space
- Use random search for large parameter spaces (more efficient)
- Set reasonable min/max bounds for parameters
- Include baseline configurations (default parameters)

### 2. Performance

- Use parallel execution (`max_workers=4-8`)
- Enable duplicate detection (config hashing)
- Resume on errors for robust execution
- Monitor SQLite database size

### 3. AI Optimization

- Provide experiment history (10+ experiments minimum)
- Set clear optimization targets (sharpe_ratio, total_return, etc.)
- Use convergence thresholds to stop early
- Validate AI proposals before deployment

### 4. Promotion

- Always validate before promotion
- Use git for audit trails
- Set conservative thresholds initially
- Document why configurations were selected

### 5. Testing

- Run unit tests: `pytest tests/test_backtest_service.py -v`
- Test individual strategies separately
- Verify reproducible results with config hashing
- Check error logs: `python scripts/show_experiments.py errors`

## Troubleshooting

### Common Issues

**1. "No module named 'backtesting.service'"**
```bash
# Ensure PYTHONPATH includes project root
export PYTHONPATH="/path/to/trading_bot:$PYTHONPATH"
```

**2. "Database is locked"**
- Check for existing connections
- Use `resume_on_error=True` in orchestrator
- Verify file permissions

**3. "Git working directory is not clean"**
```bash
git add .
git commit -m "Work in progress"
# Or
git stash
```

**4. "Experiment not found"**
- Check experiment ID: `python scripts/show_experiments.py list`
- Use latest experiments: `python scripts/show_experiments.py top sharpe_ratio --n 1`

**5. "DeepSeek API error"**
- Verify API key: `echo $DEEPSEEK_API_KEY`
- Check base URL: `echo $DEEPSEEK_BASE_URL`
- Optimizer will fallback to demo mode

### Debug Mode

**Verbose backtesting:**
```python
import logging
logging.basicConfig(level=logging.DEBUG)

result = run_backtest(config)
```

**Show experiment details:**
```bash
python scripts/show_experiments.py show 42 --json
```

**Check database contents:**
```bash
sqlite3 data/experiments.sqlite
> SELECT COUNT(*) FROM experiments;
> SELECT * FROM experiments ORDER BY sharpe_ratio DESC LIMIT 5;
```

## Rollback Procedure

If a promoted configuration causes issues:

1. **Revert Git Commit:**
```bash
git log --oneline
git revert <commit-hash>
```

2. **Remove Profile:**
```bash
python scripts/promote_strategy.py show profile_name  # Note experiment_id
# Edit config/strategy_profiles.yaml and remove the profile
git commit -am "rollback: Remove failing profile"
```

3. **Promote Previous Version:**
```bash
python scripts/promote_strategy.py show profile_name  # Get previous experiment_id
python scripts/promote_strategy.py promote <previous_id> profile_name
```

4. **Run Tests:**
```bash
python scripts/show_experiments.py top sharpe_ratio --n 5
```

## Advanced Usage

### Custom Strategies

Extend `backtesting/service.py`:

```python
def generate_signals_your_strategy(data: pd.DataFrame, params: dict) -> pd.DataFrame:
    # Implement your strategy logic
    # Return DataFrame with 'signal' column (1 for long, -1 for short, 0 for hold)
    signals = pd.Series(0, index=data.index)
    # ... strategy logic
    return data.assign(signal=signals)

# Register in generate_signals() function
```

### Custom Metrics

Extend `BacktestResult` calculation in `run_backtest()`:

```python
# Calculate custom metric
custom_metric = your_calculation(trades, data)

result = BacktestResult(
    # ... existing fields
    custom_metric=custom_metric
)
```

### Batch Scripts

Create reusable experiment scripts:

```python
#!/usr/bin/env python3
# run_experiment_batch.py

from research.orchestrator import ExperimentRunner
from research.config_generators import generate_configs
from research.experiment_store import ExperimentStore
from datetime import datetime

# Configuration
SYMBOL = "BTCUSDT"
TIMEFRAME = "15m"
START = datetime(2024, 1, 1)
END = datetime(2024, 12, 31)
STRATEGIES = ["sma", "rsi"]
SEARCH_TYPE = "grid"

# Generate configurations
configs = generate_configs(
    symbol=SYMBOL,
    timeframe=TIMEFRAME,
    start=START,
    end=END,
    strategies=STRATEGIES,
    param_grids={...},  # Your parameter grids
    search_type=SEARCH_TYPE
)

# Run experiments
runner = ExperimentRunner(
    store=ExperimentStore("data/experiments.sqlite"),
    max_workers=4
)

results = runner.run_experiments(configs)
print(f"Completed {len(results)} experiments")
```

## Performance Benchmarks

**Typical Performance:**
- Single backtest: 50-200ms
- 100 experiments (parallel, 4 workers): 5-10 seconds
- 1000 experiments (parallel, 8 workers): 1-2 minutes
- Database size: ~1KB per experiment
- SQLite query time: <10ms for 10k experiments

**Optimization:**
- Use parallel execution for batches > 10
- Increase `max_workers` based on CPU cores
- Monitor memory usage with large batches
- Consider database indexing for specific queries

## Support

For issues or questions:
1. Check this documentation
2. Review error logs: `python scripts/show_experiments.py errors`
3. Run test suite: `pytest tests/test_backtest_service.py -v`
4. Check git history for recent changes

## Related Documentation

- `QUICKSTART.md` - Basic setup and usage
- `CONVERGENCE_STRATEGY_COMPLETION_REPORT.md` - Convergence strategy details
- `docs/QA_CHECKLIST.md` - Testing procedures
- `vibecoder_implementation.json` - Implementation plan
