"""
FastAPI Wrapper for Backtesting Service
HTTP API for running backtests.
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
import json
import uuid
from loguru import logger

from .service import run_backtest, run_backtest_batch, BacktestConfig, BacktestResult
from research.experiment_store import ExperimentStore

app = FastAPI(
    title="Trading Bot Backtesting API",
    description="API for running backtests on trading strategies",
    version="1.0.0"
)


# Pydantic models for API
class BacktestConfigRequest(BaseModel):
    symbol: str = Field(..., description="Trading symbol (e.g., BTCUSDT)")
    timeframe: str = Field(..., description="Timeframe (e.g., 1h, 1d)")
    start: str = Field(..., description="Start date (ISO format)")
    end: str = Field(..., description="End date (ISO format)")
    strategy: str = Field(..., description="Strategy name (sma, rsi, macd, convergence)")
    params: Dict[str, Any] = Field(default_factory=dict, description="Strategy parameters")
    initial_capital: float = Field(10000.0, description="Initial capital")
    commission_rate: float = Field(0.001, description="Commission rate")
    slippage: float = Field(0.0005, description="Slippage")


class BatchBacktestRequest(BaseModel):
    configs: List[BacktestConfigRequest] = Field(..., description="List of backtest configurations")


class BacktestResponse(BaseModel):
    success: bool
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    config_hash: Optional[str] = None


class BatchBacktestResponse(BaseModel):
    success: bool
    total_configs: int
    successful: int
    failed: int
    results: List[Dict[str, Any]] = Field(default_factory=list)
    errors: List[Dict[str, Any]] = Field(default_factory=list)


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Trading Bot Backtesting API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "/backtest": "POST - Run single backtest",
            "/batch-backtest": "POST - Run batch backtests",
            "/strategies": "GET - List available strategies"
        }
    }


@app.get("/strategies")
async def list_strategies():
    """List available strategies and their parameters."""
    strategies = {
        "sma": {
            "name": "Simple Moving Average",
            "description": "Crossover strategy using fast and slow moving averages",
            "parameters": {
                "fast_period": {"type": "int", "default": 20, "description": "Fast MA period"},
                "slow_period": {"type": "int", "default": 50, "description": "Slow MA period"}
            }
        },
        "rsi": {
            "name": "RSI Mean Reversion",
            "description": "Mean reversion strategy using RSI",
            "parameters": {
                "period": {"type": "int", "default": 14, "description": "RSI period"},
                "oversold": {"type": "int", "default": 30, "description": "Oversold threshold"},
                "overbought": {"type": "int", "default": 70, "description": "Overbought threshold"}
            }
        },
        "macd": {
            "name": "MACD Trend Following",
            "description": "Trend following strategy using MACD",
            "parameters": {
                "fast_period": {"type": "int", "default": 12, "description": "Fast EMA period"},
                "slow_period": {"type": "int", "default": 26, "description": "Slow EMA period"},
                "signal_period": {"type": "int", "default": 9, "description": "Signal line period"}
            }
        },
        "convergence": {
            "name": "Multi-Timeframe Convergence",
            "description": "Multi-timeframe convergence strategy",
            "parameters": {}
        }
    }
    return strategies


@app.post("/backtest", response_model=BacktestResponse)
async def run_backtest_endpoint(config: BacktestConfigRequest):
    """
    Run a single backtest with the given configuration.

    Args:
        config: BacktestConfigRequest with all parameters

    Returns:
        BacktestResponse with results or error
    """
    try:
        # Parse dates
        try:
            start_date = datetime.fromisoformat(config.start)
            end_date = datetime.fromisoformat(config.end)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail="Invalid date format. Use ISO format (YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS)"
            )

        # Create BacktestConfig
        backtest_config = BacktestConfig(
            symbol=config.symbol,
            timeframe=config.timeframe,
            start=start_date,
            end=end_date,
            strategy=config.strategy,
            params=config.params,
            initial_capital=config.initial_capital,
            commission_rate=config.commission_rate,
            slippage=config.slippage
        )

        # Initialize experiment store
        store = ExperimentStore("data/experiments.sqlite")

        # Check if experiment already exists
        config_hash = backtest_config.get_config_hash()
        if store.config_exists(config_hash):
            logger.info(f"Experiment already exists for config hash: {config_hash}")
            existing_exp = store.get_config_by_hash(config_hash)
            # Return existing results
            return BacktestResponse(
                success=True,
                result=existing_exp,
                config_hash=config_hash
            )

        # Run backtest
        logger.info(f"Running backtest via API: {config.symbol} {config.timeframe} {config.strategy}")
        result = run_backtest(backtest_config)

        # Log experiment to database
        try:
            store.log_experiment(backtest_config, result)
            logger.info(f"Logged experiment to database: {config.symbol} {config.strategy}")
        except Exception as e:
            logger.error(f"Failed to log experiment: {e}")
            # Continue anyway, don't fail the request

        # Return success response
        return BacktestResponse(
            success=True,
            result=result.to_dict(),
            config_hash=result.config_hash
        )

    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Backtest error: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.post("/batch-backtest", response_model=BatchBacktestResponse)
async def run_batch_backtest_endpoint(
    request: BatchBacktestRequest,
    background_tasks: BackgroundTasks
):
    """
    Run multiple backtests in batch.

    Args:
        request: BatchBacktestRequest with list of configurations

    Returns:
        BatchBacktestResponse with results summary
    """
    try:
        logger.info(f"Running batch backtest via API: {len(request.configs)} configurations")

        # Convert configs
        configs = []
        for i, config in enumerate(request.configs):
            try:
                start_date = datetime.fromisoformat(config.start)
                end_date = datetime.fromisoformat(config.end)

                backtest_config = BacktestConfig(
                    symbol=config.symbol,
                    timeframe=config.timeframe,
                    start=start_date,
                    end=end_date,
                    strategy=config.strategy,
                    params=config.params,
                    initial_capital=config.initial_capital,
                    commission_rate=config.commission_rate,
                    slippage=config.slippage
                )
                configs.append(backtest_config)
            except Exception as e:
                logger.error(f"Error parsing config {i}: {e}")
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid configuration at index {i}: {str(e)}"
                )

        # Run batch
        results = run_backtest_batch(configs)

        # Process results
        successful_results = []
        failed_errors = []

        for config, result, error in results:
            if result:
                successful_results.append({
                    "config": {
                        "symbol": config.symbol,
                        "timeframe": config.timeframe,
                        "strategy": config.strategy,
                        "start": config.start.isoformat(),
                        "end": config.end.isoformat()
                    },
                    "result": result.to_dict(),
                    "config_hash": result.config_hash
                })
            else:
                failed_errors.append({
                    "config": {
                        "symbol": config.symbol,
                        "timeframe": config.timeframe,
                        "strategy": config.strategy
                    },
                    "error": error
                })

        # Return response
        return BatchBacktestResponse(
            success=True,
            total_configs=len(configs),
            successful=len(successful_results),
            failed=len(failed_errors),
            results=successful_results,
            errors=failed_errors
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Batch backtest error: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
