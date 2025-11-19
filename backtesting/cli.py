"""
CLI Interface for Backtesting Service
Command-line interface using Typer.
"""

import typer
from rich.console import Console
from datetime import datetime
from typing import Optional
import json
from loguru import logger

from .service import run_backtest, BacktestConfig

console = Console()

app = typer.Typer()


@app.command()
def run(
    symbol: str = typer.Argument(..., help="Trading symbol (e.g., BTCUSDT)"),
    timeframe: str = typer.Option("1h", help="Timeframe (e.g., 1m, 5m, 15m, 1h, 4h, 1d)"),
    strategy: str = typer.Option("sma", help="Strategy name (sma, rsi, macd, convergence)"),
    start: str = typer.Option(..., help="Start date (YYYY-MM-DD)"),
    end: str = typer.Option(..., help="End date (YYYY-MM-DD)"),
    params: str = typer.Option("{}", help="Strategy parameters as JSON string"),
    capital: float = typer.Option(10000.0, help="Initial capital"),
    commission: float = typer.Option(0.001, help="Commission rate"),
    slippage: float = typer.Option(0.0005, help="Slippage"),
    output: Optional[str] = typer.Option(None, help="Output JSON file path"),
    verbose: bool = typer.Option(False, help="Enable verbose logging")
):
    """
    Run a single backtest with specified parameters.
    """
    if verbose:
        logger.add("backtest_cli.log", level="DEBUG")

    try:
        # Parse dates
        start_date = datetime.strptime(start, "%Y-%m-%d")
        end_date = datetime.strptime(end, "%Y-%m-%d")

        # Parse params
        try:
            params_dict = json.loads(params) if params else {}
        except json.JSONDecodeError:
            console.print(f"[red]Error: Invalid JSON for params: {params}[/red]")
            raise typer.Exit(1)

        # Create config
        config = BacktestConfig(
            symbol=symbol,
            timeframe=timeframe,
            start=start_date,
            end=end_date,
            strategy=strategy,
            params=params_dict,
            initial_capital=capital,
            commission_rate=commission,
            slippage=slippage
        )

        # Run backtest
        console.print(f"Running backtest: {symbol} {timeframe} {strategy}")
        console.print(f"Period: {start_date.date()} to {end_date.date()}")
        console.print(f"Initial capital: ${capital:,.2f}")

        result = run_backtest(config)

        # Display results
        console.print("\n" + "="*60)
        console.print("[bold green]BACKTEST RESULTS[/bold green]")
        console.print("="*60 + "\n")

        console.print(f"Total Trades: {result.total_trades}")
        console.print(f"[yellow]Win Rate: {result.win_rate:.2%}[/yellow]")
        console.print(f"[green]Total Return: {result.total_return_pct:.2f}%[/green]")
        console.print(f"Annualized Return: {result.annualized_return:.2%}")
        console.print(f"[red]Max Drawdown: {result.max_drawdown:.2%}[/red]")
        console.print(f"[cyan]Sharpe Ratio: {result.sharpe_ratio:.2f}[/cyan]")
        console.print(f"Profit Factor: {result.profit_factor:.2f}")
        console.print(f"Initial Capital: ${result.initial_capital:,.2f}")
        console.print(f"Final Capital: ${result.final_capital:,.2f}")
        console.print(f"Total P&L: ${result.final_capital - result.initial_capital:,.2f}")

        # Winning/Losing trades
        console.print(f"\nWinning Trades: {result.winning_trades}")
        console.print(f"Losing Trades: {result.losing_trades}")
        console.print(f"Average Win: ${result.avg_win:,.2f}")
        console.print(f"Average Loss: ${result.avg_loss:,.2f}")

        # Save to file if requested
        if output:
            with open(output, 'w') as f:
                f.write(result.to_json())
            console.print(f"\n[blue]Results saved to: {output}[/blue]")

        console.print("\n[green]✓ Backtest completed successfully![/green]")

    except ValueError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Unexpected error: {e}[/red]")
        if verbose:
            import traceback
            traceback.print_exc()
        raise typer.Exit(1)


@app.command()
def batch(
    config_file: str = typer.Argument(..., help="Path to JSON config file with array of configs"),
    output_dir: str = typer.Option("./results", help="Output directory for results"),
    verbose: bool = typer.Option(False, help="Enable verbose logging")
):
    """
    Run multiple backtests from a configuration file.
    """
    from .service import run_backtest_batch
    import os

    if verbose:
        logger.add("backtest_cli.log", level="DEBUG")

    try:
        # Load configs
        with open(config_file, 'r') as f:
            configs_data = json.load(f)

        if not isinstance(configs_data, list):
            typer.echo("Error: Config file must contain an array of configurations", fg=typer.colors.RED)
            raise typer.Exit(1)

        # Parse configs
        configs = []
        for i, config_data in enumerate(configs_data, 1):
            try:
                config = BacktestConfig.from_dict(config_data)
                configs.append(config)
            except Exception as e:
                typer.echo(f"Error parsing config {i}: {e}", fg=typer.colors.RED)
                raise typer.Exit(1)

        typer.echo(f"Loaded {len(configs)} configurations from {config_file}")

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Run batch
        typer.echo(f"Running {len(configs)} backtests...")
        results = run_backtest_batch(configs)

        # Save results
        successful = 0
        failed = 0
        for i, (config, result, error) in enumerate(results, 1):
            filename = f"{i:03d}_{config.symbol}_{config.strategy}.json"
            filepath = os.path.join(output_dir, filename)

            if result:
                with open(filepath, 'w') as f:
                    f.write(result.to_json())
                successful += 1
                typer.echo(f"✓ {filename}: {result.total_return_pct:.2f}% return, Sharpe: {result.sharpe_ratio:.2f}")
            else:
                failed += 1
                typer.echo(f"✗ {filename}: {error}", fg=typer.colors.RED)

        typer.echo("\n" + "="*60)
        typer.echo(f"BATCH COMPLETE: {successful} successful, {failed} failed", fg=typer.colors.GREEN if failed == 0 else typer.colors.YELLOW)
        typer.echo(f"Results saved to: {output_dir}")

    except FileNotFoundError:
        typer.echo(f"Error: Config file not found: {config_file}", fg=typer.colors.RED)
        raise typer.Exit(1)
    except Exception as e:
        typer.echo(f"Error: {e}", fg=typer.colors.RED)
        if verbose:
            import traceback
            traceback.print_exc()
        raise typer.Exit(1)


@app.command()
def strategies():
    """
    List available strategies and their parameters.
    """
    console.print("\n[bold]Available Strategies:[/bold]\n")

    strategies = {
        "sma": {
            "description": "Simple Moving Average crossover",
            "params": {
                "fast_period": "Fast MA period (default: 20)",
                "slow_period": "Slow MA period (default: 50)"
            }
        },
        "rsi": {
            "description": "RSI mean reversion",
            "params": {
                "period": "RSI period (default: 14)",
                "oversold": "Oversold threshold (default: 30)",
                "overbought": "Overbought threshold (default: 70)"
            }
        },
        "macd": {
            "description": "MACD trend following",
            "params": {
                "fast_period": "Fast EMA period (default: 12)",
                "slow_period": "Slow EMA period (default: 26)",
                "signal_period": "Signal line period (default: 9)"
            }
        },
        "convergence": {
            "description": "Multi-timeframe convergence strategy",
            "params": {
                "description": "No additional parameters required"
            }
        }
    }

    for name, info in strategies.items():
        console.print(f"\n[bold blue]{name.upper()}:[/bold blue]")
        console.print(f"  {info['description']}")
        console.print("  Parameters:")
        for param, desc in info['params'].items():
            console.print(f"    - {param}: {desc}")


if __name__ == "__main__":
    app()
