#!/usr/bin/env python3
"""
Show Experiments
CLI tool for viewing and analyzing experiment results.
"""

import sys
from pathlib import Path

# Add project root to sys.path for imports
project_root = Path(__file__).parent.parent.resolve()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import typer
from typing import Optional, List, Dict
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import print as rprint
from rich.text import Text
from rich.align import Align
from datetime import datetime
import json

from research.experiment_store import ExperimentStore

app = typer.Typer()
console = Console()


@app.command()
def top(
    metric: str = typer.Argument(..., help="Metric to sort by (sharpe_ratio, total_return, win_rate, etc.)"),
    n: int = typer.Option(10, help="Number of top results to show"),
    db: str = typer.Option("data/experiments.sqlite", help="Database path"),
    symbol: Optional[str] = typer.Option(None, help="Filter by symbol"),
    strategy: Optional[str] = typer.Option(None, help="Filter by strategy"),
    json_output: bool = typer.Option(False, help="Output as JSON")
):
    """
    Show top N experiments by metric.

    Examples:
        python show_experiments.py top sharpe_ratio --n 20
        python show_experiments.py top total_return --symbol BTCUSDT
        python show_experiments.py top win_rate --strategy sma
    """
    try:
        store = ExperimentStore(db)

        # Build filter criteria
        filters = {}
        if symbol:
            filters['symbol'] = symbol
        if strategy:
            filters['strategy'] = strategy

        # Get results
        results = store.get_experiments(metric=metric, top_n=n, filter_criteria=filters)

        if not results:
            console.print("[yellow]No experiments found with the given criteria[/yellow]")
            raise typer.Exit(0)

        # Output as JSON if requested
        if json_output:
            console.print_json(json.dumps(results, default=str))
            raise typer.Exit(0)

        # Display as table
        _display_results_table(results, metric)

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def summary(
    db: str = typer.Option("data/experiments.sqlite", help="Database path")
):
    """
    Show summary statistics of all experiments.
    """
    try:
        store = ExperimentStore(db)
        summary = store.get_summary()

        if not summary:
            console.print("[yellow]No experiments found[/yellow]")
            raise typer.Exit(0)

        # Display summary
        console.print("\n[bold green]EXPERIMENT SUMMARY[/bold green]\n")

        # Basic stats
        table = Table(show_header=False, box=None, padding=(0, 2))
        table.add_column("Metric", style="cyan", no_wrap=True)
        table.add_column("Value", style="white")

        table.add_row("Total Experiments", str(summary['total_experiments']))
        table.add_row("Total Errors", str(summary['total_errors']))
        table.add_row("Success Rate", f"{summary['success_rate']:.2%}")

        console.print(table)
        console.print()

        # Strategies and symbols
        console.print("[bold cyan]Strategies Tested:[/bold cyan]")
        console.print(", ".join(summary['strategies']) if summary['strategies'] else "None")
        console.print()

        console.print("[bold cyan]Symbols Tested:[/bold cyan]")
        console.print(", ".join(summary['symbols']) if summary['symbols'] else "None")
        console.print()

        # Best results
        if summary.get('best_results'):
            console.print("[bold green]Best Results:[/bold green]\n")
            best_table = Table(show_header=True, box=None)
            best_table.add_column("Metric", style="cyan")
            best_table.add_column("Value", style="white")
            best_table.add_column("Strategy", style="yellow")
            best_table.add_column("Symbol", style="magenta")

            for metric, data in summary['best_results'].items():
                best_table.add_row(
                    metric.replace('_', ' ').title(),
                    f"{data['value']:.4f}",
                    data['strategy'],
                    data['symbol']
                )

            console.print(best_table)
            console.print()

        # Average metrics
        if summary.get('average_metrics'):
            console.print("[bold green]Average Metrics:[/bold green]\n")
            avg_table = Table(show_header=False, box=None, padding=(0, 2))
            avg_table.add_column("Metric", style="cyan", no_wrap=True)
            avg_table.add_column("Value", style="white")

            for metric, value in summary['average_metrics'].items():
                metric_name = metric.replace('avg_', '').replace('_', ' ').title()
                if 'return' in metric or 'drawdown' in metric:
                    avg_table.add_row(metric_name, f"{value:.4f}")
                else:
                    avg_table.add_row(metric_name, f"{value:.4f}")

            console.print(avg_table)

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def list(
    db: str = typer.Option("data/experiments.sqlite", help="Database path"),
    symbol: Optional[str] = typer.Option(None, help="Filter by symbol"),
    strategy: Optional[str] = typer.Option(None, help="Filter by strategy"),
    limit: int = typer.Option(50, help="Number of results to show"),
    json_output: bool = typer.Option(False, help="Output as JSON")
):
    """
    List all experiments with optional filters.
    """
    try:
        store = ExperimentStore(db)

        # Build filter criteria
        filters = {}
        if symbol:
            filters['symbol'] = symbol
        if strategy:
            filters['strategy'] = strategy

        # Get results
        results = store.get_experiments(top_n=limit, filter_criteria=filters)

        if not results:
            console.print("[yellow]No experiments found with the given criteria[/yellow]")
            raise typer.Exit(0)

        # Output as JSON if requested
        if json_output:
            console.print_json(json.dumps(results, default=str))
            raise typer.Exit(0)

        # Display as table
        _display_results_table(results, "timestamp")

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def show(
    experiment_id: int = typer.Argument(..., help="Experiment ID"),
    db: str = typer.Option("data/experiments.sqlite", help="Database path")
):
    """
    Show detailed information about a specific experiment.
    """
    try:
        store = ExperimentStore(db)

        # Get experiment by ID (simulate with top_n=experiment_id)
        # In a real implementation, you'd have a get_by_id method
        results = store.get_experiments(top_n=experiment_id)

        if experiment_id > len(results):
            console.print(f"[red]Experiment {experiment_id} not found[/red]")
            raise typer.Exit(1)

        exp = results[experiment_id - 1]  # ID is 1-based

        # Create detailed panel
        content = f"""
[cyan]Symbol:[/cyan] {exp['symbol']}
[cyan]Strategy:[/cyan] {exp['strategy']}
[cyan]Timeframe:[/cyan] {exp['timeframe']}
[cyan]Period:[/cyan] {exp['start']} to {exp['end']}

[bold green]Performance Metrics:[/bold green]
[white]Total Return:[/white] {exp['total_return_pct']:.2f}%
[white]Annualized Return:[/white] {exp['annualized_return']:.2f}%
[white]Sharpe Ratio:[/white] {exp['sharpe_ratio']:.4f}
[white]Max Drawdown:[/white] {exp['max_drawdown']:.2f}%
[white]Win Rate:[/white] {exp['win_rate']:.2%}
[white]Profit Factor:[/white] {exp['profit_factor']:.4f}

[bold green]Trade Statistics:[/bold green]
[white]Total Trades:[/white] {exp['total_trades']}
[white]Winning Trades:[/white] {exp['winning_trades']}
[white]Losing Trades:[/white] {exp['losing_trades']}
[white]Average Win:[/white] ${exp['avg_win']:.2f}
[white]Average Loss:[/white] ${exp['avg_loss']:.2f}

[bold green]Capital:[/bold green]
[white]Initial:[/white] ${exp['initial_capital']:,.2f}
[white]Final:[/white] ${exp['final_capital']:,.2f}
[white]P&L:[/white] ${exp['final_capital'] - exp['initial_capital']:,.2f}
        """

        console.print(Panel(
            content,
            title=f"Experiment #{exp['id']} - {exp['config_hash'][:16]}",
            border_style="green"
        ))

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def errors(
    db: str = typer.Option("data/experiments.sqlite", help="Database path"),
    limit: int = typer.Option(20, help="Number of errors to show")
):
    """
    Show recent experiment errors.
    """
    try:
        store = ExperimentStore(db)
        errors = store.get_errors()[:limit]

        if not errors:
            console.print("[green]No errors found![/green]")
            raise typer.Exit(0)

        # Create table
        table = Table(title="Experiment Errors")
        table.add_column("ID", style="cyan", no_wrap=True)
        table.add_column("Symbol", style="yellow")
        table.add_column("Strategy", style="magenta")
        table.add_column("Error", style="red")
        table.add_column("Time", style="white")

        for err in errors:
            # Truncate error message
            error_msg = err['error_message'][:100]
            if len(err['error_message']) > 100:
                error_msg += "..."

            table.add_row(
                str(err['id']),
                err['symbol'],
                err['strategy'],
                error_msg,
                err['timestamp'][:19]  # Remove microseconds
            )

        console.print(table)

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def export(
    filepath: str = typer.Argument(..., help="Output JSON file path"),
    db: str = typer.Option("data/experiments.sqlite", help="Database path"),
    symbol: Optional[str] = typer.Option(None, help="Filter by symbol"),
    strategy: Optional[str] = typer.Option(None, help="Filter by strategy")
):
    """
    Export experiments to JSON file.
    """
    try:
        store = ExperimentStore(db)

        # Build filter criteria
        filters = {}
        if symbol:
            filters['symbol'] = symbol
        if strategy:
            filters['strategy'] = strategy

        store.export_to_json(filepath, filter_criteria=filters)
        console.print(f"[green]Exported experiments to {filepath}[/green]")

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


def _display_results_table(results: List[Dict], sort_metric: str):
    """Helper to display results in a formatted table."""
    # Create table
    table = Table(title=f"Top Experiments by {sort_metric.replace('_', ' ').title()}")
    table.add_column("ID", style="cyan", no_wrap=True)
    table.add_column("Symbol", style="yellow", no_wrap=True)
    table.add_column("Strategy", style="magenta")
    table.add_column("Timeframe", style="blue")
    table.add_column("Return %", style="green")
    table.add_column("Sharpe", style="cyan")
    table.add_column("Win Rate", style="blue")
    table.add_column("Drawdown %", style="red")
    table.add_column("Trades", style="white")

    for exp in results:
        # Color code the return
        return_style = "green" if exp['total_return_pct'] >= 0 else "red"
        drawdown_style = "red"

        table.add_row(
            str(exp['id']),
            exp['symbol'],
            exp['strategy'],
            exp['timeframe'],
            f"[{return_style}]{exp['total_return_pct']:.2f}%[/{return_style}]",
            f"{exp['sharpe_ratio']:.3f}",
            f"{exp['win_rate']:.2%}",
            f"[{drawdown_style}]{exp['max_drawdown']:.2f}%[/{drawdown_style}]",
            str(exp['total_trades'])
        )

    console.print(table)


if __name__ == "__main__":
    app()
