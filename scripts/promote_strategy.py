#!/usr/bin/env python3
"""
Promote Strategy
Safely promote winning experiment configurations to production.
"""

import sys
from pathlib import Path

# Add project root to sys.path for imports
project_root = Path(__file__).parent.parent.resolve()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import typer
from typing import Optional, Dict, Any
from pathlib import Path
import json
import yaml
from datetime import datetime
from loguru import logger
import subprocess
from rich.console import Console

from research.experiment_store import ExperimentStore

app = typer.Typer()
console = Console()


class StrategyPromoter:
    """Handle promotion of experiment configurations to production."""

    def __init__(
        self,
        db_path: str = "data/experiments.sqlite",
        config_path: str = "config/strategy_profiles.yaml"
    ):
        """
        Initialize promoter.

        Args:
            db_path: Path to experiments database
            config_path: Path to strategy profiles config
        """
        self.store = ExperimentStore(db_path)
        self.config_path = Path(config_path)
        self.config_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"StrategyPromoter initialized: db={db_path}, config={config_path}")

    def validate_experiment(
        self,
        experiment_id: int,
        min_sharpe: float = 1.0,
        max_drawdown: float = 0.1,
        min_trades: int = 5,
        require_git_clean: bool = True
    ) -> bool:
        """
        Validate experiment meets promotion criteria.

        Args:
            experiment_id: Experiment ID to validate
            min_sharpe: Minimum Sharpe ratio required
            max_drawdown: Maximum drawdown allowed
            min_trades: Minimum number of trades required
            require_git_clean: Whether to require clean git status

        Returns:
            True if experiment meets all criteria
        """
        # Check git status
        if require_git_clean:
            try:
                result = subprocess.run(
                    ['git', 'status', '--porcelain'],
                    capture_output=True,
                    text=True,
                    check=True
                )

                if result.stdout.strip():
                    logger.error("Git working directory is not clean")
                    logger.error("Please commit or stash changes before promoting")
                    return False

            except subprocess.CalledProcessError:
                logger.error("Failed to check git status")
                return False
            except FileNotFoundError:
                logger.warning("Git not found, skipping clean check")

        # Get experiment
        results = self.store.get_experiments(top_n=experiment_id)

        if experiment_id > len(results):
            logger.error(f"Experiment {experiment_id} not found")
            return False

        exp = results[experiment_id - 1]  # ID is 1-based

        # Validate metrics
        checks = []

        # Sharpe ratio
        sharpe = exp.get('sharpe_ratio', 0)
        sharpe_ok = sharpe >= min_sharpe
        checks.append(("Sharpe ratio", sharpe_ok, f"{sharpe:.3f} >= {min_sharpe}"))
        if not sharpe_ok:
            logger.error(f"Sharpe ratio {sharpe:.3f} < {min_sharpe}")

        # Max drawdown
        drawdown = exp.get('max_drawdown', 1.0)
        drawdown_ok = drawdown <= max_drawdown
        checks.append(("Max drawdown", drawdown_ok, f"{drawdown:.2%} <= {max_drawdown:.2%}"))
        if not drawdown_ok:
            logger.error(f"Max drawdown {drawdown:.2%} > {max_drawdown:.2%}")

        # Trade count
        trades = exp.get('total_trades', 0)
        trades_ok = trades >= min_trades
        checks.append(("Trade count", trades_ok, f"{trades} >= {min_trades}"))
        if not trades_ok:
            logger.error(f"Trade count {trades} < {min_trades}")

        # Total return (should be positive)
        total_return = exp.get('total_return', 0)
        return_ok = total_return > 0
        checks.append(("Total return", return_ok, f"{total_return:.2%} > 0%"))
        if not return_ok:
            logger.error(f"Total return {total_return:.2%} is not positive")

        # All checks must pass
        all_passed = all(check[1] for check in checks)

        if not all_passed:
            logger.error("Validation failed")
            for name, passed, description in checks:
                status = "✓" if passed else "✗"
                logger.error(f"  {status} {name}: {description}")

        return all_passed

    def get_experiment_config(self, experiment_id: int) -> Optional[Dict[str, Any]]:
        """
        Get configuration from experiment.

        Args:
            experiment_id: Experiment ID

        Returns:
            Configuration dict or None
        """
        results = self.store.get_experiments(top_n=experiment_id)

        if experiment_id > len(results):
            logger.error(f"Experiment {experiment_id} not found")
            return None

        exp = results[experiment_id - 1]

        # Parse config
        config_json = exp.get('config_json', '{}')
        try:
            config = json.loads(config_json)

            return {
                'symbol': exp['symbol'],
                'timeframe': exp['timeframe'],
                'strategy': exp['strategy'],
                'params': config.get('params', {}),
                'initial_capital': exp['initial_capital'],
                'experiment_id': experiment_id,
                'metrics': {
                    'total_return': exp['total_return_pct'],
                    'sharpe_ratio': exp['sharpe_ratio'],
                    'max_drawdown': exp['max_drawdown'],
                    'win_rate': exp['win_rate'],
                    'total_trades': exp['total_trades']
                }
            }

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse config: {e}")
            return None

    def promote(
        self,
        experiment_id: int,
        profile_name: str,
        min_sharpe: float = 1.0,
        max_drawdown: float = 0.1,
        min_trades: int = 5,
        auto_commit: bool = False
    ) -> bool:
        """
        Promote experiment configuration to production profile.

        Args:
            experiment_id: Experiment ID to promote
            profile_name: Name of the target profile
            min_sharpe: Minimum Sharpe ratio required
            max_drawdown: Maximum drawdown allowed
            min_trades: Minimum trades required
            auto_commit: Automatically commit changes to git

        Returns:
            True if promotion succeeded
        """
        logger.info(f"Promoting experiment {experiment_id} to profile '{profile_name}'")

        # Validate experiment
        if not self.validate_experiment(
            experiment_id,
            min_sharpe=min_sharpe,
            max_drawdown=max_drawdown,
            min_trades=min_trades
        ):
            logger.error("Validation failed, aborting promotion")
            return False

        # Get configuration
        config = self.get_experiment_config(experiment_id)
        if not config:
            logger.error("Failed to get experiment configuration")
            return False

        # Load existing profiles
        profiles = self._load_profiles()

        # Create promotion entry
        promotion_entry = {
            'promoted_at': datetime.now().isoformat(),
            'experiment_id': experiment_id,
            'strategy': config['strategy'],
            'symbol': config['symbol'],
            'timeframe': config['timeframe'],
            'params': config['params'],
            'performance': config['metrics'],
            'validation_criteria': {
                'min_sharpe': min_sharpe,
                'max_drawdown': max_drawdown,
                'min_trades': min_trades
            }
        }

        # Add to profiles
        profiles[profile_name] = promotion_entry

        # Save profiles
        self._save_profiles(profiles)

        logger.info(f"Successfully promoted experiment {experiment_id} to profile '{profile_name}'")
        logger.info(f"Profile saved to: {self.config_path}")

        # Commit to git if requested
        if auto_commit:
            try:
                self._git_commit(promotion_entry, profile_name)
            except Exception as e:
                logger.warning(f"Git commit failed: {e}")

        return True

    def _load_profiles(self) -> Dict[str, Any]:
        """Load existing strategy profiles."""
        if not self.config_path.exists():
            logger.info("Creating new strategy profiles file")
            return {}

        try:
            with open(self.config_path, 'r') as f:
                if self.config_path.suffix.lower() == '.yaml':
                    return yaml.safe_load(f) or {}
                else:
                    return json.load(f)

        except Exception as e:
            logger.error(f"Failed to load profiles: {e}")
            return {}

    def _save_profiles(self, profiles: Dict[str, Any]):
        """Save strategy profiles."""
        try:
            with open(self.config_path, 'w') as f:
                if self.config_path.suffix.lower() == '.yaml':
                    yaml.dump(profiles, f, default_flow_style=False, indent=2)
                else:
                    json.dump(profiles, f, indent=2)

            logger.info(f"Profiles saved to {self.config_path}")

        except Exception as e:
            logger.error(f"Failed to save profiles: {e}")
            raise

    def _git_commit(self, promotion_entry: Dict[str, Any], profile_name: str):
        """Commit promotion to git."""
        message = f"promote: {profile_name} - experiment {promotion_entry['experiment_id']}"

        subprocess.run(['git', 'add', str(self.config_path)], check=True)
        subprocess.run(['git', 'commit', '-m', message], check=True)

        logger.info(f"Committed promotion to git: {message}")

    def list_profiles(self) -> Dict[str, Any]:
        """List all strategy profiles."""
        return self._load_profiles()

    def get_profile(self, profile_name: str) -> Optional[Dict[str, Any]]:
        """Get a specific profile."""
        profiles = self._load_profiles()
        return profiles.get(profile_name)


@app.command()
def promote(
    experiment_id: int = typer.Argument(..., help="Experiment ID to promote"),
    profile: str = typer.Argument(..., help="Profile name to create/update"),
    db: str = typer.Option("data/experiments.sqlite", help="Database path"),
    config: str = typer.Option("config/strategy_profiles.yaml", help="Config path"),
    min_sharpe: float = typer.Option(1.0, help="Minimum Sharpe ratio"),
    max_drawdown: float = typer.Option(0.1, help="Maximum drawdown"),
    min_trades: int = typer.Option(5, help="Minimum trades"),
    auto_commit: bool = typer.Option(False, help="Auto-commit to git"),
    yes: bool = typer.Option(False, help="Skip confirmation")
):
    """
    Promote an experiment configuration to a production profile.

    Example:
        python promote_strategy.py 42 scalp --min-sharpe 1.5 --max-drawdown 0.08
    """
    promoter = StrategyPromoter(db_path=db, config_path=config)

    # Show experiment details
    config_data = promoter.get_experiment_config(experiment_id)
    if not config_data:
        console.print(f"[red]Error: Experiment {experiment_id} not found[/red]")
        raise typer.Exit(1)

    # Display promotion preview
    console.print("\n" + "="*60)
    console.print("[bold cyan]PROMOTION PREVIEW[/bold cyan]")
    console.print("="*60 + "\n")

    console.print(f"[white]Experiment ID: {experiment_id}[/white]")
    console.print(f"[white]Profile Name: {profile}[/white]")
    console.print(f"[white]Strategy: {config_data['strategy']}[/white]")
    console.print(f"[white]Symbol: {config_data['symbol']}[/white]")
    console.print(f"[white]Timeframe: {config_data['timeframe']}[/white]")

    console.print("\n[yellow]Parameters:[/yellow]")
    for key, value in config_data['params'].items():
        console.print(f"[white]  {key}: {value}[/white]")

    console.print("\n[yellow]Performance:[/yellow]")
    for key, value in config_data['metrics'].items():
        console.print(f"[white]  {key}: {value}[/white]")

    console.print("\n[yellow]Validation Criteria:[/yellow]")
    console.print(f"[white]  Min Sharpe: {min_sharpe}[/white]")
    console.print(f"[white]  Max Drawdown: {max_drawdown:.2%}[/white]")
    console.print(f"[white]  Min Trades: {min_trades}[/white]")

    console.print("\n" + "="*60 + "\n")

    # Confirm
    if not yes:
        confirm = typer.confirm("Proceed with promotion?")
        if not confirm:
            console.print("[yellow]Promotion cancelled[/yellow]")
            raise typer.Exit(0)

    # Promote
    try:
        success = promoter.promote(
            experiment_id=experiment_id,
            profile_name=profile,
            min_sharpe=min_sharpe,
            max_drawdown=max_drawdown,
            min_trades=min_trades,
            auto_commit=auto_commit
        )

        if success:
            console.print("\n[green]✓ Promotion successful![/green]")
            console.print(f"[green]Profile '{profile}' created in {config}[/green]")
        else:
            console.print("\n[red]✗ Promotion failed[/red]")
            raise typer.Exit(1)

    except Exception as e:
        console.print(f"\n[red]✗ Error: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def list(
    config: str = typer.Option("config/strategy_profiles.yaml", help="Config path")
):
    """List all strategy profiles."""
    promoter = StrategyPromoter(config_path=config)
    profiles = promoter.list_profiles()

    if not profiles:
        console.print("[yellow]No profiles found[/yellow]")
        raise typer.Exit(0)

    console.print("\n[bold cyan]Strategy Profiles:[/bold cyan]\n")

    for name, profile_data in profiles.items():
        console.print(f"\n[bold yellow]{name}:[/bold yellow]")
        console.print(f"[white]  Strategy: {profile_data.get('strategy', 'N/A')}[/white]")
        console.print(f"[white]  Symbol: {profile_data.get('symbol', 'N/A')}[/white]")
        console.print(f"[white]  Promoted: {profile_data.get('promoted_at', 'N/A')}[/white]")

        if 'performance' in profile_data:
            perf = profile_data['performance']
            console.print(f"  Sharpe: {perf.get('sharpe_ratio', 0):.3f}", style="white")
            console.print(f"  Return: {perf.get('total_return', 0):.2f}%", style="white")


@app.command()
def show(
    profile_name: str = typer.Argument(..., help="Profile name"),
    config: str = typer.Option("config/strategy_profiles.yaml", help="Config path")
):
    """Show detailed information about a profile."""
    promoter = StrategyPromoter(config_path=config)
    profile = promoter.get_profile(profile_name)

    if not profile:
        console.print(f"[red]Profile '{profile_name}' not found[/red]")
        raise typer.Exit(1)

    console.print("\n" + "="*60)
    console.print(f"[bold cyan]Profile: {profile_name}[/bold cyan]")
    console.print("="*60 + "\n")

    console.print(json.dumps(profile, indent=2, default=str))


if __name__ == "__main__":
    app()
