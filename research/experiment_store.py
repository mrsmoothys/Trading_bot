"""
Experiment Store
SQLite-based storage for experiment results and metadata.
"""

import sqlite3
import json
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from loguru import logger
import hashlib

from backtesting.service import BacktestConfig, BacktestResult


class ExperimentStore:
    """
    SQLite-based storage for experiment results.

    Schema:
    - experiments: Stores successful backtest results
    - errors: Stores failed experiment attempts
    - metadata: Stores additional experiment metadata
    """

    def __init__(self, db_path: str):
        """
        Initialize experiment store.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self._ensure_db_dir()
        self._init_database()
        logger.info(f"ExperimentStore initialized: {db_path}")

    def _ensure_db_dir(self):
        """Ensure database directory exists."""
        db_file = Path(self.db_path)
        db_file.parent.mkdir(parents=True, exist_ok=True)

    def _init_database(self):
        """Initialize database schema."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Create experiments table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS experiments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                config_hash TEXT UNIQUE NOT NULL,
                timestamp TEXT NOT NULL,
                symbol TEXT NOT NULL,
                timeframe TEXT NOT NULL,
                strategy TEXT NOT NULL,
                start TEXT NOT NULL,
                end TEXT NOT NULL,
                initial_capital REAL NOT NULL,
                final_capital REAL NOT NULL,
                total_return REAL NOT NULL,
                total_return_pct REAL NOT NULL,
                annualized_return REAL NOT NULL,
                max_drawdown REAL NOT NULL,
                sharpe_ratio REAL NOT NULL,
                win_rate REAL NOT NULL,
                profit_factor REAL NOT NULL,
                total_trades INTEGER NOT NULL,
                winning_trades INTEGER NOT NULL,
                losing_trades INTEGER NOT NULL,
                avg_win REAL NOT NULL,
                avg_loss REAL NOT NULL,
                config_json TEXT NOT NULL,
                result_json TEXT NOT NULL
            )
        """)

        # Create index on config_hash for fast lookups
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_experiments_config_hash
            ON experiments(config_hash)
        """)

        # Create index on strategy for filtering
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_experiments_strategy
            ON experiments(strategy)
        """)

        # Create index on symbol for filtering
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_experiments_symbol
            ON experiments(symbol)
        """)

        # Create errors table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS errors (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                config_hash TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                symbol TEXT NOT NULL,
                timeframe TEXT NOT NULL,
                strategy TEXT NOT NULL,
                error_message TEXT NOT NULL,
                config_json TEXT NOT NULL
            )
        """)

        # Create index on config_hash for errors
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_errors_config_hash
            ON errors(config_hash)
        """)

        conn.commit()
        conn.close()

    def log_experiment(self, config: BacktestConfig, result: BacktestResult):
        """
        Log a successful experiment.

        Args:
            config: BacktestConfig that was run
            result: BacktestResult from the run
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            cursor.execute("""
                INSERT OR REPLACE INTO experiments (
                    config_hash, timestamp, symbol, timeframe, strategy,
                    start, end, initial_capital, final_capital,
                    total_return, total_return_pct, annualized_return,
                    max_drawdown, sharpe_ratio, win_rate, profit_factor,
                    total_trades, winning_trades, losing_trades,
                    avg_win, avg_loss, config_json, result_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                result.config_hash,
                result.timestamp.isoformat(),
                result.symbol,
                result.timeframe,
                result.strategy,
                result.start.isoformat(),
                result.end.isoformat(),
                result.initial_capital,
                result.final_capital,
                result.total_return,
                result.total_return_pct,
                result.annualized_return,
                result.max_drawdown,
                result.sharpe_ratio,
                result.win_rate,
                result.profit_factor,
                result.total_trades,
                result.winning_trades,
                result.losing_trades,
                result.avg_win,
                result.avg_loss,
                config.to_json(),
                result.to_json()
            ))

            conn.commit()
            logger.debug(f"Logged experiment: {config.symbol} {config.strategy}")

        except sqlite3.Error as e:
            logger.error(f"Database error logging experiment: {e}")
        finally:
            conn.close()

    def log_error(self, config: BacktestConfig, error_message: str):
        """
        Log a failed experiment.

        Args:
            config: BacktestConfig that failed
            error_message: Error message from failure
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            cursor.execute("""
                INSERT INTO errors (
                    config_hash, timestamp, symbol, timeframe, strategy,
                    error_message, config_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                config.get_config_hash(),
                datetime.now().isoformat(),
                config.symbol,
                config.timeframe,
                config.strategy,
                error_message,
                config.to_json()
            ))

            conn.commit()
            logger.debug(f"Logged error: {config.symbol} {config.strategy}")

        except sqlite3.Error as e:
            logger.error(f"Database error logging error: {e}")
        finally:
            conn.close()

    def config_exists(self, config_hash: str) -> bool:
        """
        Check if a configuration has already been run.

        Args:
            config_hash: Hash of the configuration

        Returns:
            True if configuration exists, False otherwise
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            cursor.execute("""
                SELECT COUNT(*) FROM experiments WHERE config_hash = ?
            """, (config_hash,))

            count = cursor.fetchone()[0]
            return count > 0

        except sqlite3.Error as e:
            logger.error(f"Database error checking config: {e}")
            return False
        finally:
            conn.close()

    def get_experiments(
        self,
        metric: str = "sharpe_ratio",
        top_n: Optional[int] = None,
        filter_criteria: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve experiment results.

        Args:
            metric: Metric to sort by (default: sharpe_ratio)
            top_n: Number of top results to return
            filter_criteria: Optional filters (symbol, strategy, etc.)

        Returns:
            List of experiment result dictionaries
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            # Build query
            query = """
                SELECT * FROM experiments
            """
            params = []

            # Add filters
            if filter_criteria:
                conditions = []
                for key, value in filter_criteria.items():
                    if key in ['symbol', 'timeframe', 'strategy']:
                        conditions.append(f"{key} = ?")
                        params.append(value)

                if conditions:
                    query += " WHERE " + " AND ".join(conditions)

            # Add sorting
            valid_metrics = [
                'total_return', 'total_return_pct', 'sharpe_ratio',
                'max_drawdown', 'win_rate', 'profit_factor', 'total_trades'
            ]

            if metric not in valid_metrics:
                logger.warning(f"Unknown metric {metric}, using sharpe_ratio")
                metric = "sharpe_ratio"

            query += f" ORDER BY {metric} DESC"

            # Add limit
            if top_n is not None:
                query += " LIMIT ?"
                params.append(top_n)

            cursor.execute(query, params)
            columns = [description[0] for description in cursor.description]

            results = []
            for row in cursor.fetchall():
                results.append(dict(zip(columns, row)))

            return results

        except sqlite3.Error as e:
            logger.error(f"Database error retrieving experiments: {e}")
            return []
        finally:
            conn.close()

    def get_best_experiments(
        self,
        metric: str = "sharpe_ratio",
        top_n: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Get top N performing experiments.

        Args:
            metric: Metric to optimize
            top_n: Number of top experiments to return

        Returns:
            List of top experiments
        """
        return self.get_experiments(metric=metric, top_n=top_n)

    def get_experiments_by_strategy(self, strategy: str) -> List[Dict[str, Any]]:
        """
        Get all experiments for a specific strategy.

        Args:
            strategy: Strategy name

        Returns:
            List of experiments for the strategy
        """
        return self.get_experiments(filter_criteria={'strategy': strategy})

    def get_experiments_by_symbol(self, symbol: str) -> List[Dict[str, Any]]:
        """
        Get all experiments for a specific symbol.

        Args:
            symbol: Trading symbol

        Returns:
            List of experiments for the symbol
        """
        return self.get_experiments(filter_criteria={'symbol': symbol})

    def get_errors(self) -> List[Dict[str, Any]]:
        """
        Get all failed experiments.

        Returns:
            List of error records
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            cursor.execute("""
                SELECT * FROM errors ORDER BY timestamp DESC
            """)

            columns = [description[0] for description in cursor.description]
            results = []

            for row in cursor.fetchall():
                results.append(dict(zip(columns, row)))

            return results

        except sqlite3.Error as e:
            logger.error(f"Database error retrieving errors: {e}")
            return []
        finally:
            conn.close()

    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics of all experiments.

        Returns:
            Dictionary with summary statistics
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            # Total experiments
            cursor.execute("SELECT COUNT(*) FROM experiments")
            total_experiments = cursor.fetchone()[0]

            # Total errors
            cursor.execute("SELECT COUNT(*) FROM errors")
            total_errors = cursor.fetchone()[0]

            # Strategies used
            cursor.execute("""
                SELECT DISTINCT strategy FROM experiments
            """)
            strategies = [row[0] for row in cursor.fetchall()]

            # Symbols tested
            cursor.execute("""
                SELECT DISTINCT symbol FROM experiments
            """)
            symbols = [row[0] for row in cursor.fetchall()]

            # Best results by metric
            best_results = {}

            for metric in ['total_return', 'sharpe_ratio', 'win_rate', 'profit_factor']:
                cursor.execute(f"""
                    SELECT symbol, strategy, {metric}
                    FROM experiments
                    ORDER BY {metric} DESC
                    LIMIT 1
                """)
                row = cursor.fetchone()
                if row:
                    best_results[metric] = {
                        'symbol': row[0],
                        'strategy': row[1],
                        'value': row[2]
                    }

            # Average metrics
            cursor.execute("""
                SELECT
                    AVG(total_return) as avg_return,
                    AVG(sharpe_ratio) as avg_sharpe,
                    AVG(win_rate) as avg_win_rate,
                    AVG(max_drawdown) as avg_drawdown
                FROM experiments
            """)
            row = cursor.fetchone()
            avg_metrics = {
                'avg_return': row[0] if row[0] else 0,
                'avg_sharpe': row[1] if row[1] else 0,
                'avg_win_rate': row[2] if row[2] else 0,
                'avg_drawdown': row[3] if row[3] else 0
            }

            return {
                'total_experiments': total_experiments,
                'total_errors': total_errors,
                'strategies': strategies,
                'symbols': symbols,
                'best_results': best_results,
                'average_metrics': avg_metrics,
                'success_rate': total_experiments / (total_experiments + total_errors) if (total_experiments + total_errors) > 0 else 0
            }

        except sqlite3.Error as e:
            logger.error(f"Database error getting summary: {e}")
            return {}
        finally:
            conn.close()

    def export_to_json(self, filepath: str, filter_criteria: Optional[Dict[str, Any]] = None):
        """
        Export experiments to JSON file.

        Args:
            filepath: Path to output JSON file
            filter_criteria: Optional filters
        """
        experiments = self.get_experiments(filter_criteria=filter_criteria)

        with open(filepath, 'w') as f:
            json.dump(experiments, f, indent=2, default=str)

        logger.info(f"Exported {len(experiments)} experiments to {filepath}")

    def clear_experiments(self, confirm: bool = False):
        """
        Clear all experiment data.

        Args:
            confirm: Must be True to actually clear
        """
        if not confirm:
            logger.warning("clear_experiments() requires confirm=True")
            return

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            cursor.execute("DELETE FROM experiments")
            cursor.execute("DELETE FROM errors")
            conn.commit()

            logger.warning("Cleared all experiment data")

        except sqlite3.Error as e:
            logger.error(f"Database error clearing experiments: {e}")
        finally:
            conn.close()

    def get_config_by_hash(self, config_hash: str) -> Optional[Dict[str, Any]]:
        """
        Get experiment by config hash.

        Args:
            config_hash: Configuration hash

        Returns:
            Experiment dict or None if not found
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            cursor.execute("""
                SELECT * FROM experiments WHERE config_hash = ?
            """, (config_hash,))

            row = cursor.fetchone()
            if row:
                columns = [description[0] for description in cursor.description]
                return dict(zip(columns, row))

            return None

        except sqlite3.Error as e:
            logger.error(f"Database error retrieving config: {e}")
            return None
        finally:
            conn.close()
