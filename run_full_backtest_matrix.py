#!/usr/bin/env python3
"""
Full Backtest Matrix Runner
Runs all combinations of strategies and timeframes for comprehensive analysis.
"""

import os
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import pandas as pd

# Add to path
sys.path.insert(0, '/Users/mrsmoothy/Downloads/Trading_bot')

# Set backtest mode to prevent sample data fallback
os.environ['BACKTEST_MODE'] = 'true'

from backtesting.service import BacktestConfig, run_backtest, BacktestResult
from loguru import logger
import json


# Configuration
SYMBOL = "BTCUSDT"
START_DATE = datetime(2024, 1, 1)
END_DATE = datetime.now()
INITIAL_CAPITAL = 10000

STRATEGIES = [
    ('convergence', 'Convergence Strategy'),
    ('scalp_15m_4h', 'Scalp 15m/4h'),
    ('ma_crossover', 'MA Crossover'),
    ('rsi_divergence', 'RSI Divergence')
]

TIMEFRAMES = [
    ('1m', '1 Minute'),
    ('5m', '5 Minutes'),
    ('15m', '15 Minutes'),
    ('1h', '1 Hour'),
    ('4h', '4 Hours'),
    ('1d', '1 Day')
]


def format_result_summary(result: BacktestResult) -> Dict:
    """Format result for reporting."""
    return {
        'total_return_pct': round(result.total_return_pct, 2),
        'total_trades': result.total_trades,
        'win_rate': round(result.win_rate * 100, 1),
        'max_drawdown': round(result.max_drawdown, 2),
        'sharpe_ratio': round(result.sharpe_ratio, 2),
        'profit_factor': round(result.profit_factor, 2),
        'final_capital': round(result.final_capital, 2),
        'initial_capital': round(result.initial_capital, 2),
        'winning_trades': result.winning_trades,
        'losing_trades': result.losing_trades,
        'avg_win': round(result.avg_win, 2),
        'avg_loss': round(result.avg_loss, 2),
        'equity_curve': result.equity_curve,
        'trades': result.trades
    }


def run_all_backtests() -> Dict:
    """Run all backtest combinations and return results."""
    logger.info("=" * 80)
    logger.info("STARTING FULL BACKTEST MATRIX")
    logger.info(f"Symbol: {SYMBOL}")
    logger.info(f"Period: {START_DATE.date()} to {END_DATE.date()}")
    logger.info(f"Strategies: {len(STRATEGIES)}")
    logger.info(f"Timeframes: {len(TIMEFRAMES)}")
    logger.info(f"Total Combinations: {len(STRATEGIES) * len(TIMEFRAMES)}")
    logger.info("=" * 80)

    results = {}
    successful = 0
    failed = 0

    for strategy_id, strategy_name in STRATEGIES:
        results[strategy_id] = {}

        for timeframe_id, timeframe_name in TIMEFRAMES:
            logger.info(f"\n[{successful + failed + 1}/{len(STRATEGIES) * len(TIMEFRAMES)}] Running: {strategy_name} on {timeframe_name}")

            try:
                config = BacktestConfig(
                    symbol=SYMBOL,
                    timeframe=timeframe_id,
                    start=START_DATE,
                    end=END_DATE,
                    strategy=strategy_id,
                    params={},
                    initial_capital=INITIAL_CAPITAL
                )

                result = run_backtest(config)
                results[strategy_id][timeframe_id] = {
                    'success': True,
                    'summary': format_result_summary(result),
                    'error': None,
                    'config': {
                        'symbol': SYMBOL,
                        'timeframe': timeframe_id,
                        'strategy': strategy_id,
                        'start': START_DATE.isoformat(),
                        'end': END_DATE.isoformat(),
                        'initial_capital': INITIAL_CAPITAL
                    }
                }

                summary = results[strategy_id][timeframe_id]['summary']
                logger.info(f"✓ SUCCESS: {summary['total_return_pct']:+.2f}%, {summary['total_trades']} trades, {summary['win_rate']}% win rate")
                successful += 1

            except Exception as e:
                logger.error(f"✗ FAILED: {str(e)}")
                results[strategy_id][timeframe_id] = {
                    'success': False,
                    'summary': None,
                    'error': str(e),
                    'config': {
                        'symbol': SYMBOL,
                        'timeframe': timeframe_id,
                        'strategy': strategy_id,
                        'start': START_DATE.isoformat(),
                        'end': END_DATE.isoformat(),
                        'initial_capital': INITIAL_CAPITAL
                    }
                }
                failed += 1

    logger.info("\n" + "=" * 80)
    logger.info("BACKTEST MATRIX COMPLETE")
    logger.info(f"Successful: {successful}/{successful + failed}")
    logger.info(f"Failed: {failed}/{successful + failed}")
    logger.info("=" * 80)

    return results


def generate_report(results: Dict) -> str:
    """Generate markdown report from results."""
    report = []
    report.append("# Backtest Matrix Results Report")
    report.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"\n**Period:** {START_DATE.date()} to {END_DATE.date()}")
    report.append(f"**Symbol:** {SYMBOL}")
    report.append(f"**Initial Capital:** ${INITIAL_CAPITAL:,}")
    report.append(f"\n**Total Combinations:** {len(STRATEGIES) * len(TIMEFRAMES)}")

    # Summary table
    report.append("\n## Summary Table")
    report.append("\n| Strategy | Timeframe | Return % | Trades | Win Rate | Sharpe | Max DD |")
    report.append("|----------|-----------|----------|--------|----------|--------|--------|")

    for strategy_id, strategy_name in STRATEGIES:
        report.append(f"\n### {strategy_name}")

        for timeframe_id, timeframe_name in TIMEFRAMES:
            if strategy_id in results and timeframe_id in results[strategy_id]:
                r = results[strategy_id][timeframe_id]
                if r['success']:
                    s = r['summary']
                    report.append(f"| {strategy_name} | {timeframe_name} | **{s['total_return_pct']:+.2f}%** | {s['total_trades']} | {s['win_rate']}% | {s['sharpe_ratio']} | {s['max_drawdown']:.2f}% |")
                else:
                    report.append(f"| {strategy_name} | {timeframe_name} | **FAILED** | - | - | - | - |")

    # Best performers
    report.append("\n## Best Performers")
    report.append("\n### Highest Returns")
    best_returns = []
    for strategy_id, strategy_name in STRATEGIES:
        for timeframe_id, timeframe_name in TIMEFRAMES:
            if strategy_id in results and timeframe_id in results[strategy_id]:
                r = results[strategy_id][timeframe_id]
                if r['success']:
                    best_returns.append((strategy_name, timeframe_name, r['summary']['total_return_pct'], r['summary']['total_trades']))

    best_returns.sort(key=lambda x: x[2], reverse=True)
    for i, (strategy, tf, ret, trades) in enumerate(best_returns[:5], 1):
        report.append(f"{i}. **{ret:+.2f}%** - {strategy} on {tf} ({trades} trades)")

    report.append("\n### Highest Win Rates")
    best_winrates = []
    for strategy_id, strategy_name in STRATEGIES:
        for timeframe_id, timeframe_name in TIMEFRAMES:
            if strategy_id in results and timeframe_id in results[strategy_id]:
                r = results[strategy_id][timeframe_id]
                if r['success'] and r['summary']['total_trades'] > 10:
                    best_winrates.append((strategy_name, timeframe_name, r['summary']['win_rate'], r['summary']['total_trades']))

    best_winrates.sort(key=lambda x: x[2], reverse=True)
    for i, (strategy, tf, wr, trades) in enumerate(best_winrates[:5], 1):
        report.append(f"{i}. **{wr:.1f}%** - {strategy} on {tf} ({trades} trades)")

    # Failed runs
    failed_count = 0
    report.append("\n## Failed Runs")
    for strategy_id, strategy_name in STRATEGIES:
        for timeframe_id, timeframe_name in TIMEFRAMES:
            if strategy_id in results and timeframe_id in results[strategy_id]:
                r = results[strategy_id][timeframe_id]
                if not r['success']:
                    failed_count += 1
                    report.append(f"\n- **{strategy_name} on {timeframe_name}**")
                    report.append(f"  - Error: {r['error']}")

    if failed_count == 0:
        report.append("\n✓ All backtests completed successfully!")

    # Data integrity note
    report.append("\n## Data Integrity")
    report.append("\n**IMPORTANT:** All backtests were run with LIVE DATA only.")
    report.append("Sample data fallback was disabled (BACKTEST_MODE=true).")

    # Chat reliability note
    report.append("\n## Chat Reliability")
    report.append(f"\n**CHAT_RESPONSE_TIMEOUT** set to 60 seconds (up from 8 seconds).")
    report.append("This allows DeepSeek AI more time to generate comprehensive responses.")

    return "\n".join(report)


if __name__ == "__main__":
    # Setup logging
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level="INFO"
    )

    # Run all backtests
    results = run_all_backtests()

    # Save results to JSON
    with open('/Users/mrsmoothy/Downloads/Trading_bot/backtest_matrix_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)

    # Generate report
    report = generate_report(results)

    # Save report
    with open('/Users/mrsmoothy/Downloads/Trading_bot/BACKTEST_AND_CHAT_REPORT.md', 'w') as f:
        f.write(report)

    logger.info("\n✓ Results saved to: backtest_matrix_results.json")
    logger.info("✓ Report saved to: BACKTEST_AND_CHAT_REPORT.md")
    logger.info("\nDone!")
