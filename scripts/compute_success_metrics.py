"""
Success Metrics Calculator
Computes trading success metrics: win rate, Sharpe ratio, drawdown, etc.
"""

import json
import math
import statistics
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np


class SuccessMetricsCalculator:
    """
    Calculate comprehensive trading success metrics.
    """

    def __init__(self, trades_file: Optional[str] = None):
        """
        Initialize calculator.
        
        Args:
            trades_file: Path to trades JSON file (optional)
        """
        self.trades_file = trades_file
        self.trades = []
        
        if trades_file and Path(trades_file).exists():
            self.load_trades(trades_file)

    def load_trades(self, trades_file: str):
        """Load trades from JSON file."""
        with open(trades_file, 'r') as f:
            self.trades = json.load(f)

    def compute_win_rate(self, trades: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """
        Calculate win rate metrics.
        
        Args:
            trades: List of trades (optional, uses self.trades if not provided)
            
        Returns:
            Win rate statistics
        """
        trades = trades or self.trades
        
        if not trades:
            return {'win_rate': 0, 'total_trades': 0, 'winning_trades': 0, 'losing_trades': 0}
        
        winning_trades = [t for t in trades if t.get('pnl', 0) > 0]
        losing_trades = [t for t in trades if t.get('pnl', 0) < 0]
        
        total_trades = len(trades)
        win_count = len(winning_trades)
        loss_count = len(losing_trades)
        
        win_rate = (win_count / total_trades * 100) if total_trades > 0 else 0
        
        return {
            'win_rate': round(win_rate, 2),
            'total_trades': total_trades,
            'winning_trades': win_count,
            'losing_trades': loss_count,
            'win_count': win_count,
            'loss_count': loss_count
        }

    def compute_sharpe_ratio(self, trades: Optional[List[Dict]] = None, 
                            risk_free_rate: float = 0.02) -> Dict[str, Any]:
        """
        Calculate Sharpe ratio.
        
        Args:
            trades: List of trades (optional)
            risk_free_rate: Annual risk-free rate (default: 2%)
            
        Returns:
            Sharpe ratio statistics
        """
        trades = trades or self.trades
        
        if not trades or len(trades) < 2:
            return {'sharpe_ratio': 0, 'avg_return': 0, 'std_return': 0}
        
        # Calculate returns
        returns = []
        for trade in trades:
            pnl = trade.get('pnl', 0)
            entry_value = trade.get('entry_value', 1000)  # Default entry value
            if entry_value > 0:
                return_pct = (pnl / entry_value) * 100
                returns.append(return_pct)
        
        if not returns or len(returns) < 2:
            return {'sharpe_ratio': 0, 'avg_return': 0, 'std_return': 0}
        
        # Calculate statistics
        avg_return = statistics.mean(returns)
        std_return = statistics.stdev(returns) if len(returns) > 1 else 0
        
        # Calculate Sharpe ratio
        if std_return == 0:
            sharpe_ratio = 0
        else:
            # Assume daily returns, annualize (252 trading days)
            annualized_return = avg_return * 252
            annualized_std = std_return * math.sqrt(252)
            sharpe_ratio = (annualized_return - risk_free_rate * 100) / annualized_std
        
        return {
            'sharpe_ratio': round(sharpe_ratio, 4),
            'avg_return': round(avg_return, 4),
            'std_return': round(std_return, 4),
            'annualized_return': round(annualized_return, 2),
            'annualized_std': round(annualized_std, 2)
        }

    def compute_max_drawdown(self, trades: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """
        Calculate maximum drawdown.
        
        Args:
            trades: List of trades (optional)
            
        Returns:
            Maximum drawdown statistics
        """
        trades = trades or self.trades
        
        if not trades:
            return {'max_drawdown': 0, 'max_drawdown_pct': 0, 'current_drawdown': 0}
        
        # Calculate cumulative PnL
        cumulative_pnl = []
        running_total = 0
        
        for trade in trades:
            running_total += trade.get('pnl', 0)
            cumulative_pnl.append(running_total)
        
        if not cumulative_pnl:
            return {'max_drawdown': 0, 'max_drawdown_pct': 0, 'current_drawdown': 0}
        
        # Calculate drawdowns
        peak = cumulative_pnl[0]
        max_drawdown = 0
        max_drawdown_pct = 0
        
        for value in cumulative_pnl:
            if value > peak:
                peak = value
            
            drawdown = peak - value
            drawdown_pct = (drawdown / peak * 100) if peak > 0 else 0
            
            if drawdown > max_drawdown:
                max_drawdown = drawdown
                max_drawdown_pct = drawdown_pct
        
        # Current drawdown
        current_value = cumulative_pnl[-1]
        current_peak = max(cumulative_pnl)
        current_drawdown = current_peak - current_value
        current_drawdown_pct = (current_drawdown / current_peak * 100) if current_peak > 0 else 0
        
        return {
            'max_drawdown': round(max_drawdown, 2),
            'max_drawdown_pct': round(max_drawdown_pct, 2),
            'current_drawdown': round(current_drawdown, 2),
            'current_drawdown_pct': round(current_drawdown_pct, 2),
            'peak_value': round(current_peak, 2),
            'current_value': round(current_value, 2)
        }

    def compute_profit_factor(self, trades: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """
        Calculate profit factor.
        
        Args:
            trades: List of trades (optional)
            
        Returns:
            Profit factor statistics
        """
        trades = trades or self.trades
        
        if not trades:
            return {'profit_factor': 0, 'gross_profit': 0, 'gross_loss': 0}
        
        winning_trades = [t for t in trades if t.get('pnl', 0) > 0]
        losing_trades = [t for t in trades if t.get('pnl', 0) < 0]
        
        gross_profit = sum(t.get('pnl', 0) for t in winning_trades)
        gross_loss = abs(sum(t.get('pnl', 0) for t in losing_trades))
        
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float('inf') if gross_profit > 0 else 0
        
        return {
            'profit_factor': round(profit_factor, 2) if profit_factor != float('inf') else '∞',
            'gross_profit': round(gross_profit, 2),
            'gross_loss': round(gross_loss, 2),
            'net_profit': round(gross_profit - gross_loss, 2)
        }

    def compute_average_metrics(self, trades: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """
        Calculate average trade metrics.
        
        Args:
            trades: List of trades (optional)
            
        Returns:
            Average metrics
        """
        trades = trades or self.trades
        
        if not trades:
            return {
                'avg_win': 0,
                'avg_loss': 0,
                'avg_trade': 0,
                'avg_holding_period': 0
            }
        
        winning_trades = [t for t in trades if t.get('pnl', 0) > 0]
        losing_trades = [t for t in trades if t.get('pnl', 0) < 0]
        
        avg_win = statistics.mean([t.get('pnl', 0) for t in winning_trades]) if winning_trades else 0
        avg_loss = statistics.mean([t.get('pnl', 0) for t in losing_trades]) if losing_trades else 0
        avg_trade = statistics.mean([t.get('pnl', 0) for t in trades]) if trades else 0
        
        # Holding period (if available)
        holding_periods = []
        for trade in trades:
            entry_time = trade.get('entry_time')
            exit_time = trade.get('exit_time')
            if entry_time and exit_time:
                try:
                    entry_dt = datetime.fromisoformat(entry_time)
                    exit_dt = datetime.fromisoformat(exit_time)
                    holding_period = (exit_dt - entry_dt).total_seconds() / 60  # minutes
                    holding_periods.append(holding_period)
                except:
                    pass
        
        avg_holding_period = statistics.mean(holding_periods) if holding_periods else 0
        
        return {
            'avg_win': round(avg_win, 2),
            'avg_loss': round(avg_loss, 2),
            'avg_trade': round(avg_trade, 2),
            'avg_holding_period_minutes': round(avg_holding_period, 2)
        }

    def compute_consecutive_metrics(self, trades: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """
        Calculate consecutive wins/losses.
        
        Args:
            trades: List of trades (optional)
            
        Returns:
            Consecutive metrics
        """
        trades = trades or self.trades
        
        if not trades:
            return {
                'max_consecutive_wins': 0,
                'max_consecutive_losses': 0,
                'current_consecutive': 0,
                'current_consecutive_type': 'none'
            }
        
        max_wins = 0
        max_losses = 0
        current_streak = 0
        current_type = 'none'
        
        # Count consecutive wins/losses
        for trade in trades:
            pnl = trade.get('pnl', 0)
            
            if pnl > 0:  # Win
                if current_type == 'win':
                    current_streak += 1
                else:
                    current_streak = 1
                    current_type = 'win'
                
                max_wins = max(max_wins, current_streak)
                
            elif pnl < 0:  # Loss
                if current_type == 'loss':
                    current_streak += 1
                else:
                    current_streak = 1
                    current_type = 'loss'
                
                max_losses = max(max_losses, current_streak)
        
        return {
            'max_consecutive_wins': max_wins,
            'max_consecutive_losses': max_losses,
            'current_consecutive': current_streak,
            'current_consecutive_type': current_type
        }

    def compute_all_metrics(self, trades: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """
        Compute all success metrics.
        
        Args:
            trades: List of trades (optional)
            
        Returns:
            Complete metrics report
        """
        trades = trades or self.trades
        
        if not trades:
            return {'error': 'No trades available'}
        
        # Calculate all metrics
        win_rate_metrics = self.compute_win_rate(trades)
        sharpe_metrics = self.compute_sharpe_ratio(trades)
        drawdown_metrics = self.compute_max_drawdown(trades)
        profit_factor_metrics = self.compute_profit_factor(trades)
        avg_metrics = self.compute_average_metrics(trades)
        consecutive_metrics = self.compute_consecutive_metrics(trades)
        
        # Time period
        if trades:
            timestamps = []
            for trade in trades:
                entry_time = trade.get('entry_time')
                if entry_time:
                    try:
                        timestamps.append(datetime.fromisoformat(entry_time))
                    except:
                        pass
            
            if timestamps:
                start_date = min(timestamps)
                end_date = max(timestamps)
                duration_days = (end_date - start_date).days
            else:
                start_date = None
                end_date = None
                duration_days = 0
        else:
            start_date = None
            end_date = None
            duration_days = 0
        
        # Compile complete report
        report = {
            'summary': {
                'total_trades': len(trades),
                'start_date': start_date.isoformat() if start_date else None,
                'end_date': end_date.isoformat() if end_date else None,
                'duration_days': duration_days
            },
            'win_rate': win_rate_metrics,
            'sharpe_ratio': sharpe_metrics,
            'max_drawdown': drawdown_metrics,
            'profit_factor': profit_factor_metrics,
            'averages': avg_metrics,
            'consecutive': consecutive_metrics,
            'total_pnl': round(sum(t.get('pnl', 0) for t in trades), 2),
            'generated_at': datetime.now().isoformat()
        }
        
        return report

    def export_report(self, trades: Optional[List[Dict]] = None, 
                      output_file: Optional[str] = None) -> str:
        """
        Export metrics report to JSON file.
        
        Args:
            trades: List of trades (optional)
            output_file: Output file path (optional)
            
        Returns:
            Path to output file
        """
        report = self.compute_all_metrics(trades)
        
        if not output_file:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = f'success_metrics_{timestamp}.json'
        
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        return output_file

    def print_report(self, trades: Optional[List[Dict]] = None):
        """
        Print formatted metrics report.
        
        Args:
            trades: List of trades (optional)
        """
        report = self.compute_all_metrics(trades)
        
        if 'error' in report:
            print(f"Error: {report['error']}")
            return
        
        print("\n" + "="*70)
        print("Trading Success Metrics Report")
        print("="*70)
        
        # Summary
        summary = report['summary']
        print(f"\n📊 Summary:")
        print(f"   Total Trades: {summary['total_trades']}")
        if summary['start_date']:
            print(f"   Period: {summary['start_date'][:10]} to {summary['end_date'][:10]}")
            print(f"   Duration: {summary['duration_days']} days")
        
        # Win Rate
        wr = report['win_rate']
        print(f"\n🎯 Win Rate:")
        print(f"   {wr['win_rate']}% ({wr['winning_trades']}/{wr['total_trades']})")
        
        # Sharpe Ratio
        sr = report['sharpe_ratio']
        print(f"\n📈 Sharpe Ratio:")
        print(f"   {sr['sharpe_ratio']}")
        print(f"   Avg Return: {sr['avg_return']}%")
        print(f"   Return Std: {sr['std_return']}%")
        
        # Max Drawdown
        dd = report['max_drawdown']
        print(f"\n📉 Maximum Drawdown:")
        print(f"   {dd['max_drawdown_pct']}% (${dd['max_drawdown']})")
        print(f"   Current: {dd['current_drawdown_pct']}% (${dd['current_drawdown']})")
        
        # Profit Factor
        pf = report['profit_factor']
        print(f"\n💰 Profit Factor:")
        print(f"   {pf['profit_factor']}")
        print(f"   Gross Profit: ${pf['gross_profit']}")
        print(f"   Gross Loss: ${pf['gross_loss']}")
        
        # Averages
        avg = report['averages']
        print(f"\n📊 Averages:")
        print(f"   Avg Win: ${avg['avg_win']}")
        print(f"   Avg Loss: ${avg['avg_loss']}")
        print(f"   Avg Trade: ${avg['avg_trade']}")
        if avg['avg_holding_period_minutes'] > 0:
            print(f"   Avg Holding: {avg['avg_holding_period_minutes']:.1f} minutes")
        
        # Consecutive
        consec = report['consecutive']
        print(f"\n🔥 Consecutive:")
        print(f"   Max Wins: {consec['max_consecutive_wins']}")
        print(f"   Max Losses: {consec['max_consecutive_losses']}")
        if consec['current_consecutive'] > 0:
            print(f"   Current: {consec['current_consecutive']} {consec['current_consecutive_type']}(s)")
        
        # Total PnL
        print(f"\n💵 Total P&L:")
        print(f"   ${report['total_pnl']}")
        
        print("\n" + "="*70)
        print(f"Generated: {report['generated_at']}")
        print("="*70 + "\n")


def generate_sample_trades(num_trades: int = 100) -> List[Dict]:
    """
    Generate sample trades for testing.
    
    Args:
        num_trades: Number of trades to generate
        
    Returns:
        List of sample trades
    """
    import random
    random.seed(42)  # For reproducible results
    
    trades = []
    pnl_running = 0
    
    for i in range(num_trades):
        # Random PnL (more wins than losses)
        is_win = random.random() > 0.4  # 60% win rate
        if is_win:
            pnl = random.uniform(10, 100)
        else:
            pnl = random.uniform(-100, -10)
        
        pnl_running += pnl
        
        # Random entry time
        base_time = datetime.now() - timedelta(days=30)
        entry_time = base_time + timedelta(minutes=i * 60)
        exit_time = entry_time + timedelta(minutes=random.randint(5, 60))
        
        trade = {
            'trade_id': f'T{i+1:04d}',
            'symbol': 'BTCUSDT' if random.random() > 0.5 else 'ETHUSDT',
            'side': 'LONG' if is_win else 'SHORT',
            'pnl': round(pnl, 2),
            'entry_price': round(random.uniform(40000, 50000), 2),
            'exit_price': round(random.uniform(40000, 50000), 2),
            'size': round(random.uniform(0.1, 1.0), 2),
            'entry_value': round(random.uniform(1000, 10000), 2),
            'entry_time': entry_time.isoformat(),
            'exit_time': exit_time.isoformat(),
            'duration_minutes': (exit_time - entry_time).total_seconds() / 60
        }
        
        trades.append(trade)
    
    return trades


def main():
    """Main function for CLI usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Compute trading success metrics')
    parser.add_argument('--trades', type=str, help='Path to trades JSON file')
    parser.add_argument('--output', type=str, help='Output JSON file')
    parser.add_argument('--sample', type=int, help='Generate N sample trades')
    parser.add_argument('--export', action='store_true', help='Export to JSON')
    parser.add_argument('--no-print', action='store_true', help='Do not print report')
    
    args = parser.parse_args()
    
    # Create calculator
    if args.trades:
        calculator = SuccessMetricsCalculator(args.trades)
    elif args.sample:
        trades = generate_sample_trades(args.sample)
        calculator = SuccessMetricsCalculator()
        calculator.trades = trades
    else:
        # Try to load default trades file
        default_file = 'trades.json'
        if Path(default_file).exists():
            calculator = SuccessMetricsCalculator(default_file)
        else:
            # Generate sample trades
            trades = generate_sample_trades(50)
            calculator = SuccessMetricsCalculator()
            calculator.trades = trades
            print("⚠️  No trades file found. Using sample data (50 trades).\n")
    
    # Print report
    if not args.no_print:
        calculator.print_report()
    
    # Export to JSON
    if args.export or args.output:
        output_file = args.output or 'success_metrics.json'
        exported_file = calculator.export_report(output_file=output_file)
        print(f"✅ Report exported to: {exported_file}")


if __name__ == '__main__':
    main()
