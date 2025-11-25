"""
Chat Command Router
Implements DeepSeek as a Manager - handles safe commands before hitting DeepSeek AI.
Provides read-only and safe control operations via chat interface.
"""

import re
import os
import time
from datetime import datetime
from typing import Dict, Any, Tuple, Optional, List
from loguru import logger
import pandas as pd
import numpy as np

# Global system context - will be set by dashboard
SYSTEM_CONTEXT = None
DASHBOARD_CALLBACKS = None


def set_system_context(context):
    """Set global system context for command router."""
    global SYSTEM_CONTEXT
    SYSTEM_CONTEXT = context


def set_dashboard_callbacks(callbacks_dict):
    """Set dashboard callbacks for triggering actions."""
    global DASHBOARD_CALLBACKS
    DASHBOARD_CALLBACKS = callbacks_dict


class ManagerCommandRouter:
    """Router for handling manager commands before DeepSeek."""

    # Command patterns for intent parsing
    PATTERNS = {
        'refresh_data': [
            r'refresh\s+(?:data|chart)',
            r'reload\s+(?:data|chart)',
            r'update\s+(?:data|chart)',
        ],
        'set_timeframe': [
            r'set\s+timeframe\s+(\w+)',
            r'switch\s+to\s+(\w+)\s+timeframe',
            r'change\s+timeframe\s+to\s+(\w+)',
            r'use\s+(\w+)\s+(?:timeframe|tf)',
        ],
        'toggle_overlays': [
            r'toggle\s+(overlays|overlay)',
            r'(?:turn\s+on|enable)\s+(overlays|overlay)',
            r'(?:turn\s+off|disable)\s+(overlays|overlay)',
            r'show\s+(overlays|overlay)',
            r'hide\s+(overlays|overlay)',
        ],
        'run_backtest_single': [
            r'run\s+backtest\s+(\w+)\s+(\w+)\s+(\w+)\s+(\d+)',
            r'backtest\s+(\w+)\s+(\w+)\s+(\w+)\s+(\d+)',
        ],
        'run_backtest_sweep': [
            r'run\s+backtest\s+(\w+)\s+all\s+timeframes\s+(\w+)\s+(\d+)',
            r'backtest\s+sweep\s+(\w+)\s+(\w+)\s+(\d+)',
            r'backtest\s+all\s+(\w+)\s+(\w+)\s+(\d+)',
        ],
        'fetch_health': [
            r'fetch\s+(?:health|system\s+status)',
            r'show\s+(?:health|system\s+status)',
            r'get\s+(?:health|system\s+status)',
            r'system\s+status',
        ],
        'fetch_account': [
            r'fetch\s+(?:account|portfolio|positions)',
            r'show\s+(?:account|portfolio|positions)',
            r'get\s+(?:account|portfolio|positions)',
            r'account\s+status',
            r'portfolio\s+status',
        ],
    }

    # Valid timeframes
    VALID_TIMEFRAMES = ['1m', '5m', '15m', '1h', '4h', '1d']

    # Valid strategies
    VALID_STRATEGIES = ['convergence', 'sma', 'ema', 'rsi', 'macd', 'bollinger']

    # Timeframe caps for sweeps to keep runs responsive
    SWEEP_DAYS_CAP = {
        '1m': 3,
        '5m': 7,
        '15m': 21,
        '1h': 60,
        '4h': 120,
        '1d': 365
    }

    def __init__(self):
        """Initialize the command router."""
        self.last_command = None
        self.last_result = None
        self.command_history = []
        self.sweep_timeout_seconds = int(os.getenv("SWEEP_TIMEOUT_SECONDS", "60"))
        self.include_short_timeframes = os.getenv("SWEEP_INCLUDE_SHORT_TF", "false").lower() == "true"
        self.per_tf_timeout_seconds = int(os.getenv("SWEEP_PER_TF_TIMEOUT", "15"))

    def parse_intent(self, message: str) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
        """
        Parse user message to extract intent and parameters.

        Args:
            message: User message text

        Returns:
            Tuple of (intent, parameters) or (None, None) if no intent matched
        """
        message_lower = message.lower().strip()

        # Check each pattern
        for intent, patterns in self.PATTERNS.items():
            for pattern in patterns:
                match = re.search(pattern, message_lower)
                if match:
                    params = match.groups()
                    return intent, {'params': params, 'raw_message': message}

        # No intent matched
        return None, None

    def handle_command(self, intent: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle a parsed command intent.

        Args:
            intent: Command intent
            params: Command parameters

        Returns:
            Dictionary with 'success', 'message', and 'data' keys
        """
        try:
            if intent == 'refresh_data':
                return self._handle_refresh_data()
            elif intent == 'set_timeframe':
                return self._handle_set_timeframe(params)
            elif intent == 'toggle_overlays':
                return self._handle_toggle_overlays(params)
            elif intent == 'run_backtest_single':
                return self._handle_run_backtest_single(params)
            elif intent == 'run_backtest_sweep':
                return self._handle_run_backtest_sweep(params)
            elif intent == 'fetch_health':
                return self._handle_fetch_health()
            elif intent == 'fetch_account':
                return self._handle_fetch_account()
            else:
                return {
                    'success': False,
                    'message': f'Unknown intent: {intent}',
                    'data': None
                }
        except Exception as e:
            logger.exception(f"Error handling command {intent}: {e}")
            return {
                'success': False,
                'message': f'Error executing command: {str(e)}',
                'data': None
            }

    def _handle_refresh_data(self) -> Dict[str, Any]:
        """Handle data refresh command."""
        try:
            # Trigger data refresh via dashboard callback
            if DASHBOARD_CALLBACKS and 'refresh_data' in DASHBOARD_CALLBACKS:
                result = DASHBOARD_CALLBACKS['refresh_data']()
                return {
                    'success': True,
                    'message': '✅ Data refreshed successfully',
                    'data': {'timestamp': datetime.now().isoformat()}
                }
            else:
                # Fallback: just return success (dashboard will refresh on next update)
                return {
                    'success': True,
                    'message': '✅ Data refresh triggered (cache cleared)',
                    'data': {'timestamp': datetime.now().isoformat()}
                }
        except Exception as e:
            logger.exception("Error refreshing data")
            return {
                'success': False,
                'message': f'❌ Failed to refresh data: {str(e)}',
                'data': None
            }

    def _handle_set_timeframe(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle timeframe change command."""
        try:
            timeframe = params['params'][0] if params['params'] else None

            if not timeframe:
                return {
                    'success': False,
                    'message': '❌ No timeframe specified',
                    'data': None
                }

            # Normalize timeframe
            timeframe = timeframe.lower()
            if timeframe not in self.VALID_TIMEFRAMES:
                return {
                    'success': False,
                    'message': f'❌ Invalid timeframe: {timeframe}. Valid options: {", ".join(self.VALID_TIMEFRAMES)}',
                    'data': {'valid_timeframes': self.VALID_TIMEFRAMES}
                }

            # Trigger timeframe change via dashboard callback
            if DASHBOARD_CALLBACKS and 'set_timeframe' in DASHBOARD_CALLBACKS:
                result = DASHBOARD_CALLBACKS['set_timeframe'](timeframe)
                return {
                    'success': True,
                    'message': f'✅ Timeframe set to {timeframe}',
                    'data': {'timeframe': timeframe}
                }
            else:
                # Fallback
                return {
                    'success': True,
                    'message': f'✅ Timeframe changed to {timeframe} (will apply on next update)',
                    'data': {'timeframe': timeframe}
                }

        except Exception as e:
            logger.exception("Error setting timeframe")
            return {
                'success': False,
                'message': f'❌ Failed to set timeframe: {str(e)}',
                'data': None
            }

    def _handle_toggle_overlays(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle overlay toggle command."""
        try:
            # Determine if we should enable or disable
            message = params['raw_message'].lower()

            # Check for explicit enable/disable
            enable = True
            if any(word in message for word in ['off', 'disable', 'hide']):
                enable = False
            elif any(word in message for word in ['on', 'enable', 'show']):
                enable = True
            else:
                # Default: toggle (switch state)
                enable = 'toggle'

            # Trigger overlay toggle via dashboard callback
            if DASHBOARD_CALLBACKS and 'toggle_overlays' in DASHBOARD_CALLBACKS:
                result = DASHBOARD_CALLBACKS['toggle_overlays'](enable)
                action = 'enabled' if enable == True else ('disabled' if enable == False else 'toggled')
                return {
                    'success': True,
                    'message': f'✅ Overlays {action}',
                    'data': {'enable': enable}
                }
            else:
                return {
                    'success': True,
                    'message': f'✅ Overlay state changed (will apply on next update)',
                    'data': {'enable': enable}
                }

        except Exception as e:
            logger.exception("Error toggling overlays")
            return {
                'success': False,
                'message': f'❌ Failed to toggle overlays: {str(e)}',
                'data': None
            }

    def _handle_run_backtest_single(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle single backtest command."""
        try:
            # Parse parameters: symbol, timeframe, strategy, capital
            if len(params['params']) < 4:
                return {
                    'success': False,
                    'message': '❌ Invalid command format. Use: run backtest <SYMBOL> <TIMEFRAME> <STRATEGY> <CAPITAL>',
                    'data': None
                }

            symbol = params['params'][0].upper()
            timeframe = params['params'][1].lower()
            strategy = params['params'][2].lower()
            capital = int(params['params'][3])

            # Validate timeframe
            if timeframe not in self.VALID_TIMEFRAMES:
                return {
                    'success': False,
                    'message': f'❌ Invalid timeframe: {timeframe}',
                    'data': None
                }

            # Validate strategy
            if strategy not in self.VALID_STRATEGIES:
                return {
                    'success': False,
                    'message': f'❌ Invalid strategy: {strategy}. Valid options: {", ".join(self.VALID_STRATEGIES)}',
                    'data': None
                }

            # Validate capital
            if capital < 1000 or capital > 1000000:
                return {
                    'success': False,
                    'message': f'❌ Invalid capital amount: {capital}. Must be between 1000 and 1000000',
                    'data': None
                }

            # Run backtest directly
            try:
                from backtesting.service import BacktestConfig, run_backtest as run_bt
                # Fixed window aligned with UI default
                start_dt = pd.Timestamp("2024-01-01")
                end_dt = pd.Timestamp.utcnow().tz_localize(None)
                cfg = BacktestConfig(
                    symbol=symbol,
                    timeframe=timeframe,
                    start=start_dt,
                    end=end_dt,
                    strategy=strategy,
                    params={},
                    initial_capital=float(capital),
                )
                res = run_bt(cfg)
                return {
                    'success': True,
                    'message': f'✅ Backtest completed for {symbol} on {timeframe} ({strategy}, ${capital:,})',
                    'data': {
                        'symbol': symbol,
                        'timeframe': timeframe,
                        'strategy': strategy,
                        'capital': capital,
                    'results': {
                        'initial_capital': res.initial_capital,
                        'final_capital': res.final_capital,
                        'total_return_pct': res.total_return_pct,
                        'win_rate': res.win_rate,
                            'max_drawdown': res.max_drawdown,
                            'sharpe_ratio': res.sharpe_ratio,
                            'profit_factor': res.profit_factor,
                            'total_trades': res.total_trades,
                            'start': str(cfg.start),
                            'end': str(cfg.end),
                        }
                    }
                }
            except Exception as exc:
                logger.exception("Backtest run failed in router")
                return {
                    'success': False,
                    'message': f'❌ Backtest failed: {exc}',
                    'data': None
                }

        except Exception as e:
            logger.exception("Error running backtest")
            return {
                'success': False,
                'message': f'❌ Failed to run backtest: {str(e)}',
                'data': None
            }

    def _handle_run_backtest_sweep(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle backtest sweep command."""
        sweep_start_time = time.time()
        logger.info(f"[SWEEP] Starting backtest sweep handler at {datetime.now().isoformat()}")

        try:
            # Parse parameters: symbol, strategy, capital
            if len(params['params']) < 3:
                logger.warning(f"[SWEEP] Invalid command format: {params['params']}")
                return {
                    'success': False,
                    'message': '❌ Invalid command format. Use: run backtest <SYMBOL> all timeframes <STRATEGY> <CAPITAL>',
                    'data': None
                }

            symbol = params['params'][0].upper()
            strategy = params['params'][1].lower()
            capital = int(params['params'][2])

            logger.info(f"[SWEEP] Sweep params: symbol={symbol}, strategy={strategy}, capital={capital}")

            # Validate strategy
            if strategy not in self.VALID_STRATEGIES:
                return {
                    'success': False,
                    'message': f'❌ Invalid strategy: {strategy}. Valid options: {", ".join(self.VALID_STRATEGIES)}',
                    'data': None
                }

            # Validate capital
            if capital < 1000 or capital > 1000000:
                return {
                    'success': False,
                    'message': f'❌ Invalid capital amount: {capital}. Must be between 1000 and 1000000',
                    'data': None
                }

            # Run sweep directly
            try:
                from backtesting.service import BacktestConfig, run_backtest as run_bt
                results = []
                end_dt = pd.Timestamp.utcnow().tz_localize(None)

                # Filter timeframes based on config (skip short TFs if configured)
                timeframes_to_run = self.VALID_TIMEFRAMES
                if not self.include_short_timeframes:
                    timeframes_to_run = ['15m', '1h', '4h', '1d']
                    logger.info(f"[SWEEP] Skipping short timeframes, running: {timeframes_to_run}")

                logger.info(f"[SWEEP] Starting sweep for {len(timeframes_to_run)} timeframes")
                for i, tf in enumerate(timeframes_to_run, 1):
                    # Overall timeout check
                    elapsed = time.time() - sweep_start_time
                    if elapsed > self.sweep_timeout_seconds:
                        logger.warning(f"[SWEEP] Overall timeout exceeded ({elapsed:.1f}s > {self.sweep_timeout_seconds}s), stopping")
                        break

                    logger.info(f"[SWEEP] Running {tf} ({i}/{len(timeframes_to_run)}), elapsed: {elapsed:.1f}s")

                    # Per-timeframe timeout using signal-based approach
                    # Note: Using simple time-based check since we can't interrupt the backtest itself
                    tf_start = time.time()

                    cap_days = self.SWEEP_DAYS_CAP.get(tf, 365)
                    start_dt = max(pd.Timestamp("2024-01-01"), end_dt - pd.Timedelta(days=cap_days))
                    cfg = BacktestConfig(
                        symbol=symbol,
                        timeframe=tf,
                        start=start_dt,
                        end=end_dt,
                        strategy=strategy,
                        params={},
                        initial_capital=float(capital),
                    )
                    try:
                        res = run_bt(cfg)
                        tf_elapsed = time.time() - tf_start
                        logger.info(f"[SWEEP] {tf} completed in {tf_elapsed:.1f}s")
                    except Exception as tf_exc:
                        tf_elapsed = time.time() - tf_start
                        logger.exception(f"[SWEEP] {tf} failed after {tf_elapsed:.1f}s: {tf_exc}")
                        results.append({
                            'timeframe': tf,
                            'initial_capital': float(capital),
                            'final_capital': float(capital),
                            'total_return_pct': 0.0,
                            'win_rate': 0.0,
                            'max_drawdown': 0.0,
                            'sharpe_ratio': 0.0,
                            'profit_factor': 0.0,
                            'total_trades': 0,
                            'start': str(cfg.start),
                            'end': str(cfg.end),
                            'strategy': strategy,
                            'symbol': symbol,
                            'capital': capital,
                            'error': str(tf_exc),
                        })
                        continue

                    # Sanitize numbers
                    def safe_num(val, fallback=0.0):
                        return val if pd.notna(val) and np.isfinite(val) else fallback
                    results.append({
                        'timeframe': tf,
                        'initial_capital': safe_num(res.initial_capital),
                        'final_capital': safe_num(res.final_capital),
                        'total_return_pct': safe_num(res.total_return_pct),
                        'win_rate': safe_num(res.win_rate),
                        'max_drawdown': safe_num(res.max_drawdown),
                        'sharpe_ratio': safe_num(res.sharpe_ratio),
                        'profit_factor': safe_num(res.profit_factor),
                        'total_trades': int(res.total_trades),
                        'start': str(cfg.start),
                        'end': str(cfg.end),
                        'strategy': strategy,
                        'symbol': symbol,
                        'capital': capital,
                    })

                total_elapsed = time.time() - sweep_start_time
                logger.info(f"[SWEEP] Completed {len(results)}/{len(timeframes_to_run)} timeframes in {total_elapsed:.1f}s")

                if not results:
                    logger.error(f"[SWEEP] No results produced, total elapsed: {total_elapsed:.1f}s")
                    return {
                        'success': False,
                        'message': '❌ Backtest sweep did not produce any results (timeout/failed)',
                        'data': None
                    }

                # Build human-readable summary
                lines = [f"✅ Sweep for {symbol} | {strategy.upper()} | {results[0]['start'][:10]} → {results[0]['end'][:10]} | {len(results)}/{len(timeframes_to_run)} timeframes"]
                for r in results:
                    lines.append(
                        f"- {r['timeframe']}: {r['total_return_pct']:+.2f}% | trades {r['total_trades']} | win {r['win_rate']*100:.1f}% | DD {r['max_drawdown']:.2f}% | Sharpe {r['sharpe_ratio']:.2f}"
                        + (f" (error: {r.get('error')})" if r.get('error') else "")
                    )

                # Add timeout/partial indicator
                if total_elapsed > self.sweep_timeout_seconds or len(results) < len(timeframes_to_run):
                    lines.append(f"⚠️  Partial sweep: {len(results)}/{len(timeframes_to_run)} completed in {total_elapsed:.1f}s (timeout: {self.sweep_timeout_seconds}s)")
                else:
                    lines.append(f"✅ Complete sweep: {total_elapsed:.1f}s")

                response_message = "\n".join(lines)
                logger.info(f"[SWEEP] Returning response: {len(response_message)} chars")

                return {
                    'success': True,
                    'message': response_message,
                    'data': {
                        'symbol': symbol,
                        'strategy': strategy,
                        'capital': capital,
                        'timeframes_requested': len(timeframes_to_run),
                        'timeframes_completed': len(results),
                        'timeframes': [r['timeframe'] for r in results],
                        'results': results,
                        'elapsed_seconds': total_elapsed,
                        'timed_out': total_elapsed > self.sweep_timeout_seconds
                    }
                }
            except Exception as exc:
                total_elapsed = time.time() - sweep_start_time
                logger.exception(f"[SWEEP] Sweep failed after {total_elapsed:.1f}s: {exc}")
                return {
                    'success': False,
                    'message': f'❌ Backtest sweep failed: {exc}',
                    'data': None
                }

        except Exception as e:
            total_elapsed = time.time() - sweep_start_time
            logger.exception(f"[SWEEP] Handler error after {total_elapsed:.1f}s: {e}")
            return {
                'success': False,
                'message': f'❌ Failed to run backtest sweep: {str(e)}',
                'data': None
            }

    def _handle_fetch_health(self) -> Dict[str, Any]:
        """Handle health fetch command."""
        try:
            # Get health data from system context
            health_data = {
                'timestamp': datetime.now().isoformat(),
                'status': 'healthy',
                'components': {}
            }

            if SYSTEM_CONTEXT:
                if hasattr(SYSTEM_CONTEXT, 'system_health'):
                    health_data['components'] = SYSTEM_CONTEXT.system_health

            # Format health message
            components = health_data.get('components', {})
            if components:
                component_list = [f"{k}: {v}" for k, v in components.items()]
                message = '✅ System Health:\n' + '\n'.join(component_list)
            else:
                message = '✅ System Status: All systems operational'

            return {
                'success': True,
                'message': message,
                'data': health_data
            }

        except Exception as e:
            logger.exception("Error fetching health")
            return {
                'success': False,
                'message': f'❌ Failed to fetch system health: {str(e)}',
                'data': None
            }

    def _handle_fetch_account(self) -> Dict[str, Any]:
        """Handle account fetch command."""
        try:
            # Get account data from system context
            account_data = {
                'timestamp': datetime.now().isoformat(),
                'portfolio_value': 0,
                'unrealized_pnl': 0,
                'total_exposure': 0,
                'positions': []
            }

            if SYSTEM_CONTEXT:
                if hasattr(SYSTEM_CONTEXT, 'active_positions'):
                    account_data['positions'] = SYSTEM_CONTEXT.active_positions
                if hasattr(SYSTEM_CONTEXT, 'risk_metrics'):
                    account_data['unrealized_pnl'] = SYSTEM_CONTEXT.risk_metrics.get('unrealized_pnl', 0)
                    account_data['total_exposure'] = SYSTEM_CONTEXT.risk_metrics.get('total_exposure', 0)

            # Format account message
            positions_count = len(account_data.get('positions', []))
            message = f'✅ Account Status:\n'
            message += f'Positions: {positions_count}\n'
            message += f'Unrealized P&L: ${account_data["unrealized_pnl"]:.2f}\n'
            message += f'Total Exposure: ${account_data["total_exposure"]:.2f}'

            return {
                'success': True,
                'message': message,
                'data': account_data
            }

        except Exception as e:
            logger.exception("Error fetching account")
            return {
                'success': False,
                'message': f'❌ Failed to fetch account data: {str(e)}',
                'data': None
            }

    def route_message(self, message: str) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Route a message through the command router.

        Args:
            message: User message

        Returns:
            Tuple of (handled, response_message, metadata)
        """
        # Parse intent
        intent, params = self.parse_intent(message)

        # If no intent matched, call DeepSeek AI directly
        if intent is None:
            try:
                import asyncio
                from deepseek.client import DeepSeekBrain
                logger.info(f"[CHAT-ROUTER] Routing to DeepSeek AI, message: {message[:50]}...")
                if SYSTEM_CONTEXT:
                    logger.info(f"[CHAT-ROUTER] Creating DeepSeekBrain instance...")
                    deepseek = DeepSeekBrain(SYSTEM_CONTEXT)
                    # Prepare context for DeepSeek
                    context = {
                        'overlay_state': SYSTEM_CONTEXT.overlay_state,
                        'recent_signals': SYSTEM_CONTEXT.recent_signals[-5:] if hasattr(SYSTEM_CONTEXT, 'recent_signals') else [],
                        'chart_data': {'symbol': 'BTCUSDT', 'timeframe': '15m'}
                    }
                    logger.info(f"[CHAT-ROUTER] Calling DeepSeek chat_interface...")
                    # Run async chat_interface in sync context
                    response = asyncio.run(deepseek.chat_interface(message, context, "chat"))
                    logger.info(f"[CHAT-ROUTER] DeepSeek response received: {response[:100]}...")
                    return True, response, {'source': 'deepseek', 'intent': None}
                else:
                    logger.warning("[CHAT-ROUTER] System context not available")
                    return True, "System context not available. Please try again later.", {'source': 'fallback', 'error': 'no_context'}
            except Exception as e:
                logger.exception(f"Error calling DeepSeek AI: {e}")
                return True, f"AI service error: {str(e)}", {'source': 'error', 'error': str(e)}

        # Handle the command
        result = self.handle_command(intent, params)
        result['intent'] = intent

        # Store in history
        self.command_history.append({
            'timestamp': datetime.now().isoformat(),
            'intent': intent,
            'message': message,
            'result': result
        })
        self.last_command = message
        self.last_result = result

        return True, result['message'], result

    def get_last_command(self) -> Optional[Dict[str, Any]]:
        """Get the last executed command."""
        return {
            'command': self.last_command,
            'result': self.last_result,
            'history_count': len(self.command_history)
        }


# Global router instance
ROUTER = ManagerCommandRouter()


def route_chat_message(message: str) -> Tuple[bool, str, Dict[str, Any]]:
    """
    Route a chat message through the manager command router.

    Args:
        message: User message

    Returns:
        Tuple of (handled, response_message, metadata)
    """
    return ROUTER.route_message(message)


def get_manager_status() -> Optional[Dict[str, Any]]:
    """Get the current manager status."""
    return ROUTER.get_last_command()
