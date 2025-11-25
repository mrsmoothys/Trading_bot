**Vibecoder Implementation Plan (Backtest Lab + Data Integrity) — Extremely Explicit**

Think of this as instructions for a beginner: do exactly these steps, check the dashboard via Chrome MCP after each change, and run the specified tests. Include the code snippets as-is unless you have a better fix.

Environment (don’t change):
- Start dashboard with: `DASH_AUTH_DISABLED=true CHAT_RESPONSE_TIMEOUT=60 python -m ui.dashboard`
- UI checks via Chrome MCP; if clicks fail, set values with JS: `evaluate_script(() => { sel.value='…'; sel.dispatchEvent(new Event('change',{bubbles:true})); })`
- Tests via `pytest` (smoke/backtest tests noted below)

Files you may edit: `ui/dashboard.py`, `backtesting/service.py`, `core/strategies/ma_crossover.py`, `core/strategies/rsi_divergence.py`, `tests/...`
Files you must NOT delete: legacy backups (`ui/dashboard.py.backup*`), `.env`

Step 0: Environment Setup (Conda)
- If `environment.yml` exists:
  ```bash
  conda env create -f environment.yml
  conda activate trading_bot
  ```
- If not, create and install deps:
  ```bash
  conda create -y -n trading_bot python=3.10
  conda activate trading_bot
  pip install -r requirements-extra.txt  # active requirements file in repo
  ```
- Verify core deps:
  ```bash
  python -c "import dash, pandas, numpy; print('deps ok')"
  ```
- Keep secrets (ANTHROPIC_*, etc.) out of git; set tokens via environment, not tracked files.

Step 1: Data Freshness & Badge Guardrails
- In `update_backtest_data_badge`, force-refresh data. Compute thresholds per timeframe:
  ```python
  tf_minutes = {'1m':1,'5m':5,'15m':15,'1h':60,'4h':240,'1d':1440}.get(timeframe, 60)
  ok = tf_minutes * 1.5
  stale = tf_minutes * 3
  if used_sample or last_age > ok:
      raise ValueError("Stale or sample data; aborting backtest")
  # badge color: green if <= ok; yellow if <= stale; red otherwise
  ```
- MCP check: Open Backtest Lab → badge should be green for 1h/4h after refresh (force refresh if needed).
- Test: Add/adjust a smoke test that fetches 1h and asserts `used_sample_data is False` and `last_candle_age_min < 120`.

Step 2: Persist Results & Render Modal
- On run, store trimmed results in `backtest-result-store`:
  ```python
  result_dict = result.to_dict()
  result_dict["trades"] = result_dict.get("trades", [])[:50]
  result_dict["equity_curve"] = result_dict.get("equity_curve", [])[:500]
  ```
- Modal should read the store and render summary, trades table, equity chart (no placeholders).
- MCP check: Run RSI Divergence 1h → click View Details → see summary + trades + equity (equity will appear after Step 4).
- Test: Add a callback unit test to ensure the modal builds when the store has data.

Step 3: Normalize Strategies (fix extreme/zero results)
- MA Crossover defaults (so it trades):
  ```python
  fast = params.get("fast_period", 10)
  slow = params.get("slow_period", 30)
  ```
- RSI Divergence throttle (reduce overtrading):
  ```python
  signals['throttle'] = signals['signal'].rolling(5).max().fillna(0)
  signals.loc[signals['throttle'] == 0, 'signal'] = 0
  ```
- Convergence: require multi-TF confirmation + regime filter to cut trade count (adjust in strategy logic).
- MCP check: Run Convergence 4h, RSI 1h, MA 15m → ensure metrics are plausible and MA shows nonzero trades.
- Test: Add a small-sample backtest test that caps trade counts on short TFs and asserts no NaN/inf metrics.

Step 4: Emit Equity Curve & Trades
- In the backtester, accumulate per-bar equity and attach to `BacktestResult.equity_curve`; ensure trades include timestamps, pnl, pnl_percent:
  ```python
  equity_curve.append({"timestamp": bar_time, "equity": equity_value})
  trades.append({
    "entry_time": ..., "exit_time": ..., "side": ...,
    "entry_price": ..., "exit_price": ...,
    "pnl": ..., "pnl_percent": ...
  })
  ```
- MCP check: Re-run backtest (e.g., RSI 1h), open View Details → equity chart should render.
- Test: Unit test that a sample backtest returns non-empty equity_curve and trades.

Step 5: Enforce Live Data Only in Backtests
- Keep `BACKTEST_MODE=true` during runs; in `get_real_market_data`, raise if live data is missing; surface error in status alert.
- MCP check: Force stale/missing data → expect a clear error and no result card.
- Test: Simulate missing data, expect ValueError and no result.

Step 6: Regression Check
- Run: `pytest tests/test_dashboard_smoke.py tests/test_dashboard_chart_verification.py` (adjust if your suite differs). Fix any failures.

Step 7: Final MCP Validation
- Badge green for target TFs (1h/4h).
- Run & View Details:
  - RSI Divergence 1h (see equity + trades)
  - Convergence 4h
  - MA Crossover 15m (ensure trades > 0)
- Confirm no 500s; badges accurate.

Step 8: Reporting
- Update `BACKTEST_AND_CHAT_REPORT.md` with fresh, plausible metrics/screenshots; note any remaining gaps (e.g., equity missing for certain strategies).

### Acceptance Criteria (Do not mark done unless all met)
- Step 1: Badge green for 1h/4h in MCP after refresh AND smoke test asserts `used_sample_data=False` and `last_candle_age_min<120`.
- Step 2: View Details shows summary + trades + equity (after Step 4) AND modal callback test passes.
- Step 3: MA Crossover has nonzero trades; RSI returns are plausible (no extreme values); short-TF trade counts under cap in test; no NaN/inf.
- Step 4: Backtester emits equity_curve and trades; UI shows equity chart; unit test for equity/trades passes.
- Step 5: Intentional missing/stale data run fails with clear error; test asserts ValueError; no result card.
- Step 6: `pytest tests/test_dashboard_smoke.py tests/test_dashboard_chart_verification.py` passes.
- Step 7: MCP runs and verifies View Details for RSI 1h, Convergence 4h, MA 15m; no 500s in logs; badges accurate.
- Step 8: `BACKTEST_AND_CHAT_REPORT.md` updated with new metrics/screenshots and remaining gaps noted.
