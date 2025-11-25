# Vibecoder Implementation Plan – Modern Dashboard Restoration

## Goals
- Restore full trading dashboard functionality (charts, overlays, controls, strategies, backtest, chat) with modernized UI while preserving previous features (plotly interactions, zoom/pan/modebar, download buttons, overlays).
- Remove reliance on `/tmp/dash_server.py`; use a canonical entry point and repeatable verification flow in Chrome MCP.
- Ensure backend ↔ frontend wiring is intact: data fetch, stores, callbacks, and UI elements mapped 1:1 with callback outputs.

## Manager Directives for Vibecoder
- At every step: run `python -m ui.dashboard`, open `http://127.0.0.1:8050` via Chrome MCP, actively interact (modebar zoom/pan/autoscale/download, overlays, timeframes, buttons, modals, chat, backtest) and confirm behavior. Do not skip MCP validation.
- After each step, append a concise status entry to `vibecoder_mission_report.md` only: include step number/name, actions taken, MCP observations (working/broken), and any screenshots/snapshot filenames if captured.
- Keep canonical entry `python -m ui.dashboard` (or `scripts/run_dashboard.sh` if already present for PID) and avoid `/tmp` ad-hoc scripts. Do not delete or clean unrelated files.
- If DeepSeek/backtest services are unavailable, note it in the mission report and verify demo fallbacks render clearly.
- Use Chrome MCP tools (`navigate_page`, `take_snapshot`, `take_screenshot`, interactions) to observe the live dashboard after each change; do not rely on static inspection.

## Working Commands
- Run dashboard: `python -m ui.dashboard` (with a real SystemContext; remove DummySystemContext debug runs).
- Open in Chrome MCP for live verification: navigate to `http://127.0.0.1:8050` via MCP (`navigate_page`), then `take_snapshot`/`take_screenshot` to confirm UI and plotly controls (zoom/pan/modebar) render.
- After each step below, re-run the dashboard and validate changes live in Chrome MCP (especially chart zoom/pan/modebar, overlay toggles, timeframes, and control buttons). Log findings to `vibecoder_mission_report.md`.

## Implementation Steps
1) **Bootstrap & Callback Registration**
   - Make `python -m ui.dashboard` the canonical entry. Remove DummySystemContext debug runs; use real SystemContext or a guarded demo flag.
   - Ensure callbacks are bound to the Dash app (no orphaned global callbacks). Verify `app.callback_map` is populated. Remove stray files (`ui/dashboard.py.newcallback.txt`, `ui/dashboard_mainch art_callback.py`) or integrate their logic with real IDs.
   - Keep required `dcc.Interval`/`dcc.Store` nodes and modals in layout; ensure every callback output has a rendered component.
   - **Chrome MCP check after this step**: start the app, open `http://127.0.0.1:8050`, and confirm the page loads without callback registration errors. Log observations in `vibecoder_mission_report.md`.

2) **Data Flow & Error Handling**
   - Guard convergence/performance callbacks against `None` values; fix `_last_update` reference and any `None > int` issues.
   - Reuse a singleton DataStore inside `fetch_market_data`; avoid repeated initialization/log spam.
   - Correct freshness/connection labels (don’t show “Live Binance” when using demo data). Add badges for live/demo and freshness.
   - **Chrome MCP check after this step**: reload and verify health/freshness badges and chart load without server errors. Log observations in `vibecoder_mission_report.md`.

3) **UI Modernization (preserve features)**
   - Apply design tokens (background/surface/border/accent) and modern typography; wrap sections in `.ds-card` with consistent padding and shadows.
   - Group hero (symbol/price/timeframes/status) + overlays + chart into cohesive cards; tighten vertical gaps; use responsive grid (chart vs. side panels).
   - Ensure plotly modebar and interactions (zoom/pan/autoscale/download) remain visible; set `config` to keep modebar.
   - Restyle controls (timeframes, overlays, trading CTAs) and tables; add skeletons/placeholders when data is loading/absent.
   - **Chrome MCP check after this step**: confirm visual updates, modebar presence, and mouse zoom/pan still work. Log observations in `vibecoder_mission_report.md`.

4) **State & Content Population**
   - Keep demo fallbacks labeled; do not overwrite live trading state. Render positions/metrics/feature tables with informative placeholders when live data is missing.
   - Add badges/sparklines for P&L/win-rate where feasible; surface freshness/connection in health panel.
   - **Chrome MCP check after this step**: verify populated cards/tables show demo/live badges and no empty shells. Log observations in `vibecoder_mission_report.md`.

5) **Strategy/Backtest/Chat Wiring**
   - Verify strategy panels (convergence/scalp) render all fields and buttons; make callbacks None-safe.
   - Backtest modal: ensure run/promote buttons and modal render; handle API failure gracefully with inline error.
   - Chat: enable send when text is present; keep quick actions and status messages; log to file.
   - **Chrome MCP check after this step**: exercise strategy buttons, backtest modal, and chat send/quick actions. Log observations in `vibecoder_mission_report.md`.

6) **Verification**
   - Run `python -m ui.dashboard`, open `http://127.0.0.1:8050` in Chrome MCP, and actively test: modebar zoom/pan, overlay toggles, timeframe buttons, refresh/emergency stop, chat send enablement.
   - Add/update smoke test (e.g., in `tests/`) asserting key IDs and that the chart has traces (non-empty figure data); update test IDs/text to match the real layout (`price-chart`, `feature-toggles`, `chat-send-button`, H1 “DeepSeek Trading Dashboard”).
   - Run targeted `pytest`/`make lint` if feasible.
   - **Chrome MCP check after this step**: final end-to-end validation with chart interactions and control flows. Log observations in `vibecoder_mission_report.md`.

7) **Cleanup & Consistency**
   - Remove or archive legacy/backup files not used by the current app (e.g., `ui/dashboard.py.backup` if obsolete) and ensure only one active layout/callback set remains.
   - Align titles/text with tests (H1 “DeepSeek Trading System”) and ensure smoke tests target current IDs (`main-chart`, `overlay-toggles`, `chat-send-btn`, badges).
   - Run `pytest tests/test_dashboard_smoke.py` to confirm the suite passes; fix any ID/text mismatches.
   - **Chrome MCP check after this step**: rerun the app and confirm the cleaned layout still renders correctly with plotly interactions intact. Log observations in `vibecoder_mission_report.md`.

8) **Backtest Tab UI & Wiring**
   - Build a full Backtest Lab UI: inputs (symbol, timeframe, strategy, start/end dates, capital), run/promo buttons, results modal with PnL/drawdown/win-rate/trades, and demo fallback when API is unavailable.
   - Wire callbacks to call the backtest helper/API, show loading/error states, and render results.
   - **Chrome MCP check after this step**: open Backtest tab, verify controls render, trigger a run (or demo run) and confirm UI updates without errors. Log observations in `vibecoder_mission_report.md`.

9) **Account & Trading Data Population**
   - Feed Account & Trading cards from SystemContext when available; show clear “Demo Data” badges/placeholders when not. Render portfolio value, unrealized PnL, exposure, drawdown, positions, trade history, and system health with real or clearly labeled demo values.
   - **Chrome MCP check after this step**: open Account & Trading tab, verify metrics populate (or demo badges show), no empty/0.00 unlabeled placeholders. Log observations in `vibecoder_mission_report.md`.

10) **Chat Fallback & Responses**
    - Ensure chat sends render visible responses: use DeepSeek when available; otherwise return a helpful demo response and update status text. Keep quick actions and history functional; surface errors inline if send fails.
    - **Chrome MCP check after this step**: send a test message and confirm history/status update (no “Awaiting message” stall). Log observations in `vibecoder_mission_report.md`.

11) **PID Tracking & Server Control**
    - Add a helper/script to start the dashboard and write PID for reliable stop, or document a run/stop command with PID output while keeping `python -m ui.dashboard` canonical.
    - **Chrome MCP check after this step**: start via the new helper/command, open the app, then stop using the recorded PID; ensure no orphaned server remains. Log observations in `vibecoder_mission_report.md`.

12) **Final Verification**
   - Update smoke tests if needed for Backtest/chat changes; run `pytest tests/test_dashboard_smoke.py` if feasible.
   - Final Chrome MCP pass through Market, Account, and Backtest tabs: confirm chart interactions (modebar/zoom/pan), overlays/timeframes, backtest UI, populated metrics, and chat responses.
   - Log final MCP observations and any discrepancies in `vibecoder_mission_report.md`.

13) **Backtest Accuracy & Labeling**
   - Ensure Backtest callbacks call the real `backtesting.service.run_backtest` (not demo RNG). Pass symbol/timeframe/start/end/strategy/capital; enforce sensible defaults (e.g., 1-year range if missing) and validate inputs.
   - Update Backtest result text to reflect live/real computation vs demo; only show “Demo” when the service fails and a fallback is used.
   - Add a clear summary: total return %, trades, win rate, drawdown, sharpe, profit factor, initial/final capital, period.
   - **Chrome MCP check after this step**: run a backtest (e.g., 1y on 15m) and confirm the card shows real metrics without demo wording; verify errors show inline if the service fails. Log observations in `vibecoder_mission_report.md`.

14) **Account Tab Data & Labels**
   - Wire Account & Trading tab to real `SystemContext` data where available (portfolio value, unrealized PnL, exposure, drawdown, positions, trade history, health). When data is missing, show clearly labeled demo placeholders (no unlabeled zeros).
   - **Chrome MCP check after this step**: open Account & Trading and confirm live data shows when present; demo badges/labels display when not. Log observations in `vibecoder_mission_report.md`.

15) **Chat Response Clarity**
   - Ensure chat always shows a response: DeepSeek when available, otherwise a structured demo fallback with context (symbol/timeframe/overlays).
   - Update status text to reflect success/fallback/error states visibly.
   - **Chrome MCP check after this step**: send a message and confirm history/status update (no “Awaiting message” stall). Log observations in `vibecoder_mission_report.md`.

16) **Test & Doc Updates**
   - Update smoke tests to cover Backtest success/error rendering and chat fallback rendering (IDs stable). Note `chromedriver` is required for dash_duo tests.
   - Add a short README/note about running tests with chromedriver installed.
   - **Chrome MCP check after this step**: final sanity across all tabs and interactions. Log observations in `vibecoder_mission_report.md`.

17) **Binance Live Data Switch (Manager Priority)**
   - Update `core/data/binance_client.py` to honor env overrides: use `BINANCE_TESTNET` flag, but allow `BINANCE_FUTURES_URL`/`BINANCE_SPOT_URL` to override base URLs when provided.
   - Default behavior: if `BINANCE_TESTNET=true`, use testnet URLs (existing); if false, use `https://fapi.binance.com/fapi/v1` unless overridden by env. Log which endpoint is chosen on init.
   - Update dashboard status badge copy to reflect live vs testnet based on the resolved URL/flag (no hardcoded “Demo”).
   - Ensure `.env` is set for production read-only access: `BINANCE_TESTNET=false`, `BINANCE_FUTURES_URL=https://fapi.binance.com/fapi/v1`, `BINANCE_SPOT_URL=https://api.binance.com/api/v3`, plus API key/secret.
   - Restart `python -m ui.dashboard` after changes and verify in Chrome MCP that data badges show “LIVE” and the chart pulls real data without testnet labeling.
   - **Chrome MCP check after this step**: open dashboard, confirm LIVE badge and real data load; log observations in `vibecoder_mission_report.md`.

18) **DeepSeek as Manager (Safe Command Layer)**
   - Add a command router for chat to handle safe “manager” actions before hitting DeepSeek: allowed commands = backtest (single + sweep), refresh data, switch timeframe, toggle overlays, and (read-only) fetch account/positions/health. Do NOT expose order placement or strategy start/stop unless explicitly whitelisted.
   - Implement a lightweight intent parser (regex/keywords) and a dispatcher mapping intents → functions (e.g., `force_refresh`, `set_timeframe`, `set_overlays`, `run_backtest`, `run_backtest_sweep`, `get_health_summary`). Ensure unrecognized commands fall back to DeepSeek.
   - Structure chat responses so operator sees the action outcome (e.g., “Refreshed data,” “Timeframe set to 1m,” “Backtest result: …”) and log them. Keep DeepSeek chat for general Q&A only.
   - Add a “Manager Actions” section in the Backtest/Account tab to surface last command/status (from a `dcc.Store` and card). Include safety text: “Read-only manager mode; no order placement.”
   - Add a cache-bust/version query suggestion in the UI (e.g., `?v=2`) or bump `assets_url_path` to ensure new JS is loaded.
   - **Chrome MCP check after this step**: issue commands via chat: (1) refresh data, (2) set timeframe to 1m, (3) toggle overlays on/off, (4) run backtest single, (5) run backtest all timeframes, (6) fetch health. Confirm results in chat and manager status card; log observations in `vibecoder_mission_report.md`.

## Notes for Vibecoder
- Preserve all interactive plotly controls (zoom/pan/modebar/download) and ensure they’re visible by default.
- Use assets/custom.css tokens where possible; avoid removing existing accessibility helpers (skip link, ARIA live region).
- Keep demo vs live states clearly labeled to avoid operator confusion.

## New Plan: Backtest Stability & Parity (Chrome MCP validation every step)
- **S1 Fix Sweep Instability**: Stabilize `models/backtester` to prevent NaN/overflow during multi-timeframe sweeps. Add guardrails (price bounds/position sizing/pct_change fill_method=None) and rerun chat sweep to completion without warnings. Validate via Chrome MCP chat sweep.
- **S2 Align Chat/UI Backtests**: Ensure chat single/sweep runs use the same default window as UI (2024-01-01 → now). Chat should render summary cards (including sweeps) in Backtest tab; no demo text. Validate by running chat single + sweep and seeing summaries update.
- **S3 Strategy/TF Matrix Run**: In Backtest Lab UI, run at least three combos (e.g., Convergence 1h, MA Crossover 15m, RSI Divergence 4h). Record Return/Trades/Win/DD/Sharpe/PF. Mirror each via chat commands and compare metrics; note drift in `vibecoder_mission_report.md`.
- **S4 Controls Regression**: After backtest fixes, reconfirm timeframes/overlays/range slider/modebar in Chrome MCP; chart must react to chat manager commands and UI buttons.
- **S5 Health & Account Load**: With heavy backtests running, confirm Feature Health and Account & Trading still render (no crashes/empty cards). Note any performance lag.
- **S6 Reporting**: For each step, append timestamped findings to `vibecoder_mission_report.md` (what ran, MCP observation, metrics, any drift between chat vs UI).

> Always restart `python -m ui.dashboard`, open via Chrome MCP, and actively interact (chat + Backtest tab) after each change before proceeding.
