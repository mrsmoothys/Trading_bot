# Remediation Plan: DeepSeek Integrated Trading System

This document outlines the critical steps required to fix the architectural flaws, security vulnerabilities, and operational risks identified in the system.

## 📋 To-Do List

### 1. Fix "Split-Brain" State (Shared Persistence)
**Priority: CRITICAL**
The Dashboard and Trading Engine currently run as separate processes with separate memory. We need them to share a "Brain". We will implement a file-based synchronization mechanism using JSON.

*   **Task:** Modify `core/system_context.py`.
*   **Approach:**
    *   Add a `save_to_disk()` method that dumps `self.active_positions`, `self.risk_metrics`, and `self.trade_history` to a `state.json` file.
    *   Add a `load_from_disk()` method that reads this file.
    *   **Trading Engine (`main.py`):** Call `save_to_disk()` every time a trade is opened/closed or metrics update.
    *   **Dashboard (`ui/dashboard.py`):** Call `load_from_disk()` inside the main update callback (every 1-2 seconds) to fetch the latest reality.

**Code Concept (`core/system_context.py`):**
```python
import json
import os

STATE_FILE = "data/system_state.json"

class SystemContext:
    def save_to_disk(self):
        state = {
            "active_positions": self.active_positions,
            "risk_metrics": self.risk_metrics,
            # ... other critical fields
        }
        # Atomic write pattern to prevent corruption
        temp_file = STATE_FILE + ".tmp"
        with open(temp_file, 'w') as f:
            json.dump(state, f, default=str)
        os.replace(temp_file, STATE_FILE)

    def load_from_disk(self):
        if os.path.exists(STATE_FILE):
            with open(STATE_FILE, 'r') as f:
                state = json.load(f)
                self.active_positions = state.get("active_positions", {})
                self.risk_metrics = state.get("risk_metrics", {})
```

### 2. Fix Hardcoded Paths
**Priority: HIGH**
The absolute paths (e.g., `/Users/mrsmoothy/...`) prevent the code from being portable and cause crashes if folders are moved.

*   **Task:** Clean up `ui/dashboard.py`, `ui/chat_interface.py`, and `ui/__main__.py`.
*   **Approach:**
    *   Remove `sys.path.insert(...)` lines.
    *   Ensure all imports are relative to the project root (e.g., `from core.system_context import SystemContext`).
    *   **Execution:** You must run the dashboard from the project root using the module flag: `python -m ui.dashboard`. This automatically sets the correct path.

### 3. Database Persistence for Positions
**Priority: HIGH**
Currently, if the bot crashes, it "forgets" it has open trades on Binance. It needs to check the database on startup.

*   **Task:** Update `execution/position_manager.py` and `ops/db.py`.
*   **Approach:**
    *   In `ops/db.py`, create a table for `active_positions` (separate from historical signals).
    *   In `PositionManager.__init__`, query this table.
    *   If the database says we have a position, but `SystemContext` is empty, populate `SystemContext` from the DB.
    *   Verify against Binance API: On startup, fetch active positions from Binance and reconcile them with the DB.

### 4. Secure the Dashboard (Auth)
**Priority: HIGH**
Prevent unauthorized access to your trading controls.

*   **Task:** Update `ui/dashboard.py` and `ui/chat_interface.py`.
*   **Approach:**
    *   Install `dash-auth`: `pip install dash-auth`
    *   Add Basic Auth using credentials from `.env`.

**Code Concept:**
```python
import dash_auth
import os

VALID_USERNAME_PASSWORD_PAIRS = {
    os.getenv("DASH_USER", "admin"): os.getenv("DASH_PASS", "admin")
}

app = dash.Dash(__name__)
auth = dash_auth.BasicAuth(
    app,
    VALID_USERNAME_PASSWORD_PAIRS
)
```

### 5. Production Data Safety
**Priority: MEDIUM**
Prevent the dashboard from showing fake "sample data" when the real API fails, which can be misleading.

*   **Task:** Update `ui/dashboard.py`.
*   **Approach:**
    *   In `fetch_market_data`, check a flag `os.getenv('PRODUCTION_MODE')`.
    *   If `True` and the API call fails, return an empty DataFrame or raise an explicit error displayed on the UI ("API CONNECTION LOST"), rather than generating random numbers.

### 6. AI Circuit Breaker
**Priority: MEDIUM**
Ensure trading continues even if DeepSeek API goes down.

*   **Task:** Update `execution/risk_manager.py`.
*   **Approach:**
    *   Wrap DeepSeek API calls in a `try/except` block.
    *   Track consecutive failures. If failures > 3, enable `technical_fallback_mode`.
    *   In fallback mode, `validate_trade` skips the AI check and relies solely on technical limits (Drawdown < 5%, Confidence > 0.8).

### 7. Unified Launcher
**Priority: LOW (Quality of Life)**
Simplify starting the complex system.

*   **Task:** Create `start.sh`.
*   **Approach:**
    *   A script that sets up the environment and launches the Engine, Dashboard, and Chat in parallel background processes.

**Code Concept (`start.sh`):**
```bash
#!/bin/bash
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Start Trading Engine
python main.py &
PID_ENGINE=$!

# Start Dashboard
python -m ui.dashboard &
PID_DASH=$!

# Start Chat
python -m ui.chat_interface &
PID_CHAT=$!

echo "System started. Press Ctrl+C to stop."
wait $PID_ENGINE $PID_DASH $PID_CHAT
```

## 🚀 Execution Strategy

1.  **Start with Tasks 1 & 2:** These are the "plumbing" fixes. Without them, the system is fundamentally broken (split-brain).
2.  **Test Persistence:** Open a trade in the engine (or mock one), restart the engine, and verify the position is still remembered (Task 3).
3.  **Secure It:** Add the auth (Task 4) before deploying to any server.
4.  **Run It:** Use the `python -m` command or the new `start.sh` (Task 7) to run the system cleanly.
