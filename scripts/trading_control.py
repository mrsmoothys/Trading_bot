"""
Trading Control Helpers
Manages the lifecycle of the trading system process.
"""
import os
import json
import subprocess
import signal
from datetime import datetime
from pathlib import Path

CACHE_DIR = Path("cache")
PID_FILE = CACHE_DIR / "trading_process.json"


def get_status():
    """
    Get the current status of the trading process.

    Returns:
        dict: Status information including running state, PID, and uptime
    """
    # Ensure cache directory exists
    CACHE_DIR.mkdir(exist_ok=True)

    # Check if PID file exists
    if not PID_FILE.exists():
        return {
            "running": False,
            "pid": None,
            "status": "stopped",
            "started_at": None,
            "uptime_seconds": 0
        }

    try:
        with open(PID_FILE, 'r') as f:
            pid_data = json.load(f)

        pid = pid_data.get('pid')
        started_at = pid_data.get('started_at')

        if not pid:
            return {
                "running": False,
                "pid": None,
                "status": "stopped",
                "started_at": None,
                "uptime_seconds": 0
            }

        # Check if process is actually running
        try:
            # Check if process exists
            os.kill(pid, 0)
            process_running = True
        except OSError:
            # Process doesn't exist
            process_running = False
            # Clean up stale PID file
            PID_FILE.unlink(missing_ok=True)

        if process_running:
            # Calculate uptime
            uptime_seconds = 0
            if started_at:
                try:
                    start_time = datetime.fromisoformat(started_at)
                    uptime_seconds = (datetime.now() - start_time).total_seconds()
                except:
                    uptime_seconds = 0

            return {
                "running": True,
                "pid": pid,
                "status": "running",
                "started_at": started_at,
                "uptime_seconds": uptime_seconds
            }
        else:
            return {
                "running": False,
                "pid": None,
                "status": "stopped",
                "started_at": None,
                "uptime_seconds": 0
            }

    except Exception as e:
        return {
            "running": False,
            "pid": None,
            "status": f"error: {str(e)}",
            "started_at": None,
            "uptime_seconds": 0
        }


def start_trading():
    """
    Start the trading system in a subprocess.

    Returns:
        dict: Result with success status and message
    """
    # Check if already running
    status = get_status()
    if status["running"]:
        return {
            "success": False,
            "message": f"Trading system already running (PID: {status['pid']})",
            "pid": status['pid']
        }

    try:
        # Ensure cache directory exists
        CACHE_DIR.mkdir(exist_ok=True)

        # Start the trading system
        process = subprocess.Popen(
            ["python", "main.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True  # Detach from parent process
        )

        # Save PID information
        pid_data = {
            "pid": process.pid,
            "started_at": datetime.now().isoformat(),
            "command": "python main.py"
        }

        with open(PID_FILE, 'w') as f:
            json.dump(pid_data, f, indent=2)

        return {
            "success": True,
            "message": f"Trading system started successfully (PID: {process.pid})",
            "pid": process.pid
        }

    except Exception as e:
        return {
            "success": False,
            "message": f"Failed to start trading system: {str(e)}",
            "pid": None
        }


def stop_trading():
    """
    Stop the trading system process.

    Returns:
        dict: Result with success status and message
    """
    # Check if running
    status = get_status()
    if not status["running"]:
        return {
            "success": False,
            "message": "Trading system is not running",
            "pid": None
        }

    try:
        pid = status["pid"]

        # Try to terminate gracefully first
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError:
            # Process might already be dead
            pass

        # Wait a moment for graceful shutdown
        import time
        time.sleep(2)

        # Check if still running, force kill if needed
        try:
            os.kill(pid, 0)
            # Still running, force kill
            try:
                os.kill(pid, signal.SIGKILL)
            except OSError:
                pass
        except OSError:
            # Process already dead
            pass

        # Clean up PID file
        PID_FILE.unlink(missing_ok=True)

        return {
            "success": True,
            "message": f"Trading system stopped (PID: {pid})",
            "pid": pid
        }

    except Exception as e:
        return {
            "success": False,
            "message": f"Failed to stop trading system: {str(e)}",
            "pid": status["pid"]
        }