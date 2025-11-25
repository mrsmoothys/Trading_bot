#!/usr/bin/env bash
# Unified launcher for engine, dashboard, and chat
set -e

# Load .env if present
if [ -f .env ]; then
  # shellcheck disable=SC2046
  export $(grep -v '^#' .env | xargs)
fi

export PYTHONPATH="$PYTHONPATH:$(pwd)"

echo "Starting DeepSeek engine..."
python main.py &
PID_ENGINE=$!

echo "Starting dashboard..."
python -m ui.dashboard &
PID_DASH=$!

echo "Starting chat..."
python -m ui.chat_interface &
PID_CHAT=$!

echo "PIDs -> engine: $PID_ENGINE, dashboard: $PID_DASH, chat: $PID_CHAT"
echo "Press Ctrl+C to stop all."
trap "kill $PID_ENGINE $PID_DASH $PID_CHAT" INT TERM
wait
