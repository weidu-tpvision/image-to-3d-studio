#!/usr/bin/env bash
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

# Source virtual environment
source .venv/bin/activate

# Expose on 0.0.0.0 so phone / tablet / laptop can connect directly over LAN
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"

echo "Starting Idol Image-to-3D Backend on http://$HOST:$PORT ..."
exec uvicorn backend.app:app --host "$HOST" --port "$PORT" --reload
