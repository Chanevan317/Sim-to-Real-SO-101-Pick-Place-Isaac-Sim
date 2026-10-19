#!/usr/bin/env bash
# Start SO-Arm Data Studio: the server (env_isaaclab) if it is not running yet, then the app window.
set -uo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_BASE="${CONDA_BASE:-}"
if [[ -z "$CONDA_BASE" ]]; then
  for base in "$HOME/miniconda3" "$HOME/anaconda3"; do [[ -d "$base/envs" ]] && CONDA_BASE="$base" && break; done
fi
if [[ -z "$CONDA_BASE" ]] && command -v conda > /dev/null; then
  CONDA_BASE="$(conda info --base 2>/dev/null)"
fi
ENV_NAME=$(sed -n 's/.*"isaaclab_env": *"\([^"]*\)".*/\1/p' "$DIR/studio_config.json")
PORT=$(sed -n 's/.*"port": *\([0-9]*\).*/\1/p' "$DIR/studio_config.json")
PY="$CONDA_BASE/envs/$ENV_NAME/bin/python"
URL="http://127.0.0.1:$PORT"
LOG="$HOME/.cache/so101_studio.log"
mkdir -p "$HOME/.cache"

if ! curl -s -o /dev/null "$URL/api/ping"; then
  if [[ ! -x "$PY" ]]; then
    notify-send "SO-Arm Data Studio" "Python not found: $PY" 2>/dev/null || echo "Python not found: $PY" >&2
    exit 1
  fi
  # The server needs the desktop (DISPLAY) to open Isaac Sim windows, so it inherits this session.
  setsid "$PY" "$DIR/studio.py" >> "$LOG" 2>&1 < /dev/null &
  for _ in $(seq 1 60); do curl -s -o /dev/null "$URL/api/ping" && break; sleep 0.5; done
fi

for browser in google-chrome google-chrome-stable chromium chromium-browser; do
  if command -v "$browser" > /dev/null; then
    exec "$browser" --app="$URL" --user-data-dir="$HOME/.config/so101-studio-window" \
      --class=so101-data-studio --window-size=1500,950 --no-first-run --no-default-browser-check
  fi
done
xdg-open "$URL"
