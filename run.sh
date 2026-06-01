#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────
#  NSE Nifty Dashboard — one-click launcher
#  Usage:  double-click, or run  ./run.sh  from Terminal
# ─────────────────────────────────────────────────────────────
set -e

# Move to the folder this script lives in
cd "$(dirname "$0")"

# Pick a Python (prefer python3)
PY="$(command -v python3 || command -v python)"
if [ -z "$PY" ]; then
  echo "❌ Python is not installed. Get it from https://www.python.org/downloads/"
  read -p "Press Enter to close..."
  exit 1
fi

echo "📦 Checking dependencies..."
if ! "$PY" -c "import streamlit" 2>/dev/null; then
  echo "   Installing required packages (first run only)..."
  "$PY" -m pip install -r requirements.txt
fi

echo "🚀 Launching NSE Nifty Dashboard..."
echo "   Your browser will open at http://localhost:8501"
echo "   Press Ctrl+C in this window to stop the app."
echo ""

exec "$PY" -m streamlit run streamlit_app.py
