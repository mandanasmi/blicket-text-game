#!/bin/bash

# Run the conjunctive passive case app with step-by-step statements and questions.
# Port 8505. Run from project root so .streamlit/secrets.toml and .env are found.
# Local: http://localhost:8505

set -e
cd "$(dirname "$0")"
echo "Starting Nexiom conjunctive case app ..."
echo "Port 8505"
echo ""

if [ -d "venv" ]; then
  if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
  else
    echo "Warning: venv exists but venv/bin/activate was not found; continuing without activation."
  fi
fi

if command -v streamlit >/dev/null 2>&1; then
  streamlit run passive_app/conjunctive_case_app.py --server.port 8505
  exit 0
fi

if command -v uv >/dev/null 2>&1; then
  echo "streamlit not found in PATH; using uv."
  uv run python -m streamlit run passive_app/conjunctive_case_app.py --server.port 8505
  exit 0
fi

echo "Error: neither streamlit nor uv is available in PATH."
echo "Activate your Python environment (or install dependencies), then rerun this script."
exit 1
