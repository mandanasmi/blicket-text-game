#!/bin/bash
# Launch all four extension experiment apps locally on separate ports so they can
# run concurrently (mirrors the four Streamlit Cloud links). Ctrl-C stops them all.
#
#   active-conjunctive    http://localhost:8511
#   active-disjunctive    http://localhost:8512
#   passive-conjunctive   http://localhost:8513
#   passive-disjunctive   http://localhost:8514
#
# Locally all four read Firebase from the repo's .streamlit/secrets.toml. On
# Streamlit Cloud each app has its OWN secrets, so they hit four separate databases.

set -e
cd "$(dirname "$0")/.."   # repo root, so .streamlit/secrets.toml and env/ resolve

for _env in .venv venv; do
  if [ -f "$_env/bin/activate" ] && [ -x "$_env/bin/streamlit" ]; then
    source "$_env/bin/activate"; break
  fi
done
command -v streamlit >/dev/null 2>&1 || { echo "streamlit not found; activate an env with streamlit installed"; exit 1; }

pids=()
cleanup() { kill "${pids[@]}" 2>/dev/null || true; }
trap cleanup EXIT INT TERM

streamlit run extension_experiments/active_conjunctive.py  --server.port 8511 & pids+=($!)
streamlit run extension_experiments/active_disjunctive.py  --server.port 8512 & pids+=($!)
streamlit run extension_experiments/passive_conjunctive.py --server.port 8513 & pids+=($!)
streamlit run extension_experiments/passive_disjunctive.py --server.port 8514 & pids+=($!)

echo "Started 4 apps on ports 8511-8514. Press Ctrl-C to stop all."
wait
