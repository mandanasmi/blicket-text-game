#!/bin/bash

# Run the passive app: no comprehension, action history as text only,
# blicket questions and rule inference only. Data saved to Firebase.
#
# Port 8504. Run from project root so .streamlit/secrets.toml and .env are found.
# Local: http://localhost:8504

set -e
cd "$(dirname "$0")"
echo "Starting Nexiom passive app (no exploration, Firebase) ..."
echo "Port 8504"
echo ""

if [ -d "venv" ]; then
  source venv/bin/activate
fi

streamlit run passive_app/app.py --server.port 8504
