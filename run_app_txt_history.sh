#!/bin/bash

# Run the Nexiom Action-History app: load action history from a txt file,
# view it on screen, then answer object identification and rule inference.
# No object interaction; participants update the txt file externally.
#
# Port 8503 so it can run alongside main app (8501) and conjunctive app (8502).
# Local: http://localhost:8503

echo "Starting Nexiom Action-History app (txt file + Q&A only) ..."
echo "Port 8503 - no Firebase required."
echo ""

if [ -d "venv" ]; then
  source venv/bin/activate
fi

streamlit run passive_app/app_txt_history.py --server.port 8503
