#!/bin/bash

# Run the Nexiom Text Game with CONJUNCTIVE rules only (main experiment).
# Use a different port (8502) so you can run this and the default app (8501) at the same time.
#
# Local:  Default http://localhost:8501  |  Conjunctive http://localhost:8502
# Cloud:  Default https://nexiom-text-game.streamlit.app/
#         Conjunctive-only https://nexiom-text-game.streamlit.app/?e=nex1

echo "Starting Nexiom Text Game (conjunctive rules only)..."
echo "Port 8502 - run run_app.sh for default (disjunctive) on port 8501."
echo "Streamlit Cloud: use ?e=nex1 for conjunctive-only link."
echo "Make sure you have configured your .env file with Firebase credentials"
echo ""

# Activate virtual environment
source venv/bin/activate

# Conjunctive-only main game; run on port 8502 so both experiments can run simultaneously
export NEXIOM_MAIN_RULE=conjunctive
streamlit run app.py --server.port 8502
