#!/bin/bash

# Script to run the Nexiom Text Game
# Make sure to activate the virtual environment and run the Streamlit app

echo "Starting Nexiom Text Game..."
echo "Make sure you have configured your .env file with Firebase credentials"
echo ""

# Activate virtual environment
source venv/bin/activate

# Run the Streamlit app
streamlit run app.py

