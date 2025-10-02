#!/bin/bash

# Script to run the Text-Only Blicket Game
# This version uses only text descriptions - no images

echo "Starting Blicket Text Adventure..."
echo "This version uses only text descriptions - no visual images"
echo ""

# Activate virtual environment
source venv/bin/activate

# Run the Text Streamlit app
streamlit run app_text.py

