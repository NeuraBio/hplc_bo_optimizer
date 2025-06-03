#!/bin/bash
# Development startup script for HPLC BO Optimizer
# This script installs dependencies and starts the Streamlit app with hot reloading enabled

set -e

echo "Starting HPLC BO Optimizer development environment..."

# Create log directory if it doesn't exist
mkdir -p /app/logs

# Set environment variables
export PYTHONPATH=/app

# Ensure we're in the app directory
cd /app

# Start Streamlit with hot reloading enabled
echo "Starting Streamlit app with hot reloading..."
cd /app
poetry run streamlit run app/main.py \
    --server.port=8501 \
    --server.headless=true \
    --server.enableCORS=false \
    --server.enableXsrfProtection=false \
    --server.runOnSave=true \
    --browser.gatherUsageStats=false
