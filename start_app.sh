#!/bin/bash

echo "========================================"
echo "Starting Emotion Detection Chatbot"
echo "========================================"
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ] || [ ! -f "venv/bin/activate" ]; then
    echo "[ERROR] Virtual environment not found!"
    echo "Please run ./setup.sh first to install dependencies."
    exit 1
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate
if [ $? -ne 0 ]; then
    echo "[ERROR] Failed to activate virtual environment!"
    exit 1
fi

# Change to backend directory
echo "Changing to backend directory..."
cd backend

# Start Flask application
echo ""
echo "Starting Flask application..."
echo "Open your browser and navigate to: http://localhost:5000"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""
python app.py

