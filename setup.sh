#!/bin/bash

echo "========================================"
echo "Emotion Detection Chatbot - Setup"
echo "========================================"
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "[ERROR] Python 3 is not installed or not in PATH!"
    echo "Please install Python 3.8 or higher"
    exit 1
fi

echo "[1/8] Checking Python version..."
python3 --version
echo ""

# Check if virtual environment exists
if [ -d "venv" ] && [ -f "venv/bin/activate" ]; then
    echo "[2/8] Virtual environment already exists. Skipping creation."
else
    echo "[2/8] Creating virtual environment..."
    python3 -m venv venv
    if [ $? -ne 0 ]; then
        echo "[ERROR] Failed to create virtual environment!"
        exit 1
    fi
    echo "Virtual environment created successfully!"
fi
echo ""

# Activate virtual environment
echo "[3/8] Activating virtual environment..."
source venv/bin/activate
if [ $? -ne 0 ]; then
    echo "[ERROR] Failed to activate virtual environment!"
    exit 1
fi
echo ""

# Upgrade pip
echo "[4/8] Upgrading pip..."
python -m pip install --upgrade pip --quiet
if [ $? -ne 0 ]; then
    echo "[WARNING] Failed to upgrade pip. Continuing anyway..."
else
    echo "pip upgraded!"
fi
echo ""

# Install requirements
echo "[5/8] Installing Python packages from requirements.txt..."
echo "This may take several minutes..."
python -m pip install -r requirements.txt
if [ $? -ne 0 ]; then
    echo "[ERROR] Failed to install requirements!"
    exit 1
fi
echo ""
echo "All packages installed successfully!"
echo ""

# Download NLTK data and pre-download transformer model
echo "[6/8] Downloading NLTK data and pre-loading transformer model..."
echo "This may take a few minutes (downloading ~500MB model)..."
python setup_helper.py
if [ $? -ne 0 ]; then
    echo "[WARNING] Some downloads may have failed. The application will download missing resources on first run."
fi
echo ""

# Check/create necessary directories
echo "[7/8] Creating necessary directories..."
mkdir -p models
mkdir -p backend/static
echo "Directories ready!"
echo ""

# Verify and regenerate data files if needed
echo "[8/8] Verifying and preparing data files..."
if [ -f "data/emotion_responses.json" ]; then
    echo "[OK] emotion_responses.json found"
else
    echo "[WARNING] emotion_responses.json not found. Application may still work with defaults."
fi

# Generate emotion_dataset.csv if it doesn't exist
if [ ! -f "data/emotion_dataset.csv" ]; then
    echo "Generating emotion_dataset.csv from source files..."
    python src/convert_to_csv.py
    if [ $? -ne 0 ]; then
        echo "[WARNING] Failed to generate emotion_dataset.csv. Training scripts may not work."
    else
        echo "[OK] emotion_dataset.csv generated successfully!"
    fi
else
    echo "[OK] emotion_dataset.csv already exists"
fi
echo ""

echo "========================================"
echo "Setup completed successfully!"
echo "========================================"
echo ""
echo "Next steps:"
echo "1. Run ./start_app.sh to start the application"
echo "2. Open your browser and go to http://localhost:5000"
echo ""
echo "Note: If you get permission errors, run: chmod +x start_app.sh"
echo ""

