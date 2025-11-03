@echo off
echo ========================================
echo Emotion Detection Chatbot - Setup
echo ========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH!
    echo Please install Python 3.8 or higher from https://www.python.org/
    pause
    exit /b 1
)

echo [1/8] Checking Python version...
python --version
echo.

REM Check if virtual environment exists
if exist "venv\Scripts\activate.bat" (
    echo [2/8] Virtual environment already exists. Skipping creation.
) else (
    echo [2/8] Creating virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment!
        pause
        exit /b 1
    )
    echo Virtual environment created successfully!
)
echo.

REM Activate virtual environment
echo [3/8] Activating virtual environment...
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo [ERROR] Failed to activate virtual environment!
    pause
    exit /b 1
)
echo.

REM Upgrade pip
echo [4/8] Upgrading pip...
python -m pip install --upgrade pip --quiet
echo pip upgraded!
echo.

REM Install requirements
echo [5/8] Installing Python packages from requirements.txt...
echo This may take several minutes...
python -m pip install -r requirements.txt
if errorlevel 1 (
    echo [ERROR] Failed to install requirements!
    pause
    exit /b 1
)
echo.
echo All packages installed successfully!
echo.

REM Download NLTK data and pre-download transformer model
echo [6/8] Downloading NLTK data and pre-loading transformer model...
echo This may take a few minutes (downloading ~500MB model)...
python setup_helper.py
if errorlevel 1 (
    echo [WARNING] Some downloads may have failed. The application will download missing resources on first run.
)
echo.

REM Check/create necessary directories
echo [7/8] Creating necessary directories...
if not exist "models" mkdir models
if not exist "backend\static" mkdir backend\static
echo Directories ready!
echo.

REM Verify and regenerate data files if needed
echo [8/8] Verifying and preparing data files...
if exist "data\emotion_responses.json" (
    echo [OK] emotion_responses.json found
) else (
    echo [WARNING] emotion_responses.json not found. Application may still work with defaults.
)

REM Generate emotion_dataset.csv if it doesn't exist
if not exist "data\emotion_dataset.csv" (
    echo Generating emotion_dataset.csv from source files...
    python src\convert_to_csv.py
    if errorlevel 1 (
        echo [WARNING] Failed to generate emotion_dataset.csv. Training scripts may not work.
    ) else (
        echo [OK] emotion_dataset.csv generated successfully!
    )
) else (
    echo [OK] emotion_dataset.csv already exists
)
echo.

echo ========================================
echo Setup completed successfully!
echo ========================================
echo.
echo Next steps:
echo 1. Run start_app.bat to start the application
echo 2. Open your browser and go to http://localhost:5000
echo.
pause

