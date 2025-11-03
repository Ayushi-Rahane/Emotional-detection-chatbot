@echo off
echo ========================================
echo Starting Emotion Detection Chatbot
echo ========================================
echo.

REM Check if virtual environment exists
if not exist "venv\Scripts\activate.bat" (
    echo [ERROR] Virtual environment not found!
    echo Please run setup.bat first to install dependencies.
    pause
    exit /b 1
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo [ERROR] Failed to activate virtual environment!
    pause
    exit /b 1
)

REM Change to backend directory
echo Changing to backend directory...
cd backend

REM Start Flask application
echo.
echo Starting Flask application...
echo Open your browser and navigate to: http://localhost:5000
echo.
echo Press Ctrl+C to stop the server
echo.
python app.py

pause

