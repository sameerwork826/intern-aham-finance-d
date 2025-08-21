@echo off
echo ========================================
echo AHM Finance Loan Officer Project Setup
echo ========================================
echo.

echo Installing Python dependencies...
pip install -r requirements.txt

echo.
echo Checking if Ollama is installed...
ollama --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Ollama not found. Please install Ollama from https://ollama.ai/
    echo After installation, run: ollama pull gemma:2b
    pause
    exit /b 1
)

echo.
echo Pulling required Ollama model...
ollama pull gemma:2b

echo.
echo Setup complete! To run the application:
echo 1. Open Command Prompt (cmd)
echo 2. Navigate to this directory
echo 3. Run: python scripts/app.py
echo.
pause
