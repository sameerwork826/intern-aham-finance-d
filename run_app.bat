@echo off
echo ========================================
echo Starting AHM Finance Loan Officer App (API Version)
echo ========================================
echo.

echo Starting Streamlit application with API approach...
echo The app will open in your default browser at http://localhost:8501
echo.
echo This version uses Ollama REST API to avoid encoding issues.
echo Make sure Ollama is running with: ollama serve
echo.
echo Press Ctrl+C to stop the application
echo.

cd /d "%~dp0"
streamlit run scripts/app.py

pause
