@echo off
cd /d "%~dp0"

if not exist "rag_env\Scripts\python.exe" (
    echo.
    echo rag_env was not found.
    echo Please create the environment and install requirements first.
    echo.
    pause
    exit /b 1
)

call rag_env\Scripts\activate.bat
python -m streamlit run streamlit_app.py
pause