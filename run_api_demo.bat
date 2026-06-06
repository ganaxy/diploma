@echo off
setlocal
cd /d "%~dp0"
set API_FORCE_CPU=1
echo Starting MN-BERT API demo...
echo.
echo Open this in the browser after the server starts:
echo   http://127.0.0.1:8000/
echo.
python -m uvicorn api_app.main:app --host 127.0.0.1 --port 8000
echo.
echo Server stopped.
pause
