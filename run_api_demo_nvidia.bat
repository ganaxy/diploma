@echo off
setlocal
cd /d "%~dp0"

if not exist .venv_api_nvidia\Scripts\python.exe (
  echo NVIDIA virtual environment was not found.
  echo Run setup_api_demo_nvidia.bat first.
  pause
  exit /b 1
)

set API_FORCE_CPU=
set GRADIO_FORCE_CPU=

echo Starting MN-BERT API demo with NVIDIA/CUDA PyTorch...
echo.
echo Open this in the browser after the server starts:
echo   http://127.0.0.1:8000/
echo.

.venv_api_nvidia\Scripts\python.exe -m uvicorn api_app.main:app --host 127.0.0.1 --port 8000
echo.
echo Server stopped.
pause
