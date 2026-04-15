@echo off
title XM360 Trade Dashboard
echo.
echo  ==========================================
echo   XM360 Trade Dashboard
echo  ==========================================
echo.
echo  [1/2] Installing dependencies...
pip install -r requirements.txt --quiet
echo.
echo  [2/2] Starting server...
echo.
echo  Dashboard ready at:  http://localhost:5000
echo  Press Ctrl+C to stop.
echo.
python backend/server.py
pause
