@echo off
title XM360 Dashboard - MT5 Live Edition
echo.
echo  ==========================================
echo   XM360 Trade Dashboard - LIVE DATA
echo   Connected directly to XM's servers
echo  ==========================================
echo.
echo  IMPORTANT: Make sure MetaTrader5 (XM) is
echo  open and logged in before continuing!
echo.
pause

echo  Installing packages...
pip install MetaTrader5 pandas numpy flask flask-cors yfinance --quiet

echo.
echo  Starting live dashboard...
echo  Open browser: http://localhost:5000
echo.
python backend\server_mt5.py
pause
