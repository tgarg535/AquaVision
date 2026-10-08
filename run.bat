@echo off
title Aqua Vision - Underwater Marine Species Detector
cd /d "%~dp0"
echo Activating virtual environment...
call venv\Scripts\activate.bat
echo Starting Streamlit app...
streamlit run app.py
pause
