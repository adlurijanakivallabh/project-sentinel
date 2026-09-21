@echo off
echo ===================================================
echo   Project Sentinel - Full Stack PyTorch Web Dashboard
echo ===================================================
echo Starting Flask Python backend and Neural Network wrapper...
echo Server running on Port 8082
echo Press CTRL+C to stop.
echo ===================================================
start http://localhost:8082/
.\env\Scripts\python.exe app.py
