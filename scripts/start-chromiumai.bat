@echo off
echo Starting ChromiumAI - AI-Native Browser
echo.

echo [1/3] Starting FastAPI service...
cd ai-integration\api-service\python
start "FastAPI Service" cmd /k "python main.py"
cd ..\..\..

echo [2/3] Waiting for FastAPI service to start...
timeout /t 5 /nobreak > nul

echo [3/3] Starting ChromiumAI Electron browser...
cd ai-browser-electron
start "ChromiumAI Browser" cmd /k "npm start"
cd ..

echo.
echo ChromiumAI is starting up!
echo - FastAPI service: http://localhost:3456
echo - AI Browser: chrome://ai-browser/
echo.
echo Press any key to exit...
pause > nul
