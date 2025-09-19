#!/bin/bash
echo "Starting ChromiumAI - AI-Native Browser"
echo

echo "[1/3] Starting FastAPI service..."
cd ai-integration/api-service/python
gnome-terminal --title="FastAPI Service" -- bash -c "python main.py; exec bash" &
cd ../../..

echo "[2/3] Waiting for FastAPI service to start..."
sleep 5

echo "[3/3] Starting ChromiumAI Electron browser..."
cd ai-browser-electron
gnome-terminal --title="ChromiumAI Browser" -- bash -c "npm start; exec bash" &
cd ..

echo
echo "ChromiumAI is starting up!"
echo "- FastAPI service: http://localhost:3456"
echo "- AI Browser: chrome://ai-browser/"
echo
echo "Press Ctrl+C to exit..."
wait
