#!/bin/bash
# Start script with fallback for Railway deployment

echo "🚀 Starting Synthetic Data Generator..."

# Try to start the main application
echo "Attempting to start main application (app.py)..."
python app.py 2>&1 &
MAIN_PID=$!

# Wait a few seconds to see if it started successfully
sleep 5

# Check if the main process is still running
if kill -0 $MAIN_PID 2>/dev/null; then
    echo "✅ Main application started successfully (PID: $MAIN_PID)"
    wait $MAIN_PID
else
    echo "⚠️ Main application failed, starting fallback (app-simple.py)..."
    python app-simple.py
fi