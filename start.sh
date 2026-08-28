#!/bin/bash
# Start script with fallback for Railway deployment

echo "🚀 Starting Synthetic Data Generator..."

# Start the application. (This fallback used to try app-simple.py if
# app.py crashed on startup; app-simple.py has been removed as dead code,
# and in any case Railway's own startCommand runs `python app.py` directly
# and never invokes this script -- restarts on crash are handled by
# railway.toml's restartPolicy instead.)
echo "Starting application (app.py)..."
exec python app.py