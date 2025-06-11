import os
import sys
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

# Add the current directory to Python path
sys.path.insert(0, '/app')
sys.path.insert(0, '/app/dashboard/backend')


def main():
    """Main application entry point for Railway"""
    try:
        # Try to import the dashboard backend
        try:
            from dashboard.backend.main import app
            logger.info("Successfully imported dashboard backend")
        except ImportError as e:
            logger.warning(f"Dashboard backend import failed: {e}")
            # Create a basic FastAPI app as fallback
            from fastapi import FastAPI
            app = FastAPI(title="Synthetic Data Generator - Basic Mode")

            @app.get("/")
            async def root():
                return {"message": "Synthetic Data Generator API - Basic Mode"}

            @app.get("/api/health")
            async def health():
                return {"status": "healthy", "mode": "basic"}

        # Railway uses the PORT environment variable
        import uvicorn
        port = int(os.environ.get("PORT", 8000))
        host = "0.0.0.0"

        logger.info(f"Starting server on {host}:{port}")
        uvicorn.run(app, host=host, port=port)

    except Exception as e:
        logger.error(f"Failed to start application: {e}")
        raise


if __name__ == "__main__":
    main()