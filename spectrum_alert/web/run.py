#!/usr/bin/env python3
"""
SpectrumAlert Web Interface Launcher
"""

import sys
import asyncio
import logging
from pathlib import Path

# Add the parent directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Launch the web interface"""
    try:
        # Import FastAPI dependencies
        import uvicorn
        from spectrum_alert.web.app import SpectrumAlertWebApp
        
        # Create web app
        web_app = SpectrumAlertWebApp()
        app_instance = web_app.app
        
        logger.info("Starting SpectrumAlert Web Dashboard...")
        logger.info("Dashboard will be available at: http://localhost:8000")
        logger.info("Press Ctrl+C to stop")
        
        # Run the server
        uvicorn.run(
            app_instance,
            host="0.0.0.0",
            port=8000,
            log_level="info",
            access_log=True
        )
        
    except ImportError as e:
        logger.error("Missing dependencies. Please install with:")
        logger.error("pip install fastapi uvicorn jinja2 python-multipart")
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("Shutting down web dashboard...")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Failed to start web dashboard: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
