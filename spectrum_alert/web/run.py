#!/usr/bin/env python3
"""
SpectrumAlert Web Interface Launcher
"""

import os
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
        
        # Get port from environment variable, argument, or use default
        port = int(os.getenv("SPECTRUM_ALERT_PORT", "8000"))
        host = os.getenv("SPECTRUM_ALERT_HOST", "0.0.0.0")
        
        # Allow command line override via sys.argv if provided
        if "--port" in sys.argv:
            port_index = sys.argv.index("--port")
            if port_index + 1 < len(sys.argv):
                port = int(sys.argv[port_index + 1])
        
        if "--host" in sys.argv:
            host_index = sys.argv.index("--host")
            if host_index + 1 < len(sys.argv):
                host = sys.argv[host_index + 1]
        
        # Create web app
        web_app = SpectrumAlertWebApp()
        app_instance = web_app.app
        
        logger.info("Starting SpectrumAlert Web Dashboard...")
        logger.info(f"Dashboard will be available at: http://localhost:{port}")
        logger.info("Press Ctrl+C to stop")
        
        # Run the server
        uvicorn.run(
            app_instance,
            host=host,
            port=port,
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
