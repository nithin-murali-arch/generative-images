"""
API Server wrapper for the Academic Multimodal LLM Experiment System.

This module provides a wrapper around the FastAPI server to integrate
with the system controller and provide a consistent interface.
"""

import logging
import threading
import time
from typing import Optional

import uvicorn

logger = logging.getLogger(__name__)


class APIServer:
    """
    API Server wrapper that integrates with the system controller.
    
    This class provides a consistent interface for starting and stopping
    the FastAPI server and integrating it with the system integration layer.
    """
    
    def __init__(self, system_controller=None, host: str = "127.0.0.1", port: int = 8000):
        """
        Initialize the API server.
        
        Args:
            system_controller: System integration controller
            host: Server host address
            port: Server port
        """
        self.system_controller = system_controller
        self.host = host
        self.port = port
        self.server_thread: Optional[threading.Thread] = None
        self.is_running = False
        
        logger.info(f"APIServer created for {host}:{port}")
    
    def start(self) -> None:
        """Start the API server in a background thread."""
        if self.is_running:
            logger.warning("API server is already running")
            return
        
        logger.info(f"Starting API server on {self.host}:{self.port}")
        
        # Update the API state with our system controller
        if self.system_controller:
            from .dependencies import api_state
            api_state.system_integration = self.system_controller
            api_state.llm_controller = self.system_controller.llm_controller
            api_state.hardware_config = self.system_controller.hardware_config
            api_state.image_pipeline = self.system_controller.image_pipeline
            api_state.video_pipeline = self.system_controller.video_pipeline
            api_state.experiment_tracker = self.system_controller.experiment_tracker
        
        # Start server in background thread
        self.server_thread = threading.Thread(
            target=self._run_server,
            daemon=True
        )
        self.server_thread.start()
        self.is_running = True
        
        # Give server time to start
        time.sleep(2)
        logger.info("API server started successfully")
    
    def _run_server(self) -> None:
        """Run the FastAPI server."""
        try:
            uvicorn.run(
                "src.api.server:app",
                host=self.host,
                port=self.port,
                log_level="info",
                access_log=False
            )
        except Exception as e:
            logger.error(f"API server error: {e}")
            self.is_running = False
    
    def shutdown(self) -> None:
        """Shutdown the API server."""
        if not self.is_running:
            return
        
        logger.info("Shutting down API server...")
        self.is_running = False
        
        # Note: uvicorn doesn't provide a clean way to shutdown from external thread
        # In a production system, you'd want to use a more sophisticated approach
        logger.info("API server shutdown requested")
    
    def get_status(self) -> dict:
        """Get API server status."""
        return {
            'running': self.is_running,
            'host': self.host,
            'port': self.port,
            'url': f"http://{self.host}:{self.port}"
        }