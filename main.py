#!/usr/bin/env python3
"""
Academic Multimodal LLM Experiment System - Main Application Entry Point

This is the primary entry point for the Academic Multimodal LLM Experiment System.
It provides a command-line interface for launching the system with various modes
and configurations, handles system initialization, and manages graceful startup
and shutdown procedures.
"""

import sys
import os
import argparse
import logging
import signal
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import asdict

# Add src to Python path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Core system imports
from src.core.interfaces import ComplianceMode, HardwareConfig, SystemError
from src.core.system_integration import SystemIntegration
from src.ui.research_interface import ResearchInterface
from src.api.api_server import APIServer

# Ensure logs directory exists
Path("logs").mkdir(exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('logs/system.log', mode='a')
    ]
)

logger = logging.getLogger(__name__)


class SystemController:
    """
    Main system controller that coordinates all components of the Academic
    Multimodal LLM Experiment System.
    """
    
    def __init__(self, config_path: str = "config/system_config.json"):
        """
        Initialize the system controller.
        
        Args:
            config_path: Path to system configuration file
        """
        self.config_path = config_path
        self.config: Dict[str, Any] = {}
        
        # Core system integration
        self.system_integration: Optional[SystemIntegration] = None
        self.research_interface: Optional[ResearchInterface] = None
        self.api_server: Optional[APIServer] = None
        
        # System state
        self.is_initialized = False
        self.is_running = False
        self.shutdown_requested = False
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        logger.info("SystemController created")
    
    def initialize(self) -> bool:
        """
        Initialize the complete system with proper error handling.
        
        Returns:
            bool: True if initialization successful, False otherwise
        """
        try:
            logger.info("Starting system initialization...")
            
            # Step 1: Load configuration
            if not self._load_configuration():
                logger.error("Failed to load system configuration")
                return False
            
            # Step 2: Create necessary directories
            if not self._create_directories():
                logger.error("Failed to create system directories")
                return False
            
            # Step 3: Initialize system integration (handles hardware, pipelines, etc.)
            if not self._initialize_system_integration():
                logger.error("Failed to initialize system integration")
                return False
            
            # Step 4: Initialize user interface
            if not self._initialize_user_interface():
                logger.error("Failed to initialize user interface")
                return False
            
            # Step 5: Initialize API server (optional)
            if self.config.get("enable_api", True):
                if not self._initialize_api_server():
                    logger.warning("Failed to initialize API server - continuing without API")
            
            self.is_initialized = True
            logger.info("System initialization completed successfully")
            
            # Log system summary
            self._log_system_summary()
            
            return True
            
        except Exception as e:
            logger.error(f"System initialization failed: {e}")
            return False
    
    def _load_configuration(self) -> bool:
        """Load system configuration from file."""
        try:
            config_file = Path(self.config_path)
            
            if config_file.exists():
                with open(config_file, 'r') as f:
                    self.config = json.load(f)
                logger.info(f"Configuration loaded from {self.config_path}")
            else:
                # Create default configuration
                self.config = self._get_default_config()
                
                # Ensure config directory exists
                config_file.parent.mkdir(parents=True, exist_ok=True)
                
                # Save default configuration
                with open(config_file, 'w') as f:
                    json.dump(self.config, f, indent=2)
                
                logger.info(f"Created default configuration at {self.config_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            return False
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default system configuration."""
        return {
            "data_dir": "data",
            "models_dir": "models",
            "experiments_dir": "experiments",
            "cache_dir": "cache",
            "logs_dir": "logs",
            "hardware_config": None,
            "default_compliance_mode": "research_safe",
            "max_concurrent_generations": 1,
            "memory_cleanup_threshold": 0.85,
            "log_level": "INFO",
            "log_file": "logs/system.log",
            "enable_api": True,
            "api_host": "127.0.0.1",
            "api_port": 8000,
            "ui_host": "127.0.0.1",
            "ui_port": 7860,
            "ui_share": False,
            "auto_detect_hardware": True,
            "enable_telemetry": False
        }
    
    def _create_directories(self) -> bool:
        """Create necessary system directories."""
        try:
            directories = [
                self.config.get("data_dir", "data"),
                self.config.get("models_dir", "models"),
                self.config.get("experiments_dir", "experiments"),
                self.config.get("cache_dir", "cache"),
                self.config.get("logs_dir", "logs")
            ]
            
            for directory in directories:
                Path(directory).mkdir(parents=True, exist_ok=True)
                logger.debug(f"Created directory: {directory}")
            
            logger.info("System directories created successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create directories: {e}")
            return False
    
    def _initialize_system_integration(self) -> bool:
        """Initialize the system integration layer."""
        try:
            logger.info("Initializing system integration...")
            
            # Create system integration
            self.system_integration = SystemIntegration()
            
            # Initialize with configuration
            if not self.system_integration.initialize(self.config):
                logger.error("System integration initialization failed")
                return False
            
            # Update configuration with detected hardware
            if self.system_integration.hardware_config:
                self.config["hardware_config"] = asdict(self.system_integration.hardware_config)
            
            logger.info("System integration initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"System integration initialization failed: {e}")
            return False
    
    def _initialize_user_interface(self) -> bool:
        """Initialize the research user interface."""
        try:
            logger.info("Initializing research interface...")
            
            # Create research interface
            self.research_interface = ResearchInterface(
                system_controller=self.system_integration,
                experiment_tracker=self.system_integration.experiment_tracker if self.system_integration else None,
                compliance_engine=None  # Will be initialized later
            )
            
            # Initialize the interface
            if not self.research_interface.initialize():
                logger.error("Failed to initialize research interface")
                return False
            
            logger.info("Research interface initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"User interface initialization failed: {e}")
            return False
    
    def _initialize_api_server(self) -> bool:
        """Initialize the API server."""
        try:
            logger.info("Initializing API server...")
            
            # Create API server
            self.api_server = APIServer(
                system_controller=self.system_integration,
                host=self.config.get("api_host", "127.0.0.1"),
                port=self.config.get("api_port", 8000)
            )
            
            logger.info("API server initialized")
            return True
            
        except Exception as e:
            logger.error(f"API server initialization failed: {e}")
            return False
    
    def _log_system_summary(self) -> None:
        """Log a summary of the initialized system."""
        logger.info("=== System Summary ===")
        
        if self.system_integration and self.system_integration.hardware_config:
            hw_config = self.system_integration.hardware_config
            logger.info(f"Hardware: {hw_config.gpu_model}")
            logger.info(f"VRAM: {hw_config.vram_size}MB")
            logger.info(f"CPU Cores: {hw_config.cpu_cores}")
            logger.info(f"RAM: {hw_config.ram_size}MB")
            logger.info(f"CUDA Available: {hw_config.cuda_available}")
            logger.info(f"Optimization Level: {hw_config.optimization_level}")
        
        logger.info(f"Compliance Mode: {self.config.get('default_compliance_mode')}")
        logger.info(f"UI Port: {self.config.get('ui_port')}")
        if self.api_server:
            logger.info(f"API Port: {self.config.get('api_port')}")
        logger.info("=====================")
    
    def run(self, mode: str = "ui") -> None:
        """
        Run the system in the specified mode.
        
        Args:
            mode: Running mode ("ui", "api", "both", "cli")
        """
        if not self.is_initialized:
            logger.error("System not initialized - call initialize() first")
            return
        
        try:
            self.is_running = True
            logger.info(f"Starting system in {mode} mode...")
            
            if mode == "ui":
                self._run_ui_mode()
            elif mode == "api":
                self._run_api_mode()
            elif mode == "both":
                self._run_both_modes()
            elif mode == "cli":
                self._run_cli_mode()
            else:
                logger.error(f"Unknown mode: {mode}")
                return
            
        except KeyboardInterrupt:
            logger.info("Received keyboard interrupt")
        except Exception as e:
            logger.error(f"System runtime error: {e}")
        finally:
            self.shutdown()
    
    def _run_ui_mode(self) -> None:
        """Run in UI-only mode."""
        logger.info("Starting research interface...")
        
        self.research_interface.launch(
            share=self.config.get("ui_share", False),
            server_name=self.config.get("ui_host", "127.0.0.1"),
            server_port=self.config.get("ui_port", 7860)
        )
    
    def _run_api_mode(self) -> None:
        """Run in API-only mode."""
        if not self.api_server:
            logger.error("API server not initialized")
            return
        
        logger.info("Starting API server...")
        self.api_server.start()
        
        # Keep running until shutdown
        try:
            while self.is_running and not self.shutdown_requested:
                time.sleep(1)
        except KeyboardInterrupt:
            pass
    
    def _run_both_modes(self) -> None:
        """Run both UI and API modes."""
        import threading
        
        if not self.api_server:
            logger.error("API server not initialized")
            return
        
        # Start API server in background thread
        api_thread = threading.Thread(
            target=self.api_server.start,
            daemon=True
        )
        api_thread.start()
        
        # Start UI in main thread
        self._run_ui_mode()
    
    def _run_cli_mode(self) -> None:
        """Run in CLI-only mode for testing and debugging."""
        logger.info("Starting CLI mode...")
        
        print("\nAcademic Multimodal LLM Experiment System - CLI Mode")
        print("Type 'help' for available commands, 'quit' to exit")
        
        while self.is_running and not self.shutdown_requested:
            try:
                command = input("\n> ").strip().lower()
                
                if command == "quit" or command == "exit":
                    break
                elif command == "help":
                    self._show_cli_help()
                elif command == "status":
                    self._show_system_status()
                elif command == "hardware":
                    self._show_hardware_info()
                elif command.startswith("generate"):
                    self._handle_cli_generation(command)
                else:
                    print(f"Unknown command: {command}")
                    
            except KeyboardInterrupt:
                break
            except EOFError:
                break
    
    def _show_cli_help(self) -> None:
        """Show CLI help information."""
        print("\nAvailable commands:")
        print("  help     - Show this help message")
        print("  status   - Show system status")
        print("  hardware - Show hardware information")
        print("  generate - Generate content (image/video)")
        print("  quit     - Exit the system")
    
    def _show_system_status(self) -> None:
        """Show current system status."""
        print(f"\nSystem Status:")
        print(f"  Initialized: {self.is_initialized}")
        print(f"  Running: {self.is_running}")
        
        if self.system_integration:
            status = self.system_integration.get_system_status()
            if status.get('hardware'):
                hw = status['hardware']
                print(f"  Hardware: {hw.get('gpu_model', 'Unknown')}")
                print(f"  VRAM: {hw.get('vram_size', 0)}MB")
            print(f"  Active Generations: {status.get('active_generations', 0)}")
        
        print(f"  Compliance Mode: {self.config.get('default_compliance_mode')}")
    
    def _show_hardware_info(self) -> None:
        """Show detailed hardware information."""
        if self.system_integration and self.system_integration.hardware_detector:
            detailed_info = self.system_integration.hardware_detector.get_detailed_info()
            print(f"\nDetailed Hardware Information:")
            print(json.dumps(detailed_info, indent=2))
    
    def _handle_cli_generation(self, command: str) -> None:
        """Handle CLI generation commands."""
        print("CLI generation not yet implemented - use UI mode for generation")
    
    def _signal_handler(self, signum, frame) -> None:
        """Handle system signals for graceful shutdown."""
        logger.info(f"Received signal {signum}")
        self.shutdown_requested = True
        self.shutdown()
    
    def shutdown(self) -> None:
        """Perform graceful system shutdown."""
        if not self.is_running:
            return
        
        logger.info("Starting system shutdown...")
        self.is_running = False
        
        try:
            # Shutdown API server
            if self.api_server:
                logger.info("Shutting down API server...")
                self.api_server.shutdown()
            
            # Clean up system integration
            if self.system_integration:
                logger.info("Cleaning up system integration...")
                self.system_integration.cleanup()
            
            # Save configuration
            self._save_configuration()
            
            logger.info("System shutdown completed")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
    
    def _save_configuration(self) -> None:
        """Save current configuration to file."""
        try:
            with open(self.config_path, 'w') as f:
                json.dump(self.config, f, indent=2)
            logger.info("Configuration saved")
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")


def create_argument_parser() -> argparse.ArgumentParser:
    """Create command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Academic Multimodal LLM Experiment System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                    # Start with UI interface
  python main.py --mode api         # Start API server only
  python main.py --mode both        # Start both UI and API
  python main.py --mode cli         # Start CLI mode for testing
  python main.py --config custom.json  # Use custom configuration
        """
    )
    
    parser.add_argument(
        "--mode",
        choices=["ui", "api", "both", "cli"],
        default="ui",
        help="System running mode (default: ui)"
    )
    
    parser.add_argument(
        "--config",
        default="config/system_config.json",
        help="Path to configuration file (default: config/system_config.json)"
    )
    
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO)"
    )
    
    parser.add_argument(
        "--ui-port",
        type=int,
        default=7860,
        help="UI server port (default: 7860)"
    )
    
    parser.add_argument(
        "--api-port",
        type=int,
        default=8000,
        help="API server port (default: 8000)"
    )
    
    parser.add_argument(
        "--ui-share",
        action="store_true",
        help="Create public UI link (Gradio share)"
    )
    
    parser.add_argument(
        "--no-hardware-detection",
        action="store_true",
        help="Skip automatic hardware detection"
    )
    
    parser.add_argument(
        "--compliance-mode",
        choices=[mode.value for mode in ComplianceMode],
        help="Default copyright compliance mode"
    )
    
    return parser


def main():
    """Main application entry point."""
    # Parse command-line arguments
    parser = create_argument_parser()
    args = parser.parse_args()
    
    # Set logging level
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    try:
        logger.info("Starting Academic Multimodal LLM Experiment System")
        logger.info(f"Python version: {sys.version}")
        logger.info(f"Working directory: {os.getcwd()}")
        
        # Create system controller
        system = SystemController(config_path=args.config)
        
        # Override configuration with command-line arguments
        if args.ui_port != 7860:
            system.config["ui_port"] = args.ui_port
        if args.api_port != 8000:
            system.config["api_port"] = args.api_port
        if args.ui_share:
            system.config["ui_share"] = True
        if args.no_hardware_detection:
            system.config["auto_detect_hardware"] = False
        if args.compliance_mode:
            system.config["default_compliance_mode"] = args.compliance_mode
        
        # Initialize system
        if not system.initialize():
            logger.error("System initialization failed")
            sys.exit(1)
        
        # Run system
        system.run(mode=args.mode)
        
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt")
    except Exception as e:
        logger.error(f"Application error: {e}")
        sys.exit(1)
    finally:
        logger.info("Application terminated")


if __name__ == "__main__":
    main()