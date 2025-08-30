#!/usr/bin/env python3
"""
Modern AI Content Generator - Main Application

A clean, user-friendly interface for generating images and videos using the latest AI models.
Features easy/advanced mode toggle and automatic hardware optimization.
"""

import sys
import os
import argparse
import logging
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def main():
    """Main application entry point."""
    parser = argparse.ArgumentParser(
        description="AI Content Generator - Generate images and videos with AI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python app.py                    # Launch with default settings
  python app.py --port 8080        # Launch on custom port
  python app.py --share            # Create public link
  python app.py --debug            # Enable debug logging
        """
    )
    
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Port to run the interface on (default: 7860)"
    )
    
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host to bind to (default: 127.0.0.1)"
    )
    
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create a public Gradio link"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )
    
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        logger.info("🚀 Starting AI Content Generator")
        
        # Import and create the modern interface
        from src.ui.modern_interface import ModernInterface
        
        # Create interface
        interface = ModernInterface()
        
        # Initialize
        if not interface.initialize():
            logger.error("❌ Failed to initialize interface")
            sys.exit(1)
        
        logger.info(f"🌐 Launching interface on http://{args.host}:{args.port}")
        if args.share:
            logger.info("🔗 Creating public link...")
        
        # Launch
        interface.launch(
            share=args.share,
            server_name=args.host,
            server_port=args.port
        )
        
    except KeyboardInterrupt:
        logger.info("👋 Shutting down...")
    except Exception as e:
        logger.error(f"❌ Application error: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()