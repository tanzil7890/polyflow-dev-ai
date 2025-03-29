#!/usr/bin/env python3
"""
Runner script for PolyFlow Sheets application
This script helps set up and run the PolyFlow Sheets application
"""

import os
import sys
import argparse
import logging
from app import app, configure_polyflow

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('polyflow_sheets')

def check_requirements():
    """Check if all required packages are installed"""
    try:
        import flask
        import pandas
        import numpy
        import plotly
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        import polyflow
        logger.info("All required packages are installed.")
        return True
    except ImportError as e:
        logger.error(f"Missing required package: {e}")
        logger.error("Please install all requirements with: pip install -r requirements.txt")
        return False

def check_api_key():
    """Check if OpenAI API key is set"""
    api_key = os.environ.get('OPENAI_API_KEY')
    if not api_key:
        logger.warning("OPENAI_API_KEY environment variable not set.")
        logger.warning("You can still run the application, but LLM operations will not work.")
        logger.warning("Set your API key with: export OPENAI_API_KEY=your_api_key")
    else:
        logger.info("API key found.")

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Run PolyFlow Sheets application')
    parser.add_argument('--host', default='0.0.0.0', help='Host to run the server on')
    parser.add_argument('--port', type=int, default=8000, help='Port to run the server on')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode')
    return parser.parse_args()

def main():
    """Main entry point"""
    logger.info("Starting PolyFlow Sheets application...")
    
    # Check requirements
    if not check_requirements():
        sys.exit(1)
        
    # Check API key
    check_api_key()
    
    # Parse arguments
    args = parse_arguments()
    
    # Configure PolyFlow
    configure_result = configure_polyflow()
    logger.info(configure_result)
    
    # Run the application
    logger.info(f"Running server on http://{args.host}:{args.port}")
    app.run(
        host=args.host,
        port=args.port,
        debug=args.debug
    )

if __name__ == '__main__':
    main() 