#!/usr/bin/env python3
"""
RoViT-KAN FastAPI Server Launcher
Run this script to start the FastAPI web server
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

import uvicorn
from api.main import app


def main():
    parser = argparse.ArgumentParser(
        description='RoViT-KAN FastAPI Server',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default settings (port 8000)
  python run_api.py
  
  # Run on custom port
  python run_api.py --port 8080
  
  # Enable auto-reload for development
  python run_api.py --reload
  
  # Run in production mode
  python run_api.py --host 0.0.0.0 --port 8000 --workers 4
        """
    )
    
    parser.add_argument(
        '--host',
        type=str,
        default='127.0.0.1',
        help='Host to bind to (default: 127.0.0.1)'
    )
    
    parser.add_argument(
        '--port',
        type=int,
        default=8000,
        help='Port to bind to (default: 8000)'
    )
    
    parser.add_argument(
        '--reload',
        action='store_true',
        help='Enable auto-reload for development'
    )
    
    parser.add_argument(
        '--workers',
        type=int,
        default=1,
        help='Number of worker processes (default: 1, for production use 4+)'
    )
    
    parser.add_argument(
        '--log-level',
        type=str,
        default='info',
        choices=['debug', 'info', 'warning', 'error', 'critical'],
        help='Logging level (default: info)'
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("🌹 RoViT-KAN FastAPI Server")
    print("=" * 60)
    print(f"Starting server at http://{args.host}:{args.port}")
    print(f"Mode: {'Development (auto-reload)' if args.reload else 'Production'}")
    print(f"Workers: {args.workers}")
    print(f"Log Level: {args.log_level}")
    print("=" * 60)
    print("\nAvailable endpoints:")
    print(f"  📊 Web UI:       http://{args.host}:{args.port}/")
    print(f"  🔍 API Docs:     http://{args.host}:{args.port}/docs")
    print(f"  📈 Health Check: http://{args.host}:{args.port}/health")
    print(f"  🤖 Model Info:   http://{args.host}:{args.port}/model-info")
    print("=" * 60)
    print("\nPress Ctrl+C to stop the server\n")
    
    uvicorn.run(
        "api.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=args.workers if not args.reload else 1,
        log_level=args.log_level
    )


if __name__ == "__main__":
    main()
