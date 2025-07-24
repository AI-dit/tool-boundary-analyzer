"""
Server module for Tool Visualizer

This module provides Flask application factory and server management functions.
It wraps the existing backend/app.py without modifying its core functionality.
"""

import os
import sys
from pathlib import Path

def create_app():
    """
    Create and configure the Flask application.
    
    This function imports and returns the Flask app from backend/app.py
    without modifying any of its functionality.
    
    Returns:
        Flask: Configured Flask application instance
    """
    # Add the project root to Python path so we can import from backend
    project_root = Path(__file__).parent.parent
    backend_path = project_root / "backend"
    
    if str(backend_path) not in sys.path:
        sys.path.insert(0, str(backend_path))
    
    try:
        # Import the existing Flask app from backend/app.py
        from app import app
        return app
    except ImportError as e:
        raise ImportError(
            f"Failed to import Flask app from backend/app.py: {e}\n"
            f"Make sure backend/app.py exists in the project root."
        )

def run_server(host='127.0.0.1', port=5000, debug=False):
    """
    Run the Tool Visualizer server.
    
    Args:
        host (str): Host to bind to (default: 127.0.0.1)
        port (int): Port to bind to (default: 5000)
        debug (bool): Enable debug mode (default: False)
    """
    app = create_app()
    
    print(f"🚀 Starting Tool Boundary Analyzer server...")
    print(f"📍 Server URL: http://{host}:{port}")
    print(f"📊 Open the web interface at: http://{host}:{port}")
    
    if debug:
        print("🔧 Debug mode enabled")
    
    try:
        app.run(host=host, port=port, debug=debug)
    except KeyboardInterrupt:
        print("\n👋 Server stopped by user")
    except Exception as e:
        print(f"❌ Server error: {e}")
        raise

def get_app():
    """
    Get the Flask application instance.
    
    Returns:
        Flask: The Flask application instance
    """
    return create_app()

# For backward compatibility and direct import
app = None

def get_or_create_app():
    """Get existing app or create new one"""
    global app
    if app is None:
        app = create_app()
    return app
