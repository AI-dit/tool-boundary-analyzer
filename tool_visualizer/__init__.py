"""
Tool Boundary Analyzer - Advanced Tool Boundary Analysis

A comprehensive tool for analyzing and visualizing tool boundaries, capabilities,
and interactions using machine learning and natural language processing.

This package provides:
- Flask web server for interactive tool analysis
- Machine learning models for semantic similarity
- REST API for tool boundary detection
- Web interface for visualization and exploration
- Command line interface for easy server management
"""

__version__ = "0.1.0"
__author__ = "Tool Visualizer Team"
__email__ = "contact@toolvisualizer.com"
__description__ = "Advanced Tool Boundary Analysis with ML and NLP"

# Import main components for easy access
from .server import create_app, run_server
from .cli import main as cli_main

__all__ = ['create_app', 'run_server', 'cli_main', '__version__']
