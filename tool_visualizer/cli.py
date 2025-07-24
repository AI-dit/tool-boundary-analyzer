"""
Command Line Interface for Tool Boundary Analyzer

This module provides a CLI for the Tool Boundary Analyzer,
allowing users to analyze tools from files or start a web server.
"""

import argparse
import sys
import os
import json
from pathlib import Path


def _ensure_spacy_model():
    """Ensure spaCy model is available, download if needed"""
    try:
        import spacy
        try:
            spacy.load('en_core_web_sm')
            return True
        except OSError:
            print("📥 Downloading required spaCy model...")
            from .install_models import download_spacy_model
            return download_spacy_model()
    except ImportError:
        print("❌ spaCy not installed")
        return False


def main():
    """Main CLI entry point"""
    # Ensure spaCy model is available
    _ensure_spacy_model()
    
    parser = argparse.ArgumentParser(
        description="Tool Boundary Analyzer - Advanced Tool Boundary Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze tools from file
  tool-boundary-analyzer analyze tools.json
  
  # Analyze with lightweight mode (no advanced ML)
  tool-boundary-analyzer analyze tools.json --lightweight
  
  # Start web server
  tool-boundary-analyzer serve --port 8080
  
  # Show version
  tool-boundary-analyzer --version
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze tools from file')
    analyze_parser.add_argument('file', help='JSON file containing tool definitions')
    analyze_parser.add_argument('--output', '-o', help='Output file for results (default: print to stdout)')
    analyze_parser.add_argument('--lightweight', action='store_true', 
                               help='Use lightweight mode (TF-IDF only, no advanced ML)')
    
    # Serve command (optional web interface)
    serve_parser = subparsers.add_parser('serve', help='Start web server')
    serve_parser.add_argument('--host', default='127.0.0.1', help='Host to bind to (default: 127.0.0.1)')
    serve_parser.add_argument('--port', type=int, default=5000, help='Port to bind to (default: 5000)')
    serve_parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    
    # Version
    parser.add_argument('--version', action='version', version='%(prog)s 0.1.0')
    
    args = parser.parse_args()
    
    # If no command specified, default to serve for backward compatibility
    if not args.command:
        args.command = 'serve'
        args.host = '127.0.0.1'
        args.port = 5000
        args.debug = False
    
    try:
        if args.command == 'analyze':
            handle_analyze(args)
        elif args.command == 'serve':
            handle_serve(args)
        else:
            parser.print_help()
            
    except ImportError as e:
        print(f"❌ Failed to start Tool Boundary Analyzer: {e}")
        print("Make sure all dependencies are installed:")
        print("  pip install tool-boundary-analyzer")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)


def handle_analyze(args):
    """Handle the analyze command"""
    from .analyzer import analyze_from_file
    
    print(f"🔍 Analyzing tools from {args.file}...")
    
    # Determine mode
    use_advanced = not args.lightweight
    if args.lightweight:
        print("🚀 Using lightweight mode (TF-IDF only)")
    else:
        print("🧠 Using advanced mode (ML models)")
    
    # Analyze
    results = analyze_from_file(args.file, use_advanced=use_advanced)
    
    # Handle errors
    if 'error' in results:
        print(f"❌ Analysis failed: {results['error']}")
        sys.exit(1)
    
    # Output results
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"✅ Results saved to {args.output}")
    else:
        # Print summary to stdout
        print_analysis_summary(results)


def print_analysis_summary(results):
    """Print a summary of analysis results"""
    metadata = results.get('metadata', {})
    overlaps = results.get('overlaps', [])
    recommendations = results.get('recommendations', [])
    
    print("\n" + "="*60)
    print("📊 TOOL BOUNDARY ANALYSIS RESULTS")
    print("="*60)
    
    print(f"🔧 Tools analyzed: {metadata.get('tool_count', 0)}")
    print(f"📈 Average similarity: {metadata.get('average_similarity', 0):.3f}")
    print(f"⚙️ Analysis method: {metadata.get('method', 'Unknown')}")
    print(f"🎯 Mode: {metadata.get('mode', 'Unknown')}")
    
    if overlaps:
        print(f"\n🔍 Found {len(overlaps)} tool overlaps:")
        for overlap in overlaps[:10]:  # Show top 10
            similarity = overlap['similarity'] * 100
            print(f"  • {overlap['tool1']} ↔ {overlap['tool2']}")
            print(f"    Similarity: {similarity:.1f}% ({overlap['overlap_type']})")
            if overlap.get('commonWords'):
                print(f"    Common words: {', '.join(overlap['commonWords'][:3])}")
    else:
        print("\n✅ No significant overlaps found")
    
    if recommendations:
        print(f"\n💡 Recommendations:")
        for rec in recommendations:
            print(f"  📋 {rec['message']}")
            for item in rec.get('items', [])[:3]:
                print(f"    - {item}")
    
    print("\n" + "="*60)


def handle_serve(args):
    """Handle the serve command (web server)"""
    try:
        from .server import run_server
        print("🌐 Starting web server mode...")
        run_server(host=args.host, port=args.port, debug=args.debug)
    except ImportError:
        print("❌ Web server dependencies not available")
        print("The analyze command works without Flask, but serve requires:")
        print("  pip install flask flask-cors")
        sys.exit(1)


def show_info():
    """Show information about the Tool Boundary Analyzer installation"""
    try:
        from . import __version__, __description__
        print(f"Tool Boundary Analyzer v{__version__}")
        print(f"{__description__}")
        print()
        print("📋 Commands:")
        print("  tool-boundary-analyzer analyze <file>  # Analyze tools from JSON file")
        print("  tool-boundary-analyzer serve           # Start web interface")
        print("📚 Documentation: Check README.md for detailed usage")
        print("🔧 Source Code: https://github.com/your-username/tool-boundary-analyzer")
    except ImportError:
        print("Tool Boundary Analyzer - Advanced Tool Boundary Analysis")


if __name__ == "__main__":
    main()
