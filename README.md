# Tool Boundary Analyzer

An ML-powered tool that analyzes tool definitions to identify overlaps, similarities, and potential confusion points for AI agents. Built with Flask backend, NLP models, and interactive D3.js visualizations.

##  Quick Start

### Installation

Install directly from PyPI:

```bash
pip install tool-boundary-analyzer
```

Or install from source:

```bash
git clone https://github.com/AI-dit/tool-boundary-analyzer.git
cd tool-boundary-analyzer
pip install -e .
```

> **Note**: The installation automatically includes the English spaCy model (`en_core_web_sm`) required for advanced NLP features.



### Usage

Start the server with the command line tool:

```bash
# Start server on default port 5000
tool-boundary-analyzer

# Start on custom port
tool-boundary-analyzer --port 8080

# Allow external connections
tool-boundary-analyzer --host 0.0.0.0

# Enable debug mode
tool-boundary-analyzer --debug
```

Then open your browser to `http://localhost:5000` to access the web interface.

### Python API

Use the tool programmatically in your Python code:

```python
from tool_visualizer import create_app, run_server

# Create Flask app instance
app = create_app()

# Or run the server directly
run_server(host='127.0.0.1', port=5000, debug=False)
```

##  Features

### ML based  Analysis
- **Semantic Similarity**: Uses Sentence Transformers for  semantic understanding
- **Intent Analysis**: Extracts and compares tool purposes using spaCy NLP
- **Parameter Structure**: Analyzes function signatures and parameter patterns
- **Context Detection**: Identifies usage patterns and operational contexts
- **Confusion Risk**: Calculates likelihood of AI agent confusion between tools

### Interactive Visualizations
- **Network Graph**: D3.js force-directed graph showing tool relationships
- **Similarity Matrix**: Heatmap visualization of tool overlaps
- **Real-time Updates**: Dynamic visualizations that update as you analyze


