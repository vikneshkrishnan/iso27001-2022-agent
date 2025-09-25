#!/bin/bash
#
# Knowledge Base CLI - Command-line interface for managing ISO Agent knowledge base
# This wrapper script ensures the virtual environment is used
#

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Check if virtual environment exists
if [ ! -f "$SCRIPT_DIR/venv/bin/python" ]; then
    echo "❌ Virtual environment not found at $SCRIPT_DIR/venv/"
    echo "Please create and activate virtual environment first:"
    echo "  python3 -m venv venv"
    echo "  source venv/bin/activate"
    echo "  pip install -r requirements.txt"
    exit 1
fi

# Check if required packages are installed
if ! "$SCRIPT_DIR/venv/bin/python" -c "import pydantic_settings" 2>/dev/null; then
    echo "❌ Required packages not installed in virtual environment"
    echo "Please install dependencies:"
    echo "  source venv/bin/activate"
    echo "  pip install -r requirements.txt"
    exit 1
fi

# Run the Python CLI with virtual environment Python
exec "$SCRIPT_DIR/venv/bin/python" "$SCRIPT_DIR/kb_cli_core.py" "$@"