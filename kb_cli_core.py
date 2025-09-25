#!/usr/bin/env python3
"""
Knowledge Base CLI Core - Python implementation for managing ISO Agent knowledge base
This file is called by the kb_cli.py wrapper script
"""

import sys
import os

# Add the src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    from src.document_extraction_system.iso_agent.knowledge_base_manager import kb
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("This usually means:")
    print("1. Virtual environment is not properly activated")
    print("2. Dependencies are not installed")
    print("3. Project structure has changed")
    print("")
    print("To fix this, try:")
    print("  source venv/bin/activate")
    print("  pip install -r requirements.txt")
    sys.exit(1)

if __name__ == "__main__":
    try:
        kb()
    except Exception as e:
        print(f"❌ CLI error: {e}")
        print("Check your configuration and try again")
        sys.exit(1)