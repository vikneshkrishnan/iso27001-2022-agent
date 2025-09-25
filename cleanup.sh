#!/bin/bash
#
# Cleanup Script for ISO 27001:2022 Agent
# Removes temporary files, cache files, and cleans up development artifacts
#

set -e

echo "🧹 Starting codebase cleanup..."

# Remove Python cache files from source code
echo "Removing Python cache files..."
find src/ -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
find src/ -name "*.pyc" -type f -delete 2>/dev/null || true
find src/ -name "*.pyo" -type f -delete 2>/dev/null || true

# Remove temporary files
echo "Removing temporary files..."
find . -name "*.tmp" -type f -delete 2>/dev/null || true
find . -name "*.temp" -type f -delete 2>/dev/null || true
find . -name "*~" -type f -delete 2>/dev/null || true

# Remove macOS system files
echo "Removing macOS system files..."
find . -name ".DS_Store" -type f -delete 2>/dev/null || true
find . -name "._*" -type f -delete 2>/dev/null || true

# Clean up log files (keep the directory)
echo "Cleaning old log files..."
find logs/ -name "*.log" -type f -mtime +30 -delete 2>/dev/null || true

# Clean up old uploaded files (keep the directory)
echo "Cleaning old uploaded files..."
find uploads/ -type f ! -name "README.md" ! -name ".gitkeep" -mtime +7 -delete 2>/dev/null || true

# Remove IDE files
echo "Removing IDE files..."
find . -name "*.swp" -type f -delete 2>/dev/null || true
find . -name "*.swo" -type f -delete 2>/dev/null || true

# Show final status
echo ""
echo "✅ Cleanup completed successfully!"
echo ""
echo "📊 Current directory status:"
echo "Source code cache files: $(find src/ -name "__pycache__" -type d | wc -l | xargs) directories"
echo "Uploads directory: $(ls -1 uploads/ | grep -v README.md | grep -v .gitkeep | wc -l | xargs) files"
echo "Logs directory: $(ls -1 logs/ | grep -v README.md | grep -v .gitkeep | wc -l | xargs) files"
echo ""
echo "🎯 Your codebase is now clean and optimized!"
echo ""
echo "💡 Next steps:"
echo "  • Check knowledge base: ./kb_cli.py status"
echo "  • Initialize knowledge base: ./kb_cli.py init (requires Pinecone API key)"
echo "  • Start application: ./bash.sh"