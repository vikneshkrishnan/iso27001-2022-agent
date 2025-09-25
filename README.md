# Document Extraction System

A simple FastAPI-based document extraction and processing system for file uploads and basic document handling.

## 🚀 Features

### Simple Document Processing
- **File Upload API**: Upload documents via REST API
- **File Validation**: Support for PDF, PNG, JPG, JPEG, TIFF, BMP files
- **Basic Text Extraction**: Foundation for document text processing
- **File Size Limits**: Configurable maximum file sizes (default 100MB)

### Production Ready
- **FastAPI Framework**: Modern async web framework with automatic API documentation
- **Environment Configuration**: Environment-based configuration with Pydantic
- **Health Checks**: Built-in health monitoring endpoints
- **CORS Support**: Cross-origin request handling
- **Error Handling**: Comprehensive error responses

## 📋 Requirements

- Python 3.9+
- Optional: Tesseract OCR (for future OCR functionality)

## 🛠 Installation

### 1. Clone the Repository
```bash
git clone <repository-url>
cd document-extraction-agent
```

### 2. Set Up Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements-minimal.txt
```

### 4. Optional: Install Tesseract OCR

#### Ubuntu/Debian
```bash
sudo apt-get update
sudo apt-get install tesseract-ocr tesseract-ocr-eng
```

#### macOS
```bash
brew install tesseract
```

### 5. Configuration
The system uses `.env` for configuration. Key settings:
```bash
DEBUG=true
API_HOST=0.0.0.0
API_PORT=8000
MAX_FILE_SIZE=104857600  # 100MB
```

## 🚀 Quick Start

### Start the API Server

#### Option A: Using the Startup Script (Recommended) 🚀
```bash
# Basic startup
./bash.sh

# With custom configuration
./bash.sh --port 8080 --debug

# Install dependencies and start
./bash.sh --install

# Production mode
./bash.sh --prod --workers 4

# Run health check only
./bash.sh --check
```

#### Option B: Manual Startup
```bash
cd src
python -m document_extraction_system.main_simple
```

The API will be available at `http://localhost:8000`

### Stop the Server
```bash
# Stop the server gracefully
./stop.sh

# Force stop all processes
./stop.sh --force

# Stop specific port
./stop.sh --port 8080
```

### Access API Documentation
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## 📚 API Usage

### Upload Document
```bash
curl -X POST "http://localhost:8000/api/v1/extract" \
  -F "file=@document.pdf"
```

### Check Processing Status
```bash
curl "http://localhost:8000/api/v1/status/{document_id}"
```

### Get Extraction Results
```bash
curl "http://localhost:8000/api/v1/results/{document_id}"
```

### Health Check
```bash
curl "http://localhost:8000/health"
```

### System Status
```bash
curl "http://localhost:8000/"
```

## 🏗 Architecture

### Simple Processing Flow
1. **File Upload** → Validates file type and size
2. **File Storage** → Temporarily stores uploaded file
3. **Response** → Returns file information and processing status
4. **Status Check** → Monitor processing via API endpoints

### Supported File Types
- PDF documents (`.pdf`)
- Image files (`.png`, `.jpg`, `.jpeg`, `.tiff`, `.bmp`)
- Text files (`.txt`)

## 🔧 Configuration

Key configuration options in `.env`:

```env
# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
DEBUG=true

# File Upload Configuration
MAX_FILE_SIZE=104857600  # 100MB
UPLOAD_DIR=uploads
OUTPUT_DIR=data

# Optional OCR Configuration
TESSERACT_PATH=/opt/homebrew/bin/tesseract
OCR_CONFIDENCE_THRESHOLD=0.6
```

## 📁 Project Structure
```
src/document_extraction_system/
├── __init__.py
├── main_simple.py      # FastAPI application
└── config/
    ├── __init__.py
    └── settings.py     # Configuration management
```

## 🧪 Testing

### Test API Endpoints
```bash
# Test with sample file
curl -X POST "http://localhost:8000/api/v1/extract" \
  -F "file=@test_sample.txt"

# Check health
curl "http://localhost:8000/health"
```

## 🐛 Troubleshooting

### Common Issues

**Port already in use**
```bash
# Change port in .env
API_PORT=8001
```

**File too large**
```bash
# Increase max file size in .env
MAX_FILE_SIZE=209715200  # 200MB
```

**Tesseract not found (optional)**
```bash
# Install Tesseract and set path in .env
TESSERACT_PATH=/usr/local/bin/tesseract
```

## 🧹 Maintenance & Cleanup

### Automatic Cleanup
Run the cleanup script to remove temporary files, cache files, and development artifacts:

```bash
# Clean up the codebase
./cleanup.sh
```

### Manual Maintenance
- **Knowledge Base Status**: `./kb_cli.py status` - Check vector database health
- **Knowledge Base Init**: `./kb_cli.py init` - Populate ISO knowledge base (requires API keys)
- **Knowledge Base Validation**: `./kb_cli.py validate` - Verify knowledge base integrity
- **Logs**: Check `logs/` directory for application logs
- **Uploads**: Temporary uploaded files are stored in `uploads/`

### Prerequisites for Knowledge Base
The knowledge base CLI requires:
- Virtual environment with dependencies installed
- Pinecone API key (set in `.env` file)
- OpenAI API key (set in `.env` file)

### What Gets Cleaned
- Python cache files (`__pycache__/`, `*.pyc`)
- Temporary files (`*.tmp`, `*.temp`, `*~`)
- macOS system files (`.DS_Store`, `._*`)
- Old log files (>30 days)
- Old uploaded files (>7 days)
- IDE temporary files (`*.swp`, `*.swo`)

## 📄 License

MIT License - see LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test your changes
5. Submit a pull request

## 📞 Support

For issues and questions:
- Create an issue on GitHub
- Check the API documentation at `/docs`