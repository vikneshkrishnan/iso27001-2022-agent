# ISO 27001:2022 Expert Agent System

A comprehensive AI-powered document extraction and ISO 27001:2022 compliance analysis system, featuring an expert agent for compliance consultation and automated assessment.

## 🎯 Overview

This system combines:
- **Document Processing**: FastAPI-based document extraction and processing
- **ISO Expert Agent**: AI-powered compliance analysis and consultation
- **93 Annex A Controls**: Complete knowledge of all organizational, people, physical, and technological controls
- **Management Clauses 4-10**: Mandatory ISMS requirements (Context, Leadership, Planning, Support, Operations, Performance Evaluation, Improvement)
- **Automated Analysis**: Document compliance assessment against ISO 27001:2022 requirements
- **Expert Consultation**: Interactive Q&A on ISO topics with implementation guidance

## 🚀 Key Features

### Document Processing
- **File Upload API**: Upload documents via REST API
- **File Validation**: Support for PDF, PNG, JPG, JPEG, TIFF, BMP files
- **Text Extraction**: Foundation for document text processing
- **File Size Limits**: Configurable maximum file sizes (default 100MB)

### ISO 27001:2022 Expert Capabilities
- **Document Compliance Analysis**: Upload documents and receive comprehensive ISO compliance assessments
- **Expert Consultation**: Ask questions about ISO 27001:2022 and receive expert guidance
- **Control Management**: Browse, search, and get detailed information about all 93 Annex A controls
- **Gap Analysis**: Identify compliance gaps with prioritized recommendations
- **Implementation Roadmap**: Get structured implementation plans with timelines
- **Agent Card Responses**: Structured, comprehensive analysis reports
- **Risk Assessment**: Identify risk areas and quick wins
- **Maturity Assessment**: Evaluate organizational security maturity levels

### Production Ready
- **FastAPI Framework**: Modern async web framework with automatic API documentation
- **Environment Configuration**: Environment-based configuration with Pydantic
- **Health Checks**: Built-in health monitoring endpoints
- **CORS Support**: Cross-origin request handling
- **Error Handling**: Comprehensive error responses

## 📋 Requirements

- Python 3.9+
- Optional: Tesseract OCR (for future OCR functionality)
- Optional: OpenAI API key (for enhanced LLM analysis)
- Optional: Anthropic API key (for enhanced LLM analysis)
- Optional: Pinecone API key (for knowledge base)

## 🛠 Installation & Setup

### 1. Clone the Repository
```bash
git clone <repository-url>
cd iso27001-2022-agent
```

### 2. Set Up Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
# Basic installation
pip install -r requirements-minimal.txt

# Full installation with ISO features
pip install -r requirements.txt
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

### 5. Environment Configuration
Create a `.env` file with your configuration:

```bash
# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
DEBUG=true

# File Upload Configuration
MAX_FILE_SIZE=104857600  # 100MB
UPLOAD_DIR=uploads
OUTPUT_DIR=data

# Optional: Enhanced LLM Analysis
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key

# Optional: Knowledge Base (Pinecone)
PINECONE_API_KEY=your-pinecone-api-key-here
PINECONE_ENVIRONMENT=us-east-1
PINECONE_INDEX_NAME=iso27001
PINECONE_DIMENSION=1024

# Optional OCR Configuration
TESSERACT_PATH=/opt/homebrew/bin/tesseract
OCR_CONFIDENCE_THRESHOLD=0.6
```

## 🚀 Quick Start

### Start the Server

#### Option A: Using the Startup Script (Recommended) 🚀
```bash
# Basic startup with automatic health checks
./bash.sh

# With custom configuration
./bash.sh --port 8080 --debug

# Install dependencies and start
./bash.sh --install

# Production mode with multiple workers
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

## 📊 API Endpoints

### Document Processing
```bash
# Upload and extract document
curl -X POST "http://localhost:8000/api/v1/extract" \
  -F "file=@document.pdf"

# Extract with ISO analysis
curl -X POST "http://localhost:8000/api/v1/extract?iso_analysis=true" \
  -F "file=@policy.pdf"

# Check processing status
curl "http://localhost:8000/api/v1/status/{document_id}"

# Get extraction results
curl "http://localhost:8000/api/v1/results/{document_id}"
```

### ISO 27001:2022 Expert Agent
```bash
# Analyze document for ISO compliance
curl -X POST "http://localhost:8000/api/v1/iso-analyze" \
  -F "file=@security_policy.pdf" \
  -F "analysis_scope=comprehensive"

# Expert consultation
curl -X POST "http://localhost:8000/api/v1/iso-consult" \
  -H "Content-Type: application/json" \
  -d '{"question": "How do I implement access control policies according to ISO 27001?"}'

# Browse all controls
curl "http://localhost:8000/api/v1/iso-controls"

# Filter controls by category
curl "http://localhost:8000/api/v1/iso-controls?category=organizational"

# Search controls
curl "http://localhost:8000/api/v1/iso-controls?search=access"

# Get detailed control info
curl "http://localhost:8000/api/v1/iso-controls/5.1"
```

### Health & Status
```bash
# Health check
curl "http://localhost:8000/health"

# System status
curl "http://localhost:8000/"
```

## 📋 ISO 27001:2022 Coverage

### Annex A Controls (93 total)
- **Organizational Controls (37)** - Policies, procedures, governance
- **People Controls (8)** - HR security, awareness, training
- **Physical Controls (14)** - Physical security, environmental protection
- **Technological Controls (34)** - System security, network protection

### Management Clauses
- **Clause 4**: Context of the organization
- **Clause 5**: Leadership
- **Clause 6**: Planning (including risk management)
- **Clause 7**: Support (resources, competence, awareness)
- **Clause 8**: Operation
- **Clause 9**: Performance evaluation
- **Clause 10**: Improvement

## 💡 Usage Examples

### Document Analysis
```python
import requests

# Analyze a policy document
with open('security_policy.pdf', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/api/v1/iso-analyze',
        files={'file': f},
        data={'analysis_scope': 'comprehensive'}
    )

analysis = response.json()['analysis']
print(f"Compliance Score: {analysis['compliance_overview']['overall_score']}")
print(f"Total Gaps: {analysis['gap_analysis']['total_gaps']}")
```

### Expert Consultation
```python
# Ask the expert agent
consultation = requests.post(
    'http://localhost:8000/api/v1/iso-consult',
    json={"question": "What are the key requirements for incident management?"}
)

answer = consultation.json()['consultation']
print(f"Answer: {answer['answer']}")
print(f"Relevant Controls: {answer['relevant_controls']}")
```

### Control Lookup
```python
# Get specific control details
control = requests.get('http://localhost:8000/api/v1/iso-controls/8.2')
print(f"Control: {control.json()['control']['title']}")
print(f"Implementation: {control.json()['control']['implementation_guidance']}")
```

## 📈 Agent Card Response Format

The agent returns structured "Agent Cards" with comprehensive analysis:

```json
{
  "analysis_id": "uuid",
  "timestamp": "2024-01-01T00:00:00",
  "document_classification": {
    "type": "policy",
    "iso_relevance": "high",
    "confidence": 0.9
  },
  "compliance_overview": {
    "overall_score": 75.5,
    "overall_maturity": "managed",
    "key_strengths": ["Strong policy framework"],
    "major_concerns": ["Incident response gaps"]
  },
  "gap_analysis": {
    "total_gaps": 10,
    "critical_gaps": 2,
    "top_gap_areas": ["incident management", "access control"]
  },
  "recommendations": [
    {
      "title": "Implement incident response procedures",
      "priority": "critical",
      "timeline": "short-term",
      "implementation_steps": [...]
    }
  ],
  "implementation_roadmap": [...],
  "executive_summary": "...",
  "next_steps": [...]
}
```

## 🔧 Knowledge Base Setup (Optional but Recommended)

For enhanced analysis with semantic search and pattern recognition:

### 1. Get API Keys
- **Pinecone**: Sign up at https://www.pinecone.io/
- **OpenAI**: Sign up at https://platform.openai.com/

### 2. Configure Environment
Add to your `.env` file:
```bash
PINECONE_API_KEY=your-pinecone-api-key-here
PINECONE_ENVIRONMENT=us-east-1
PINECONE_INDEX_NAME=iso27001
OPENAI_API_KEY=your-openai-api-key-here
```

### 3. Initialize Knowledge Base
```bash
# Check current status
./kb_cli.py status

# Initialize (creates index and populates with ISO knowledge)
./kb_cli.py init

# Validate setup
./kb_cli.py validate
```

This indexes all 93 controls, management clauses, and enables:
- Semantic similarity matching
- Historical pattern recognition
- Intelligent control selection
- Context-aware recommendations

**Note**: Initial indexing uses ~$2-5 in OpenAI credits and takes 5-10 minutes.

## 🏗 Architecture

### Project Structure
```
src/document_extraction_system/
├── __init__.py
├── main_simple.py              # FastAPI application
├── config/
│   ├── __init__.py
│   └── settings.py            # Configuration management
├── iso_knowledge/             # ISO 27001:2022 Knowledge Base
│   ├── annex_a_controls.py    # 93 Annex A controls database
│   ├── management_clauses.py   # Management clauses 4-10
│   └── agent_response.py      # Structured response formats
└── iso_agent/                 # ISO Expert Agent
    ├── iso_expert.py          # Main expert agent
    ├── iso_helpers.py         # Analysis helper functions
    ├── enhanced_llm.py        # LLM integration
    ├── semantic_search.py     # Semantic search engine
    ├── recommendation_engine.py # Smart recommendations
    ├── intelligent_assessor.py # Control assessment
    └── vector_store.py        # Vector database interface
```

### Processing Flow
1. **File Upload** → Validates file type and size
2. **Document Analysis** → Extracts text and analyzes content
3. **ISO Assessment** → Maps content to controls and clauses
4. **Agent Card Generation** → Creates structured analysis report
5. **Expert Consultation** → Provides interactive guidance

### Supported File Types
- PDF documents (`.pdf`)
- Image files (`.png`, `.jpg`, `.jpeg`, `.tiff`, `.bmp`)
- Text files (`.txt`)

## 🧪 System Verification & Testing

### Health Checks
```bash
# Comprehensive system status
./status.sh --test

# Basic health check
curl "http://localhost:8000/health"

# Knowledge base status (if configured)
./kb_cli.py status
```

### Test API Endpoints
```bash
# Test document upload
curl -X POST "http://localhost:8000/api/v1/extract" \
  -F "file=@test_sample.txt"

# Test ISO analysis
curl -X POST "http://localhost:8000/api/v1/iso-analyze" \
  -F "file=@policy_sample.pdf"

# Test expert consultation
curl -X POST "http://localhost:8000/api/v1/iso-consult" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is required for access control?"}'
```

## 🐛 Troubleshooting

### Common Issues

**Port already in use**
```bash
# Change port in .env or use custom port
./bash.sh --port 8001
```

**File too large**
```bash
# Increase max file size in .env
MAX_FILE_SIZE=209715200  # 200MB
```

**Virtual environment issues**
```bash
# Recreate virtual environment
rm -rf venv
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**API key issues**
- Verify API keys are correct in `.env`
- Check that you have credits/billing set up
- The system works without API keys but with reduced capabilities

**Knowledge base errors**
```bash
# Check status and reinitialize if needed
./kb_cli.py status
./kb_cli.py init
```

## 🧹 Maintenance & Cleanup

### Automatic Cleanup
```bash
# Clean up temporary files, cache files, and development artifacts
./cleanup.sh
```

### Manual Maintenance Commands
- **Knowledge Base Status**: `./kb_cli.py status` - Check vector database health
- **Knowledge Base Init**: `./kb_cli.py init` - Populate ISO knowledge base
- **Knowledge Base Validation**: `./kb_cli.py validate` - Verify knowledge base integrity
- **Logs**: Check `logs/` directory for application logs
- **Uploads**: Temporary uploaded files are stored in `uploads/`

### What Gets Cleaned
- Python cache files (`__pycache__/`, `*.pyc`)
- Temporary files (`*.tmp`, `*.temp`, `*~`)
- macOS system files (`.DS_Store`, `._*`)
- Old log files (>30 days)
- Old uploaded files (>7 days)
- IDE temporary files (`*.swp`, `*.swo`)

## 🎯 Best Practices

### For Document Analysis
1. **Use well-structured documents** for better analysis accuracy
2. **Provide complete policies** rather than fragments
3. **Include implementation details** in documents for comprehensive assessment
4. **Regular analysis** of updated documents to track improvement

### For Expert Consultation
1. **Be specific** in questions for better answers
2. **Provide context** when asking about implementation scenarios
3. **Ask follow-up questions** to dive deeper into specific areas
4. **Reference control IDs** when asking about specific requirements

### For Implementation
1. **Start with critical gaps** identified in analysis
2. **Use the implementation roadmap** as a structured guide
3. **Focus on quick wins** for immediate improvements
4. **Regular re-assessment** to track progress

## 🔄 Integration Patterns

### With Existing Systems
- REST API integration with any system
- JSON response format for easy parsing
- Webhook support for automated workflows
- Batch processing capabilities

### CI/CD Integration
```bash
# Automated policy analysis in pipelines
curl -X POST "http://iso-agent/api/v1/iso-analyze" \
     -F "file=@policy.pdf" \
     -F "analysis_scope=comprehensive"
```

### Enterprise Integration
- Integration with document management systems
- Compliance dashboard integration
- Risk management system connectivity
- Audit trail and reporting

## ✅ Setup Status: COMPLETE & WORKING

Your ISO 27001:2022 Expert Agent system has been successfully implemented with:

### ✅ What's Working
- **93 Annex A Controls** - Complete database implemented
- **Management Clauses 4-10** - Full coverage of mandatory ISMS requirements
- **AI-Powered Analysis** - LLM integration ready (optional API keys)
- **Document Processing** - PDF/TXT analysis against ISO requirements
- **Expert Consultation** - Interactive Q&A system operational
- **Professional Scripts** - Complete bash script suite for management
- **Health Monitoring** - Comprehensive status checking and validation

### 🚀 Ready to Use Commands
```bash
# Start the ISO expert agent (recommended)
./bash.sh --port 8080

# Check comprehensive system status
./status.sh --test

# Stop service gracefully
./stop.sh
```

Visit `http://localhost:8080/docs` for interactive API documentation.

## 📞 Support & Documentation

- **API Documentation**: Available at `/docs` when server is running
- **Interactive Testing**: Use `/docs` Swagger UI for testing endpoints
- **Sample Documents**: Test with various policy types for best results
- **Error Handling**: Comprehensive error messages for troubleshooting
- **Health Monitoring**: Built-in status checks and validation

For issues and questions:
- Create an issue on GitHub
- Check the API documentation at `/docs`
- Use the health check endpoints for diagnostics

## 📄 License

MIT License - see LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test your changes
5. Submit a pull request

---

**Built with ❤️ for ISO 27001:2022 compliance professionals**

Transform your compliance journey with AI-powered expert analysis and guidance.