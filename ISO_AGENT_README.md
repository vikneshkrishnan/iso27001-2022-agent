# ISO 27001:2022 Expert Agent

A comprehensive AI-powered agent system for ISO 27001:2022 compliance analysis and consultation, built on top of the document extraction system.

## 🎯 Overview

This expert agent specializes in:
- **93 Annex A Controls** - Complete knowledge of all organizational, people, physical, and technological controls
- **Management Clauses 4-10** - Mandatory ISMS requirements (Context, Leadership, Planning, Support, Operations, Performance Evaluation, Improvement)
- **Document Analysis** - Automated compliance assessment against ISO 27001:2022 requirements
- **Expert Consultation** - Interactive Q&A on ISO topics with implementation guidance

## 🚀 Features

### Core Capabilities
- **Document Compliance Analysis**: Upload documents and receive comprehensive ISO compliance assessments
- **Expert Consultation**: Ask questions about ISO 27001:2022 and receive expert guidance
- **Control Management**: Browse, search, and get detailed information about all 93 Annex A controls
- **Gap Analysis**: Identify compliance gaps with prioritized recommendations
- **Implementation Roadmap**: Get structured implementation plans with timelines

### Advanced Analysis
- **Agent Card Responses**: Structured, comprehensive analysis reports
- **Risk Assessment**: Identify risk areas and quick wins
- **Maturity Assessment**: Evaluate organizational security maturity levels
- **Recommendation Engine**: Prioritized action items with implementation guidance

## 📊 API Endpoints

### Document Analysis
```bash
# Analyze document for ISO compliance
POST /api/v1/iso-analyze
- Upload PDF/TXT files
- Get comprehensive compliance analysis
- Agent card with gaps, recommendations, roadmap

# Extract text with optional ISO analysis
POST /api/v1/extract?iso_analysis=true
- Combined extraction and ISO analysis
- Single endpoint for both operations
```

### Expert Consultation
```bash
# Ask ISO expert questions
POST /api/v1/iso-consult
{
  "question": "How do I implement access control policies according to ISO 27001?"
}
```

### Control Management
```bash
# Browse all controls
GET /api/v1/iso-controls

# Filter by category
GET /api/v1/iso-controls?category=organizational

# Search controls
GET /api/v1/iso-controls?search=access

# Get detailed control info
GET /api/v1/iso-controls/5.1
```

## 🏗️ Architecture

### Knowledge Base
```
src/document_extraction_system/iso_knowledge/
├── annex_a_controls.py      # 93 Annex A controls database
├── management_clauses.py    # Management clauses 4-10
└── agent_response.py        # Structured response formats
```

### Core Agent
```
src/document_extraction_system/iso_agent/
├── iso_expert.py           # Main expert agent
└── iso_helpers.py         # Analysis helper functions
```

### Integration
- FastAPI endpoints in `main_simple.py`
- LLM integration (OpenAI/Anthropic) for advanced analysis
- Fallback rule-based analysis when LLMs unavailable

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

## 🛠️ Setup & Configuration

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Environment Configuration
```bash
# Optional: Add API keys for enhanced LLM analysis
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key

# Default configuration
DEBUG=true
API_HOST=0.0.0.0
API_PORT=8000
```

### 3. Start the Server

#### Quick Start with Scripts
```bash
# Basic startup with automatic setup
./bash.sh

# Install dependencies and start
./bash.sh --install

# Custom configuration
./bash.sh --host 127.0.0.1 --port 8080 --debug

# Production mode with multiple workers
./bash.sh --prod --workers 4

# Health check only
./bash.sh --check
```

#### Manual Startup
```bash
cd src
python -m document_extraction_system.main_simple
```

#### Stop the Server
```bash
# Graceful shutdown
./stop.sh

# Force stop all processes
./stop.sh --force
```

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

## 🔧 Customization

### Adding New Controls
```python
# Add custom control to annex_a_controls.py
"X.1": Control(
    id="X.1",
    title="Custom Control",
    description="...",
    # ... other attributes
)
```

### Extending Analysis Logic
```python
# Customize analysis in iso_helpers.py
def custom_analysis_logic(document_text, controls):
    # Your custom analysis logic
    return assessment_results
```

### LLM Integration
- Supports OpenAI GPT models
- Supports Anthropic Claude models
- Fallback to rule-based analysis
- Easy to extend with other LLMs

## 📊 Analysis Capabilities

### Document Classification
- Automatic document type detection (policy, procedure, standard, etc.)
- ISO relevance scoring
- Applicable category identification

### Compliance Assessment
- Control-by-control analysis
- Management clause assessment
- Evidence identification
- Gap analysis with severity ranking

### Risk Analysis
- Risk area identification
- Quick win opportunities
- Implementation priority scoring
- Timeline recommendations

### Maturity Assessment
- 5-level maturity model (basic → optimizing)
- Category-specific maturity scoring
- Overall organizational assessment

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

## 🚀 Future Enhancements

- **Multi-language support** for international standards
- **Sector-specific analysis** (healthcare, finance, etc.)
- **Integration with other ISO standards** (27002, 27003, etc.)
- **Advanced ML models** for improved analysis accuracy
- **Collaborative features** for team-based compliance management

## 📞 Support & Documentation

- **API Documentation**: Available at `/docs` when server is running
- **Interactive Testing**: Use `/docs` Swagger UI for testing endpoints
- **Sample Documents**: Test with various policy types for best results
- **Error Handling**: Comprehensive error messages for troubleshooting

---

**Built with ❤️ for ISO 27001:2022 compliance professionals**

Transform your compliance journey with AI-powered expert analysis and guidance.