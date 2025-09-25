# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## System Overview

This is an ISO 27001:2022 Expert Agent System that combines document processing with AI-powered compliance analysis. The system consists of two main layers:

1. **Document Processing Layer**: FastAPI-based document extraction system
2. **ISO Expert Agent Layer**: AI-powered compliance analysis and consultation system

The architecture follows a modular design with the ISO expert agent orchestrating multiple specialized components for document classification, control assessment, gap analysis, and recommendations.

## Key Commands

### Application Management
```bash
# Start the server (recommended - includes health checks and setup)
./bash.sh

# Start with custom configuration
./bash.sh --port 8080 --debug

# Production mode with multiple workers
./bash.sh --prod --workers 4

# Install dependencies and start
./bash.sh --install

# Stop the server gracefully
./stop.sh

# Check comprehensive system status
./status.sh --test

# Health check only
./bash.sh --check
```

### Knowledge Base Management
```bash
# Check knowledge base status
./kb_cli.py status

# Initialize knowledge base (requires API keys)
./kb_cli.py init

# Validate knowledge base integrity
./kb_cli.py validate
```

### Development and Testing
```bash
# Run test scripts
cd tests && python test_crawler.py

# Clean up temporary files and cache
./cleanup.sh

# Manual startup (for development)
cd src && python -m document_extraction_system.main_simple
```

## Architecture

### Core Agent Flow
The `ISOExpertAgent` (`iso_expert.py`) is the orchestration hub that:
1. Uses `EnhancedLLMProcessor` for document classification
2. Delegates to `IntelligentControlAssessor` for control-specific analysis
3. Leverages `SmartRecommendationEngine` for prioritized recommendations
4. Coordinates with `SemanticSearchEngine` and `VectorIndexManager` for knowledge retrieval
5. Validates responses through `ISOResponseValidator`

### Knowledge Base Structure
- **Static Knowledge**: 93 Annex A controls (`annex_a_controls.py`) and management clauses (`management_clauses.py`)
- **Vector Storage**: Pinecone-based semantic search via `PineconeVectorStore`
- **Caching Layer**: `KnowledgeCache` for performance optimization
- **Response Format**: Structured via `ISOAgentCard` in `agent_response.py`

### Document Processing Pipeline
1. Upload via FastAPI endpoints (`main_simple.py`)
2. Text extraction (PyMuPDF for PDFs)
3. Classification through `EnhancedLLMProcessor`
4. Control mapping via `SemanticAnalyzer`
5. Assessment through `IntelligentControlAssessor`
6. Recommendation generation via `SmartRecommendationEngine`

## Configuration

The system uses environment-based configuration via `.env`:
- **Required**: No API keys required for basic functionality
- **Enhanced**: OpenAI/Anthropic keys for LLM features
- **Advanced**: Pinecone key for semantic search/knowledge base

Configuration is managed through `settings.py` using Pydantic settings with proper environment variable mapping.

## Key Integrations

### LLM Integration
- **OpenAI**: Document classification, enhanced analysis
- **Anthropic**: Alternative LLM provider
- **Fallback**: Rule-based analysis when LLMs unavailable

### Vector Database
- **Pinecone**: Semantic search and pattern recognition
- **Optional**: System works without but with reduced capabilities

## API Structure

### Core Endpoints
- `POST /api/v1/extract` - Basic document extraction
- `POST /api/v1/iso-analyze` - ISO compliance analysis
- `POST /api/v1/iso-consult` - Expert consultation
- `GET /api/v1/iso-controls` - Browse/search controls

### Response Format
All ISO analysis returns structured `ISOAgentCard` responses containing:
- Document classification
- Compliance overview with scores
- Gap analysis with prioritized issues
- Recommendations with implementation steps
- Executive summary and next steps

## Development Notes

### Testing Strategy
- Manual test scripts in `tests/` directory
- Health check endpoints for system validation
- Status monitoring via dedicated script (`status.sh`)

### Knowledge Base Initialization
The knowledge base requires one-time setup with API keys but operates independently afterward. The `KnowledgeBaseManager` handles initialization and health monitoring.

### Vector Store Architecture
The system uses a multi-namespace approach in Pinecone:
- `knowledge`: General ISO knowledge
- `controls`: Specific control information
- `clauses`: Management clause requirements

### Error Handling
- Graceful degradation when external services unavailable
- Comprehensive validation through `ISOResponseValidator`
- Fallback to rule-based analysis when LLMs fail