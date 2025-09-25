"""Simplified Document Extraction System - Main Application"""

import logging
import tempfile
import os
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
from contextlib import asynccontextmanager

import fitz  # PyMuPDF
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .config.settings import get_settings
from .iso_agent.iso_expert import ISOExpertAgent
from .iso_agent.iso_helpers import ISOAnalysisHelpers

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get settings
settings = get_settings()

# Initialize ISO Expert Agent
iso_agent = ISOExpertAgent()


async def initialize_knowledge_base_on_startup():
    """Initialize knowledge base on application startup if needed"""
    try:
        from .iso_agent.knowledge_base_manager import KnowledgeBaseManager

        kb_manager = KnowledgeBaseManager()

        # Check knowledge base status
        logger.info("Checking knowledge base status...")
        status = await kb_manager.get_status()

        if status.is_healthy and not status.initialization_required:
            logger.info(f"✅ Knowledge base is healthy with {status.total_documents} documents")
            return

        if status.errors:
            logger.warning(f"Knowledge base has errors: {', '.join(status.errors)}")
            # Don't auto-initialize if there are configuration errors
            return

        if status.initialization_required:
            logger.info("🚀 Knowledge base initialization required - starting background initialization...")

            # Run initialization in the background to not block startup
            import asyncio
            asyncio.create_task(auto_initialize_knowledge_base(kb_manager))

        else:
            logger.info("Knowledge base status check completed")

    except Exception as e:
        logger.error(f"Failed to check knowledge base status: {e}")
        # Don't fail startup due to knowledge base issues


async def auto_initialize_knowledge_base(kb_manager):
    """Auto-initialize knowledge base in background"""
    try:
        logger.info("Starting automatic knowledge base initialization...")
        success = await kb_manager.initialize(force=False)

        if success:
            logger.info("🎉 Knowledge base auto-initialization completed successfully")
        else:
            logger.warning("⚠️ Knowledge base auto-initialization failed - manual initialization may be required")

    except Exception as e:
        logger.error(f"Auto-initialization failed: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    # Startup
    logger.info(f"Starting {settings.app_name} v{settings.app_version}")

    # Ensure directories exist
    settings.ensure_directories()

    # Initialize knowledge base if needed
    await initialize_knowledge_base_on_startup()

    logger.info("Application started successfully")

    yield

    # Shutdown
    logger.info("Shutting down application")


# Create FastAPI app
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Simplified Document Extraction and Processing System",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def calculate_reading_time(word_count: int, words_per_minute: int = 250) -> str:
    """Calculate estimated reading time based on word count."""
    minutes = word_count / words_per_minute
    if minutes < 1:
        return "Less than 1 minute"
    elif minutes < 60:
        return f"{int(minutes)} minutes"
    else:
        hours = int(minutes / 60)
        remaining_minutes = int(minutes % 60)
        if remaining_minutes == 0:
            return f"{hours} hour{'s' if hours > 1 else ''}"
        return f"{hours} hour{'s' if hours > 1 else ''} {remaining_minutes} minutes"


def detect_language(text: str) -> str:
    """Simple language detection based on common words."""
    text_lower = text.lower()

    # Common words in different languages
    malay_indicators = ['dan', 'yang', 'ini', 'dengan', 'untuk', 'pada', 'dalam', 'adalah', 'akan', 'dari']
    english_indicators = ['the', 'and', 'is', 'in', 'to', 'of', 'a', 'that', 'it', 'with']
    chinese_indicators = ['的', '了', '在', '是', '我', '有', '他', '这', '中', '就']

    # Count indicators
    malay_count = sum(1 for word in malay_indicators if f' {word} ' in text_lower)
    english_count = sum(1 for word in english_indicators if f' {word} ' in text_lower)
    chinese_count = sum(1 for char in chinese_indicators if char in text)

    # Determine primary language
    counts = {'malay': malay_count, 'english': english_count, 'chinese': chinese_count}
    primary_lang = max(counts, key=counts.get)

    # If counts are close, it's likely multilingual
    total_indicators = sum(counts.values())
    if total_indicators > 0 and max(counts.values()) / total_indicators < 0.6:
        return "multilingual"

    return primary_lang if total_indicators > 0 else "unknown"


def extract_table_of_contents(text: str) -> List[str]:
    """Extract potential table of contents or section headers."""
    toc_patterns = [
        r'^(\d+\.?\d*\.?\s+.+)$',  # Numbered sections (1. Introduction, 1.1 Overview)
        r'^([A-Z][A-Za-z\s&-]+)$',  # Title case headers
        r'^(\d+\.\s+[A-Z][A-Za-z\s&-]+)$',  # Numbered title case
    ]

    toc_items = []
    lines = text.split('\n')

    for line in lines[:50]:  # Check first 50 lines for TOC
        line = line.strip()
        if len(line) > 3 and len(line) < 100:  # Reasonable header length
            for pattern in toc_patterns:
                if re.match(pattern, line):
                    toc_items.append(line)
                    break

    return toc_items[:15]  # Return max 15 TOC items


def identify_document_type(text: str, filename: str) -> str:
    """Identify document type based on content and filename."""
    text_lower = text.lower()
    filename_lower = filename.lower()

    # Document type indicators
    types = {
        'technical_documentation': ['api', 'documentation', 'technical', 'system', 'architecture', 'specification'],
        'academic_paper': ['abstract', 'introduction', 'methodology', 'conclusion', 'references', 'research'],
        'business_report': ['executive summary', 'financial', 'quarterly', 'annual report', 'business plan'],
        'legal_document': ['agreement', 'contract', 'legal', 'clause', 'whereas', 'liability'],
        'manual': ['manual', 'guide', 'instruction', 'how to', 'step by step', 'tutorial'],
        'policy': ['policy', 'procedure', 'guidelines', 'compliance', 'standard', 'regulation']
    }

    # Check filename first
    for doc_type, keywords in types.items():
        if any(keyword in filename_lower for keyword in keywords):
            return doc_type

    # Check content
    for doc_type, keywords in types.items():
        keyword_count = sum(1 for keyword in keywords if keyword in text_lower)
        if keyword_count >= 2:  # At least 2 keyword matches
            return doc_type

    return 'general_document'


def format_text_preview(text: str, preview_length: int = 500) -> Dict[str, str]:
    """Generate formatted text preview and ending."""
    text_cleaned = re.sub(r'\n\s*\n', '\n\n', text.strip())  # Clean up excessive line breaks

    preview = text_cleaned[:preview_length]
    if len(text_cleaned) > preview_length:
        preview += "..."

    ending = ""
    if len(text_cleaned) > preview_length * 2:
        ending = "..." + text_cleaned[-preview_length:]

    return {
        "preview": preview.strip(),
        "ending": ending.strip() if ending else None
    }


def extract_key_phrases(text: str, top_n: int = 10) -> List[str]:
    """Extract key phrases from the text."""
    # Simple keyword extraction - remove common stop words
    stop_words = {
        'the', 'and', 'is', 'in', 'to', 'of', 'a', 'that', 'it', 'with', 'for', 'as', 'was', 'on', 'are', 'you',
        'dan', 'yang', 'ini', 'dengan', 'untuk', 'pada', 'dalam', 'adalah', 'akan', 'dari', 'ke', 'di', 'tidak'
    }

    # Extract words and count frequency
    words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
    word_freq = {}

    for word in words:
        if word not in stop_words and len(word) > 2:
            word_freq[word] = word_freq.get(word, 0) + 1

    # Return top N words
    return sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:top_n]


async def extract_pdf_text(file_content: bytes, filename: str = "document.pdf") -> dict:
    """Extract text from PDF content using PyMuPDF."""
    try:
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
            temp_file.write(file_content)
            temp_file_path = temp_file.name

        try:
            # Open PDF with PyMuPDF
            doc = fitz.open(temp_file_path)

            # Extract text from all pages
            full_text = ""
            page_count = len(doc)

            for page_num in range(page_count):
                page = doc[page_num]
                page_text = page.get_text()
                full_text += page_text + "\n\n"

            doc.close()

            # Calculate basic metrics
            word_count = len(full_text.split())
            char_count = len(full_text)

            # Generate enhanced analysis
            text_preview = format_text_preview(full_text)
            toc_items = extract_table_of_contents(full_text)
            language = detect_language(full_text)
            reading_time = calculate_reading_time(word_count)
            key_phrases = extract_key_phrases(full_text)
            document_type = identify_document_type(full_text, filename)

            return {
                "status": "completed",
                "summary": {
                    "preview": text_preview["preview"],
                    "ending": text_preview["ending"],
                    "table_of_contents": toc_items
                },
                "full_text": full_text.strip(),
                "metadata": {
                    "document_info": {
                        "pages": page_count,
                        "words": word_count,
                        "characters": char_count,
                        "confidence": 1.0
                    },
                    "analysis": {
                        "estimated_read_time": reading_time,
                        "language": language,
                        "document_type": document_type,
                        "key_phrases": [{"phrase": phrase, "frequency": freq} for phrase, freq in key_phrases],
                        "has_table_of_contents": len(toc_items) > 0
                    }
                }
            }

        finally:
            # Clean up temporary file
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)

    except Exception as e:
        logger.error(f"Error extracting PDF text: {e}")
        return {
            "status": "failed",
            "summary": {
                "preview": "",
                "ending": None,
                "table_of_contents": []
            },
            "full_text": "",
            "metadata": {
                "document_info": {
                    "pages": 0,
                    "words": 0,
                    "characters": 0,
                    "confidence": 0.0
                },
                "analysis": {
                    "estimated_read_time": "0 minutes",
                    "language": "unknown",
                    "key_phrases": [],
                    "has_table_of_contents": False
                }
            },
            "error": str(e)
        }


async def extract_text_file(file_content: bytes, filename: str = "document.txt") -> dict:
    """Extract text from plain text files."""
    try:
        # Decode text content
        text_content = file_content.decode('utf-8')

        # Calculate basic metrics
        word_count = len(text_content.split())
        char_count = len(text_content)

        # Generate enhanced analysis
        text_preview = format_text_preview(text_content)
        toc_items = extract_table_of_contents(text_content)
        language = detect_language(text_content)
        reading_time = calculate_reading_time(word_count)
        key_phrases = extract_key_phrases(text_content)
        document_type = identify_document_type(text_content, filename)

        return {
            "status": "completed",
            "summary": {
                "preview": text_preview["preview"],
                "ending": text_preview["ending"],
                "table_of_contents": toc_items
            },
            "full_text": text_content,
            "metadata": {
                "document_info": {
                    "pages": 1,
                    "words": word_count,
                    "characters": char_count,
                    "confidence": 1.0
                },
                "analysis": {
                    "estimated_read_time": reading_time,
                    "language": language,
                    "document_type": document_type,
                    "key_phrases": [{"phrase": phrase, "frequency": freq} for phrase, freq in key_phrases],
                    "has_table_of_contents": len(toc_items) > 0
                }
            }
        }

    except UnicodeDecodeError as e:
        logger.error(f"Error decoding text file: {e}")
        return {
            "status": "failed",
            "summary": {
                "preview": "",
                "ending": None,
                "table_of_contents": []
            },
            "full_text": "",
            "metadata": {
                "document_info": {
                    "pages": 0,
                    "words": 0,
                    "characters": 0,
                    "confidence": 0.0
                },
                "analysis": {
                    "estimated_read_time": "0 minutes",
                    "language": "unknown",
                    "key_phrases": [],
                    "has_table_of_contents": False
                }
            },
            "error": f"Could not decode text file: {str(e)}"
        }


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "status": "running",
        "docs": "/docs" if settings.debug else "disabled in production",
        "message": "Welcome to the Document Extraction System"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "app_name": settings.app_name,
        "app_version": settings.app_version,
        "debug_mode": settings.debug
    }


@app.get("/api/v1/knowledge-base/status")
async def knowledge_base_status():
    """Get knowledge base status and health information."""
    try:
        from .iso_agent.knowledge_base_manager import KnowledgeBaseManager

        kb_manager = KnowledgeBaseManager()
        status = await kb_manager.get_status()

        return {
            "status": "success",
            "data": {
                "is_healthy": status.is_healthy,
                "total_documents": status.total_documents,
                "documents_by_namespace": status.documents_by_namespace,
                "last_updated": status.last_updated.isoformat() if status.last_updated else None,
                "storage_usage_mb": status.storage_usage_mb,
                "initialization_required": status.initialization_required,
                "errors": status.errors,
                "recommendations": status.recommendations
            }
        }
    except Exception as e:
        logger.error(f"Failed to get knowledge base status: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve knowledge base status")


@app.post("/api/v1/knowledge-base/initialize")
async def initialize_knowledge_base(force: bool = False):
    """Initialize or refresh the knowledge base."""
    try:
        from .iso_agent.knowledge_base_manager import KnowledgeBaseManager

        kb_manager = KnowledgeBaseManager()

        # Check if already healthy and not forcing
        if not force:
            status = await kb_manager.get_status()
            if status.is_healthy and not status.initialization_required:
                return {
                    "status": "success",
                    "message": "Knowledge base is already healthy",
                    "data": {
                        "already_initialized": True,
                        "total_documents": status.total_documents
                    }
                }

        success = await kb_manager.initialize(force=force)

        if success:
            # Get updated status
            status = await kb_manager.get_status()
            return {
                "status": "success",
                "message": "Knowledge base initialized successfully",
                "data": {
                    "total_documents": status.total_documents,
                    "documents_by_namespace": status.documents_by_namespace
                }
            }
        else:
            return {
                "status": "error",
                "message": "Knowledge base initialization failed",
                "data": {"success": False}
            }

    except Exception as e:
        logger.error(f"Failed to initialize knowledge base: {e}")
        raise HTTPException(status_code=500, detail="Knowledge base initialization failed")


@app.post("/api/v1/extract")
async def extract_document(
    file: UploadFile = File(...),
    iso_analysis: bool = False
):
    """
    Extract text from a document with optional ISO 27001:2022 analysis.
    Supports PDF and text files with actual content extraction.

    Args:
        file: Document to extract text from
        iso_analysis: If True, also perform ISO 27001:2022 compliance analysis
    """
    try:
        # Validate file extension
        if not file.filename:
            raise HTTPException(status_code=400, detail="File must have a filename")

        file_extension = Path(file.filename).suffix.lower()

        # Extended validation for supported extraction types
        allowed_extensions = [".pdf", ".txt", ".png", ".jpg", ".jpeg"]
        if file_extension not in allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type. Allowed types: {', '.join(allowed_extensions)}"
            )

        # Read file content
        content = await file.read()
        file_size = len(content)

        # Validate file size (100MB limit from settings)
        if file_size > settings.max_file_size:
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Maximum size: {settings.max_file_size / (1024*1024):.1f}MB"
            )

        # Perform actual text extraction based on file type
        extraction_result = {"status": "pending", "message": "Extraction type not supported yet"}

        if file_extension == ".pdf":
            logger.info(f"Extracting text from PDF: {file.filename}")
            extraction_result = await extract_pdf_text(content, file.filename)

        elif file_extension == ".txt":
            logger.info(f"Extracting text from text file: {file.filename}")
            extraction_result = await extract_text_file(content, file.filename)

        elif file_extension in [".png", ".jpg", ".jpeg"]:
            # Placeholder for future OCR implementation
            extraction_result = {
                "status": "pending",
                "message": "OCR extraction for images not yet implemented",
                "summary": {
                    "preview": "OCR extraction for images not yet implemented",
                    "ending": None,
                    "table_of_contents": []
                },
                "full_text": "",
                "metadata": {
                    "document_info": {
                        "pages": 1,
                        "words": 0,
                        "characters": 0,
                        "confidence": 0.0
                    },
                    "analysis": {
                        "estimated_read_time": "0 minutes",
                        "language": "unknown",
                        "document_type": "image_document",
                        "key_phrases": [],
                        "has_table_of_contents": False
                    }
                }
            }

        # Determine overall success status
        success_status = "success" if extraction_result["status"] == "completed" else "partial"
        base_message = "Text extracted successfully" if extraction_result["status"] == "completed" else "File processed with limited extraction"

        # Prepare response
        response = {
            "status": success_status,
            "message": base_message,
            "file_info": {
                "filename": file.filename,
                "size": file_size,
                "type": file_extension,
                "content_type": file.content_type
            },
            "extraction": extraction_result
        }

        # Perform ISO analysis if requested and extraction was successful
        if iso_analysis and extraction_result["status"] == "completed":
            try:
                document_text = extraction_result.get("full_text", "")

                if document_text.strip():
                    logger.info(f"Performing ISO analysis for: {file.filename}")

                    document_info = {
                        "filename": file.filename,
                        "size": file_size,
                        "type": file_extension,
                        "content_type": file.content_type,
                        "word_count": len(document_text.split()),
                        "character_count": len(document_text)
                    }

                    agent_card = await iso_agent.analyze_document(
                        document_text=document_text,
                        document_info=document_info,
                        analysis_scope="comprehensive"
                    )

                    response["iso_analysis"] = agent_card.to_dict()
                    response["message"] += " with ISO 27001:2022 compliance analysis"

                    logger.info(f"ISO analysis completed for {file.filename}. Analysis ID: {agent_card.analysis_id}")
                else:
                    response["iso_analysis"] = {
                        "status": "skipped",
                        "message": "No text content available for ISO analysis"
                    }

            except Exception as e:
                logger.error(f"ISO analysis failed for {file.filename}: {e}")
                response["iso_analysis"] = {
                    "status": "failed",
                    "message": f"ISO analysis failed: {str(e)}"
                }

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing file {file.filename if file.filename else 'unknown'}: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")


@app.get("/api/v1/status/{document_id}")
async def get_processing_status(document_id: str):
    """Get processing status (simplified version)."""
    return {
        "document_id": document_id,
        "status": "completed",
        "message": "This is a simplified endpoint - actual processing not implemented"
    }


@app.get("/api/v1/results/{document_id}")
async def get_extraction_results(document_id: str):
    """Get extraction results (simplified version)."""
    return {
        "document_id": document_id,
        "status": "success",
        "results": {
            "text": "Sample extracted text would appear here",
            "metadata": {
                "pages": 1,
                "words": 100,
                "confidence": 0.95
            }
        },
        "message": "This is a simplified endpoint - actual results not implemented"
    }


@app.post("/api/v1/iso-analyze")
async def analyze_iso_compliance(
    file: UploadFile = File(...),
    analysis_scope: str = "comprehensive"
):
    """
    Analyze document for ISO 27001:2022 compliance

    Args:
        file: Document to analyze
        analysis_scope: "comprehensive", "controls_only", "clauses_only"

    Returns:
        ISO Agent Card with complete compliance analysis
    """
    try:
        # Validate file and extract text (reuse existing extraction logic)
        if not file.filename:
            raise HTTPException(status_code=400, detail="File must have a filename")

        file_extension = Path(file.filename).suffix.lower()
        allowed_extensions = [".pdf", ".txt"]

        if file_extension not in allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"ISO analysis supports only PDF and TXT files. Provided: {file_extension}"
            )

        content = await file.read()
        file_size = len(content)

        if file_size > settings.max_file_size:
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Maximum size: {settings.max_file_size / (1024*1024):.1f}MB"
            )

        # Extract text from document
        document_text = ""
        if file_extension == ".pdf":
            extraction_result = await extract_pdf_text(content, file.filename)
            if extraction_result["status"] == "completed":
                document_text = extraction_result["full_text"]
            else:
                raise HTTPException(status_code=422, detail="Failed to extract text from PDF")

        elif file_extension == ".txt":
            extraction_result = await extract_text_file(content, file.filename)
            if extraction_result["status"] == "completed":
                document_text = extraction_result["full_text"]
            else:
                raise HTTPException(status_code=422, detail="Failed to extract text from file")

        if not document_text.strip():
            raise HTTPException(status_code=422, detail="No text content found in document")

        # Prepare document info
        document_info = {
            "filename": file.filename,
            "size": file_size,
            "type": file_extension,
            "content_type": file.content_type,
            "word_count": len(document_text.split()),
            "character_count": len(document_text)
        }

        # Perform ISO compliance analysis
        logger.info(f"Starting ISO compliance analysis for: {file.filename}")
        agent_card = await iso_agent.analyze_document(
            document_text=document_text,
            document_info=document_info,
            analysis_scope=analysis_scope
        )

        logger.info(f"ISO analysis completed for {file.filename}. Analysis ID: {agent_card.analysis_id}")

        return {
            "status": "success",
            "message": "ISO 27001:2022 compliance analysis completed",
            "analysis": agent_card.to_dict()
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during ISO analysis for {file.filename if file.filename else 'unknown'}: {e}")
        raise HTTPException(status_code=500, detail=f"ISO analysis failed: {str(e)}")


@app.post("/api/v1/iso-consult")
async def iso_consultation(request: Dict[str, Any]):
    """
    Get expert consultation on ISO 27001:2022 topics

    Args:
        request: {"question": "Your ISO 27001:2022 question"}

    Returns:
        ISO Consultation Card with expert response
    """
    try:
        question = request.get("question", "").strip()

        if not question:
            raise HTTPException(status_code=400, detail="Question is required")

        if len(question) > 1000:
            raise HTTPException(status_code=400, detail="Question too long (max 1000 characters)")

        logger.info(f"Processing ISO consultation request: {question[:100]}...")

        consultation_card = await iso_agent.consult(question)

        logger.info(f"ISO consultation completed. Query ID: {consultation_card.query_id}")

        return {
            "status": "success",
            "message": "ISO consultation completed",
            "consultation": consultation_card.to_dict()
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during ISO consultation: {e}")
        raise HTTPException(status_code=500, detail=f"ISO consultation failed: {str(e)}")


@app.get("/api/v1/iso-controls")
async def get_iso_controls(
    category: Optional[str] = None,
    search: Optional[str] = None
):
    """
    Get ISO 27001:2022 Annex A controls

    Args:
        category: Filter by category (organizational, people, physical, technological)
        search: Search controls by title or description

    Returns:
        List of ISO controls matching criteria
    """
    try:
        from .iso_knowledge.annex_a_controls import (
            get_all_controls, get_controls_by_category, search_controls, get_control_categories
        )

        if search:
            controls = search_controls(search)
            return {
                "status": "success",
                "message": f"Found {len(controls)} controls matching '{search}'",
                "controls": [
                    {
                        "id": ctrl.id,
                        "title": ctrl.title,
                        "category": ctrl.category,
                        "description": ctrl.description[:200] + "..." if len(ctrl.description) > 200 else ctrl.description
                    }
                    for ctrl in controls
                ]
            }

        if category:
            if category not in get_control_categories():
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid category. Available: {', '.join(get_control_categories())}"
                )

            controls = get_controls_by_category(category)
            return {
                "status": "success",
                "message": f"Found {len(controls)} controls in '{category}' category",
                "controls": [
                    {
                        "id": ctrl.id,
                        "title": ctrl.title,
                        "category": ctrl.category,
                        "description": ctrl.description[:200] + "..." if len(ctrl.description) > 200 else ctrl.description
                    }
                    for ctrl in controls
                ]
            }

        # Return all controls summary
        all_controls = get_all_controls()
        return {
            "status": "success",
            "message": f"Total {len(all_controls)} controls available",
            "summary": {
                "total_controls": len(all_controls),
                "categories": get_control_categories()
            },
            "controls": [
                {
                    "id": ctrl.id,
                    "title": ctrl.title,
                    "category": ctrl.category
                }
                for ctrl in list(all_controls.values())[:20]  # First 20 for overview
            ]
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving ISO controls: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve controls: {str(e)}")


@app.get("/api/v1/iso-controls/{control_id}")
async def get_iso_control_detail(control_id: str):
    """
    Get detailed information about a specific ISO control

    Args:
        control_id: ISO control ID (e.g., "5.1", "8.2")

    Returns:
        Detailed control information including implementation guidance
    """
    try:
        from .iso_knowledge.annex_a_controls import get_control_by_id

        control = get_control_by_id(control_id)
        if not control:
            raise HTTPException(status_code=404, detail=f"Control {control_id} not found")

        return {
            "status": "success",
            "message": f"Control {control_id} details retrieved",
            "control": {
                "id": control.id,
                "title": control.title,
                "description": control.description,
                "purpose": control.purpose,
                "category": control.category,
                "control_type": [ct.value for ct in control.control_type],
                "security_properties": [sp.value for sp in control.security_properties],
                "cybersecurity_concepts": [cc.value for cc in control.cybersecurity_concepts],
                "implementation_guidance": control.implementation_guidance,
                "related_controls": control.related_controls
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving control {control_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve control: {str(e)}")


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """General exception handler."""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "An internal error occurred"}
    )


def main():
    """Main entry point for running the application."""
    import uvicorn

    logger.info(f"Starting server on {settings.api_host}:{settings.api_port}")

    uvicorn.run(
        "document_extraction_system.main_simple:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug,
        log_level=settings.log_level.lower()
    )


if __name__ == "__main__":
    main()