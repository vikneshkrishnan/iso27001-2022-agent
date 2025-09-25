"""
Pinecone Vector Store for ISO 27001:2022 Agent
Manages vector embeddings and semantic search capabilities
"""

import json
import logging
import hashlib
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum

try:
    import pinecone
    from pinecone import Pinecone, ServerlessSpec
    PINECONE_AVAILABLE = True
except ImportError:
    PINECONE_AVAILABLE = False
    logging.warning("Pinecone client not available")

try:
    import openai
    from openai import AsyncOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

from ..config.settings import get_settings

logger = logging.getLogger(__name__)


class VectorNamespace(Enum):
    """Vector namespaces for different data types"""
    DOCUMENTS = "documents"
    CONTROLS = "controls"
    CLAUSES = "clauses"
    RECOMMENDATIONS = "recommendations"
    CONSULTATIONS = "consultations"
    KNOWLEDGE = "knowledge"
    ANALYSES = "analyses"


@dataclass
class VectorDocument:
    """Document to be stored in vector database"""
    id: str
    content: str
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = None
    namespace: str = VectorNamespace.DOCUMENTS.value

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class SearchResult:
    """Search result from vector database"""
    id: str
    content: str
    score: float
    metadata: Dict[str, Any]
    namespace: str


@dataclass
class SearchQuery:
    """Search query configuration"""
    text: str
    namespace: str = VectorNamespace.DOCUMENTS.value
    top_k: int = 5
    filter_metadata: Optional[Dict[str, Any]] = None
    include_content: bool = True
    min_score: float = 0.0


class PineconeVectorStore:
    """Pinecone vector store implementation"""

    def __init__(self, api_key: Optional[str] = None, environment: Optional[str] = None,
                 index_name: Optional[str] = None):
        """Initialize Pinecone vector store"""
        self.settings = get_settings()
        self.api_key = api_key or self.settings.pinecone_api_key
        self.environment = environment or self.settings.pinecone_environment
        self.index_name = index_name or self.settings.pinecone_index_name

        self.pc = None
        self.index = None
        self.openai_client = None

        if not PINECONE_AVAILABLE:
            logger.error("Pinecone client not available - install pinecone-client")
            return

        if not self.api_key:
            logger.error("Pinecone API key not provided")
            return

        self._initialize_pinecone()
        self._initialize_openai()

    def _initialize_pinecone(self):
        """Initialize Pinecone client and index"""
        try:
            self.pc = Pinecone(api_key=self.api_key)

            # Check if index exists, create if not
            if self.index_name not in self.pc.list_indexes().names():
                logger.info(f"Creating Pinecone index: {self.index_name}")
                # Use the environment setting for region (us-east-1 for your free plan)
                region = "us-east-1"  # Free plan supported region
                self.pc.create_index(
                    name=self.index_name,
                    dimension=self.settings.pinecone_dimension,
                    metric="cosine",
                    spec=ServerlessSpec(
                        cloud="aws",
                        region=region
                    )
                )

            self.index = self.pc.Index(self.index_name)
            logger.info(f"Pinecone index '{self.index_name}' initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize Pinecone: {e}")
            self.pc = None
            self.index = None

    def _initialize_openai(self):
        """Initialize OpenAI client for embeddings"""
        try:
            if OPENAI_AVAILABLE and (self.settings.openai_api_key or os.getenv("OPENAI_API_KEY")):
                from openai import AsyncOpenAI
                self.openai_client = AsyncOpenAI(
                    api_key=self.settings.openai_api_key or os.getenv("OPENAI_API_KEY")
                )
                logger.info("OpenAI client initialized for embeddings")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}")

    async def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for text using OpenAI"""
        if not self.openai_client:
            raise Exception("OpenAI client not available for embedding generation")

        try:
            # Truncate text if too long
            max_tokens = self.settings.max_embedding_tokens
            if len(text.split()) > max_tokens:
                text = " ".join(text.split()[:max_tokens])

            response = await self.openai_client.embeddings.create(
                model=self.settings.embedding_model,
                input=text,
                dimensions=self.settings.pinecone_dimension
            )

            return response.data[0].embedding

        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            raise

    def _generate_id(self, content: str, namespace: str = None, prefix: str = None) -> str:
        """Generate unique ID for document"""
        content_hash = hashlib.md5(content.encode()).hexdigest()[:12]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        parts = []
        if namespace:
            parts.append(namespace)
        if prefix:
            parts.append(prefix)
        parts.extend([timestamp, content_hash])

        return "_".join(parts)

    async def store_document(self, document: VectorDocument) -> bool:
        """Store document in vector database"""
        if not self.index:
            logger.error("Pinecone index not available")
            return False

        try:
            # Generate embedding if not provided
            if not document.embedding:
                document.embedding = await self.generate_embedding(document.content)

            # Generate ID if not provided
            if not document.id:
                document.id = self._generate_id(document.content, document.namespace)

            # Add timestamp to metadata
            document.metadata["stored_at"] = datetime.now().isoformat()
            document.metadata["content_length"] = len(document.content)

            # Store in Pinecone
            self.index.upsert(
                vectors=[(
                    document.id,
                    document.embedding,
                    document.metadata
                )],
                namespace=document.namespace
            )

            logger.info(f"Stored document {document.id} in namespace {document.namespace}")
            return True

        except Exception as e:
            logger.error(f"Failed to store document: {e}")
            return False

    async def store_documents_batch(self, documents: List[VectorDocument]) -> Dict[str, bool]:
        """Store multiple documents in batch"""
        results = {}

        if not self.index:
            logger.error("Pinecone index not available")
            return {doc.id or "unknown": False for doc in documents}

        try:
            # Group documents by namespace
            namespace_docs = {}
            for doc in documents:
                if doc.namespace not in namespace_docs:
                    namespace_docs[doc.namespace] = []
                namespace_docs[doc.namespace].append(doc)

            # Process each namespace separately
            for namespace, docs in namespace_docs.items():
                vectors_to_upsert = []

                for doc in docs:
                    try:
                        # Generate embedding if not provided
                        if not doc.embedding:
                            doc.embedding = await self.generate_embedding(doc.content)

                        # Generate ID if not provided
                        if not doc.id:
                            doc.id = self._generate_id(doc.content, doc.namespace)

                        # Add metadata
                        doc.metadata["stored_at"] = datetime.now().isoformat()
                        doc.metadata["content_length"] = len(doc.content)

                        vectors_to_upsert.append((
                            doc.id,
                            doc.embedding,
                            doc.metadata
                        ))

                        results[doc.id] = True

                    except Exception as e:
                        logger.error(f"Failed to process document {doc.id}: {e}")
                        results[doc.id or "unknown"] = False

                # Batch upsert for this namespace
                if vectors_to_upsert:
                    self.index.upsert(vectors=vectors_to_upsert, namespace=namespace)
                    logger.info(f"Batch stored {len(vectors_to_upsert)} documents in namespace {namespace}")

            return results

        except Exception as e:
            logger.error(f"Failed to store documents batch: {e}")
            return {doc.id or "unknown": False for doc in documents}

    async def search(self, query: SearchQuery) -> List[SearchResult]:
        """Search for similar documents"""
        if not self.index:
            logger.error("Pinecone index not available")
            return []

        try:
            # Generate embedding for query
            query_embedding = await self.generate_embedding(query.text)

            # Perform search
            search_response = self.index.query(
                vector=query_embedding,
                top_k=query.top_k,
                namespace=query.namespace,
                filter=query.filter_metadata,
                include_metadata=True,
                include_values=False
            )

            # Convert results
            results = []
            for match in search_response.matches:
                if match.score >= query.min_score:
                    result = SearchResult(
                        id=match.id,
                        content=match.metadata.get("content", "") if query.include_content else "",
                        score=match.score,
                        metadata=match.metadata,
                        namespace=query.namespace
                    )
                    results.append(result)

            logger.info(f"Found {len(results)} results for query in namespace {query.namespace}")
            return results

        except Exception as e:
            logger.error(f"Failed to search: {e}")
            return []

    async def get_document(self, document_id: str, namespace: str) -> Optional[VectorDocument]:
        """Retrieve specific document by ID"""
        if not self.index:
            logger.error("Pinecone index not available")
            return None

        try:
            fetch_response = self.index.fetch([document_id], namespace=namespace)

            if document_id in fetch_response.vectors:
                vector = fetch_response.vectors[document_id]
                return VectorDocument(
                    id=document_id,
                    content=vector.metadata.get("content", ""),
                    embedding=vector.values if vector.values else None,
                    metadata=vector.metadata,
                    namespace=namespace
                )

            return None

        except Exception as e:
            logger.error(f"Failed to get document {document_id}: {e}")
            return None

    def delete_document(self, document_id: str, namespace: str) -> bool:
        """Delete document from vector database"""
        if not self.index:
            logger.error("Pinecone index not available")
            return False

        try:
            self.index.delete(ids=[document_id], namespace=namespace)
            logger.info(f"Deleted document {document_id} from namespace {namespace}")
            return True

        except Exception as e:
            logger.error(f"Failed to delete document {document_id}: {e}")
            return False

    def delete_namespace(self, namespace: str) -> bool:
        """Delete all documents in a namespace"""
        if not self.index:
            logger.error("Pinecone index not available")
            return False

        try:
            self.index.delete(delete_all=True, namespace=namespace)
            logger.info(f"Deleted all documents in namespace {namespace}")
            return True

        except Exception as e:
            logger.error(f"Failed to delete namespace {namespace}: {e}")
            return False

    def get_index_stats(self) -> Dict[str, Any]:
        """Get index statistics"""
        if not self.index:
            logger.error("Pinecone index not available")
            return {}

        try:
            stats = self.index.describe_index_stats()
            return {
                "total_vector_count": stats.total_vector_count,
                "dimension": stats.dimension,
                "index_fullness": stats.index_fullness,
                "namespaces": {
                    ns: {"vector_count": info.vector_count}
                    for ns, info in stats.namespaces.items()
                }
            }

        except Exception as e:
            logger.error(f"Failed to get index stats: {e}")
            return {}

    def health_check(self) -> Dict[str, Any]:
        """Check health of vector store"""
        health = {
            "pinecone_available": PINECONE_AVAILABLE,
            "openai_available": self.openai_client is not None,
            "index_available": self.index is not None,
            "api_key_configured": bool(self.api_key)
        }

        if self.index:
            try:
                stats = self.get_index_stats()
                health["index_stats"] = stats
                health["operational"] = True
            except:
                health["operational"] = False
        else:
            health["operational"] = False

        return health


# Import fix
import os