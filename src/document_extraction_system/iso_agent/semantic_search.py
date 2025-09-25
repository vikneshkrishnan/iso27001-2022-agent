"""
Semantic Search Engine for ISO 27001:2022 Agent
Advanced semantic search capabilities using Pinecone vector database
"""

import logging
import json
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum

from .vector_store import PineconeVectorStore, VectorDocument, SearchQuery, SearchResult, VectorNamespace
from ..iso_knowledge.agent_response import (
    ISOAgentCard, ControlAssessment, ClauseAssessment,
    Recommendation, ComplianceStatus, DocumentType
)

logger = logging.getLogger(__name__)


class SearchType(Enum):
    """Types of semantic searches available"""
    SIMILAR_DOCUMENTS = "similar_documents"
    SIMILAR_ASSESSMENTS = "similar_assessments"
    RELATED_CONTROLS = "related_controls"
    HISTORICAL_PATTERNS = "historical_patterns"
    KNOWLEDGE_LOOKUP = "knowledge_lookup"
    RECOMMENDATION_CONTEXT = "recommendation_context"


@dataclass
class EnhancedSearchQuery:
    """Enhanced search query with ISO-specific parameters"""
    text: str
    search_type: SearchType
    namespace: str = VectorNamespace.DOCUMENTS.value
    top_k: int = 5
    min_score: float = 0.7
    document_type: Optional[str] = None
    compliance_status: Optional[str] = None
    control_categories: Optional[List[str]] = None
    date_range: Optional[Tuple[datetime, datetime]] = None
    organization_context: Optional[str] = None


@dataclass
class ContextualResult:
    """Search result with additional context"""
    result: SearchResult
    relevance_explanation: str
    iso_context: Dict[str, Any]
    confidence_factors: List[str]
    related_items: List[str]


class SemanticSearchEngine:
    """Advanced semantic search engine for ISO compliance data"""

    def __init__(self, vector_store: Optional[PineconeVectorStore] = None):
        """Initialize semantic search engine"""
        self.vector_store = vector_store or PineconeVectorStore()
        self.search_history: List[Dict[str, Any]] = []

    async def search_similar_documents(self, document_content: str, document_type: str = None,
                                     top_k: int = 5) -> List[ContextualResult]:
        """Find documents similar to the given content"""

        query = SearchQuery(
            text=document_content,
            namespace=VectorNamespace.DOCUMENTS.value,
            top_k=top_k,
            filter_metadata={
                "document_type": document_type
            } if document_type else None,
            min_score=0.7
        )

        results = await self.vector_store.search(query)
        contextual_results = []

        for result in results:
            # Add contextual information
            iso_context = {
                "document_type": result.metadata.get("document_type"),
                "analysis_date": result.metadata.get("analysis_date"),
                "compliance_overview": result.metadata.get("compliance_overview"),
                "control_count": result.metadata.get("assessed_controls", 0)
            }

            confidence_factors = []
            if result.score > 0.9:
                confidence_factors.append("Very high semantic similarity")
            elif result.score > 0.8:
                confidence_factors.append("High semantic similarity")

            if result.metadata.get("document_type") == document_type:
                confidence_factors.append("Same document type")

            contextual_result = ContextualResult(
                result=result,
                relevance_explanation=f"Similar content patterns with {result.score:.2f} similarity",
                iso_context=iso_context,
                confidence_factors=confidence_factors,
                related_items=result.metadata.get("related_controls", [])
            )
            contextual_results.append(contextual_result)

        return contextual_results

    async def search_control_implementations(self, control_id: str,
                                           implementation_context: str = None) -> List[ContextualResult]:
        """Find similar control implementations"""

        # Build search text
        search_text = f"Control {control_id}"
        if implementation_context:
            search_text += f" {implementation_context}"

        query = SearchQuery(
            text=search_text,
            namespace=VectorNamespace.CONTROLS.value,
            top_k=10,
            filter_metadata={
                "control_id": control_id
            },
            min_score=0.6
        )

        results = await self.vector_store.search(query)
        contextual_results = []

        for result in results:
            # Extract control-specific context
            iso_context = {
                "control_id": result.metadata.get("control_id"),
                "compliance_status": result.metadata.get("compliance_status"),
                "implementation_approach": result.metadata.get("implementation_approach"),
                "evidence_strength": result.metadata.get("evidence_count", 0),
                "gap_areas": result.metadata.get("gaps_identified", [])
            }

            # Determine relevance explanation
            status = result.metadata.get("compliance_status")
            explanation = f"Control {control_id} implementation"
            if status == "compliant":
                explanation += " - successful implementation example"
            elif status == "partially_compliant":
                explanation += " - partial implementation with identified gaps"

            confidence_factors = [
                f"Control match: {control_id}",
                f"Implementation score: {result.score:.2f}"
            ]

            contextual_result = ContextualResult(
                result=result,
                relevance_explanation=explanation,
                iso_context=iso_context,
                confidence_factors=confidence_factors,
                related_items=result.metadata.get("related_controls", [])
            )
            contextual_results.append(contextual_result)

        return contextual_results

    async def search_compliance_patterns(self, gap_description: str,
                                       category: str = None) -> List[ContextualResult]:
        """Find similar compliance gaps and their solutions"""

        query = SearchQuery(
            text=f"compliance gap {gap_description}",
            namespace=VectorNamespace.RECOMMENDATIONS.value,
            top_k=8,
            filter_metadata={
                "category": category
            } if category else None,
            min_score=0.65
        )

        results = await self.vector_store.search(query)
        contextual_results = []

        for result in results:
            iso_context = {
                "gap_category": result.metadata.get("category"),
                "solution_effectiveness": result.metadata.get("effectiveness_score"),
                "implementation_effort": result.metadata.get("effort_estimate"),
                "success_rate": result.metadata.get("success_rate"),
                "affected_controls": result.metadata.get("related_controls", [])
            }

            explanation = f"Similar compliance gap in {result.metadata.get('category', 'general')} category"
            if result.metadata.get("success_rate"):
                explanation += f" with {result.metadata['success_rate']}% success rate"

            confidence_factors = [
                f"Gap similarity: {result.score:.2f}",
                f"Category match: {category == result.metadata.get('category')}"
            ]

            contextual_result = ContextualResult(
                result=result,
                relevance_explanation=explanation,
                iso_context=iso_context,
                confidence_factors=confidence_factors,
                related_items=result.metadata.get("related_recommendations", [])
            )
            contextual_results.append(contextual_result)

        return contextual_results

    async def search_expert_consultations(self, question: str, topic_area: str = None) -> List[ContextualResult]:
        """Find similar expert consultation responses"""

        query = SearchQuery(
            text=question,
            namespace=VectorNamespace.CONSULTATIONS.value,
            top_k=5,
            filter_metadata={
                "topic_area": topic_area
            } if topic_area else None,
            min_score=0.75
        )

        results = await self.vector_store.search(query)
        contextual_results = []

        for result in results:
            iso_context = {
                "topic_area": result.metadata.get("topic_area"),
                "response_quality": result.metadata.get("confidence_score"),
                "consultation_date": result.metadata.get("consultation_date"),
                "referenced_controls": result.metadata.get("relevant_controls", []),
                "referenced_clauses": result.metadata.get("relevant_clauses", [])
            }

            explanation = f"Similar consultation on {result.metadata.get('topic_area', 'ISO compliance')}"
            if result.metadata.get("confidence_score"):
                explanation += f" (confidence: {result.metadata['confidence_score']:.2f})"

            confidence_factors = [
                f"Question similarity: {result.score:.2f}",
                f"Expert confidence: {result.metadata.get('confidence_score', 'unknown')}"
            ]

            contextual_result = ContextualResult(
                result=result,
                relevance_explanation=explanation,
                iso_context=iso_context,
                confidence_factors=confidence_factors,
                related_items=result.metadata.get("related_topics", [])
            )
            contextual_results.append(contextual_result)

        return contextual_results

    async def hybrid_search(self, enhanced_query: EnhancedSearchQuery) -> List[ContextualResult]:
        """Perform hybrid search combining multiple search strategies"""

        all_results = []

        # Primary semantic search
        base_query = SearchQuery(
            text=enhanced_query.text,
            namespace=enhanced_query.namespace,
            top_k=enhanced_query.top_k * 2,  # Get more results for filtering
            min_score=enhanced_query.min_score,
            filter_metadata=self._build_filter_metadata(enhanced_query)
        )

        primary_results = await self.vector_store.search(base_query)

        # Add contextual searches based on search type
        if enhanced_query.search_type == SearchType.SIMILAR_ASSESSMENTS:
            # Also search in controls and recommendations
            control_query = SearchQuery(
                text=enhanced_query.text,
                namespace=VectorNamespace.CONTROLS.value,
                top_k=5,
                min_score=0.7
            )
            control_results = await self.vector_store.search(control_query)
            primary_results.extend(control_results)

        elif enhanced_query.search_type == SearchType.RECOMMENDATION_CONTEXT:
            # Include historical patterns
            pattern_query = SearchQuery(
                text=enhanced_query.text,
                namespace=VectorNamespace.ANALYSES.value,
                top_k=3,
                min_score=0.75
            )
            pattern_results = await self.vector_store.search(pattern_query)
            primary_results.extend(pattern_results)

        # Convert to contextual results with enhanced metadata
        for result in primary_results[:enhanced_query.top_k]:
            contextual_result = await self._enrich_result(result, enhanced_query)
            all_results.append(contextual_result)

        # Sort by combined relevance score
        all_results.sort(key=lambda x: x.result.score, reverse=True)

        return all_results

    def _build_filter_metadata(self, query: EnhancedSearchQuery) -> Optional[Dict[str, Any]]:
        """Build metadata filter for enhanced query"""
        filters = {}

        if query.document_type:
            filters["document_type"] = query.document_type

        if query.compliance_status:
            filters["compliance_status"] = query.compliance_status

        if query.control_categories:
            filters["category"] = {"$in": query.control_categories}

        if query.date_range:
            start_date, end_date = query.date_range
            filters["analysis_date"] = {
                "$gte": start_date.isoformat(),
                "$lte": end_date.isoformat()
            }

        if query.organization_context:
            filters["organization_type"] = query.organization_context

        return filters if filters else None

    async def _enrich_result(self, result: SearchResult, query: EnhancedSearchQuery) -> ContextualResult:
        """Enrich search result with contextual information"""

        # Determine ISO context based on namespace
        iso_context = {}
        confidence_factors = []
        related_items = []

        if result.namespace == VectorNamespace.DOCUMENTS.value:
            iso_context = {
                "document_analysis": True,
                "compliance_summary": result.metadata.get("compliance_overview"),
                "control_coverage": result.metadata.get("control_coverage_percentage")
            }
        elif result.namespace == VectorNamespace.CONTROLS.value:
            iso_context = {
                "control_assessment": True,
                "implementation_status": result.metadata.get("compliance_status"),
                "evidence_quality": result.metadata.get("evidence_count")
            }
        elif result.namespace == VectorNamespace.RECOMMENDATIONS.value:
            iso_context = {
                "recommendation_data": True,
                "priority_level": result.metadata.get("priority"),
                "effort_required": result.metadata.get("effort_estimate")
            }

        # Generate relevance explanation
        explanation = self._generate_relevance_explanation(result, query)

        # Build confidence factors
        confidence_factors.append(f"Semantic similarity: {result.score:.2f}")

        if result.metadata.get("validation_status") == "validated":
            confidence_factors.append("ISO-validated content")

        if result.metadata.get("confidence_score"):
            confidence_factors.append(f"Analysis confidence: {result.metadata['confidence_score']:.2f}")

        return ContextualResult(
            result=result,
            relevance_explanation=explanation,
            iso_context=iso_context,
            confidence_factors=confidence_factors,
            related_items=result.metadata.get("related_items", [])
        )

    def _generate_relevance_explanation(self, result: SearchResult, query: EnhancedSearchQuery) -> str:
        """Generate human-readable explanation of why result is relevant"""

        base_explanation = f"Content similarity: {result.score:.2f}"

        if query.search_type == SearchType.SIMILAR_DOCUMENTS:
            return f"Similar document content with {base_explanation}"
        elif query.search_type == SearchType.SIMILAR_ASSESSMENTS:
            return f"Similar compliance assessment pattern with {base_explanation}"
        elif query.search_type == SearchType.RELATED_CONTROLS:
            return f"Related ISO control implementation with {base_explanation}"
        elif query.search_type == SearchType.HISTORICAL_PATTERNS:
            return f"Historical compliance pattern with {base_explanation}"
        elif query.search_type == SearchType.RECOMMENDATION_CONTEXT:
            return f"Relevant recommendation context with {base_explanation}"
        else:
            return f"Semantic match with {base_explanation}"

    async def get_search_suggestions(self, partial_query: str, search_type: SearchType) -> List[str]:
        """Get search suggestions based on stored data"""

        # This could be enhanced with a separate suggestions index
        suggestions = []

        # Simple implementation - get top searches and extract common patterns
        if len(partial_query) >= 3:
            query = SearchQuery(
                text=partial_query,
                namespace=VectorNamespace.KNOWLEDGE.value,
                top_k=10,
                min_score=0.5
            )

            results = await self.vector_store.search(query)

            for result in results[:5]:
                if result.metadata.get("keywords"):
                    suggestions.extend(result.metadata["keywords"][:2])

        return list(set(suggestions))[:5]

    def get_search_analytics(self) -> Dict[str, Any]:
        """Get analytics about search patterns"""

        total_searches = len(self.search_history)
        if total_searches == 0:
            return {"total_searches": 0}

        # Analyze search patterns
        search_types = {}
        avg_results = 0
        recent_searches = self.search_history[-100:]  # Last 100 searches

        for search in recent_searches:
            search_type = search.get("search_type", "unknown")
            search_types[search_type] = search_types.get(search_type, 0) + 1
            avg_results += len(search.get("results", []))

        avg_results = avg_results / len(recent_searches) if recent_searches else 0

        return {
            "total_searches": total_searches,
            "search_types": search_types,
            "avg_results_per_search": avg_results,
            "most_common_search": max(search_types.items(), key=lambda x: x[1])[0] if search_types else None
        }

    async def cleanup_old_searches(self, days_to_keep: int = 30):
        """Clean up old search history"""
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)

        original_count = len(self.search_history)
        self.search_history = [
            search for search in self.search_history
            if datetime.fromisoformat(search.get("timestamp", "1900-01-01")) > cutoff_date
        ]

        cleaned_count = original_count - len(self.search_history)
        logger.info(f"Cleaned up {cleaned_count} old search records")

        return cleaned_count

    async def find_similar_controls(self, document_text: str, top_k: int = 20, min_similarity: float = 0.6) -> List[SearchResult]:
        """Find controls similar to document content using semantic search"""
        try:
            # Search in knowledge namespace for control implementations
            query = SearchQuery(
                text=document_text,
                namespace=VectorNamespace.KNOWLEDGE.value,
                top_k=top_k,
                min_score=min_similarity,
                filter_metadata={"knowledge_type": "control"}
            )

            results = await self.vector_store.search(query)

            # Also search in controls namespace for historical assessments
            if len(results) < top_k // 2:
                controls_query = SearchQuery(
                    text=document_text,
                    namespace=VectorNamespace.CONTROLS.value,
                    top_k=top_k - len(results),
                    min_score=min_similarity
                )

                controls_results = await self.vector_store.search(controls_query)
                results.extend(controls_results)

            logger.info(f"Found {len(results)} similar controls for document")
            return results

        except Exception as e:
            logger.error(f"Failed to find similar controls: {e}")
            return []

    async def find_similar_analyses(self, document_features: Dict[str, Any], top_k: int = 10) -> List[SearchResult]:
        """Find similar historical analyses based on document features"""
        try:
            # Create search text from document features
            search_components = []

            # Extract key concepts for search
            concept_matches = document_features.get('concept_matches', {})
            for category, matches in concept_matches.items():
                if isinstance(matches, list):
                    for match in matches[:2]:  # Top 2 matches per category
                        if hasattr(match, 'concept'):
                            search_components.append(match.concept)

            if not search_components:
                return []

            search_text = " ".join(search_components)

            query = SearchQuery(
                text=search_text,
                namespace=VectorNamespace.ANALYSES.value,
                top_k=top_k,
                min_score=0.7
            )

            results = await self.vector_store.search(query)
            logger.info(f"Found {len(results)} similar historical analyses")
            return results

        except Exception as e:
            logger.error(f"Failed to find similar analyses: {e}")
            return []