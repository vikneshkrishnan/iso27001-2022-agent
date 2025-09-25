"""
Vector Index Manager for ISO 27001:2022 Agent
Manages batch operations, indexing, and maintenance of vector data
"""

import json
import logging
import asyncio
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
import hashlib

from .vector_store import PineconeVectorStore, VectorDocument, VectorNamespace
from ..iso_knowledge.agent_response import (
    ISOAgentCard, ControlAssessment, ClauseAssessment,
    Recommendation, GapAnalysis, ComplianceOverview
)
from ..iso_knowledge.annex_a_controls import get_all_controls, Control
from ..iso_knowledge.management_clauses import get_all_clauses, get_all_requirements, Clause

logger = logging.getLogger(__name__)


@dataclass
class IndexingJob:
    """Represents a batch indexing job"""
    job_id: str
    job_type: str
    status: str  # pending, running, completed, failed
    total_documents: int
    processed_documents: int
    failed_documents: int
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None


@dataclass
class IndexStats:
    """Vector index statistics"""
    total_documents: int
    documents_by_namespace: Dict[str, int]
    last_updated: datetime
    health_status: str
    storage_usage_mb: float


class VectorIndexManager:
    """Manages vector index operations and maintenance"""

    def __init__(self, vector_store: Optional[PineconeVectorStore] = None):
        """Initialize vector index manager"""
        self.vector_store = vector_store or PineconeVectorStore()
        self.active_jobs: Dict[str, IndexingJob] = {}
        self.job_history: List[IndexingJob] = []

    async def index_iso_agent_analysis(self, agent_card: ISOAgentCard) -> bool:
        """Index complete ISO agent analysis"""
        try:
            documents_to_store = []

            # 1. Store main document analysis
            main_doc = self._create_analysis_document(agent_card)
            documents_to_store.append(main_doc)

            # 2. Store individual control assessments
            for assessment in agent_card.control_assessments:
                control_doc = self._create_control_document(assessment, agent_card.analysis_id)
                documents_to_store.append(control_doc)

            # 3. Store clause assessments
            for assessment in agent_card.clause_assessments:
                clause_doc = self._create_clause_document(assessment, agent_card.analysis_id)
                documents_to_store.append(clause_doc)

            # 4. Store recommendations
            for recommendation in agent_card.recommendations:
                rec_doc = self._create_recommendation_document(recommendation, agent_card.analysis_id)
                documents_to_store.append(rec_doc)

            # 5. Store gap analysis
            if agent_card.gap_analysis:
                gap_doc = self._create_gap_analysis_document(agent_card.gap_analysis, agent_card.analysis_id)
                documents_to_store.append(gap_doc)

            # Batch store all documents
            results = await self.vector_store.store_documents_batch(documents_to_store)

            success_count = sum(1 for success in results.values() if success)
            total_count = len(results)

            logger.info(f"Indexed {success_count}/{total_count} documents for analysis {agent_card.analysis_id}")

            return success_count == total_count

        except Exception as e:
            logger.error(f"Failed to index ISO agent analysis: {e}")
            return False

    def _create_analysis_document(self, agent_card: ISOAgentCard) -> VectorDocument:
        """Create vector document for main analysis"""
        content = f"""
        Document Analysis Summary
        Analysis ID: {agent_card.analysis_id}
        Document: {agent_card.document_info.get('filename', 'Unknown')}
        Document Type: {agent_card.document_classification.document_type.value}
        Overall Compliance: {agent_card.compliance_overview.compliant_percentage:.1f}%

        Executive Summary:
        {agent_card.executive_summary}

        Key Findings:
        - Controls Assessed: {len(agent_card.control_assessments)}
        - Clauses Assessed: {len(agent_card.clause_assessments)}
        - Recommendations: {len(agent_card.recommendations)}
        - Risk Areas: {len(agent_card.risk_areas)}
        - Quick Wins: {len(agent_card.quick_wins)}

        Coverage Percentage: {agent_card.coverage_percentage:.1f}%
        Analysis Confidence: {agent_card.analysis_confidence:.2f}
        """

        metadata = {
            "analysis_id": agent_card.analysis_id,
            "document_filename": agent_card.document_info.get('filename'),
            "document_type": agent_card.document_classification.document_type.value,
            "analysis_date": agent_card.timestamp.isoformat(),
            "compliance_overview": agent_card.compliance_overview.compliant_percentage,
            "coverage_percentage": agent_card.coverage_percentage,
            "analysis_confidence": agent_card.analysis_confidence,
            "control_count": len(agent_card.control_assessments),
            "recommendation_count": len(agent_card.recommendations),
            "risk_area_count": len(agent_card.risk_areas),
            "iso_relevance": agent_card.document_classification.iso_relevance,
            "applicable_categories": agent_card.document_classification.applicable_categories,
            "validation_status": "validated"
        }

        return VectorDocument(
            id=f"analysis_{agent_card.analysis_id}",
            content=content.strip(),
            metadata=metadata,
            namespace=VectorNamespace.ANALYSES.value
        )

    def _create_control_document(self, assessment: ControlAssessment, analysis_id: str) -> VectorDocument:
        """Create vector document for control assessment"""
        content = f"""
        Control Assessment: {assessment.control_id} - {assessment.control_title}
        Compliance Status: {assessment.status.value}
        Confidence Score: {assessment.confidence_score:.2f}

        Evidence Found:
        {chr(10).join(f"- {evidence}" for evidence in assessment.evidence_found)}

        Gaps Identified:
        {chr(10).join(f"- {gap}" for gap in assessment.gaps_identified)}

        Recommendations:
        {chr(10).join(f"- {rec}" for rec in assessment.recommendations)}
        """

        metadata = {
            "analysis_id": analysis_id,
            "control_id": assessment.control_id,
            "control_title": assessment.control_title,
            "compliance_status": assessment.status.value,
            "confidence_score": assessment.confidence_score,
            "evidence_count": len(assessment.evidence_found),
            "gaps_count": len(assessment.gaps_identified),
            "recommendations_count": len(assessment.recommendations),
            "assessment_type": "control",
            "validation_status": "validated"
        }

        return VectorDocument(
            id=f"control_{analysis_id}_{assessment.control_id}",
            content=content.strip(),
            metadata=metadata,
            namespace=VectorNamespace.CONTROLS.value
        )

    def _create_clause_document(self, assessment: ClauseAssessment, analysis_id: str) -> VectorDocument:
        """Create vector document for clause assessment"""
        content = f"""
        Management Clause Assessment: {assessment.clause_id} - {assessment.requirement_title}
        Compliance Status: {assessment.status.value}
        Confidence Score: {assessment.confidence_score:.2f}

        Evidence Found:
        {chr(10).join(f"- {evidence}" for evidence in assessment.evidence_found)}

        Missing Elements:
        {chr(10).join(f"- {element}" for element in assessment.missing_elements)}

        Recommendations:
        {chr(10).join(f"- {rec}" for rec in assessment.recommendations)}
        """

        metadata = {
            "analysis_id": analysis_id,
            "clause_id": assessment.clause_id,
            "requirement_title": assessment.requirement_title,
            "compliance_status": assessment.status.value,
            "confidence_score": assessment.confidence_score,
            "evidence_count": len(assessment.evidence_found),
            "missing_elements_count": len(assessment.missing_elements),
            "recommendations_count": len(assessment.recommendations),
            "assessment_type": "clause",
            "validation_status": "validated"
        }

        return VectorDocument(
            id=f"clause_{analysis_id}_{assessment.clause_id}",
            content=content.strip(),
            metadata=metadata,
            namespace=VectorNamespace.CLAUSES.value
        )

    def _create_recommendation_document(self, recommendation: Recommendation, analysis_id: str) -> VectorDocument:
        """Create vector document for recommendation"""
        content = f"""
        Recommendation: {recommendation.title}
        Priority: {recommendation.priority.value}
        Category: {getattr(recommendation, 'category', getattr(recommendation, 'recommendation_type', 'unknown').value if hasattr(recommendation, 'recommendation_type') else 'unknown')}

        Description:
        {recommendation.description}

        Implementation Steps:
        {chr(10).join(f"{i+1}. {step}" for i, step in enumerate(getattr(recommendation, 'implementation_steps', getattr(recommendation, 'specific_actions', []))))}

        Related Controls: {', '.join(getattr(recommendation, 'related_controls', getattr(recommendation, 'compliance_impact', [])))}
        Effort Estimate: {getattr(recommendation, 'effort_estimate', f"{getattr(recommendation, 'estimated_effort_hours', 0)} hours")}
        Timeline: {getattr(recommendation, 'timeline_estimate', getattr(recommendation, 'estimated_timeline', 'unknown'))}
        """

        metadata = {
            "analysis_id": analysis_id,
            "recommendation_id": recommendation.id,
            "title": recommendation.title,
            "priority": recommendation.priority.value,
            "category": getattr(recommendation, 'category', getattr(recommendation, 'recommendation_type', 'unknown').value if hasattr(recommendation, 'recommendation_type') else 'unknown'),
            "effort_estimate": getattr(recommendation, 'effort_estimate', f"{getattr(recommendation, 'estimated_effort_hours', 0)} hours"),
            "timeline_estimate": getattr(recommendation, 'timeline_estimate', getattr(recommendation, 'estimated_timeline', 'unknown')),
            "related_controls": getattr(recommendation, 'related_controls', getattr(recommendation, 'compliance_impact', [])),
            "implementation_steps_count": len(getattr(recommendation, 'implementation_steps', getattr(recommendation, 'specific_actions', []))),
            "validation_status": "validated"
        }

        return VectorDocument(
            id=f"rec_{analysis_id}_{recommendation.id}",
            content=content.strip(),
            metadata=metadata,
            namespace=VectorNamespace.RECOMMENDATIONS.value
        )

    def _create_gap_analysis_document(self, gap_analysis: GapAnalysis, analysis_id: str) -> VectorDocument:
        """Create vector document for gap analysis"""
        content = f"""
        Gap Analysis Summary
        Total Gaps: {gap_analysis.total_gaps}
        Critical Gaps: {gap_analysis.critical_gaps}
        High Priority Gaps: {gap_analysis.high_priority_gaps}

        Gap Categories:
        {chr(10).join(f"- {category}: {count} gaps" for category, count in gap_analysis.gap_categories.items())}

        Top Gap Areas:
        {chr(10).join(f"- {area}" for area in gap_analysis.top_gap_areas)}

        Overall Risk Level: {gap_analysis.overall_risk_level}
        Priority Assessment: {gap_analysis.priority_assessment}
        """

        metadata = {
            "analysis_id": analysis_id,
            "total_gaps": gap_analysis.total_gaps,
            "critical_gaps": gap_analysis.critical_gaps,
            "high_priority_gaps": gap_analysis.high_priority_gaps,
            "gap_categories": gap_analysis.gap_categories,
            "top_gap_areas": gap_analysis.top_gap_areas,
            "overall_risk_level": gap_analysis.overall_risk_level,
            "priority_assessment": gap_analysis.priority_assessment,
            "validation_status": "validated"
        }

        return VectorDocument(
            id=f"gaps_{analysis_id}",
            content=content.strip(),
            metadata=metadata,
            namespace=VectorNamespace.ANALYSES.value
        )

    async def index_consultation_response(self, question: str, answer: str,
                                        relevant_controls: List[str], relevant_clauses: List[str],
                                        confidence_score: float) -> bool:
        """Index expert consultation response"""
        try:
            content = f"""
            Expert Consultation
            Question: {question}

            Answer: {answer}

            Relevant Controls: {', '.join(relevant_controls)}
            Relevant Clauses: {', '.join(relevant_clauses)}
            Expert Confidence: {confidence_score:.2f}
            """

            metadata = {
                "consultation_date": datetime.now().isoformat(),
                "question": question,
                "relevant_controls": relevant_controls,
                "relevant_clauses": relevant_clauses,
                "confidence_score": confidence_score,
                "topic_area": self._extract_topic_area(question),
                "validation_status": "validated"
            }

            consultation_doc = VectorDocument(
                id=f"consult_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hashlib.md5(question.encode()).hexdigest()[:8]}",
                content=content.strip(),
                metadata=metadata,
                namespace=VectorNamespace.CONSULTATIONS.value
            )

            return await self.vector_store.store_document(consultation_doc)

        except Exception as e:
            logger.error(f"Failed to index consultation response: {e}")
            return False

    def _extract_topic_area(self, question: str) -> str:
        """Extract topic area from consultation question"""
        question_lower = question.lower()

        topic_keywords = {
            "access_control": ["access", "authentication", "authorization", "privilege"],
            "incident_management": ["incident", "breach", "response", "recovery"],
            "risk_management": ["risk", "threat", "vulnerability", "assessment"],
            "policy_governance": ["policy", "governance", "management", "oversight"],
            "technical_controls": ["encryption", "firewall", "security", "technical"],
            "physical_security": ["physical", "facility", "premises", "building"],
            "human_resources": ["training", "awareness", "personnel", "staff"],
            "compliance_audit": ["audit", "compliance", "certification", "assessment"]
        }

        for topic, keywords in topic_keywords.items():
            if any(keyword in question_lower for keyword in keywords):
                return topic

        return "general"

    async def bulk_index_knowledge_base(self) -> IndexingJob:
        """Bulk index the entire ISO knowledge base"""
        job_id = f"bulk_kb_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        job = IndexingJob(
            job_id=job_id,
            job_type="knowledge_base_bulk",
            status="running",
            total_documents=0,
            processed_documents=0,
            failed_documents=0,
            started_at=datetime.now()
        )

        self.active_jobs[job_id] = job

        try:
            documents_to_store = []

            # Index all controls
            controls = get_all_controls()
            for control in controls.values():
                control_doc = self._create_knowledge_control_document(control)
                documents_to_store.append(control_doc)

            # Index all clauses
            clauses = get_all_clauses()
            for clause in clauses.values():
                clause_doc = self._create_knowledge_clause_document(clause)
                documents_to_store.append(clause_doc)

            # Index all requirements
            requirements = get_all_requirements()
            logger.info(f"Processing {len(requirements)} requirements for indexing")
            for clause_id, requirement in requirements:
                try:
                    req_doc = self._create_knowledge_requirement_document(clause_id, requirement)
                    documents_to_store.append(req_doc)
                except Exception as e:
                    logger.error(f"Failed to process requirement {clause_id}: {e}")
                    # Continue processing other requirements

            job.total_documents = len(documents_to_store)

            # Batch process in chunks to avoid overwhelming the vector store
            chunk_size = 50
            for i in range(0, len(documents_to_store), chunk_size):
                chunk = documents_to_store[i:i + chunk_size]
                results = await self.vector_store.store_documents_batch(chunk)

                for doc_id, success in results.items():
                    if success:
                        job.processed_documents += 1
                    else:
                        job.failed_documents += 1

            job.status = "completed"
            job.completed_at = datetime.now()

            logger.info(f"Bulk knowledge base indexing completed: {job.processed_documents}/{job.total_documents} successful")

        except Exception as e:
            job.status = "failed"
            job.error_message = str(e)
            job.completed_at = datetime.now()
            logger.error(f"Bulk knowledge base indexing failed: {e}")

        self.job_history.append(job)
        if job_id in self.active_jobs:
            del self.active_jobs[job_id]

        return job

    def _create_knowledge_control_document(self, control: Control) -> VectorDocument:
        """Create vector document for ISO control knowledge"""
        content = f"""
        ISO 27001:2022 Control: {control.id} - {control.title}
        Category: {control.category}

        Purpose: {control.purpose}

        Description: {control.description}

        Implementation Guidance:
        {chr(10).join(f"- {guidance}" for guidance in control.implementation_guidance)}

        Related Controls: {', '.join(control.related_controls)}
        """

        metadata = {
            "control_id": control.id,
            "title": control.title,
            "category": control.category,
            "purpose": control.purpose,
            "related_controls": control.related_controls,
            "guidance_count": len(control.implementation_guidance),
            "knowledge_type": "control",
            "iso_standard": "27001:2022"
        }

        return VectorDocument(
            id=f"kb_control_{control.id}",
            content=content.strip(),
            metadata=metadata,
            namespace=VectorNamespace.KNOWLEDGE.value
        )

    def _create_knowledge_clause_document(self, clause: Clause) -> VectorDocument:
        """Create vector document for ISO clause knowledge"""
        content = f"""
        ISO 27001:2022 Management Clause: {clause.id} - {clause.title}
        Clause Type: {clause.clause_type.value}

        Purpose: {clause.purpose}

        Requirements Summary:
        This clause contains {len(clause.requirements)} specific requirements for information security management system implementation.
        """

        metadata = {
            "clause_id": clause.id,
            "title": clause.title,
            "clause_type": clause.clause_type.value,
            "purpose": clause.purpose,
            "requirements_count": len(clause.requirements),
            "knowledge_type": "clause",
            "iso_standard": "27001:2022"
        }

        return VectorDocument(
            id=f"kb_clause_{clause.id}",
            content=content.strip(),
            metadata=metadata,
            namespace=VectorNamespace.KNOWLEDGE.value
        )

    def _create_knowledge_requirement_document(self, clause_id: str, requirement) -> VectorDocument:
        """Create vector document for ISO requirement knowledge"""

        # Safe access to verification_methods with fallback
        verification_methods = getattr(requirement, 'verification_methods', [])

        # If verification_methods doesn't exist, use documentation_required as fallback
        if not verification_methods:
            verification_methods = getattr(requirement, 'documentation_required', [])

        content = f"""
        ISO 27001:2022 Requirement: {clause_id}
        Requirement: {requirement.description}

        Implementation Guidance:
        {chr(10).join(f"- {guidance}" for guidance in requirement.implementation_guidance)}

        Verification Methods:
        {chr(10).join(f"- {method}" for method in verification_methods) if verification_methods else "- Standard audit and assessment methods"}
        """

        metadata = {
            "clause_id": clause_id,
            "requirement_description": requirement.description,
            "guidance_count": len(requirement.implementation_guidance),
            "verification_methods_count": len(verification_methods),
            "knowledge_type": "requirement",
            "iso_standard": "27001:2022"
        }

        return VectorDocument(
            id=f"kb_req_{clause_id}_{hashlib.md5(requirement.description.encode()).hexdigest()[:8]}",
            content=content.strip(),
            metadata=metadata,
            namespace=VectorNamespace.KNOWLEDGE.value
        )

    async def get_index_statistics(self) -> IndexStats:
        """Get comprehensive index statistics"""
        try:
            stats = self.vector_store.get_index_stats()

            return IndexStats(
                total_documents=stats.get("total_vector_count", 0),
                documents_by_namespace=stats.get("namespaces", {}),
                last_updated=datetime.now(),
                health_status="healthy" if stats.get("total_vector_count", 0) > 0 else "empty",
                storage_usage_mb=stats.get("index_fullness", 0) * 100  # Rough estimate
            )

        except Exception as e:
            logger.error(f"Failed to get index statistics: {e}")
            return IndexStats(
                total_documents=0,
                documents_by_namespace={},
                last_updated=datetime.now(),
                health_status="error",
                storage_usage_mb=0
            )

    async def cleanup_old_documents(self, days_to_keep: int = 90,
                                  namespace: str = None) -> int:
        """Clean up old documents from vector store"""
        # Note: This is a simplified implementation
        # In practice, you'd need to iterate through documents and check their timestamps
        logger.info(f"Cleanup old documents older than {days_to_keep} days")

        # For now, return 0 - this would need to be implemented based on
        # your specific cleanup requirements
        return 0

    def get_job_status(self, job_id: str) -> Optional[IndexingJob]:
        """Get status of indexing job"""
        if job_id in self.active_jobs:
            return self.active_jobs[job_id]

        for job in self.job_history:
            if job.job_id == job_id:
                return job

        return None

    def get_active_jobs(self) -> List[IndexingJob]:
        """Get all active indexing jobs"""
        return list(self.active_jobs.values())