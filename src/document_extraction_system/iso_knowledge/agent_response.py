"""
Agent Card Response Format for ISO 27001:2022 Analysis
Structured output format for ISO compliance analysis results
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime


class ComplianceStatus(Enum):
    COMPLIANT = "compliant"
    PARTIALLY_COMPLIANT = "partially_compliant"
    NON_COMPLIANT = "non_compliant"
    NOT_APPLICABLE = "not_applicable"
    INSUFFICIENT_INFORMATION = "insufficient_information"


class Priority(Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class DocumentType(Enum):
    POLICY = "policy"
    PROCEDURE = "procedure"
    STANDARD = "standard"
    GUIDELINE = "guideline"
    RISK_ASSESSMENT = "risk_assessment"
    AUDIT_REPORT = "audit_report"
    INCIDENT_REPORT = "incident_report"
    TRAINING_MATERIAL = "training_material"
    CONTRACT = "contract"
    TECHNICAL_SPECIFICATION = "technical_specification"
    OTHER = "other"


@dataclass
class ControlAssessment:
    """Assessment of a single ISO control"""
    control_id: str
    control_title: str
    status: ComplianceStatus
    confidence_score: float  # 0.0 to 1.0
    evidence_found: List[str]
    gaps_identified: List[str]
    recommendations: List[str]


@dataclass
class ClauseAssessment:
    """Assessment of a management clause requirement"""
    clause_id: str
    requirement_title: str
    status: ComplianceStatus
    confidence_score: float
    evidence_found: List[str]
    missing_elements: List[str]
    recommendations: List[str]


@dataclass
class CategoryScore:
    """Scoring for a category of controls"""
    category: str
    total_controls: int
    assessed_controls: int
    compliant_count: int
    partially_compliant_count: int
    non_compliant_count: int
    not_applicable_count: int
    overall_score: float  # 0.0 to 100.0
    maturity_level: str  # basic, managed, defined, quantified, optimizing


@dataclass
class Recommendation:
    """Action recommendation with priority and details"""
    id: str
    title: str
    description: str
    priority: Priority
    category: str  # organizational, people, physical, technological
    effort_estimate: str  # low, medium, high
    timeline_estimate: str  # immediate, short-term, medium-term, long-term
    related_controls: List[str]
    implementation_steps: List[str]


@dataclass
class GapAnalysis:
    """Gap analysis results"""
    total_gaps: int
    critical_gaps: int
    high_priority_gaps: int
    gap_categories: Dict[str, int]  # category -> gap count
    top_gap_areas: List[str]


@dataclass
class DocumentClassification:
    """Classification of the analyzed document"""
    document_type: DocumentType
    confidence_score: float
    iso_relevance: str  # high, medium, low
    applicable_categories: List[str]
    primary_focus_areas: List[str]


@dataclass
class ComplianceOverview:
    """High-level compliance overview"""
    overall_maturity: str
    overall_score: float
    total_controls_assessed: int
    compliant_percentage: float
    partially_compliant_percentage: float
    non_compliant_percentage: float
    key_strengths: List[str]
    major_concerns: List[str]


@dataclass
class ISOAgentCard:
    """Main agent card response structure"""
    # Metadata
    analysis_id: str
    timestamp: datetime
    document_info: Dict[str, Any]

    # Document Classification
    document_classification: DocumentClassification

    # Compliance Assessment
    compliance_overview: ComplianceOverview
    category_scores: List[CategoryScore]

    # Detailed Assessments
    control_assessments: List[ControlAssessment]
    clause_assessments: List[ClauseAssessment]

    # Analysis Results
    gap_analysis: GapAnalysis
    recommendations: List[Recommendation]

    # Additional Insights
    risk_areas: List[str]
    quick_wins: List[str]  # Easy improvements with high impact
    implementation_roadmap: List[Dict[str, Any]]

    # Quality Metrics
    analysis_confidence: float  # Overall confidence in analysis
    coverage_percentage: float  # Percentage of ISO framework covered

    # Summary
    executive_summary: str
    next_steps: List[str]

    def to_dict(self) -> Dict[str, Any]:
        """Convert agent card to dictionary for API response"""
        return {
            "analysis_id": self.analysis_id,
            "timestamp": self.timestamp.isoformat(),
            "document_info": self.document_info,

            "document_classification": {
                "type": self.document_classification.document_type.value,
                "confidence": self.document_classification.confidence_score,
                "iso_relevance": self.document_classification.iso_relevance,
                "applicable_categories": self.document_classification.applicable_categories,
                "primary_focus_areas": self.document_classification.primary_focus_areas
            },

            "compliance_overview": {
                "overall_maturity": self.compliance_overview.overall_maturity,
                "overall_score": self.compliance_overview.overall_score,
                "total_controls_assessed": self.compliance_overview.total_controls_assessed,
                "compliant_percentage": self.compliance_overview.compliant_percentage,
                "partially_compliant_percentage": self.compliance_overview.partially_compliant_percentage,
                "non_compliant_percentage": self.compliance_overview.non_compliant_percentage,
                "key_strengths": self.compliance_overview.key_strengths,
                "major_concerns": self.compliance_overview.major_concerns
            },

            "category_scores": [
                {
                    "category": score.category,
                    "total_controls": score.total_controls,
                    "assessed_controls": score.assessed_controls,
                    "compliant_count": score.compliant_count,
                    "partially_compliant_count": score.partially_compliant_count,
                    "non_compliant_count": score.non_compliant_count,
                    "not_applicable_count": score.not_applicable_count,
                    "overall_score": score.overall_score,
                    "maturity_level": score.maturity_level
                }
                for score in self.category_scores
            ],

            "control_assessments": [
                {
                    "control_id": assessment.control_id,
                    "control_title": assessment.control_title,
                    "status": assessment.status.value,
                    "confidence_score": assessment.confidence_score,
                    "evidence_found": assessment.evidence_found,
                    "gaps_identified": assessment.gaps_identified,
                    "recommendations": assessment.recommendations
                }
                for assessment in self.control_assessments
            ],

            "clause_assessments": [
                {
                    "clause_id": assessment.clause_id,
                    "requirement_title": assessment.requirement_title,
                    "status": assessment.status.value,
                    "confidence_score": assessment.confidence_score,
                    "evidence_found": assessment.evidence_found,
                    "missing_elements": assessment.missing_elements,
                    "recommendations": assessment.recommendations
                }
                for assessment in self.clause_assessments
            ],

            "gap_analysis": {
                "total_gaps": self.gap_analysis.total_gaps,
                "critical_gaps": self.gap_analysis.critical_gaps,
                "high_priority_gaps": self.gap_analysis.high_priority_gaps,
                "gap_categories": self.gap_analysis.gap_categories,
                "top_gap_areas": self.gap_analysis.top_gap_areas
            },

            "recommendations": [
                {
                    "id": rec.id,
                    "title": rec.title,
                    "description": rec.description,
                    "priority": rec.priority.value,
                    "category": getattr(rec, 'category', getattr(rec, 'recommendation_type', 'unknown').value if hasattr(rec, 'recommendation_type') else 'unknown'),
                    "effort_estimate": getattr(rec, 'effort_estimate', f"{getattr(rec, 'estimated_effort_hours', 0)} hours"),
                    "timeline_estimate": getattr(rec, 'timeline_estimate', getattr(rec, 'estimated_timeline', 'unknown')),
                    "related_controls": getattr(rec, 'related_controls', getattr(rec, 'compliance_impact', [])),
                    "implementation_steps": getattr(rec, 'implementation_steps', getattr(rec, 'specific_actions', []))
                }
                for rec in self.recommendations
            ],

            "risk_areas": self.risk_areas,
            "quick_wins": self.quick_wins,
            "implementation_roadmap": self.implementation_roadmap,

            "analysis_confidence": self.analysis_confidence,
            "coverage_percentage": self.coverage_percentage,

            "executive_summary": self.executive_summary,
            "next_steps": self.next_steps
        }


@dataclass
class ISOConsultationCard:
    """Response format for ISO consultation queries"""
    query_id: str
    timestamp: datetime
    question: str
    answer: str
    relevant_controls: List[str]
    relevant_clauses: List[str]
    implementation_guidance: List[str]
    related_topics: List[str]
    confidence_score: float
    sources: List[str]

    def to_dict(self) -> Dict[str, Any]:
        """Convert consultation card to dictionary"""
        return {
            "query_id": self.query_id,
            "timestamp": self.timestamp.isoformat(),
            "question": self.question,
            "answer": self.answer,
            "relevant_controls": self.relevant_controls,
            "relevant_clauses": self.relevant_clauses,
            "implementation_guidance": self.implementation_guidance,
            "related_topics": self.related_topics,
            "confidence_score": self.confidence_score,
            "sources": self.sources
        }


def create_sample_agent_card() -> ISOAgentCard:
    """Create a sample agent card for testing"""
    return ISOAgentCard(
        analysis_id="sample_001",
        timestamp=datetime.now(),
        document_info={
            "filename": "sample_policy.pdf",
            "size": 1024000,
            "pages": 10,
            "words": 5000
        },
        document_classification=DocumentClassification(
            document_type=DocumentType.POLICY,
            confidence_score=0.9,
            iso_relevance="high",
            applicable_categories=["organizational", "people"],
            primary_focus_areas=["access control", "information security policy"]
        ),
        compliance_overview=ComplianceOverview(
            overall_maturity="managed",
            overall_score=75.5,
            total_controls_assessed=25,
            compliant_percentage=60.0,
            partially_compliant_percentage=24.0,
            non_compliant_percentage=16.0,
            key_strengths=["Strong policy framework", "Clear roles and responsibilities"],
            major_concerns=["Lack of incident response procedures", "Insufficient access control measures"]
        ),
        category_scores=[],
        control_assessments=[],
        clause_assessments=[],
        gap_analysis=GapAnalysis(
            total_gaps=10,
            critical_gaps=2,
            high_priority_gaps=4,
            gap_categories={"organizational": 6, "technological": 4},
            top_gap_areas=["incident management", "access control", "monitoring"]
        ),
        recommendations=[],
        risk_areas=["Data breaches", "Unauthorized access", "System outages"],
        quick_wins=["Update password policy", "Implement basic monitoring"],
        implementation_roadmap=[],
        analysis_confidence=0.85,
        coverage_percentage=68.5,
        executive_summary="The analyzed policy document shows good foundational coverage of ISO 27001 requirements with several areas for improvement.",
        next_steps=["Address critical gaps", "Implement quick wins", "Develop incident response procedures"]
    )