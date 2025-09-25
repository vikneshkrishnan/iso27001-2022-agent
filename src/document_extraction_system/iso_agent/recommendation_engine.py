"""
Comprehensive Recommendation Engine for ISO 27001:2022
Generates prioritized, actionable recommendations based on gap analysis
"""

import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
try:
    import numpy as np
except ImportError:
    np = None

from ..iso_knowledge.annex_a_controls import Control, get_control_by_id
from ..iso_knowledge.agent_response import (
    Recommendation, Priority, ComplianceStatus, ControlAssessment, ClauseAssessment
)

logger = logging.getLogger(__name__)


class RecommendationType(Enum):
    POLICY_DEVELOPMENT = "policy_development"
    PROCESS_IMPROVEMENT = "process_improvement"
    TECHNICAL_IMPLEMENTATION = "technical_implementation"
    TRAINING_AWARENESS = "training_awareness"
    GOVERNANCE_STRUCTURE = "governance_structure"
    MONITORING_COMPLIANCE = "monitoring_compliance"
    DOCUMENTATION = "documentation"
    RESOURCE_ALLOCATION = "resource_allocation"


class ImplementationComplexity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


class BusinessImpact(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class RecommendationContext:
    """Context information for generating recommendations"""
    organization_maturity: str
    document_type: str
    primary_focus_areas: List[str]
    available_resources: str
    compliance_urgency: str
    industry_sector: str = "general"


@dataclass
class DetailedRecommendation:
    """Comprehensive recommendation with detailed implementation guidance"""
    id: str
    title: str
    description: str
    recommendation_type: RecommendationType
    priority: Priority
    business_impact: BusinessImpact
    implementation_complexity: ImplementationComplexity

    # Implementation details
    specific_actions: List[str]
    success_criteria: List[str]
    resources_required: Dict[str, Any]
    estimated_timeline: str
    estimated_effort_hours: int

    # Risk and impact
    risk_reduction: float  # 0.0 to 1.0
    compliance_impact: List[str]  # Which controls/clauses this addresses
    business_benefits: List[str]

    # Implementation guidance
    prerequisites: List[str]
    potential_challenges: List[str]
    mitigation_strategies: List[str]

    # Metrics and validation
    kpis: List[str]
    validation_methods: List[str]

    # Relationships
    related_recommendations: List[str]
    dependency_order: int  # Order in implementation sequence


class SmartRecommendationEngine:
    """Intelligent recommendation engine with priority scoring and implementation planning"""

    def __init__(self):
        self.recommendation_templates = self._build_recommendation_templates()
        self.priority_scoring_rules = self._build_priority_scoring_rules()
        self.implementation_patterns = self._build_implementation_patterns()
        self.control_mappings = self._build_control_recommendation_mappings()

    def _build_recommendation_templates(self) -> Dict[str, Dict[str, Any]]:
        """Build templates for different types of recommendations"""
        return {
            "policy_missing": {
                "type": RecommendationType.POLICY_DEVELOPMENT,
                "title_template": "Develop {control_area} Policy",
                "description_template": "Create a comprehensive {control_area} policy that addresses {specific_requirements}",
                "base_complexity": ImplementationComplexity.MEDIUM,
                "base_impact": BusinessImpact.HIGH,
                "typical_timeline": "4-8 weeks",
                "typical_effort": 40
            },

            "process_improvement": {
                "type": RecommendationType.PROCESS_IMPROVEMENT,
                "title_template": "Enhance {control_area} Process",
                "description_template": "Improve existing {control_area} processes to address identified gaps in {specific_areas}",
                "base_complexity": ImplementationComplexity.MEDIUM,
                "base_impact": BusinessImpact.MEDIUM,
                "typical_timeline": "6-12 weeks",
                "typical_effort": 60
            },

            "technical_control": {
                "type": RecommendationType.TECHNICAL_IMPLEMENTATION,
                "title_template": "Implement {technical_solution}",
                "description_template": "Deploy technical controls for {control_area} including {specific_technologies}",
                "base_complexity": ImplementationComplexity.HIGH,
                "base_impact": BusinessImpact.HIGH,
                "typical_timeline": "8-16 weeks",
                "typical_effort": 120
            },

            "monitoring_system": {
                "type": RecommendationType.MONITORING_COMPLIANCE,
                "title_template": "Establish {monitoring_area} Monitoring",
                "description_template": "Implement monitoring and measurement systems for {specific_controls}",
                "base_complexity": ImplementationComplexity.MEDIUM,
                "base_impact": BusinessImpact.MEDIUM,
                "typical_timeline": "4-10 weeks",
                "typical_effort": 50
            },

            "training_program": {
                "type": RecommendationType.TRAINING_AWARENESS,
                "title_template": "Develop {training_area} Training",
                "description_template": "Create training and awareness programs for {target_audience} on {training_topics}",
                "base_complexity": ImplementationComplexity.LOW,
                "base_impact": BusinessImpact.MEDIUM,
                "typical_timeline": "3-6 weeks",
                "typical_effort": 30
            },

            "documentation": {
                "type": RecommendationType.DOCUMENTATION,
                "title_template": "Document {process_area} Procedures",
                "description_template": "Create comprehensive documentation for {specific_processes} including {documentation_types}",
                "base_complexity": ImplementationComplexity.LOW,
                "base_impact": BusinessImpact.MEDIUM,
                "typical_timeline": "2-4 weeks",
                "typical_effort": 25
            }
        }

    def _build_priority_scoring_rules(self) -> Dict[str, Dict[str, float]]:
        """Build rules for priority scoring"""
        return {
            "compliance_status_weights": {
                ComplianceStatus.NON_COMPLIANT.value: 1.0,
                ComplianceStatus.PARTIALLY_COMPLIANT.value: 0.6,
                ComplianceStatus.COMPLIANT.value: 0.2,
                ComplianceStatus.INSUFFICIENT_INFORMATION.value: 0.4
            },

            "control_category_weights": {
                "organizational": 0.9,  # High impact on overall governance
                "people": 0.7,          # Important for culture and awareness
                "physical": 0.6,        # Important but often more contained
                "technological": 0.8    # High impact on security posture
            },

            "business_impact_multipliers": {
                "confidentiality": 1.0,
                "integrity": 0.9,
                "availability": 0.8,
                "compliance": 0.9,
                "reputation": 0.8
            },

            "urgency_factors": {
                "regulatory_deadline": 1.5,
                "audit_preparation": 1.3,
                "incident_response": 1.8,
                "certification_target": 1.4,
                "business_continuity": 1.6
            }
        }

    def _build_implementation_patterns(self) -> Dict[str, List[str]]:
        """Build patterns for implementation planning"""
        return {
            "foundation_first": [
                "5.1",  # Information security policies
                "5.2",  # Roles and responsibilities
                "5.3",  # Segregation of duties
                "6.1",  # Screening
                "6.2"   # Terms and conditions
            ],

            "governance_layer": [
                "5.1", "5.2", "5.4",  # Governance controls
                "9.1", "9.2", "9.3"   # Performance evaluation
            ],

            "technical_controls": [
                "8.1", "8.2", "8.3",  # Access controls
                "8.8", "8.9", "8.10", # Technical controls
                "8.16", "8.24"        # Monitoring controls
            ],

            "operational_controls": [
                "7.1", "7.2", "7.3",  # Physical controls
                "8.1", "8.2",         # Operational security
                "5.25", "5.26"        # Incident management
            ]
        }

    def _build_control_recommendation_mappings(self) -> Dict[str, Dict[str, Any]]:
        """Build mappings from controls to recommendation types"""
        return {
            "5.1": {
                "missing_recommendations": [
                    {
                        "template": "policy_missing",
                        "variables": {
                            "control_area": "Information Security",
                            "specific_requirements": "management approval, regular review, communication"
                        }
                    }
                ],
                "improvement_recommendations": [
                    {
                        "template": "process_improvement",
                        "variables": {
                            "control_area": "Policy Management",
                            "specific_areas": "review processes, communication mechanisms"
                        }
                    }
                ]
            },

            "8.2": {
                "missing_recommendations": [
                    {
                        "template": "technical_control",
                        "variables": {
                            "technical_solution": "Privileged Access Management (PAM)",
                            "control_area": "privileged access",
                            "specific_technologies": "PAM tools, session monitoring, access analytics"
                        }
                    }
                ],
                "improvement_recommendations": [
                    {
                        "template": "monitoring_system",
                        "variables": {
                            "monitoring_area": "Privileged Access",
                            "specific_controls": "administrative account usage, elevated privileges"
                        }
                    }
                ]
            },

            "6.1": {
                "missing_recommendations": [
                    {
                        "template": "process_improvement",
                        "variables": {
                            "control_area": "Personnel Screening",
                            "specific_areas": "background verification, reference checking"
                        }
                    }
                ],
                "improvement_recommendations": [
                    {
                        "template": "documentation",
                        "variables": {
                            "process_area": "Screening",
                            "specific_processes": "background check procedures",
                            "documentation_types": "forms, checklists, approval workflows"
                        }
                    }
                ]
            }
        }

    async def generate_comprehensive_recommendations(
        self,
        control_assessments: List[ControlAssessment],
        clause_assessments: List[ClauseAssessment],
        context: RecommendationContext,
        max_recommendations: int = 20
    ) -> List[DetailedRecommendation]:
        """Generate comprehensive prioritized recommendations"""

        all_recommendations = []

        # Generate control-based recommendations
        control_recommendations = await self._generate_control_recommendations(
            control_assessments, context
        )
        all_recommendations.extend(control_recommendations)

        # Generate clause-based recommendations
        clause_recommendations = await self._generate_clause_recommendations(
            clause_assessments, context
        )
        all_recommendations.extend(clause_recommendations)

        # Generate strategic recommendations
        strategic_recommendations = await self._generate_strategic_recommendations(
            control_assessments, clause_assessments, context
        )
        all_recommendations.extend(strategic_recommendations)

        # Score and prioritize all recommendations
        scored_recommendations = self._score_and_prioritize_recommendations(
            all_recommendations, context
        )

        # Apply implementation sequencing
        sequenced_recommendations = self._sequence_recommendations(scored_recommendations)

        # Return top recommendations
        return sequenced_recommendations[:max_recommendations]

    async def _generate_control_recommendations(
        self,
        assessments: List[ControlAssessment],
        context: RecommendationContext
    ) -> List[DetailedRecommendation]:
        """Generate recommendations based on control assessments"""

        recommendations = []

        for assessment in assessments:
            # Get control information
            control = get_control_by_id(assessment.control_id)
            if not control:
                continue

            # Determine recommendation type based on compliance status
            if assessment.status == ComplianceStatus.NON_COMPLIANT:
                recs = await self._create_missing_control_recommendations(
                    control, assessment, context
                )
                recommendations.extend(recs)

            elif assessment.status == ComplianceStatus.PARTIALLY_COMPLIANT:
                recs = await self._create_improvement_recommendations(
                    control, assessment, context
                )
                recommendations.extend(recs)

        return recommendations

    async def _create_missing_control_recommendations(
        self,
        control: Control,
        assessment: ControlAssessment,
        context: RecommendationContext
    ) -> List[DetailedRecommendation]:
        """Create recommendations for missing controls"""

        recommendations = []
        control_mapping = self.control_mappings.get(control.id, {})

        # Get recommendation templates for this control
        missing_recs = control_mapping.get("missing_recommendations", [])

        for rec_config in missing_recs:
            template_name = rec_config.get("template")
            template = self.recommendation_templates.get(template_name)

            if not template:
                continue

            variables = rec_config.get("variables", {})

            # Create the recommendation
            rec_id = f"REC-{control.id}-MISSING-001"
            title = template["title_template"].format(**variables)
            description = template["description_template"].format(**variables)

            recommendation = DetailedRecommendation(
                id=rec_id,
                title=title,
                description=description,
                recommendation_type=template["type"],
                priority=self._calculate_priority(control, assessment, context),
                business_impact=self._calculate_business_impact(control, context),
                implementation_complexity=template["base_complexity"],

                specific_actions=self._generate_specific_actions(control, template, variables),
                success_criteria=self._generate_success_criteria(control, template),
                resources_required=self._estimate_resources(template, context),
                estimated_timeline=template["typical_timeline"],
                estimated_effort_hours=template["typical_effort"],

                risk_reduction=self._calculate_risk_reduction(control, ComplianceStatus.NON_COMPLIANT),
                compliance_impact=[control.id],
                business_benefits=self._identify_business_benefits(control),

                prerequisites=self._identify_prerequisites(control, template),
                potential_challenges=self._identify_challenges(control, template, context),
                mitigation_strategies=self._generate_mitigation_strategies(control, template),

                kpis=self._generate_kpis(control, template),
                validation_methods=self._generate_validation_methods(control, template),

                related_recommendations=[],
                dependency_order=0
            )

            recommendations.append(recommendation)

        return recommendations

    async def _create_improvement_recommendations(
        self,
        control: Control,
        assessment: ControlAssessment,
        context: RecommendationContext
    ) -> List[DetailedRecommendation]:
        """Create recommendations for improving partially compliant controls"""

        recommendations = []
        control_mapping = self.control_mappings.get(control.id, {})

        # Focus on specific gaps identified
        for gap in assessment.gaps_identified:
            rec_id = f"REC-{control.id}-IMPROVE-{len(recommendations)+1:03d}"

            recommendation = DetailedRecommendation(
                id=rec_id,
                title=f"Address {gap} for {control.title}",
                description=f"Improve {control.title.lower()} implementation by addressing: {gap}",
                recommendation_type=RecommendationType.PROCESS_IMPROVEMENT,
                priority=Priority.MEDIUM,
                business_impact=BusinessImpact.MEDIUM,
                implementation_complexity=ImplementationComplexity.LOW,

                specific_actions=[
                    f"Review current {control.title.lower()} implementation",
                    f"Address the gap: {gap}",
                    "Update documentation and procedures",
                    "Validate improvements"
                ],
                success_criteria=[
                    f"{gap} is fully addressed",
                    f"{control.title} shows compliant status",
                    "Evidence of improvement documented"
                ],
                resources_required={"effort": "low", "skills": ["process improvement"], "budget": "low"},
                estimated_timeline="2-4 weeks",
                estimated_effort_hours=20,

                risk_reduction=0.3,
                compliance_impact=[control.id],
                business_benefits=[f"Improved {control.title.lower()}", "Better compliance posture"],

                prerequisites=["Current implementation review"],
                potential_challenges=["Resource availability", "Process complexity"],
                mitigation_strategies=["Prioritize critical gaps", "Phased implementation"],

                kpis=[f"{control.title} compliance percentage", "Gap resolution rate"],
                validation_methods=["Process review", "Documentation audit", "Testing"],

                related_recommendations=[],
                dependency_order=1
            )

            recommendations.append(recommendation)

        return recommendations

    async def _generate_clause_recommendations(
        self,
        assessments: List[ClauseAssessment],
        context: RecommendationContext
    ) -> List[DetailedRecommendation]:
        """Generate recommendations based on clause assessments"""

        recommendations = []

        # Focus on non-compliant clauses
        non_compliant_clauses = [
            a for a in assessments
            if a.status == ComplianceStatus.NON_COMPLIANT
        ]

        for assessment in non_compliant_clauses:
            rec_id = f"REC-CLAUSE-{assessment.clause_id}-001"

            recommendation = DetailedRecommendation(
                id=rec_id,
                title=f"Implement {assessment.requirement_title}",
                description=f"Address management clause requirement: {assessment.requirement_title}",
                recommendation_type=RecommendationType.GOVERNANCE_STRUCTURE,
                priority=Priority.HIGH,
                business_impact=BusinessImpact.HIGH,
                implementation_complexity=ImplementationComplexity.MEDIUM,

                specific_actions=[
                    f"Develop {assessment.requirement_title.lower()} framework",
                    "Document processes and procedures",
                    "Assign responsibilities and authorities",
                    "Implement monitoring and review"
                ],
                success_criteria=[
                    f"{assessment.requirement_title} fully implemented",
                    "Management approval obtained",
                    "Evidence documented and maintained"
                ],
                resources_required={"effort": "medium", "skills": ["management", "governance"], "budget": "medium"},
                estimated_timeline="4-8 weeks",
                estimated_effort_hours=50,

                risk_reduction=0.6,
                compliance_impact=[assessment.clause_id],
                business_benefits=["Improved governance", "Better compliance foundation"],

                prerequisites=["Management commitment", "Resource allocation"],
                potential_challenges=["Organizational change", "Process integration"],
                mitigation_strategies=["Phased rollout", "Change management", "Training"],

                kpis=["Implementation completion percentage", "Management review frequency"],
                validation_methods=["Management review", "Internal audit", "Process assessment"],

                related_recommendations=[],
                dependency_order=0
            )

            recommendations.append(recommendation)

        return recommendations

    async def _generate_strategic_recommendations(
        self,
        control_assessments: List[ControlAssessment],
        clause_assessments: List[ClauseAssessment],
        context: RecommendationContext
    ) -> List[DetailedRecommendation]:
        """Generate strategic recommendations based on overall analysis"""

        recommendations = []

        # Calculate overall compliance metrics
        total_controls = len(control_assessments)
        non_compliant_controls = len([a for a in control_assessments if a.status == ComplianceStatus.NON_COMPLIANT])

        if non_compliant_controls > total_controls * 0.3:  # More than 30% non-compliant
            # Recommend comprehensive ISMS implementation
            strategic_rec = DetailedRecommendation(
                id="REC-STRATEGIC-001",
                title="Comprehensive ISMS Implementation Program",
                description="Establish a comprehensive Information Security Management System implementation program to address multiple compliance gaps",
                recommendation_type=RecommendationType.GOVERNANCE_STRUCTURE,
                priority=Priority.CRITICAL,
                business_impact=BusinessImpact.CRITICAL,
                implementation_complexity=ImplementationComplexity.VERY_HIGH,

                specific_actions=[
                    "Establish ISMS governance structure",
                    "Develop comprehensive implementation roadmap",
                    "Allocate dedicated resources and budget",
                    "Implement project management framework",
                    "Execute phased implementation plan"
                ],
                success_criteria=[
                    "ISMS governance established",
                    "Implementation roadmap approved",
                    "Dedicated team in place",
                    "80%+ compliance achieved within timeline"
                ],
                resources_required={
                    "effort": "very_high",
                    "skills": ["ISMS expertise", "project management", "change management"],
                    "budget": "high"
                },
                estimated_timeline="12-18 months",
                estimated_effort_hours=1000,

                risk_reduction=0.8,
                compliance_impact=["multiple_controls"],
                business_benefits=[
                    "Comprehensive risk reduction",
                    "Regulatory compliance",
                    "Improved security posture",
                    "Business continuity enhancement"
                ],

                prerequisites=["Executive commitment", "Budget approval", "Resource allocation"],
                potential_challenges=[
                    "Organizational resistance",
                    "Resource constraints",
                    "Complexity management"
                ],
                mitigation_strategies=[
                    "Executive sponsorship",
                    "Change management program",
                    "Phased implementation",
                    "External expertise engagement"
                ],

                kpis=[
                    "Overall compliance percentage",
                    "Risk reduction metrics",
                    "Implementation milestone achievement"
                ],
                validation_methods=[
                    "Internal audit",
                    "External assessment",
                    "Management review",
                    "Compliance testing"
                ],

                related_recommendations=[],
                dependency_order=0
            )

            recommendations.append(strategic_rec)

        return recommendations

    def _calculate_priority(
        self,
        control: Control,
        assessment: ControlAssessment,
        context: RecommendationContext
    ) -> Priority:
        """Calculate priority based on multiple factors"""

        # Base priority from compliance status
        status_weights = self.priority_scoring_rules["compliance_status_weights"]
        base_score = status_weights.get(assessment.status.value, 0.5)

        # Control category weight
        category_weights = self.priority_scoring_rules["control_category_weights"]
        category_score = category_weights.get(control.category, 0.7)

        # Context factors
        urgency_score = 1.0
        if context.compliance_urgency == "high":
            urgency_score = 1.3
        elif context.compliance_urgency == "critical":
            urgency_score = 1.5

        # Calculate final score
        final_score = base_score * category_score * urgency_score

        # Map to priority levels
        if final_score >= 1.2:
            return Priority.CRITICAL
        elif final_score >= 0.9:
            return Priority.HIGH
        elif final_score >= 0.6:
            return Priority.MEDIUM
        else:
            return Priority.LOW

    def _calculate_business_impact(self, control: Control, context: RecommendationContext) -> BusinessImpact:
        """Calculate business impact of implementing control"""

        # High impact controls (governance, access control, incident response)
        high_impact_controls = ["5.1", "5.2", "8.1", "8.2", "5.25", "5.26"]

        if control.id in high_impact_controls:
            return BusinessImpact.HIGH

        # Category-based impact
        if control.category == "organizational":
            return BusinessImpact.HIGH
        elif control.category == "technological":
            return BusinessImpact.MEDIUM
        else:
            return BusinessImpact.MEDIUM

    def _calculate_risk_reduction(self, control: Control, current_status: ComplianceStatus) -> float:
        """Calculate risk reduction from implementing control"""

        # Base risk reduction based on control importance
        base_reduction = {
            "organizational": 0.7,
            "people": 0.5,
            "physical": 0.4,
            "technological": 0.6
        }.get(control.category, 0.5)

        # Adjust based on current status
        status_multiplier = {
            ComplianceStatus.NON_COMPLIANT: 1.0,
            ComplianceStatus.PARTIALLY_COMPLIANT: 0.6,
            ComplianceStatus.COMPLIANT: 0.2
        }.get(current_status, 0.5)

        return base_reduction * status_multiplier

    def _generate_specific_actions(self, control: Control, template: Dict[str, Any], variables: Dict[str, str]) -> List[str]:
        """Generate specific actions for a recommendation"""

        base_actions = [
            f"Review current state of {control.title.lower()}",
            f"Develop implementation plan for {control.title}",
            "Assign ownership and responsibilities",
            "Implement necessary controls and processes",
            "Document implementation evidence",
            "Validate effectiveness through testing"
        ]

        # Add control-specific actions from implementation guidance
        control_actions = [f"Implement: {guidance}" for guidance in control.implementation_guidance[:3]]

        return base_actions + control_actions

    def _generate_success_criteria(self, control: Control, template: Dict[str, Any]) -> List[str]:
        """Generate success criteria for a recommendation"""

        return [
            f"{control.title} is fully implemented",
            f"Evidence of {control.title.lower()} documented",
            f"{control.title} passes compliance assessment",
            "Implementation approved by management",
            "Regular monitoring established"
        ]

    def _estimate_resources(self, template: Dict[str, Any], context: RecommendationContext) -> Dict[str, Any]:
        """Estimate resources required for implementation"""

        base_effort = template.get("typical_effort", 40)

        # Adjust based on organization maturity
        maturity_multiplier = {
            "basic": 1.5,
            "managed": 1.2,
            "defined": 1.0,
            "quantified": 0.8,
            "optimizing": 0.7
        }.get(context.organization_maturity, 1.0)

        adjusted_effort = int(base_effort * maturity_multiplier)

        return {
            "effort_hours": adjusted_effort,
            "skills_required": self._get_required_skills(template["type"]),
            "budget_estimate": self._estimate_budget(template["type"], adjusted_effort),
            "external_support": self._assess_external_support_need(template["type"], context)
        }

    def _get_required_skills(self, rec_type: RecommendationType) -> List[str]:
        """Get required skills for recommendation type"""

        skill_map = {
            RecommendationType.POLICY_DEVELOPMENT: ["policy writing", "legal compliance", "governance"],
            RecommendationType.TECHNICAL_IMPLEMENTATION: ["technical expertise", "system administration", "security tools"],
            RecommendationType.PROCESS_IMPROVEMENT: ["process analysis", "business analysis", "change management"],
            RecommendationType.TRAINING_AWARENESS: ["training design", "communication", "adult learning"],
            RecommendationType.GOVERNANCE_STRUCTURE: ["governance", "management", "organizational design"]
        }

        return skill_map.get(rec_type, ["general"])

    def _estimate_budget(self, rec_type: RecommendationType, effort_hours: int) -> str:
        """Estimate budget requirement"""

        # Simplified budget estimation
        hourly_rate = 100  # Average rate per hour
        total_cost = effort_hours * hourly_rate

        if total_cost < 5000:
            return "low"
        elif total_cost < 20000:
            return "medium"
        elif total_cost < 50000:
            return "high"
        else:
            return "very_high"

    def _assess_external_support_need(self, rec_type: RecommendationType, context: RecommendationContext) -> str:
        """Assess if external support is needed"""

        if context.organization_maturity in ["basic", "managed"] and rec_type in [
            RecommendationType.TECHNICAL_IMPLEMENTATION,
            RecommendationType.GOVERNANCE_STRUCTURE
        ]:
            return "recommended"

        return "optional"

    def _identify_business_benefits(self, control: Control) -> List[str]:
        """Identify business benefits of implementing control"""

        # Generic benefits
        benefits = [
            "Improved security posture",
            "Enhanced compliance status",
            "Reduced security risks"
        ]

        # Control-specific benefits
        if control.category == "organizational":
            benefits.append("Better governance and oversight")
        elif control.category == "technological":
            benefits.append("Enhanced technical security")
        elif control.category == "people":
            benefits.append("Improved security awareness")

        return benefits

    def _identify_prerequisites(self, control: Control, template: Dict[str, Any]) -> List[str]:
        """Identify prerequisites for implementation"""

        return [
            "Management approval",
            "Resource allocation",
            "Stakeholder buy-in",
            f"Understanding of {control.title.lower()} requirements"
        ]

    def _identify_challenges(self, control: Control, template: Dict[str, Any], context: RecommendationContext) -> List[str]:
        """Identify potential implementation challenges"""

        challenges = [
            "Resource constraints",
            "Technical complexity",
            "Organizational resistance"
        ]

        if context.organization_maturity == "basic":
            challenges.append("Limited security expertise")

        return challenges

    def _generate_mitigation_strategies(self, control: Control, template: Dict[str, Any]) -> List[str]:
        """Generate mitigation strategies for challenges"""

        return [
            "Phased implementation approach",
            "Executive sponsorship",
            "Training and awareness programs",
            "External expert consultation",
            "Regular progress monitoring"
        ]

    def _generate_kpis(self, control: Control, template: Dict[str, Any]) -> List[str]:
        """Generate KPIs for measuring success"""

        return [
            f"{control.title} compliance percentage",
            "Implementation milestone completion",
            "Risk reduction metrics",
            "Stakeholder satisfaction",
            "Time to full implementation"
        ]

    def _generate_validation_methods(self, control: Control, template: Dict[str, Any]) -> List[str]:
        """Generate methods to validate implementation"""

        return [
            "Internal audit",
            "Process testing",
            "Documentation review",
            "Stakeholder feedback",
            "Management review"
        ]

    def _score_and_prioritize_recommendations(
        self,
        recommendations: List[DetailedRecommendation],
        context: RecommendationContext
    ) -> List[DetailedRecommendation]:
        """Score and prioritize all recommendations"""

        # Calculate composite score for each recommendation
        for rec in recommendations:
            score = self._calculate_composite_score(rec, context)
            # Store score for sorting (could add to dataclass if needed)
            rec._priority_score = score

        # Sort by priority score (higher is better)
        return sorted(recommendations, key=lambda x: x._priority_score, reverse=True)

    def _calculate_composite_score(self, recommendation: DetailedRecommendation, context: RecommendationContext) -> float:
        """Calculate composite priority score"""

        # Priority weight
        priority_weights = {
            Priority.CRITICAL: 1.0,
            Priority.HIGH: 0.8,
            Priority.MEDIUM: 0.6,
            Priority.LOW: 0.4
        }
        priority_score = priority_weights[recommendation.priority]

        # Business impact weight
        impact_weights = {
            BusinessImpact.CRITICAL: 1.0,
            BusinessImpact.HIGH: 0.8,
            BusinessImpact.MEDIUM: 0.6,
            BusinessImpact.LOW: 0.4
        }
        impact_score = impact_weights[recommendation.business_impact]

        # Risk reduction weight
        risk_score = recommendation.risk_reduction

        # Implementation complexity penalty
        complexity_penalties = {
            ImplementationComplexity.LOW: 1.0,
            ImplementationComplexity.MEDIUM: 0.9,
            ImplementationComplexity.HIGH: 0.8,
            ImplementationComplexity.VERY_HIGH: 0.7
        }
        complexity_score = complexity_penalties[recommendation.implementation_complexity]

        # Calculate composite score
        composite_score = (priority_score * 0.4 +
                          impact_score * 0.3 +
                          risk_score * 0.2 +
                          complexity_score * 0.1)

        return composite_score

    def _sequence_recommendations(self, recommendations: List[DetailedRecommendation]) -> List[DetailedRecommendation]:
        """Sequence recommendations for optimal implementation order"""

        # Foundation recommendations first
        foundation_pattern = self.implementation_patterns["foundation_first"]

        for i, rec in enumerate(recommendations):
            if any(control in foundation_pattern for control in rec.compliance_impact):
                rec.dependency_order = 1
            elif rec.recommendation_type == RecommendationType.GOVERNANCE_STRUCTURE:
                rec.dependency_order = 1
            elif rec.recommendation_type == RecommendationType.POLICY_DEVELOPMENT:
                rec.dependency_order = 2
            elif rec.recommendation_type == RecommendationType.TECHNICAL_IMPLEMENTATION:
                rec.dependency_order = 3
            else:
                rec.dependency_order = 4

        # Sort by dependency order, then by priority score
        return sorted(recommendations, key=lambda x: (x.dependency_order, -x._priority_score))

    def generate_implementation_roadmap(self, recommendations: List[DetailedRecommendation]) -> Dict[str, Any]:
        """Generate implementation roadmap from recommendations"""

        phases = {
            "Phase 1: Foundation (Months 1-3)": [],
            "Phase 2: Governance (Months 2-6)": [],
            "Phase 3: Technical Controls (Months 4-9)": [],
            "Phase 4: Advanced Controls (Months 6-12)": []
        }

        for rec in recommendations:
            if rec.dependency_order == 1:
                phases["Phase 1: Foundation (Months 1-3)"].append(rec)
            elif rec.dependency_order == 2:
                phases["Phase 2: Governance (Months 2-6)"].append(rec)
            elif rec.dependency_order == 3:
                phases["Phase 3: Technical Controls (Months 4-9)"].append(rec)
            else:
                phases["Phase 4: Advanced Controls (Months 6-12)"].append(rec)

        return {
            "phases": phases,
            "total_timeline": "12 months",
            "total_effort": sum(rec.estimated_effort_hours for rec in recommendations),
            "critical_milestones": self._identify_critical_milestones(recommendations)
        }

    def _identify_critical_milestones(self, recommendations: List[DetailedRecommendation]) -> List[str]:
        """Identify critical milestones in implementation"""

        milestones = []

        # Foundation milestones
        foundation_recs = [r for r in recommendations if r.dependency_order == 1]
        if foundation_recs:
            milestones.append("Foundation policies and governance established")

        # Technical milestones
        technical_recs = [r for r in recommendations if r.recommendation_type == RecommendationType.TECHNICAL_IMPLEMENTATION]
        if technical_recs:
            milestones.append("Critical technical controls implemented")

        # Compliance milestones
        critical_recs = [r for r in recommendations if r.priority == Priority.CRITICAL]
        if critical_recs:
            milestones.append("All critical recommendations completed")

        milestones.append("Overall compliance assessment passed")

        return milestones

    def generate_compliance_overview(self, control_assessments: List['ControlAssessment'], clause_assessments: List['ClauseAssessment'], category_scores: List['CategoryScore']) -> 'ComplianceOverview':
        """Generate compliance overview"""
        from ..iso_knowledge.agent_response import ComplianceOverview, ComplianceStatus

        # Calculate overall score from category scores
        if category_scores:
            overall_score = sum(score.overall_score for score in category_scores) / len(category_scores)
        else:
            overall_score = 0

        # Determine maturity level
        if overall_score >= 90:
            maturity_level = "optimizing"
        elif overall_score >= 75:
            maturity_level = "quantified"
        elif overall_score >= 60:
            maturity_level = "defined"
        elif overall_score >= 40:
            maturity_level = "managed"
        else:
            maturity_level = "basic"

        # Count assessments by status
        all_assessments = control_assessments + clause_assessments
        compliant_count = sum(1 for a in all_assessments if a.status == ComplianceStatus.COMPLIANT)
        partial_count = sum(1 for a in all_assessments if a.status == ComplianceStatus.PARTIALLY_COMPLIANT)
        non_compliant_count = sum(1 for a in all_assessments if a.status == ComplianceStatus.NON_COMPLIANT)

        return ComplianceOverview(
            overall_maturity=maturity_level,
            overall_score=overall_score,
            total_controls_assessed=len(control_assessments) + len(clause_assessments),
            compliant_percentage=(compliant_count / (len(control_assessments) + len(clause_assessments))) * 100 if (len(control_assessments) + len(clause_assessments)) > 0 else 0,
            partially_compliant_percentage=(partial_count / (len(control_assessments) + len(clause_assessments))) * 100 if (len(control_assessments) + len(clause_assessments)) > 0 else 0,
            non_compliant_percentage=(non_compliant_count / (len(control_assessments) + len(clause_assessments))) * 100 if (len(control_assessments) + len(clause_assessments)) > 0 else 0,
            key_strengths=[f"Strong performance in {compliant_count} controls", "High overall compliance rate"] if compliant_count > 5 else ["Basic compliance established"],
            major_concerns=[f"Critical gaps in {non_compliant_count} controls"] if non_compliant_count > 3 else []
        )

    def perform_gap_analysis(self, control_assessments: List['ControlAssessment'], clause_assessments: List['ClauseAssessment']) -> 'GapAnalysis':
        """Perform gap analysis"""
        from ..iso_knowledge.agent_response import GapAnalysis, ComplianceStatus, Priority

        high_priority_gaps = []
        medium_priority_gaps = []
        low_priority_gaps = []

        # Analyze control gaps
        for assessment in control_assessments:
            if assessment.status == ComplianceStatus.NON_COMPLIANT:
                high_priority_gaps.append(f"Control {assessment.control_id}: {assessment.control_title}")
            elif assessment.status == ComplianceStatus.PARTIALLY_COMPLIANT:
                medium_priority_gaps.append(f"Control {assessment.control_id}: {assessment.control_title}")

        # Analyze clause gaps
        for assessment in clause_assessments:
            if assessment.status == ComplianceStatus.NON_COMPLIANT:
                high_priority_gaps.append(f"Clause {assessment.clause_id}: {assessment.requirement_title}")
            elif assessment.status == ComplianceStatus.PARTIALLY_COMPLIANT:
                medium_priority_gaps.append(f"Clause {assessment.clause_id}: {assessment.requirement_title}")

        # Calculate impact scores
        total_gaps = len(high_priority_gaps) + len(medium_priority_gaps)
        compliance_impact = "high" if len(high_priority_gaps) > 5 else "medium" if len(high_priority_gaps) > 0 else "low"
        risk_level = "critical" if len(high_priority_gaps) > 10 else "high" if len(high_priority_gaps) > 5 else "medium"

        return GapAnalysis(
            total_gaps=total_gaps,
            critical_gaps=len(high_priority_gaps),
            high_priority_gaps=len(medium_priority_gaps),
            gap_categories={
                "organizational": len([g for g in high_priority_gaps + medium_priority_gaps if "A.5" in g or "Control 5" in g]),
                "people": len([g for g in high_priority_gaps + medium_priority_gaps if "A.6" in g or "Control 6" in g]),
                "physical": len([g for g in high_priority_gaps + medium_priority_gaps if "A.7" in g or "Control 7" in g]),
                "technological": len([g for g in high_priority_gaps + medium_priority_gaps if "A.8" in g or "Control 8" in g]),
                "management": len([g for g in high_priority_gaps + medium_priority_gaps if "Clause" in g])
            },
            top_gap_areas=["organizational", "people", "technological"] if high_priority_gaps else ["basic_compliance"]
        )

    def identify_risk_areas_and_quick_wins(self, control_assessments: List['ControlAssessment'], clause_assessments: List['ClauseAssessment'], gap_analysis: 'GapAnalysis') -> Tuple[List[str], List[str]]:
        """Identify risk areas and quick wins"""
        risk_areas = []
        quick_wins = []

        # Risk areas from high priority assessments
        high_risk_controls = [assessment for assessment in control_assessments
                             if assessment.status.value == "non_compliant"]

        for assessment in high_risk_controls[:5]:
            risk_areas.append(f"Critical gap: Control {assessment.control_id} - {assessment.control_title}")

        high_risk_clauses = [assessment for assessment in clause_assessments
                            if assessment.status.value == "non_compliant"]

        for assessment in high_risk_clauses[:3]:
            risk_areas.append(f"Critical gap: {assessment.requirement_title}")

        # Quick wins from partially compliant items with high confidence
        for assessment in control_assessments:
            if (assessment.status.value == "partially_compliant" and
                assessment.confidence_score > 0.7 and
                len(assessment.evidence_found) > 0):
                quick_wins.append(f"Enhance {assessment.control_title.lower()}")

        for assessment in clause_assessments:
            if (assessment.status.value == "partially_compliant" and
                assessment.confidence_score > 0.7 and
                len(assessment.evidence_found) > 0):
                quick_wins.append(f"Complete {assessment.requirement_title.lower()}")

        # Add some basic quick wins if none found
        if not quick_wins:
            quick_wins = [
                "Update information security policy",
                "Conduct security awareness training",
                "Implement basic access logging"
            ]

        return risk_areas[:5], quick_wins[:5]

    def generate_next_steps(self, recommendations: List['Recommendation'], quick_wins: List[str]) -> List[str]:
        """Generate next steps"""
        next_steps = []

        # Add immediate quick wins
        if quick_wins:
            next_steps.append("Immediate actions:")
            for quick_win in quick_wins[:3]:
                next_steps.append(f"  • {quick_win}")

        # Add priority recommendations
        critical_recs = [r for r in recommendations if r.priority.value == "critical"]
        if critical_recs:
            next_steps.append("Priority implementations:")
            for rec in critical_recs[:3]:
                next_steps.append(f"  • {rec.title}")

        # Add planning step
        next_steps.append("Strategic planning:")
        next_steps.append("  • Develop detailed implementation timeline")
        next_steps.append("  • Assign responsible teams and resources")
        next_steps.append("  • Establish monitoring and review processes")

        return next_steps

    def generate_executive_summary(self, compliance_overview: 'ComplianceOverview', gap_analysis: 'GapAnalysis', recommendations: List['Recommendation']) -> str:
        """Generate executive summary"""
        return f"""**ISO 27001:2022 Compliance Analysis Summary**

Overall compliance score: {compliance_overview.overall_score:.1f}%
Maturity level: {compliance_overview.overall_maturity}
Total gaps identified: {gap_analysis.total_gaps}
Critical recommendations: {len([r for r in recommendations if r.priority.value == 'critical'])}

The organization demonstrates {compliance_overview.overall_maturity} level maturity in information security management.
{gap_analysis.total_gaps} gaps have been identified requiring attention.
Priority should be given to addressing {gap_analysis.critical_gaps} critical gaps."""