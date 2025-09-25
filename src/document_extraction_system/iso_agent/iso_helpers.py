"""
Helper methods for ISO Expert Agent
Contains utility functions for analysis, scoring, and recommendation generation
"""

from typing import Dict, List, Optional, Any, Tuple
import logging
from datetime import datetime, timedelta

from ..iso_knowledge.annex_a_controls import search_controls, Control
from ..iso_knowledge.management_clauses import search_requirements, Clause
from ..iso_knowledge.agent_response import (
    ComplianceStatus, Priority, GapAnalysis, Recommendation,
    ComplianceOverview, ControlAssessment, ClauseAssessment
)

logger = logging.getLogger(__name__)


class ISOAnalysisHelpers:
    """Helper methods for ISO analysis operations"""

    @staticmethod
    def generate_compliance_overview(
        control_assessments: List[ControlAssessment],
        clause_assessments: List[ClauseAssessment],
        category_scores: List[Any]
    ) -> ComplianceOverview:
        """Generate overall compliance overview"""

        # Calculate overall statistics
        total_assessments = len(control_assessments) + len(clause_assessments)

        if total_assessments == 0:
            return ComplianceOverview(
                overall_maturity="unknown",
                overall_score=0.0,
                total_controls_assessed=0,
                compliant_percentage=0.0,
                partially_compliant_percentage=0.0,
                non_compliant_percentage=0.0,
                key_strengths=[],
                major_concerns=[]
            )

        # Count compliance statuses
        compliant_count = sum(
            1 for assessment in control_assessments + clause_assessments
            if assessment.status == ComplianceStatus.COMPLIANT
        )

        partially_compliant_count = sum(
            1 for assessment in control_assessments + clause_assessments
            if assessment.status == ComplianceStatus.PARTIALLY_COMPLIANT
        )

        non_compliant_count = sum(
            1 for assessment in control_assessments + clause_assessments
            if assessment.status == ComplianceStatus.NON_COMPLIANT
        )

        # Calculate percentages
        compliant_percentage = (compliant_count / total_assessments) * 100
        partially_compliant_percentage = (partially_compliant_count / total_assessments) * 100
        non_compliant_percentage = (non_compliant_count / total_assessments) * 100

        # Calculate overall score
        overall_score = compliant_percentage + (partially_compliant_percentage * 0.5)

        # Determine maturity level
        if overall_score >= 90:
            overall_maturity = "optimizing"
        elif overall_score >= 75:
            overall_maturity = "quantified"
        elif overall_score >= 60:
            overall_maturity = "defined"
        elif overall_score >= 40:
            overall_maturity = "managed"
        else:
            overall_maturity = "basic"

        # Identify key strengths
        key_strengths = []
        compliant_controls = [
            assessment for assessment in control_assessments
            if assessment.status == ComplianceStatus.COMPLIANT
        ]
        if compliant_controls:
            key_strengths.append(f"Strong performance in {len(compliant_controls)} controls")

        if compliant_percentage > 70:
            key_strengths.append("High overall compliance rate")

        if category_scores:
            best_category = max(category_scores, key=lambda x: x.overall_score)
            key_strengths.append(f"Excellent {best_category.category} controls implementation")

        # Identify major concerns
        major_concerns = []
        critical_failures = [
            assessment for assessment in control_assessments + clause_assessments
            if assessment.status == ComplianceStatus.NON_COMPLIANT and
            any(keyword in assessment.control_title.lower() or
                getattr(assessment, 'requirement_title', '').lower()
                for keyword in ['access', 'security', 'incident', 'risk'])
        ]

        if critical_failures:
            major_concerns.append(f"Critical security controls not implemented ({len(critical_failures)} identified)")

        if non_compliant_percentage > 30:
            major_concerns.append("High number of non-compliant requirements")

        if overall_score < 50:
            major_concerns.append("Overall maturity level needs significant improvement")

        return ComplianceOverview(
            overall_maturity=overall_maturity,
            overall_score=overall_score,
            total_controls_assessed=len(control_assessments),
            compliant_percentage=compliant_percentage,
            partially_compliant_percentage=partially_compliant_percentage,
            non_compliant_percentage=non_compliant_percentage,
            key_strengths=key_strengths,
            major_concerns=major_concerns
        )

    @staticmethod
    def perform_gap_analysis(
        control_assessments: List[ControlAssessment],
        clause_assessments: List[ClauseAssessment]
    ) -> GapAnalysis:
        """Perform comprehensive gap analysis"""

        # Count total gaps
        control_gaps = [
            assessment for assessment in control_assessments
            if assessment.status in [ComplianceStatus.NON_COMPLIANT, ComplianceStatus.PARTIALLY_COMPLIANT]
        ]

        clause_gaps = [
            assessment for assessment in clause_assessments
            if assessment.status in [ComplianceStatus.NON_COMPLIANT, ComplianceStatus.PARTIALLY_COMPLIANT]
        ]

        total_gaps = len(control_gaps) + len(clause_gaps)

        # Count critical gaps (non-compliant only)
        critical_gaps = len([
            assessment for assessment in control_assessments + clause_assessments
            if assessment.status == ComplianceStatus.NON_COMPLIANT
        ])

        # Count high priority gaps (based on common security keywords)
        high_priority_keywords = ['access', 'incident', 'risk', 'security', 'vulnerability', 'threat']
        high_priority_gaps = len([
            assessment for assessment in control_assessments + clause_assessments
            if assessment.status != ComplianceStatus.COMPLIANT and
            any(keyword in (assessment.control_title.lower() if hasattr(assessment, 'control_title')
                           else getattr(assessment, 'requirement_title', '').lower())
                for keyword in high_priority_keywords)
        ])

        # Categorize gaps
        gap_categories = {
            "organizational": 0,
            "people": 0,
            "physical": 0,
            "technological": 0,
            "management": 0
        }

        # Count control gaps by category (simplified categorization)
        for assessment in control_gaps:
            control_id = assessment.control_id
            if control_id.startswith('5'):
                gap_categories["organizational"] += 1
            elif control_id.startswith('6'):
                gap_categories["people"] += 1
            elif control_id.startswith('7'):
                gap_categories["physical"] += 1
            elif control_id.startswith('8'):
                gap_categories["technological"] += 1

        # Count clause gaps as management gaps
        gap_categories["management"] = len(clause_gaps)

        # Identify top gap areas
        top_gap_areas = []
        sorted_categories = sorted(gap_categories.items(), key=lambda x: x[1], reverse=True)
        for category, count in sorted_categories:
            if count > 0:
                top_gap_areas.append(category)
            if len(top_gap_areas) >= 3:
                break

        return GapAnalysis(
            total_gaps=total_gaps,
            critical_gaps=critical_gaps,
            high_priority_gaps=high_priority_gaps,
            gap_categories=gap_categories,
            top_gap_areas=top_gap_areas
        )

    @staticmethod
    async def generate_recommendations(
        gap_analysis: GapAnalysis,
        control_assessments: List[ControlAssessment],
        clause_assessments: List[ClauseAssessment]
    ) -> List[Recommendation]:
        """Generate prioritized recommendations based on gap analysis"""

        recommendations = []

        # Generate control-based recommendations
        non_compliant_controls = [
            assessment for assessment in control_assessments
            if assessment.status == ComplianceStatus.NON_COMPLIANT
        ]

        for i, assessment in enumerate(non_compliant_controls[:10]):  # Limit to top 10
            # Determine priority based on control type
            priority = Priority.HIGH
            if any(keyword in assessment.control_title.lower()
                   for keyword in ['access', 'incident', 'vulnerability']):
                priority = Priority.CRITICAL
            elif any(keyword in assessment.control_title.lower()
                     for keyword in ['monitoring', 'audit', 'review']):
                priority = Priority.MEDIUM

            # Determine category
            control_id = assessment.control_id
            category = "organizational"
            if control_id.startswith('6'):
                category = "people"
            elif control_id.startswith('7'):
                category = "physical"
            elif control_id.startswith('8'):
                category = "technological"

            # Create recommendation
            recommendation = Recommendation(
                id=f"REC-CTRL-{i+1:03d}",
                title=f"Implement {assessment.control_title}",
                description=f"Address gaps in {assessment.control_title.lower()} to achieve compliance.",
                priority=priority,
                category=category,
                effort_estimate="medium",
                timeline_estimate="short-term" if priority == Priority.CRITICAL else "medium-term",
                related_controls=[assessment.control_id],
                implementation_steps=[
                    "Review current implementation",
                    "Identify specific gaps",
                    "Develop implementation plan",
                    "Execute implementation",
                    "Validate effectiveness"
                ]
            )
            recommendations.append(recommendation)

        # Generate clause-based recommendations
        non_compliant_clauses = [
            assessment for assessment in clause_assessments
            if assessment.status == ComplianceStatus.NON_COMPLIANT
        ]

        for i, assessment in enumerate(non_compliant_clauses[:5]):  # Limit to top 5
            recommendation = Recommendation(
                id=f"REC-CLAUSE-{i+1:03d}",
                title=f"Address {getattr(assessment, 'requirement_title', 'Management Requirement')}",
                description=f"Implement missing elements for {getattr(assessment, 'requirement_title', 'requirement')}.",
                priority=Priority.HIGH,
                category="management",
                effort_estimate="high",
                timeline_estimate="medium-term",
                related_controls=[],
                implementation_steps=[
                    "Document current state",
                    "Identify missing elements",
                    "Create implementation plan",
                    "Implement requirements",
                    "Document evidence"
                ]
            )
            recommendations.append(recommendation)

        # Sort recommendations by priority
        priority_order = {
            Priority.CRITICAL: 1,
            Priority.HIGH: 2,
            Priority.MEDIUM: 3,
            Priority.LOW: 4
        }

        recommendations.sort(key=lambda x: priority_order[x.priority])

        return recommendations

    @staticmethod
    def identify_risk_areas_and_quick_wins(
        control_assessments: List[ControlAssessment],
        clause_assessments: List[ClauseAssessment],
        gap_analysis: GapAnalysis
    ) -> Tuple[List[str], List[str]]:
        """Identify risk areas and quick wins"""

        risk_areas = []
        quick_wins = []

        # Identify risk areas based on critical gaps
        high_risk_controls = [
            assessment for assessment in control_assessments
            if assessment.status == ComplianceStatus.NON_COMPLIANT and
            any(keyword in assessment.control_title.lower()
                for keyword in ['access', 'incident', 'vulnerability', 'threat', 'breach'])
        ]

        for assessment in high_risk_controls:
            risk_areas.append(f"Uncontrolled {assessment.control_title.lower()}")

        # Add generic risk areas based on gap categories
        if gap_analysis.gap_categories.get("technological", 0) > 3:
            risk_areas.append("Technology security vulnerabilities")

        if gap_analysis.gap_categories.get("organizational", 0) > 3:
            risk_areas.append("Governance and policy weaknesses")

        # Identify quick wins (partially compliant controls that need minor work)
        quick_win_candidates = [
            assessment for assessment in control_assessments
            if assessment.status == ComplianceStatus.PARTIALLY_COMPLIANT and
            assessment.confidence_score > 0.6
        ]

        for assessment in quick_win_candidates[:5]:  # Limit to top 5
            quick_wins.append(f"Complete {assessment.control_title.lower()} implementation")

        # Add some generic quick wins
        if len(quick_wins) < 3:
            quick_wins.extend([
                "Update information security policy",
                "Conduct security awareness training",
                "Implement basic access logging"
            ])

        return risk_areas[:5], quick_wins[:5]  # Limit lists

    @staticmethod
    def create_implementation_roadmap(recommendations: List[Recommendation]) -> List[Dict[str, Any]]:
        """Create implementation roadmap from recommendations"""

        roadmap = []

        # Group recommendations by timeline
        timeline_groups = {
            "immediate": [],
            "short-term": [],
            "medium-term": [],
            "long-term": []
        }

        for rec in recommendations:
            timeline_groups[rec.timeline_estimate].append(rec)

        # Create roadmap phases
        phase_num = 1
        for timeline, recs in timeline_groups.items():
            if recs:
                # Calculate phase duration
                duration_map = {
                    "immediate": "1-4 weeks",
                    "short-term": "1-3 months",
                    "medium-term": "3-6 months",
                    "long-term": "6-12 months"
                }

                roadmap_phase = {
                    "phase": phase_num,
                    "name": f"Phase {phase_num}: {timeline.replace('-', ' ').title()} Implementation",
                    "duration": duration_map[timeline],
                    "recommendations": [
                        {
                            "id": rec.id,
                            "title": rec.title,
                            "priority": rec.priority.value,
                            "effort": rec.effort_estimate
                        }
                        for rec in recs
                    ],
                    "success_criteria": [
                        f"Complete {len(recs)} recommendations",
                        "Validate implementation effectiveness",
                        "Update documentation"
                    ]
                }
                roadmap.append(roadmap_phase)
                phase_num += 1

        return roadmap

    @staticmethod
    def calculate_analysis_confidence(
        control_assessments: List[ControlAssessment],
        clause_assessments: List[ClauseAssessment]
    ) -> float:
        """Calculate overall confidence in analysis results"""

        all_assessments = control_assessments + clause_assessments

        if not all_assessments:
            return 0.0

        # Calculate average confidence across all assessments
        total_confidence = sum(assessment.confidence_score for assessment in all_assessments)
        average_confidence = total_confidence / len(all_assessments)

        # Adjust based on number of assessments (more assessments = higher confidence)
        assessment_factor = min(len(all_assessments) / 50.0, 1.0)  # Cap at 50 assessments

        # Final confidence calculation
        final_confidence = average_confidence * assessment_factor

        return round(final_confidence, 2)

    @staticmethod
    def calculate_coverage_percentage(
        control_assessments: List[ControlAssessment],
        clause_assessments: List[ClauseAssessment]
    ) -> float:
        """Calculate percentage of ISO framework covered by analysis"""

        # Total possible: 93 controls + approximately 20 clause requirements
        total_possible = 113

        total_assessed = len(control_assessments) + len(clause_assessments)

        if total_possible == 0:
            return 0.0

        coverage = (total_assessed / total_possible) * 100

        return round(min(coverage, 100.0), 1)  # Cap at 100%

    @staticmethod
    async def generate_executive_summary(
        compliance_overview: ComplianceOverview,
        gap_analysis: GapAnalysis,
        recommendations: List[Recommendation]
    ) -> str:
        """Generate executive summary of analysis results"""

        summary_parts = []

        # Overall assessment
        summary_parts.append(
            f"The analyzed document demonstrates {compliance_overview.overall_maturity} maturity "
            f"with an overall compliance score of {compliance_overview.overall_score:.1f}%."
        )

        # Key findings
        if compliance_overview.key_strengths:
            summary_parts.append(
                f"Key strengths include: {', '.join(compliance_overview.key_strengths[:2])}."
            )

        # Major concerns
        if compliance_overview.major_concerns:
            summary_parts.append(
                f"Primary concerns identified: {', '.join(compliance_overview.major_concerns[:2])}."
            )

        # Gap analysis summary
        if gap_analysis.total_gaps > 0:
            summary_parts.append(
                f"Analysis identified {gap_analysis.total_gaps} total gaps, "
                f"including {gap_analysis.critical_gaps} critical areas requiring immediate attention."
            )

        # Recommendations summary
        critical_recs = [r for r in recommendations if r.priority == Priority.CRITICAL]
        high_recs = [r for r in recommendations if r.priority == Priority.HIGH]

        if critical_recs or high_recs:
            summary_parts.append(
                f"Priority actions include {len(critical_recs)} critical and "
                f"{len(high_recs)} high-priority recommendations for implementation."
            )

        return " ".join(summary_parts)

    @staticmethod
    def generate_next_steps(
        recommendations: List[Recommendation],
        quick_wins: List[str]
    ) -> List[str]:
        """Generate prioritized next steps"""

        next_steps = []

        # Add critical recommendations
        critical_recs = [r for r in recommendations if r.priority == Priority.CRITICAL]
        for rec in critical_recs[:3]:  # Top 3 critical
            next_steps.append(f"URGENT: {rec.title}")

        # Add quick wins
        for quick_win in quick_wins[:2]:  # Top 2 quick wins
            next_steps.append(f"Quick Win: {quick_win}")

        # Add high priority recommendations
        high_recs = [r for r in recommendations if r.priority == Priority.HIGH]
        for rec in high_recs[:2]:  # Top 2 high priority
            next_steps.append(f"High Priority: {rec.title}")

        # Add general next steps
        if len(next_steps) < 5:
            next_steps.extend([
                "Conduct detailed risk assessment",
                "Develop comprehensive implementation plan",
                "Establish regular monitoring and review processes"
            ])

        return next_steps[:7]  # Limit to 7 items