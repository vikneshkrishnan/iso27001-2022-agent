"""
ISO 27001:2022 Management Clauses 4-10 Knowledge Base
Contains detailed information about mandatory ISMS requirements
"""

from typing import Dict, List, Any
from dataclasses import dataclass
from enum import Enum


class ClauseType(Enum):
    CONTEXT = "context"
    LEADERSHIP = "leadership"
    PLANNING = "planning"
    SUPPORT = "support"
    OPERATION = "operation"
    PERFORMANCE_EVALUATION = "performance_evaluation"
    IMPROVEMENT = "improvement"


@dataclass
class Requirement:
    id: str
    title: str
    description: str
    purpose: str
    mandatory_elements: List[str]
    documentation_required: List[str]
    implementation_guidance: List[str]
    related_subclauses: List[str]


@dataclass
class Clause:
    id: str
    title: str
    clause_type: ClauseType
    purpose: str
    overview: str
    requirements: List[Requirement]
    key_changes_2022: List[str]


# ISO 27001:2022 Management Clauses Database
MANAGEMENT_CLAUSES = {
    "4": Clause(
        id="4",
        title="Context of the organization",
        clause_type=ClauseType.CONTEXT,
        purpose="To establish the context within which the ISMS operates and to define the scope of the ISMS.",
        overview="Organizations must understand their internal and external context, stakeholder needs, and define the scope of their ISMS.",
        key_changes_2022=[
            "Minor wording updates for clarity",
            "Enhanced guidance on determining scope boundaries"
        ],
        requirements=[
            Requirement(
                id="4.1",
                title="Understanding the organization and its context",
                description="The organization shall determine external and internal issues that are relevant to its purpose and that affect its ability to achieve the intended outcome(s) of its information security management system.",
                purpose="To understand factors that can impact the organization's ability to achieve ISMS objectives.",
                mandatory_elements=[
                    "Identify external issues (legal, technological, competitive, market, cultural, social, economic)",
                    "Identify internal issues (values, culture, knowledge, performance)",
                    "Document relevant issues",
                    "Consider these issues when establishing and maintaining the ISMS"
                ],
                documentation_required=[
                    "Context analysis documentation",
                    "Issue register or similar records"
                ],
                implementation_guidance=[
                    "Conduct regular environmental scanning",
                    "Use SWOT or PESTLE analysis frameworks",
                    "Consider regulatory and compliance requirements",
                    "Review industry trends and threat landscape",
                    "Assess organizational culture and capabilities"
                ],
                related_subclauses=["4.2", "4.3", "6.1"]
            ),
            Requirement(
                id="4.2",
                title="Understanding the needs and expectations of interested parties",
                description="The organization shall determine: the interested parties that are relevant to the information security management system; and the requirements of these interested parties relevant to information security.",
                purpose="To identify stakeholders and their information security expectations that could affect the ISMS.",
                mandatory_elements=[
                    "Identify interested parties relevant to ISMS",
                    "Determine requirements of interested parties",
                    "Document stakeholder requirements",
                    "Monitor changes in stakeholder needs"
                ],
                documentation_required=[
                    "Stakeholder register",
                    "Stakeholder requirements documentation"
                ],
                implementation_guidance=[
                    "Map all relevant stakeholders (customers, suppliers, regulators, employees)",
                    "Conduct stakeholder analysis and engagement",
                    "Document specific security requirements from each stakeholder",
                    "Establish communication channels with key stakeholders",
                    "Regularly review and update stakeholder requirements"
                ],
                related_subclauses=["4.1", "4.3", "5.2"]
            ),
            Requirement(
                id="4.3",
                title="Determining the scope of the information security management system",
                description="The organization shall determine the boundaries and applicability of the information security management system to establish its scope.",
                purpose="To define clear boundaries of what is included and excluded from the ISMS.",
                mandatory_elements=[
                    "Consider issues identified in 4.1",
                    "Consider requirements identified in 4.2",
                    "Consider interfaces and dependencies",
                    "Document the scope as documented information"
                ],
                documentation_required=[
                    "ISMS scope statement",
                    "Scope boundary documentation"
                ],
                implementation_guidance=[
                    "Clearly define organizational boundaries included in scope",
                    "Specify locations, assets, and technologies covered",
                    "Document interfaces with external parties",
                    "Justify any exclusions from scope",
                    "Ensure scope aligns with business objectives"
                ],
                related_subclauses=["4.1", "4.2", "4.4"]
            ),
            Requirement(
                id="4.4",
                title="Information security management system",
                description="The organization shall establish, implement, maintain and continually improve an information security management system, including the processes needed and their interactions.",
                purpose="To establish the overall ISMS framework with defined processes and interactions.",
                mandatory_elements=[
                    "Establish ISMS processes",
                    "Implement the processes",
                    "Maintain the ISMS",
                    "Continually improve the ISMS",
                    "Define process interactions"
                ],
                documentation_required=[
                    "ISMS process documentation",
                    "Process interaction diagrams or descriptions"
                ],
                implementation_guidance=[
                    "Map key ISMS processes and their interactions",
                    "Establish process ownership and responsibilities",
                    "Define process inputs, outputs, and controls",
                    "Implement process measurement and monitoring",
                    "Create process improvement mechanisms"
                ],
                related_subclauses=["4.3", "10.1", "10.2"]
            )
        ]
    ),

    "5": Clause(
        id="5",
        title="Leadership",
        clause_type=ClauseType.LEADERSHIP,
        purpose="To demonstrate leadership and commitment from top management for the ISMS.",
        overview="Top management must demonstrate leadership and commitment, establish policy, and assign organizational roles and responsibilities.",
        key_changes_2022=[
            "Enhanced clarity on management responsibilities",
            "Strengthened requirements for management commitment demonstration"
        ],
        requirements=[
            Requirement(
                id="5.1",
                title="Leadership and commitment",
                description="Top management shall demonstrate leadership and commitment with respect to the information security management system.",
                purpose="To ensure top management actively leads and supports the ISMS implementation and operation.",
                mandatory_elements=[
                    "Ensure ISMS policy and objectives are compatible with strategic direction",
                    "Ensure integration of ISMS requirements into business processes",
                    "Ensure resources are available for ISMS",
                    "Communicate importance of effective ISMS",
                    "Ensure ISMS achieves intended outcomes",
                    "Direct and support persons to contribute to ISMS effectiveness",
                    "Promote continual improvement",
                    "Support other management roles"
                ],
                documentation_required=[
                    "Evidence of management commitment",
                    "Management review records"
                ],
                implementation_guidance=[
                    "Establish clear management commitment statements",
                    "Allocate adequate budget and resources for information security",
                    "Participate actively in ISMS governance activities",
                    "Communicate security expectations throughout organization",
                    "Support security awareness and training programs"
                ],
                related_subclauses=["5.2", "5.3", "9.3"]
            ),
            Requirement(
                id="5.2",
                title="Policy",
                description="Top management shall establish an information security policy.",
                purpose="To provide overall direction and principles for information security within the organization.",
                mandatory_elements=[
                    "Policy appropriate to purpose of organization",
                    "Include information security objectives or framework for setting objectives",
                    "Include commitment to satisfy applicable requirements",
                    "Include commitment to continual improvement",
                    "Be available as documented information",
                    "Be communicated within organization",
                    "Be available to interested parties as appropriate"
                ],
                documentation_required=[
                    "Information security policy document",
                    "Policy communication records",
                    "Policy approval documentation"
                ],
                implementation_guidance=[
                    "Develop comprehensive information security policy",
                    "Ensure policy reflects organizational context and objectives",
                    "Obtain formal approval from top management",
                    "Communicate policy to all personnel and relevant parties",
                    "Make policy easily accessible to all stakeholders"
                ],
                related_subclauses=["5.1", "5.3", "6.2"]
            ),
            Requirement(
                id="5.3",
                title="Organizational roles, responsibilities and authorities",
                description="Top management shall ensure that the responsibilities and authorities for roles relevant to information security are assigned and communicated.",
                purpose="To ensure clear assignment and communication of information security roles and responsibilities.",
                mandatory_elements=[
                    "Assign responsibility for ensuring ISMS conforms to ISO 27001",
                    "Assign authority for reporting on ISMS performance to top management",
                    "Assign responsibilities and authorities for information security throughout organization",
                    "Communicate assigned responsibilities and authorities"
                ],
                documentation_required=[
                    "Role and responsibility matrix",
                    "Job descriptions including security responsibilities",
                    "Authority delegation documentation"
                ],
                implementation_guidance=[
                    "Create detailed RACI matrix for information security roles",
                    "Update job descriptions to include security responsibilities",
                    "Establish clear reporting lines for security matters",
                    "Ensure adequate authority is delegated for security decisions",
                    "Communicate role assignments clearly to all personnel"
                ],
                related_subclauses=["5.1", "5.2", "6.1"]
            )
        ]
    ),

    "6": Clause(
        id="6",
        title="Planning",
        clause_type=ClauseType.PLANNING,
        purpose="To establish planning processes for the ISMS including risk management and objective setting.",
        overview="Organizations must plan for risks and opportunities, establish information security objectives, and plan changes to the ISMS.",
        key_changes_2022=[
            "New subclause 6.3 added for 'Planning of changes'",
            "Enhanced guidance on risk treatment planning"
        ],
        requirements=[
            Requirement(
                id="6.1",
                title="Actions to address risks and opportunities",
                description="When planning for the information security management system, the organization shall consider the issues and requirements referenced in 4.1 and 4.2 and determine the risks and opportunities.",
                purpose="To ensure the ISMS can achieve its intended outcomes, prevent or reduce undesired effects, and achieve continual improvement.",
                mandatory_elements=[
                    "Consider issues from 4.1 and requirements from 4.2",
                    "Determine risks and opportunities",
                    "Plan actions to address risks and opportunities",
                    "Plan how to integrate and implement actions into ISMS processes",
                    "Plan how to evaluate effectiveness of actions"
                ],
                documentation_required=[
                    "Risk assessment methodology",
                    "Risk register and treatment plans",
                    "Opportunity identification and action plans"
                ],
                implementation_guidance=[
                    "Establish risk management framework and methodology",
                    "Conduct comprehensive risk assessments",
                    "Identify improvement opportunities",
                    "Develop risk treatment plans with clear ownership",
                    "Monitor and review risk treatment effectiveness"
                ],
                related_subclauses=["4.1", "4.2", "6.2", "8.1", "8.2", "8.3"]
            ),
            Requirement(
                id="6.2",
                title="Information security objectives and planning to achieve them",
                description="The organization shall establish information security objectives at relevant functions and levels.",
                purpose="To provide clear direction for information security efforts and enable measurement of ISMS performance.",
                mandatory_elements=[
                    "Objectives consistent with information security policy",
                    "Objectives measurable",
                    "Take into account applicable requirements",
                    "Consider risk assessment and treatment results",
                    "Be communicated",
                    "Be updated as appropriate"
                ],
                documentation_required=[
                    "Information security objectives",
                    "Objective achievement plans",
                    "Performance measurement records"
                ],
                implementation_guidance=[
                    "Set SMART (Specific, Measurable, Achievable, Relevant, Time-bound) objectives",
                    "Align objectives with business strategy and risk appetite",
                    "Define clear success criteria and measurement methods",
                    "Assign ownership and accountability for objective achievement",
                    "Regularly review and update objectives"
                ],
                related_subclauses=["5.2", "6.1", "9.1", "9.3"]
            ),
            Requirement(
                id="6.3",
                title="Planning of changes",
                description="When the organization determines the need for changes to the information security management system, the changes shall be carried out in a planned manner.",
                purpose="To ensure changes to the ISMS are implemented systematically and do not compromise security.",
                mandatory_elements=[
                    "Consider purpose of changes and potential consequences",
                    "Consider integrity of ISMS",
                    "Consider availability of resources",
                    "Consider allocation or reallocation of responsibilities and authorities"
                ],
                documentation_required=[
                    "Change management procedures",
                    "Change impact assessments",
                    "Change approval records"
                ],
                implementation_guidance=[
                    "Establish formal change management process for ISMS",
                    "Conduct impact assessments for proposed changes",
                    "Ensure adequate resources are available for changes",
                    "Obtain appropriate approvals before implementing changes",
                    "Monitor change implementation and effectiveness"
                ],
                related_subclauses=["6.1", "6.2", "8.1", "10.2"]
            )
        ]
    ),

    # Continue with clauses 7-10...
    "7": Clause(
        id="7",
        title="Support",
        clause_type=ClauseType.SUPPORT,
        purpose="To ensure adequate support is provided for the ISMS operation including resources, competence, awareness, communication, and documented information.",
        overview="Organizations must provide necessary support elements including resources, competence, awareness, communication, and documentation.",
        key_changes_2022=[
            "Minor clarifications on competence requirements",
            "Enhanced guidance on documented information management"
        ],
        requirements=[
            Requirement(
                id="7.1",
                title="Resources",
                description="The organization shall determine and provide the resources needed for the establishment, implementation, maintenance and continual improvement of the information security management system.",
                purpose="To ensure adequate resources are available for effective ISMS operation.",
                mandatory_elements=[
                    "Determine resources needed for ISMS",
                    "Provide determined resources",
                    "Consider resource needs for establishment, implementation, maintenance, and improvement"
                ],
                documentation_required=[
                    "Resource allocation records",
                    "Budget documentation for information security"
                ],
                implementation_guidance=[
                    "Conduct resource planning for ISMS activities",
                    "Allocate adequate budget for information security",
                    "Ensure sufficient human resources with appropriate skills",
                    "Provide necessary technology and infrastructure",
                    "Plan for resource needs in different scenarios"
                ],
                related_subclauses=["5.1", "7.2", "7.3"]
            )
        ]
    )
}


def get_clause_by_id(clause_id: str) -> Clause:
    """Get a specific clause by its ID"""
    return MANAGEMENT_CLAUSES.get(clause_id)


def get_requirement_by_id(clause_id: str, requirement_id: str) -> Requirement:
    """Get a specific requirement within a clause"""
    clause = get_clause_by_id(clause_id)
    if clause:
        return next((req for req in clause.requirements if req.id == requirement_id), None)
    return None


def get_all_clauses() -> Dict[str, Clause]:
    """Get all management clauses"""
    return MANAGEMENT_CLAUSES


def get_clauses_by_type(clause_type: ClauseType) -> List[Clause]:
    """Get all clauses of a specific type"""
    return [clause for clause in MANAGEMENT_CLAUSES.values() if clause.clause_type == clause_type]


def search_requirements(query: str) -> List[tuple[str, Requirement]]:
    """Search requirements by title or description"""
    query_lower = query.lower()
    matching_requirements = []

    for clause_id, clause in MANAGEMENT_CLAUSES.items():
        for requirement in clause.requirements:
            if (query_lower in requirement.title.lower() or
                query_lower in requirement.description.lower() or
                query_lower in requirement.purpose.lower()):
                matching_requirements.append((clause_id, requirement))

    return matching_requirements


def get_all_requirements() -> List[tuple[str, Requirement]]:
    """Get all requirements from all clauses"""
    all_requirements = []
    for clause_id, clause in MANAGEMENT_CLAUSES.items():
        for requirement in clause.requirements:
            all_requirements.append((clause_id, requirement))
    return all_requirements