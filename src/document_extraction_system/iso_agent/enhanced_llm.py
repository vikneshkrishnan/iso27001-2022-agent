"""
Enhanced LLM Integration for ISO 27001:2022 Analysis
Provides structured prompts, response validation, and intelligent analysis
"""

import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import re

try:
    import openai
    from anthropic import Anthropic
    OPENAI_AVAILABLE = True
    ANTHROPIC_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    ANTHROPIC_AVAILABLE = False

from ..iso_knowledge.annex_a_controls import Control
from ..iso_knowledge.management_clauses import Requirement
from ..iso_knowledge.agent_response import ComplianceStatus, DocumentType, Priority

logger = logging.getLogger(__name__)


class AnalysisType(Enum):
    DOCUMENT_CLASSIFICATION = "document_classification"
    CONTROL_ASSESSMENT = "control_assessment"
    GAP_ANALYSIS = "gap_analysis"
    RECOMMENDATION_GENERATION = "recommendation_generation"
    EXECUTIVE_SUMMARY = "executive_summary"


@dataclass
class LLMRequest:
    """Structure for LLM requests"""
    analysis_type: AnalysisType
    prompt: str
    context_data: Dict[str, Any]
    expected_format: str
    max_tokens: int = 2000
    temperature: float = 0.3


@dataclass
class LLMResponse:
    """Structure for LLM responses"""
    success: bool
    content: str
    structured_data: Optional[Dict[str, Any]] = None
    confidence: float = 0.0
    validation_errors: List[str] = None
    raw_response: str = ""


class EnhancedLLMProcessor:
    """Enhanced LLM processor with structured analysis capabilities"""

    def __init__(self, openai_api_key: Optional[str] = None, anthropic_api_key: Optional[str] = None):
        self.openai_client = None
        self.anthropic_client = None

        # Initialize clients if available
        if OPENAI_AVAILABLE and openai_api_key:
            from openai import AsyncOpenAI
            self.openai_client = AsyncOpenAI(api_key=openai_api_key)

        if ANTHROPIC_AVAILABLE and anthropic_api_key:
            self.anthropic_client = Anthropic(api_key=anthropic_api_key)

        # Pre-built prompt templates
        self.prompt_templates = self._build_prompt_templates()
        self.validation_schemas = self._build_validation_schemas()

    def _build_prompt_templates(self) -> Dict[AnalysisType, str]:
        """Build structured prompt templates for different analysis types"""
        return {
            AnalysisType.DOCUMENT_CLASSIFICATION: """
You are an expert in ISO 27001:2022 information security management. Analyze the following document and provide a classification.

DOCUMENT INFORMATION:
Filename: {filename}
Size: {size} bytes
Content Preview: {content_preview}

DOCUMENT CONTENT (First 3000 characters):
{document_content}

ANALYSIS REQUIRED:
Classify this document according to ISO 27001:2022 context and provide the following information:

1. Document Type: (policy, procedure, standard, guideline, risk_assessment, audit_report, incident_report, training_material, contract, technical_specification, or other)
2. ISO Relevance: (high, medium, low) - How relevant is this document to ISO 27001:2022 compliance
3. Applicable Categories: List which apply (organizational, people, physical, technological)
4. Primary Focus Areas: List 3-5 main security areas this document addresses
5. Confidence Score: Your confidence in this analysis (0.0-1.0)
6. Key Security Topics: Identify main security topics discussed

RESPOND ONLY WITH VALID JSON:
{{
    "document_type": "policy",
    "iso_relevance": "high",
    "applicable_categories": ["organizational", "people"],
    "primary_focus_areas": ["access control", "information security policy", "roles and responsibilities"],
    "confidence_score": 0.9,
    "key_security_topics": ["access management", "policy governance", "security roles"]
}}
""",

            AnalysisType.CONTROL_ASSESSMENT: """
You are an expert ISO 27001:2022 auditor. Assess how well the document addresses the specified control.

CONTROL INFORMATION:
ID: {control_id}
Title: {control_title}
Description: {control_description}
Purpose: {control_purpose}

DOCUMENT CONTENT ANALYSIS:
Relevant Excerpts: {relevant_excerpts}
Document Context: {document_context}

SEMANTIC ANALYSIS RESULTS:
Evidence Found: {evidence_found}
Implementation Indicators: {implementation_indicators}
Gap Indicators: {gap_indicators}

ASSESSMENT REQUIRED:
Provide a detailed assessment of how this document addresses the control:

1. Compliance Status: (compliant, partially_compliant, non_compliant, insufficient_information)
2. Specific Evidence: Quote exact text from the document that supports compliance
3. Implementation Details: Describe what is implemented based on the document
4. Gaps Identified: List specific gaps or missing elements
5. Quality Assessment: Rate the implementation quality (1-5 scale)
6. Recommendations: Provide specific actionable recommendations
7. Confidence Level: Your confidence in this assessment (0.0-1.0)

RESPOND ONLY WITH VALID JSON:
{{
    "compliance_status": "partially_compliant",
    "specific_evidence": ["Quote 1 from document", "Quote 2 from document"],
    "implementation_details": ["Detail 1", "Detail 2"],
    "gaps_identified": ["Gap 1", "Gap 2"],
    "quality_assessment": 3,
    "recommendations": ["Recommendation 1", "Recommendation 2"],
    "confidence_level": 0.8
}}
""",

            AnalysisType.GAP_ANALYSIS: """
You are an expert ISO 27001:2022 consultant. Perform a comprehensive gap analysis.

ASSESSMENT RESULTS:
Controls Analyzed: {controls_count}
Compliance Summary: {compliance_summary}
Evidence Quality: {evidence_quality}

CONTROL ASSESSMENTS:
{control_assessments}

GAP ANALYSIS REQUIRED:
Based on the control assessments, provide:

1. Critical Gaps: High-impact gaps requiring immediate attention
2. Moderate Gaps: Important gaps that should be addressed soon
3. Minor Gaps: Lower priority improvement opportunities
4. Root Cause Analysis: Identify underlying causes of gaps
5. Impact Assessment: Assess potential impact of each gap category
6. Priority Ranking: Rank gaps by risk and effort to fix
7. Quick Wins: Identify easy improvements with high impact

RESPOND ONLY WITH VALID JSON:
{{
    "critical_gaps": [
        {{"description": "Gap description", "impact": "high", "controls_affected": ["5.1", "5.2"]}}
    ],
    "moderate_gaps": [
        {{"description": "Gap description", "impact": "medium", "controls_affected": ["6.1"]}}
    ],
    "minor_gaps": [],
    "root_causes": ["Lack of formal processes", "Insufficient documentation"],
    "impact_assessment": "High risk due to...",
    "priority_ranking": ["Critical gap 1", "Moderate gap 1"],
    "quick_wins": ["Easy improvement 1", "Easy improvement 2"]
}}
""",

            AnalysisType.RECOMMENDATION_GENERATION: """
You are an expert ISO 27001:2022 implementation consultant. Generate actionable recommendations.

GAP ANALYSIS RESULTS:
{gap_analysis_results}

ORGANIZATIONAL CONTEXT:
Document Type: {document_type}
Maturity Level: {maturity_level}
Primary Focus Areas: {focus_areas}

RECOMMENDATION REQUIREMENTS:
Generate specific, actionable recommendations that include:

1. Immediate Actions: Critical actions needed within 30 days
2. Short-term Goals: Actions for next 3-6 months
3. Long-term Strategy: Strategic initiatives for 6-18 months
4. Resource Requirements: Estimate effort, skills, budget needed
5. Success Metrics: How to measure implementation success
6. Implementation Order: Logical sequence for implementation
7. Risk Mitigation: How each recommendation reduces risk

RESPOND ONLY WITH VALID JSON:
{{
    "immediate_actions": [
        {{
            "title": "Action title",
            "description": "Detailed description",
            "priority": "critical",
            "effort_estimate": "low/medium/high",
            "timeline": "1-4 weeks",
            "resources_needed": ["skill/resource"],
            "success_metric": "How to measure success"
        }}
    ],
    "short_term_goals": [],
    "long_term_strategy": [],
    "implementation_sequence": ["Step 1", "Step 2"],
    "total_effort_estimate": "medium",
    "expected_timeline": "6-12 months"
}}
""",

            AnalysisType.EXECUTIVE_SUMMARY: """
You are an expert ISO 27001:2022 consultant preparing an executive summary for senior management.

ANALYSIS RESULTS:
Overall Compliance Score: {overall_score}%
Total Controls Assessed: {total_controls}
Document Classification: {document_classification}
Key Findings: {key_findings}
Gap Analysis: {gap_analysis}
Recommendations: {recommendations}

EXECUTIVE SUMMARY REQUIREMENTS:
Create a concise executive summary that includes:

1. Current State Assessment: High-level overview of compliance status
2. Key Strengths: What's working well
3. Critical Issues: Most important issues requiring attention
4. Business Impact: Potential impact of identified gaps
5. Investment Required: High-level resource requirements
6. Timeline: Realistic timeline for addressing issues
7. Strategic Recommendations: Top 3-5 strategic actions
8. Next Steps: Immediate next steps for management

Keep the summary concise but comprehensive, suitable for C-level executives.

RESPOND ONLY WITH VALID JSON:
{{
    "current_state": "Overall assessment in 2-3 sentences",
    "key_strengths": ["Strength 1", "Strength 2"],
    "critical_issues": ["Issue 1", "Issue 2"],
    "business_impact": "Potential business impact description",
    "investment_required": "High-level resource estimate",
    "timeline": "Realistic implementation timeline",
    "strategic_recommendations": ["Recommendation 1", "Recommendation 2"],
    "next_steps": ["Next step 1", "Next step 2"],
    "executive_priority": "highest/high/medium/low"
}}
"""
        }

    def _build_validation_schemas(self) -> Dict[AnalysisType, Dict[str, Any]]:
        """Build validation schemas for LLM responses"""
        return {
            AnalysisType.DOCUMENT_CLASSIFICATION: {
                "required_fields": [
                    "document_type", "iso_relevance", "applicable_categories",
                    "primary_focus_areas", "confidence_score"
                ],
                "field_types": {
                    "document_type": str,
                    "iso_relevance": str,
                    "applicable_categories": list,
                    "primary_focus_areas": list,
                    "confidence_score": float
                },
                "valid_values": {
                    "document_type": [
                        "policy", "procedure", "standard", "guideline",
                        "risk_assessment", "audit_report", "incident_report",
                        "training_material", "contract", "technical_specification", "other"
                    ],
                    "iso_relevance": ["high", "medium", "low"],
                    "applicable_categories": ["organizational", "people", "physical", "technological"]
                }
            },

            AnalysisType.CONTROL_ASSESSMENT: {
                "required_fields": [
                    "compliance_status", "specific_evidence", "implementation_details",
                    "gaps_identified", "recommendations", "confidence_level"
                ],
                "field_types": {
                    "compliance_status": str,
                    "specific_evidence": list,
                    "implementation_details": list,
                    "gaps_identified": list,
                    "recommendations": list,
                    "confidence_level": float
                },
                "valid_values": {
                    "compliance_status": ["compliant", "partially_compliant", "non_compliant", "insufficient_information"]
                }
            }
        }

    async def process_llm_request(self, request: LLMRequest) -> LLMResponse:
        """Process an LLM request with validation"""
        try:
            # Get the appropriate prompt template
            prompt_template = self.prompt_templates.get(request.analysis_type)
            if not prompt_template:
                return LLMResponse(
                    success=False,
                    content="",
                    validation_errors=[f"No template found for {request.analysis_type}"]
                )

            # Format the prompt with context data
            formatted_prompt = prompt_template.format(**request.context_data)

            # Call LLM
            raw_response = await self._call_llm(formatted_prompt, request.max_tokens, request.temperature)

            if not raw_response:
                return LLMResponse(
                    success=False,
                    content="",
                    validation_errors=["No response from LLM"]
                )

            # Parse and validate response
            parsed_response = self._parse_llm_response(raw_response)
            validation_result = self._validate_response(parsed_response, request.analysis_type)

            return LLMResponse(
                success=validation_result["valid"],
                content=raw_response,
                structured_data=parsed_response if validation_result["valid"] else None,
                confidence=parsed_response.get("confidence_score", 0.0) if parsed_response else 0.0,
                validation_errors=validation_result.get("errors", []),
                raw_response=raw_response
            )

        except Exception as e:
            logger.error(f"Error processing LLM request: {e}")
            return LLMResponse(
                success=False,
                content="",
                validation_errors=[f"Processing error: {str(e)}"]
            )

    async def _call_llm(self, prompt: str, max_tokens: int, temperature: float) -> str:
        """Call the available LLM service"""
        try:
            if self.openai_client:
                return await self._call_openai(prompt, max_tokens, temperature)
            elif self.anthropic_client:
                return await self._call_anthropic(prompt, max_tokens, temperature)
            else:
                raise ValueError("No LLM client available")
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return ""

    async def _call_openai(self, prompt: str, max_tokens: int, temperature: float) -> str:
        """Call OpenAI API"""
        try:
            response = await self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert ISO 27001:2022 consultant. Provide accurate, detailed analysis based on the standard's requirements. Always respond with valid JSON as requested."
                    },
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=temperature
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise

    async def _call_anthropic(self, prompt: str, max_tokens: int, temperature: float) -> str:
        """Call Anthropic API"""
        try:
            response = await self.anthropic_client.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=max_tokens,
                temperature=temperature,
                system="You are an expert ISO 27001:2022 consultant. Provide accurate, detailed analysis based on the standard's requirements. Always respond with valid JSON as requested.",
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text
        except Exception as e:
            logger.error(f"Anthropic API error: {e}")
            raise

    def _parse_llm_response(self, response: str) -> Optional[Dict[str, Any]]:
        """Parse LLM response into structured data"""
        try:
            # Extract JSON from response (handle cases where there's extra text)
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                return json.loads(json_str)
            else:
                # Try parsing the entire response
                return json.loads(response)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response as JSON: {e}")
            logger.debug(f"Raw response: {response}")
            return None
        except Exception as e:
            logger.error(f"Error parsing LLM response: {e}")
            return None

    def _validate_response(self, response: Optional[Dict[str, Any]], analysis_type: AnalysisType) -> Dict[str, Any]:
        """Validate parsed response against schema"""
        if not response:
            return {"valid": False, "errors": ["Could not parse response as JSON"]}

        schema = self.validation_schemas.get(analysis_type, {})
        errors = []

        # Check required fields
        required_fields = schema.get("required_fields", [])
        for field in required_fields:
            if field not in response:
                errors.append(f"Missing required field: {field}")

        # Check field types
        field_types = schema.get("field_types", {})
        for field, expected_type in field_types.items():
            if field in response:
                actual_value = response[field]
                if not isinstance(actual_value, expected_type):
                    errors.append(f"Field {field} should be {expected_type.__name__}, got {type(actual_value).__name__}")

        # Check valid values
        valid_values = schema.get("valid_values", {})
        for field, valid_list in valid_values.items():
            if field in response:
                value = response[field]
                if isinstance(value, list):
                    # For list fields, check each item
                    for item in value:
                        if item not in valid_list:
                            errors.append(f"Invalid value in {field}: {item}")
                else:
                    # For single value fields
                    if value not in valid_list:
                        errors.append(f"Invalid value for {field}: {value}")

        return {
            "valid": len(errors) == 0,
            "errors": errors
        }

    async def classify_document_llm(self, document_text: str, document_info: Dict[str, Any]) -> Dict[str, Any]:
        """Use LLM to classify document with enhanced analysis"""
        context_data = {
            "filename": document_info.get("filename", "unknown"),
            "size": document_info.get("size", 0),
            "content_preview": document_text[:500] + "..." if len(document_text) > 500 else document_text,
            "document_content": document_text[:3000]  # First 3000 chars for analysis
        }

        request = LLMRequest(
            analysis_type=AnalysisType.DOCUMENT_CLASSIFICATION,
            prompt="",  # Will be filled by template
            context_data=context_data,
            expected_format="json",
            max_tokens=1000,
            temperature=0.2
        )

        response = await self.process_llm_request(request)

        if response.success and response.structured_data:
            return response.structured_data
        else:
            logger.error(f"LLM document classification failed: {response.validation_errors}")
            raise Exception(f"ISO document classification failed: {response.validation_errors}")

    async def assess_control_llm(
        self,
        control: Control,
        document_text: str,
        evidence_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Use LLM to assess control implementation with evidence"""

        # Prepare relevant excerpts from document
        relevant_excerpts = self._extract_relevant_excerpts(document_text, control, 5)

        context_data = {
            "control_id": control.id,
            "control_title": control.title,
            "control_description": control.description,
            "control_purpose": control.purpose,
            "relevant_excerpts": relevant_excerpts,
            "document_context": document_text[:2000],
            "evidence_found": evidence_data.get("evidence_found", []),
            "implementation_indicators": evidence_data.get("implementation_indicators", []),
            "gap_indicators": evidence_data.get("gap_indicators", [])
        }

        request = LLMRequest(
            analysis_type=AnalysisType.CONTROL_ASSESSMENT,
            prompt="",
            context_data=context_data,
            expected_format="json",
            max_tokens=1500,
            temperature=0.3
        )

        response = await self.process_llm_request(request)

        if response.success and response.structured_data:
            return response.structured_data
        else:
            logger.error(f"LLM control assessment failed: {response.validation_errors}")
            raise Exception(f"ISO control assessment failed: {response.validation_errors}")

    def _extract_relevant_excerpts(self, text: str, control: Control, max_excerpts: int = 5) -> List[str]:
        """Extract text excerpts most relevant to the control"""
        excerpts = []

        # Look for control-related keywords
        control_keywords = control.title.lower().split() + control.description.lower().split()[:10]

        # Split text into paragraphs
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]

        # Score paragraphs by relevance
        scored_paragraphs = []
        for para in paragraphs:
            if len(para) < 50:  # Skip very short paragraphs
                continue

            para_lower = para.lower()
            score = sum(1 for keyword in control_keywords if keyword in para_lower and len(keyword) > 3)

            if score > 0:
                scored_paragraphs.append((score, para))

        # Sort by score and take top excerpts
        scored_paragraphs.sort(key=lambda x: x[0], reverse=True)

        for score, para in scored_paragraphs[:max_excerpts]:
            # Trim long paragraphs
            if len(para) > 300:
                para = para[:300] + "..."
            excerpts.append(para)

        return excerpts


    async def classify_document(self, text: str, document_info: Dict[str, Any]) -> 'DocumentClassification':
        """Enhanced document classification using LLM"""
        try:
            # Use the existing classify_document_llm method
            result = await self.classify_document_llm(text, document_info)

            # Convert to DocumentClassification object
            from ..iso_knowledge.agent_response import DocumentClassification, DocumentType

            return DocumentClassification(
                document_type=DocumentType(result.get('document_type', 'other')),
                confidence_score=float(result.get('confidence_score', 0.5)),
                iso_relevance=result.get('iso_relevance', 'medium'),
                applicable_categories=result.get('applicable_categories', []),
                primary_focus_areas=result.get('primary_focus_areas', [])
            )

        except Exception as e:
            logger.error(f"Enhanced LLM document classification failed: {e}")
            raise Exception(f"ISO document classification failed - cannot provide non-ISO response: {e}")

    async def generate_executive_summary(self, compliance_overview: 'ComplianceOverview', gap_analysis: 'GapAnalysis', recommendations: List['Recommendation']) -> str:
        """Generate executive summary using LLM"""
        try:
            summary_request = LLMRequest(
                analysis_type=AnalysisType.EXECUTIVE_SUMMARY,
                prompt="",  # Will be filled by template
                context_data={
                    "overall_score": compliance_overview.overall_score if hasattr(compliance_overview, 'overall_score') else 0.0,
                    "total_controls": compliance_overview.total_controls_assessed if hasattr(compliance_overview, 'total_controls_assessed') else 0,
                    "document_classification": "Information Security Policy", # Default
                    "key_findings": f"Overall maturity: {compliance_overview.overall_maturity if hasattr(compliance_overview, 'overall_maturity') else 'unknown'}",
                    "gap_analysis": str(gap_analysis),
                    "recommendations": [str(rec) for rec in recommendations[:5]]
                },
                expected_format="text",
                max_tokens=1000,
                temperature=0.3
            )

            response = await self.process_llm_request(summary_request)
            if response.success:
                # response.content is already the text string we need
                return response.content if response.content else "Executive summary generation completed but no content returned."
            else:
                raise Exception(f"ISO executive summary generation failed: {response.validation_errors}")

        except Exception as e:
            logger.error(f"LLM executive summary generation failed: {e}")
            raise Exception(f"ISO executive summary generation failed - cannot provide non-ISO response: {e}")


    async def generate_consultation_response(self, question: str, relevant_controls: List['Control'], relevant_clauses: List['Clause']) -> str:
        """Generate expert consultation response using LLM"""
        try:
            context = "Relevant ISO 27001:2022 Controls:\\n"
            for control in relevant_controls:
                context += f"- {control.id}: {control.title} - {control.purpose}\\n"

            context += "\\nRelevant Management Clauses:\\n"
            for clause in relevant_clauses:
                context += f"- Clause {clause.id}: {clause.title} - {clause.purpose}\\n"

            consultation_request = LLMRequest(
                analysis_type=AnalysisType.CONSULTATION,
                prompt="",  # Will be filled by template
                context_data={
                    "question": question,
                    "relevant_controls": [{"id": ctrl.id, "title": ctrl.title, "purpose": ctrl.purpose} for ctrl in relevant_controls],
                    "relevant_clauses": [{"id": clause.id, "title": clause.title, "purpose": clause.purpose} for clause in relevant_clauses]
                },
                expected_format="text",
                max_tokens=1500,
                temperature=0.3
            )

            response = await self.process_llm_request(consultation_request)
            if response.success:
                # response.content is already the text string we need
                return response.content if response.content else "Consultation response generated but no content returned."
            else:
                raise Exception(f"ISO consultation response generation failed: {response.validation_errors}")

        except Exception as e:
            logger.error(f"LLM consultation response failed: {e}")
            raise Exception(f"ISO consultation response failed - cannot provide non-ISO response: {e}")

