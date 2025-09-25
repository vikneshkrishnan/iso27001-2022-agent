"""
ISO 27001:2022 Response Validation System
Ensures all agent responses are strictly ISO-compliant without hallucination
"""

import re
import logging
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass
from enum import Enum

from ..iso_knowledge.annex_a_controls import get_all_controls, get_control_by_id
from ..iso_knowledge.management_clauses import get_all_clauses, get_clause_by_id, get_all_requirements
from ..iso_knowledge.agent_response import ComplianceStatus, DocumentType, Priority

logger = logging.getLogger(__name__)


class ValidationLevel(Enum):
    STRICT = "strict"  # No content allowed outside ISO standard
    GUIDED = "guided"  # ISO-focused with minimal context
    COMPREHENSIVE = "comprehensive"  # Full ISO analysis with explanations


@dataclass
class ValidationResult:
    """Result of ISO compliance validation"""
    is_valid: bool
    confidence_score: float
    validation_errors: List[str]
    iso_references: List[str]
    non_iso_content: List[str]
    compliance_level: str


class ISOResponseValidator:
    """Validates all agent responses for ISO 27001:2022 compliance"""

    def __init__(self, validation_level: ValidationLevel = ValidationLevel.STRICT):
        self.validation_level = validation_level

        # Load ISO knowledge bases
        self.controls = get_all_controls()
        self.clauses = get_all_clauses()
        self.requirements = dict(get_all_requirements())

        # Build ISO reference database
        self.iso_concepts = self._build_iso_concepts()
        self.iso_terminology = self._build_iso_terminology()

        logger.info(f"ISO Validator initialized with {len(self.iso_concepts)} concepts and {len(self.iso_terminology)} terms")

    def _build_iso_concepts(self) -> Set[str]:
        """Build database of valid ISO 27001:2022 concepts"""
        concepts = set()

        # Add control concepts
        for control in self.controls.values():
            concepts.add(control.title.lower())
            concepts.add(control.category.lower())
            concepts.update(word.lower() for word in control.description.split() if len(word) > 3)
            concepts.update(word.lower() for word in control.purpose.split() if len(word) > 3)

        # Add clause concepts
        for clause in self.clauses.values():
            concepts.add(clause.title.lower())
            concepts.update(word.lower() for word in clause.purpose.split() if len(word) > 3)

        # Add requirement concepts
        for requirement in self.requirements.values():
            concepts.update(word.lower() for word in requirement.description.split() if len(word) > 3)

        # Add standard ISO terminology
        iso_terms = {
            'information security', 'confidentiality', 'integrity', 'availability',
            'risk assessment', 'risk management', 'vulnerability', 'threat',
            'asset', 'control', 'policy', 'procedure', 'incident',
            'audit', 'monitoring', 'review', 'improvement',
            'management system', 'documentation', 'competence',
            'awareness', 'communication', 'operation', 'performance',
            'nonconformity', 'corrective action', 'continual improvement'
        }
        concepts.update(iso_terms)

        return concepts

    def _build_iso_terminology(self) -> Dict[str, List[str]]:
        """Build ISO terminology mapping"""
        return {
            'security_controls': [control.id for control in self.controls.values()],
            'management_clauses': [clause.id for clause in self.clauses.values()],
            'compliance_statuses': ['compliant', 'partially_compliant', 'non_compliant', 'not_applicable'],
            'document_types': ['policy', 'procedure', 'standard', 'guideline', 'risk_assessment'],
            'categories': ['organizational', 'people', 'physical', 'technological'],
            'priorities': ['critical', 'high', 'medium', 'low']
        }

    def validate_response(self, response_content: str, response_type: str) -> ValidationResult:
        """Validate any agent response for ISO compliance"""
        errors = []
        iso_references = []
        non_iso_content = []

        control_refs = self._extract_control_references(response_content)
        clause_refs = self._extract_clause_references(response_content)

        iso_references.extend(control_refs)
        iso_references.extend(clause_refs)

        iso_compliance = self._validate_iso_content(response_content)
        hallucination_check = self._check_for_hallucination(response_content)
        if self.validation_level == ValidationLevel.STRICT:
            non_iso_check = self._strict_iso_validation(response_content)
        else:
            non_iso_check = self._guided_iso_validation(response_content)

        # Compile validation results
        if not iso_compliance['valid']:
            errors.extend(iso_compliance['errors'])

        if hallucination_check['has_hallucination']:
            errors.extend(hallucination_check['issues'])

        if non_iso_check['has_non_iso']:
            non_iso_content.extend(non_iso_check['content'])
            if self.validation_level == ValidationLevel.STRICT:
                errors.extend([f"Non-ISO content detected: {content}" for content in non_iso_check['content']])

        # Calculate confidence score
        confidence = self._calculate_confidence(
            len(iso_references), len(errors), len(non_iso_content), len(response_content)
        )

        # Determine compliance level
        compliance_level = self._determine_compliance_level(len(errors), confidence)

        return ValidationResult(
            is_valid=len(errors) == 0,
            confidence_score=confidence,
            validation_errors=errors,
            iso_references=iso_references,
            non_iso_content=non_iso_content,
            compliance_level=compliance_level
        )

    def _extract_control_references(self, content: str) -> List[str]:
        """Extract valid ISO control references"""
        control_pattern = r'\b([A-Z]\.?\d{1,2}\.?\d{0,2})\b'
        potential_refs = re.findall(control_pattern, content)

        valid_refs = []
        for ref in potential_refs:
            # Normalize reference format
            normalized = ref.replace('.', '.')
            if get_control_by_id(normalized):
                valid_refs.append(f"Control {normalized}")

        return valid_refs

    def _extract_clause_references(self, content: str) -> List[str]:
        """Extract valid ISO clause references"""
        clause_pattern = r'\b(clause\s+)?(\d{1,2}\.?\d{0,2}\.?\d{0,2})\b'
        potential_refs = re.findall(clause_pattern, content.lower())

        valid_refs = []
        for _, ref in potential_refs:
            if get_clause_by_id(ref):
                valid_refs.append(f"Clause {ref}")

        return valid_refs

    def _validate_iso_content(self, content: str) -> Dict[str, Any]:
        """Validate content contains only ISO-compliant information"""
        errors = []
        content_lower = content.lower()

        # Check for essential ISO concepts
        iso_concept_count = sum(1 for concept in self.iso_concepts if concept in content_lower)

        if iso_concept_count < 3:
            errors.append("Insufficient ISO 27001:2022 concept coverage")

        # Check terminology consistency
        for term_type, valid_terms in self.iso_terminology.items():
            if any(term in content_lower for term in valid_terms):
                # Found usage, validate it's correct
                continue

        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'concept_coverage': iso_concept_count
        }

    def _check_for_hallucination(self, content: str) -> Dict[str, Any]:
        """Check for potential hallucinated content"""
        issues = []

        # Check for invalid control IDs
        control_pattern = r'\b([A-Z]\.?\d{1,2}\.?\d{0,2})\b'
        potential_controls = re.findall(control_pattern, content)

        for control_id in potential_controls:
            normalized = control_id.replace('.', '.')
            if not get_control_by_id(normalized):
                issues.append(f"Invalid control reference: {control_id}")

        # Check for invalid clause references
        clause_pattern = r'\bclause\s+(\d{1,2}\.?\d{0,2}\.?\d{0,2})\b'
        potential_clauses = re.findall(clause_pattern, content.lower())

        for clause_id in potential_clauses:
            if not get_clause_by_id(clause_id):
                issues.append(f"Invalid clause reference: {clause_id}")

        # Check for made-up compliance statuses
        content_lower = content.lower()
        valid_statuses = ['compliant', 'partially_compliant', 'non_compliant', 'not_applicable', 'insufficient_information']

        compliance_patterns = r'\b(fully\s+compliant|completely\s+compliant|mostly\s+compliant|somewhat\s+compliant)\b'
        invalid_statuses = re.findall(compliance_patterns, content_lower)

        for status in invalid_statuses:
            if status not in valid_statuses:
                issues.append(f"Non-standard compliance status: {status}")

        return {
            'has_hallucination': len(issues) > 0,
            'issues': issues
        }

    def _strict_iso_validation(self, content: str) -> Dict[str, Any]:
        """Strict validation - only ISO content allowed"""
        non_iso_content = []

        # Define prohibited non-ISO content patterns
        prohibited_patterns = [
            r'\b(in my opinion|i think|i believe|personally|generally speaking)\b',
            r'\b(best practice|industry standard|common approach|typical implementation)\b',
            r'\b(might|could|should|would suggest|it is recommended)\b',
            r'\b(for example|such as|like|similar to)\b(?!\s+(?:control|clause|requirement))',
            r'\b(external|third-party|vendor|commercial|proprietary)\b(?!\s+(?:audit|assessment|service))'
        ]

        content_lower = content.lower()

        for pattern in prohibited_patterns:
            matches = re.findall(pattern, content_lower)
            non_iso_content.extend(matches)

        return {
            'has_non_iso': len(non_iso_content) > 0,
            'content': non_iso_content
        }

    def _guided_iso_validation(self, content: str) -> Dict[str, Any]:
        """Guided validation - ISO-focused with minimal context"""
        non_iso_content = []

        # Less strict - allow some contextual information
        prohibited_patterns = [
            r'\b(in my opinion|i believe|personally)\b',
            r'\b(best practice|industry standard)(?!\s+(?:as defined|according to|per))\b',
        ]

        content_lower = content.lower()

        for pattern in prohibited_patterns:
            matches = re.findall(pattern, content_lower)
            non_iso_content.extend(matches)

        return {
            'has_non_iso': len(non_iso_content) > 0,
            'content': non_iso_content
        }

    def _calculate_confidence(self, iso_ref_count: int, error_count: int,
                            non_iso_count: int, content_length: int) -> float:
        """Calculate confidence score for validation"""
        if content_length == 0:
            return 0.0

        # Base confidence from ISO references
        ref_score = min(iso_ref_count * 0.2, 0.8)

        # Penalty for errors
        error_penalty = error_count * 0.3

        # Penalty for non-ISO content (based on validation level)
        if self.validation_level == ValidationLevel.STRICT:
            non_iso_penalty = non_iso_count * 0.4
        else:
            non_iso_penalty = non_iso_count * 0.2

        # Content adequacy bonus
        length_bonus = min(content_length / 500, 0.2) if content_length > 100 else 0

        confidence = ref_score + length_bonus - error_penalty - non_iso_penalty

        return max(min(confidence, 1.0), 0.0)

    def _determine_compliance_level(self, error_count: int, confidence: float) -> str:
        """Determine ISO compliance level"""
        if error_count == 0 and confidence >= 0.9:
            return "fully_compliant"
        elif error_count <= 1 and confidence >= 0.7:
            return "mostly_compliant"
        elif error_count <= 3 and confidence >= 0.5:
            return "partially_compliant"
        else:
            return "non_compliant"

    def validate_document_classification(self, classification_data: Dict[str, Any]) -> ValidationResult:
        """Validate document classification response"""
        return self._validate_structured_response(
            classification_data,
            'document_classification',
            required_fields=['document_type', 'iso_relevance', 'confidence_score'],
            valid_values={
                'document_type': ['policy', 'procedure', 'standard', 'guideline', 'risk_assessment', 'audit_report'],
                'iso_relevance': ['high', 'medium', 'low']
            }
        )

    def validate_control_assessment(self, assessment_data: Dict[str, Any]) -> ValidationResult:
        """Validate control assessment response"""
        return self._validate_structured_response(
            assessment_data,
            'control_assessment',
            required_fields=['compliance_status', 'confidence_level'],
            valid_values={
                'compliance_status': ['compliant', 'partially_compliant', 'non_compliant', 'not_applicable']
            }
        )

    def _validate_structured_response(self, data: Dict[str, Any], response_type: str,
                                    required_fields: List[str], valid_values: Dict[str, List[str]]) -> ValidationResult:
        """Validate structured response data"""
        errors = []

        # Check required fields
        for field in required_fields:
            if field not in data:
                errors.append(f"Missing required field: {field}")

        # Check valid values
        for field, valid_list in valid_values.items():
            if field in data and data[field] not in valid_list:
                errors.append(f"Invalid value for {field}: {data[field]}")

        # Validate any text content
        text_content = ""
        for key, value in data.items():
            if isinstance(value, str):
                text_content += f" {value}"
            elif isinstance(value, list):
                text_content += f" {' '.join(str(v) for v in value)}"

        text_validation = self.validate_response(text_content, response_type)

        return ValidationResult(
            is_valid=len(errors) == 0 and text_validation.is_valid,
            confidence_score=text_validation.confidence_score,
            validation_errors=errors + text_validation.validation_errors,
            iso_references=text_validation.iso_references,
            non_iso_content=text_validation.non_iso_content,
            compliance_level=text_validation.compliance_level
        )

    def enforce_iso_compliance(self, response_data: Any, response_type: str) -> Any:
        """Enforce ISO compliance by rejecting non-compliant responses"""
        if isinstance(response_data, dict):
            validation = self._validate_structured_response(response_data, response_type, [], {})
        else:
            validation = self.validate_response(str(response_data), response_type)

        if not validation.is_valid:
            error_msg = f"Response validation failed for {response_type}: {'; '.join(validation.validation_errors)}"
            logger.error(error_msg)
            raise Exception(f"ISO compliance validation failed: {error_msg}")

        if validation.confidence_score < 0.6:
            error_msg = f"Response confidence too low ({validation.confidence_score:.2f}) for {response_type}"
            logger.error(error_msg)
            raise Exception(f"ISO compliance confidence insufficient: {error_msg}")

        return response_data