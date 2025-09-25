"""
Intelligent Control Assessment Engine for ISO 27001:2022
Provides sophisticated control assessment with evidence extraction and gap analysis
"""

import logging
import re
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
try:
    import numpy as np
except ImportError:
    np = None

from .semantic_analyzer import ISOSemanticAnalyzer, TextSegment, Evidence, ConceptMatch
from ..iso_knowledge.annex_a_controls import Control, get_control_by_id
from ..iso_knowledge.management_clauses import Requirement
from ..iso_knowledge.agent_response import (
    ControlAssessment, ClauseAssessment, ComplianceStatus, Priority
)

# Import knowledge crawler and cache
try:
    from .iso_knowledge_crawler import ISOKnowledgeCrawler, ControlKnowledge
    from .knowledge_cache import KnowledgeCache
    CRAWLER_AVAILABLE = True
except ImportError:
    CRAWLER_AVAILABLE = False
    logger.info("ISO Knowledge Crawler not available")

logger = logging.getLogger(__name__)


@dataclass
class ControlEvidence:
    """Detailed evidence for a control assessment"""
    control_id: str
    direct_evidence: List[Evidence]
    contextual_evidence: List[Evidence]
    implementation_indicators: List[str]
    gap_indicators: List[str]
    compliance_score: float
    confidence_level: float


@dataclass
class GapDetail:
    """Detailed gap information"""
    gap_type: str  # missing, incomplete, inadequate
    description: str
    impact_level: str  # high, medium, low
    specific_requirements: List[str]
    recommended_actions: List[str]


class IntelligentControlAssessor:
    """Advanced control assessment using semantic analysis"""

    def __init__(self, enable_crawler: bool = True):
        self.semantic_analyzer = ISOSemanticAnalyzer()
        self.control_mappings = self._build_control_mappings()
        self.implementation_patterns = self._build_implementation_patterns()
        self.gap_analysis_rules = self._build_gap_analysis_rules()

        # Initialize crawler and cache if available
        self.crawler = None
        self.knowledge_cache = None
        if enable_crawler and CRAWLER_AVAILABLE:
            try:
                self.crawler = ISOKnowledgeCrawler()
                self.knowledge_cache = KnowledgeCache()
                logger.info("ISO Knowledge Crawler initialized")
            except Exception as e:
                logger.warning(f"Could not initialize crawler: {e}")

    def _build_control_mappings(self) -> Dict[str, Dict[str, Any]]:
        """Build mappings between controls and their assessment criteria"""
        return {
            # Organizational Controls
            '5.1': {
                'keywords': [
                    'information security policy', 'security policy', 'policy framework',
                    'policy statement', 'security governance', 'policy approval',
                    'policy review', 'policy communication', 'policy acknowledgment'
                ],
                'implementation_indicators': [
                    'approved by', 'signed by', 'reviewed annually', 'communicated to',
                    'acknowledged by', 'published', 'distributed', 'accessible',
                    'version control', 'policy register'
                ],
                'required_elements': [
                    'management approval', 'regular review', 'communication process',
                    'scope definition', 'objectives statement', 'compliance commitment'
                ],
                'evidence_types': ['policy_document', 'approval_record', 'communication_record']
            },

            '5.2': {
                'keywords': [
                    'roles and responsibilities', 'security roles', 'accountability',
                    'responsibility assignment', 'role definition', 'authority delegation',
                    'responsibility matrix', 'job descriptions'
                ],
                'implementation_indicators': [
                    'defined roles', 'assigned responsibilities', 'clear accountability',
                    'documented in', 'role descriptions', 'authority levels',
                    'reporting structure', 'escalation procedures'
                ],
                'required_elements': [
                    'role definitions', 'responsibility assignments', 'authority levels',
                    'communication of roles', 'regular review'
                ],
                'evidence_types': ['role_matrix', 'job_descriptions', 'org_chart']
            },

            '6.1': {
                'keywords': [
                    'background verification', 'screening', 'background check',
                    'employment verification', 'reference check', 'security clearance',
                    'vetting process', 'pre-employment screening'
                ],
                'implementation_indicators': [
                    'background checks conducted', 'references verified',
                    'screening process', 'clearance obtained', 'verification completed',
                    'documented results', 'approval granted'
                ],
                'required_elements': [
                    'screening procedures', 'verification process', 'documentation',
                    'legal compliance', 'proportional to risk'
                ],
                'evidence_types': ['screening_procedure', 'verification_records']
            },

            '8.2': {
                'keywords': [
                    'privileged access', 'administrative access', 'elevated privileges',
                    'privileged accounts', 'privileged users', 'access management',
                    'privileged access management', 'pam', 'admin rights'
                ],
                'implementation_indicators': [
                    'restricted access', 'managed separately', 'regular review',
                    'monitoring implemented', 'approval required', 'documented access',
                    'access logs', 'privileged session monitoring'
                ],
                'required_elements': [
                    'access restriction', 'separate management', 'regular review',
                    'monitoring capabilities', 'approval process'
                ],
                'evidence_types': ['access_procedure', 'review_records', 'monitoring_logs']
            },

            # Access Control Controls
            'A.9.1.1': {
                'keywords': [
                    'access control policy', 'access management', 'user access',
                    'access rights', 'access control', 'authorization policy',
                    'access governance', 'access control framework'
                ],
                'implementation_indicators': [
                    'documented policy', 'approved access', 'access control procedures',
                    'user access management', 'access rights defined', 'regular review'
                ],
                'required_elements': ['policy document', 'approval process', 'review procedure'],
                'evidence_types': ['policy_document', 'procedures', 'approval_records']
            },

            'A.9.1.2': {
                'keywords': [
                    'network access', 'remote access', 'network controls',
                    'network security', 'vpn', 'firewall', 'network segmentation',
                    'network access control', 'network perimeter'
                ],
                'implementation_indicators': [
                    'controlled access', 'secured networks', 'access restrictions',
                    'network monitoring', 'firewall rules', 'vpn access'
                ],
                'required_elements': ['network controls', 'access restrictions', 'monitoring'],
                'evidence_types': ['network_configs', 'access_logs', 'security_controls']
            },

            'A.9.2.1': {
                'keywords': [
                    'user registration', 'account management', 'user provisioning',
                    'user lifecycle', 'account creation', 'user enrollment',
                    'identity management', 'user registration process'
                ],
                'implementation_indicators': [
                    'formal registration', 'approval process', 'documented procedures',
                    'user verification', 'access approval', 'account lifecycle'
                ],
                'required_elements': ['registration process', 'approval workflow', 'documentation'],
                'evidence_types': ['procedures', 'approval_records', 'user_records']
            },

            'A.9.2.2': {
                'keywords': [
                    'privileged access management', 'privileged accounts', 'admin access',
                    'elevated privileges', 'administrative rights', 'privileged users',
                    'pam', 'privileged access control', 'privileged session management'
                ],
                'implementation_indicators': [
                    'restricted access', 'separate management', 'monitoring enabled',
                    'approval required', 'session recording', 'regular review'
                ],
                'required_elements': ['access restriction', 'monitoring', 'approval process'],
                'evidence_types': ['pam_system', 'access_logs', 'approval_records']
            },

            'A.9.4.2': {
                'keywords': [
                    'secure logon', 'authentication', 'login security',
                    'multi-factor authentication', 'mfa', '2fa', 'two-factor',
                    'strong authentication', 'authentication factors', 'biometric'
                ],
                'implementation_indicators': [
                    'mfa enabled', 'two-factor authentication', 'strong passwords',
                    'biometric authentication', 'token-based', 'authentication required'
                ],
                'required_elements': ['authentication method', 'password policy', 'mfa implementation'],
                'evidence_types': ['authentication_system', 'password_policy', 'mfa_config']
            },

            # Cryptographic Controls
            'A.10.1.1': {
                'keywords': [
                    'cryptographic policy', 'encryption policy', 'cryptographic controls',
                    'encryption standards', 'crypto policy', 'key management policy',
                    'cryptographic requirements', 'encryption requirements'
                ],
                'implementation_indicators': [
                    'encryption standards', 'approved algorithms', 'key management',
                    'cryptographic procedures', 'encryption implementation'
                ],
                'required_elements': ['encryption policy', 'approved algorithms', 'key management'],
                'evidence_types': ['crypto_policy', 'standards_doc', 'procedures']
            },

            'A.10.1.2': {
                'keywords': [
                    'key management', 'cryptographic keys', 'key lifecycle',
                    'key generation', 'key distribution', 'key storage',
                    'key rotation', 'key escrow', 'hsm', 'hardware security module'
                ],
                'implementation_indicators': [
                    'secure key storage', 'key rotation', 'hsm implementation',
                    'key lifecycle management', 'secure key generation'
                ],
                'required_elements': ['key storage', 'key rotation', 'key generation'],
                'evidence_types': ['key_management_system', 'hsm_config', 'key_procedures']
            },

            # Data Protection Controls
            'A.8.2.1': {
                'keywords': [
                    'data classification', 'information classification', 'data labeling',
                    'classification scheme', 'data categories', 'sensitivity levels',
                    'confidentiality levels', 'data handling', 'classification policy'
                ],
                'implementation_indicators': [
                    'classification scheme', 'labeled data', 'handling procedures',
                    'classification applied', 'data categories defined'
                ],
                'required_elements': ['classification scheme', 'labeling process', 'handling procedures'],
                'evidence_types': ['classification_policy', 'labeling_examples', 'procedures']
            },

            'A.8.2.3': {
                'keywords': [
                    'data handling', 'information handling', 'data protection',
                    'secure handling', 'data processing', 'information protection',
                    'data security measures', 'handling procedures'
                ],
                'implementation_indicators': [
                    'handling procedures', 'protection measures', 'secure processing',
                    'data protection controls', 'handling guidelines'
                ],
                'required_elements': ['handling procedures', 'protection measures', 'guidelines'],
                'evidence_types': ['procedures', 'guidelines', 'protection_measures']
            },

            # System Security Controls
            'A.12.6.1': {
                'keywords': [
                    'vulnerability management', 'vulnerability scanning', 'patch management',
                    'security vulnerabilities', 'vulnerability assessment', 'security patches',
                    'vulnerability remediation', 'security updates', 'vulnerability testing'
                ],
                'implementation_indicators': [
                    'regular scanning', 'patch deployment', 'vulnerability tracking',
                    'remediation process', 'security updates applied'
                ],
                'required_elements': ['scanning process', 'patch management', 'remediation'],
                'evidence_types': ['scan_reports', 'patch_logs', 'remediation_records']
            },

            # Incident Management Controls
            'A.16.1.1': {
                'keywords': [
                    'incident management', 'security incident', 'incident response',
                    'incident handling', 'incident procedures', 'security events',
                    'incident reporting', 'incident process', 'security incident management'
                ],
                'implementation_indicators': [
                    'incident procedures', 'response team', 'incident reporting',
                    'documented process', 'incident tracking', 'response procedures'
                ],
                'required_elements': ['incident procedures', 'response team', 'reporting process'],
                'evidence_types': ['incident_procedures', 'response_plan', 'incident_records']
            },

            'A.16.1.2': {
                'keywords': [
                    'incident reporting', 'security events', 'event reporting',
                    'incident notification', '4 hours', 'immediate reporting',
                    'escalation', 'incident escalation', 'reporting timeframes'
                ],
                'implementation_indicators': [
                    'reporting timeframes', '4 hour reporting', 'immediate notification',
                    'escalation procedures', 'reporting channels', 'timely reporting'
                ],
                'required_elements': ['reporting timeframes', 'escalation process', 'notification'],
                'evidence_types': ['reporting_procedures', 'escalation_matrix', 'incident_logs']
            },

            'A.16.1.4': {
                'keywords': [
                    'incident assessment', 'incident analysis', 'forensic analysis',
                    'evidence collection', 'incident investigation', 'forensics',
                    'digital forensics', 'incident evaluation', 'evidence preservation'
                ],
                'implementation_indicators': [
                    'forensic procedures', 'evidence collection', 'investigation process',
                    'analysis capabilities', 'preservation methods'
                ],
                'required_elements': ['investigation procedures', 'evidence collection', 'analysis'],
                'evidence_types': ['forensic_procedures', 'investigation_records', 'evidence_logs']
            },

            # Physical Security Controls
            'A.11.1.1': {
                'keywords': [
                    'physical security perimeter', 'physical barriers', 'security perimeter',
                    'physical boundaries', 'secured areas', 'physical protection',
                    'perimeter controls', 'physical access control'
                ],
                'implementation_indicators': [
                    'physical barriers', 'controlled areas', 'perimeter security',
                    'access controls', 'secured boundaries'
                ],
                'required_elements': ['physical barriers', 'access controls', 'perimeter definition'],
                'evidence_types': ['security_measures', 'access_controls', 'perimeter_map']
            },

            'A.11.1.2': {
                'keywords': [
                    'physical entry controls', 'badge access', 'access cards',
                    'biometric', 'physical access', 'entry controls',
                    'access control systems', 'physical authentication', 'card readers'
                ],
                'implementation_indicators': [
                    'badge access', 'biometric controls', 'card readers',
                    'access logging', 'physical authentication', 'controlled entry'
                ],
                'required_elements': ['access control system', 'authentication method', 'access logging'],
                'evidence_types': ['access_system', 'access_logs', 'authentication_records']
            },

            'A.11.2.3': {
                'keywords': [
                    'equipment protection', 'environmental controls', 'power protection',
                    'ups', 'backup power', 'environmental monitoring',
                    'temperature control', 'humidity control', '24/7 monitoring'
                ],
                'implementation_indicators': [
                    'environmental monitoring', 'backup power', 'temperature control',
                    '24/7 monitoring', 'power protection', 'environmental controls'
                ],
                'required_elements': ['environmental monitoring', 'power backup', 'protection measures'],
                'evidence_types': ['monitoring_system', 'power_systems', 'environmental_logs']
            },

            # Network Security Controls
            'A.13.1.1': {
                'keywords': [
                    'network controls', 'network security', 'network segregation',
                    'network segmentation', 'firewall', 'network isolation',
                    'network protection', 'network access control'
                ],
                'implementation_indicators': [
                    'network segmentation', 'firewall rules', 'network controls',
                    'access restrictions', 'network monitoring'
                ],
                'required_elements': ['network segmentation', 'access controls', 'monitoring'],
                'evidence_types': ['network_config', 'firewall_rules', 'network_diagrams']
            },

            'A.13.2.1': {
                'keywords': [
                    'information transfer', 'data transmission', 'secure transfer',
                    'encryption in transit', 'tls', 'ssl', 'tls 1.3',
                    'secure communications', 'transmission security', 'data in transit'
                ],
                'implementation_indicators': [
                    'encryption in transit', 'tls implementation', 'secure protocols',
                    'encrypted communications', 'tls 1.3', 'secure transmission'
                ],
                'required_elements': ['encryption protocols', 'secure transmission', 'protocol standards'],
                'evidence_types': ['encryption_config', 'protocol_settings', 'transmission_logs']
            },

            # Data Protection at Rest
            'A.8.2.2': {
                'keywords': [
                    'data at rest', 'encryption at rest', 'stored data encryption',
                    'aes-256', 'data encryption', 'storage encryption',
                    'encrypted storage', 'data protection at rest', 'file encryption'
                ],
                'implementation_indicators': [
                    'aes-256 encryption', 'encrypted storage', 'data encryption',
                    'storage protection', 'encryption at rest', 'encrypted files'
                ],
                'required_elements': ['encryption implementation', 'key management', 'storage protection'],
                'evidence_types': ['encryption_config', 'storage_settings', 'encryption_logs']
            }
        }

    def _build_implementation_patterns(self) -> Dict[str, List[str]]:
        """Build patterns that indicate implementation levels"""
        return {
            'fully_implemented': [
                r'fully implemented', r'completely deployed', r'comprehensive coverage',
                r'established and maintained', r'regularly monitored', r'continuous monitoring',
                r'mature process', r'well-established', r'robust implementation',
                r'effective controls in place', r'comprehensive procedures'
            ],
            'partially_implemented': [
                r'partially implemented', r'in development', r'being established',
                r'initial implementation', r'pilot phase', r'limited coverage',
                r'basic controls', r'preliminary procedures', r'developing process'
            ],
            'planned_implementation': [
                r'planned for', r'scheduled for', r'to be implemented',
                r'future implementation', r'under consideration', r'being planned',
                r'roadmap includes', r'next phase'
            ],
            'not_implemented': [
                r'not implemented', r'no coverage', r'absent',
                r'not addressed', r'lacking', r'missing controls',
                r'gaps identified', r'requires implementation'
            ]
        }

    async def enhance_control_with_crawled_knowledge(self, control_id: str) -> Dict[str, Any]:
        """Enhance control mapping with crawled knowledge"""
        if not self.crawler or not self.knowledge_cache:
            return self.control_mappings.get(control_id, {})

        try:
            # Check cache first
            cached_knowledge = self.knowledge_cache.get_control_details(control_id)

            if not cached_knowledge:
                # Crawl fresh knowledge with error handling
                logger.info(f"Crawling knowledge for control {control_id}")
                try:
                    control_knowledge = await self.crawler.crawl_control_details(control_id)
                except Exception as e:
                    logger.warning(f"Knowledge crawling failed for control {control_id}: {e}")
                    control_knowledge = None

                if control_knowledge:
                    # Store in cache
                    self.knowledge_cache.store_control_knowledge(
                        control_id,
                        {
                            "title": control_knowledge.title,
                            "description": control_knowledge.description,
                            "implementation_guidance": control_knowledge.implementation_guidance,
                            "best_practices": control_knowledge.best_practices,
                            "common_gaps": control_knowledge.common_gaps,
                            "nist_mappings": control_knowledge.nist_mappings,
                            "industry_examples": control_knowledge.industry_examples,
                            "regulatory_mappings": control_knowledge.regulatory_mappings,
                            "confidence_score": control_knowledge.confidence_score
                        },
                        control_knowledge.source_url
                    )
                    cached_knowledge = self.knowledge_cache.get_control_details(control_id)
                else:
                    # Crawling failed, use basic fallback assessment
                    logger.info(f"Using fallback assessment for control {control_id} (crawling unavailable)")

            if cached_knowledge:
                # Enhance existing mapping with crawled knowledge
                enhanced_mapping = self.control_mappings.get(control_id, {}).copy()

                # Add crawled keywords
                if cached_knowledge.get("best_practices"):
                    enhanced_mapping.setdefault("keywords", []).extend(
                        [bp.lower() for bp in cached_knowledge["best_practices"][:5]]
                    )

                # Add implementation indicators
                if cached_knowledge.get("implementation_guidance"):
                    enhanced_mapping.setdefault("implementation_indicators", []).extend(
                        cached_knowledge["implementation_guidance"][:5]
                    )

                # Add common gaps as required elements
                if cached_knowledge.get("common_gaps"):
                    enhanced_mapping.setdefault("required_elements", []).extend(
                        cached_knowledge["common_gaps"][:3]
                    )

                # Add NIST mappings
                if cached_knowledge.get("nist_mappings"):
                    enhanced_mapping["nist_mappings"] = cached_knowledge["nist_mappings"]

                # Add industry examples
                if cached_knowledge.get("industry_examples"):
                    enhanced_mapping["industry_examples"] = cached_knowledge["industry_examples"]

                # Remove duplicates
                for key in ["keywords", "implementation_indicators", "required_elements"]:
                    if key in enhanced_mapping:
                        enhanced_mapping[key] = list(set(enhanced_mapping[key]))

                return enhanced_mapping

        except Exception as e:
            logger.error(f"Error enhancing control {control_id} with crawled knowledge: {e}")

        return self.control_mappings.get(control_id, {})

    def _build_gap_analysis_rules(self) -> Dict[str, Dict[str, Any]]:
        """Build rules for gap analysis"""
        return {
            'critical_gaps': {
                'patterns': [
                    r'no (policy|procedure|control|process)',
                    r'not implemented',
                    r'significant gaps?',
                    r'major deficiency',
                    r'critical (weakness|gap|issue)'
                ],
                'impact': 'high',
                'priority': 'critical'
            },
            'moderate_gaps': {
                'patterns': [
                    r'partially implemented',
                    r'limited coverage',
                    r'needs improvement',
                    r'should be enhanced',
                    r'minor gaps?'
                ],
                'impact': 'medium',
                'priority': 'high'
            },
            'minor_gaps': {
                'patterns': [
                    r'could be improved',
                    r'enhancement opportunity',
                    r'minor improvement',
                    r'optimization needed'
                ],
                'impact': 'low',
                'priority': 'medium'
            }
        }

    async def assess_control_comprehensive(
        self,
        control: Control,
        document_text: str,
        semantic_features: Dict[str, Any],
        segments: List[TextSegment]
    ) -> ControlAssessment:
        """Perform comprehensive assessment of a single control"""

        # Get control-specific mapping
        control_mapping = self.control_mappings.get(control.id, {})

        # Try to enhance with crawled knowledge if crawler is available
        if self.crawler and self.knowledge_cache:
            try:
                enhanced_mapping = await self.enhance_control_with_crawled_knowledge(control.id)
                if enhanced_mapping:
                    control_mapping = enhanced_mapping
            except Exception as e:
                logger.warning(f"Could not enhance control {control.id}: {e}")

        # Extract evidence for this control
        evidence = await self._extract_control_evidence(
            control, document_text, semantic_features, segments, control_mapping
        )

        # Assess implementation level
        implementation_level = self._assess_implementation_level(
            evidence, document_text, control_mapping
        )

        # Identify gaps
        gaps = self._identify_control_gaps(
            control, evidence, implementation_level, control_mapping
        )

        # Generate specific recommendations
        recommendations = self._generate_control_recommendations(
            control, gaps, implementation_level
        )

        # Calculate confidence score
        confidence_score = self._calculate_assessment_confidence(
            evidence, implementation_level, len(gaps)
        )

        return ControlAssessment(
            control_id=control.id,
            control_title=control.title,
            status=implementation_level,
            confidence_score=confidence_score,
            evidence_found=evidence.implementation_indicators[:5],  # Top 5 evidence
            gaps_identified=[gap.description for gap in gaps],
            recommendations=recommendations[:3]  # Top 3 recommendations (already strings)
        )

    async def _extract_control_evidence(
        self,
        control: Control,
        document_text: str,
        semantic_features: Dict[str, Any],
        segments: List[TextSegment],
        control_mapping: Dict[str, Any]
    ) -> ControlEvidence:
        """Extract detailed evidence for a control"""

        direct_evidence = []
        contextual_evidence = []
        implementation_indicators = []
        gap_indicators = []

        # Extract evidence from semantic features
        concept_matches = semantic_features.get('concept_matches', {})

        # Look for direct keyword matches
        control_keywords = control_mapping.get('keywords', [])
        for keyword in control_keywords:
            matches = self._find_keyword_evidence(document_text, segments, keyword)
            direct_evidence.extend(matches)

        # Look for implementation indicators
        impl_patterns = control_mapping.get('implementation_indicators', [])
        for pattern in impl_patterns:
            matches = self._find_pattern_evidence(document_text, segments, pattern)
            if matches:
                implementation_indicators.extend([m.text_quote for m in matches])

        # Check for control-specific concept matches
        control_category = self._map_control_to_category(control.id)
        if control_category in concept_matches:
            for concept_match in concept_matches[control_category]:
                if self._is_relevant_to_control(concept_match, control):
                    contextual_evidence.extend(concept_match.evidence_segments)

        # Look for gap indicators
        gap_patterns = [
            'not implemented', 'missing', 'absent', 'lacks', 'requires',
            'needs improvement', 'gaps identified', 'deficient'
        ]
        for pattern in gap_patterns:
            matches = self._find_pattern_evidence(document_text, segments, pattern)
            if matches:
                gap_indicators.extend([m.text_quote for m in matches])

        # Calculate compliance score
        compliance_score = self._calculate_compliance_score(
            len(direct_evidence), len(implementation_indicators),
            len(gap_indicators), len(contextual_evidence)
        )

        # Calculate confidence level
        confidence_level = self._calculate_evidence_confidence(
            direct_evidence + contextual_evidence
        )

        return ControlEvidence(
            control_id=control.id,
            direct_evidence=direct_evidence,
            contextual_evidence=contextual_evidence,
            implementation_indicators=implementation_indicators,
            gap_indicators=gap_indicators,
            compliance_score=compliance_score,
            confidence_level=confidence_level
        )

    def _find_keyword_evidence(
        self,
        text: str,
        segments: List[TextSegment],
        keyword: str
    ) -> List[Evidence]:
        """Find evidence for a specific keyword"""
        evidence_list = []
        text_lower = text.lower()
        keyword_lower = keyword.lower()

        # Create a more flexible pattern
        # Replace spaces with flexible whitespace matching
        pattern_str = re.sub(r'\s+', r'\\s+', re.escape(keyword_lower))
        pattern = re.compile(pattern_str, re.IGNORECASE)

        for match in pattern.finditer(text):
            # Get surrounding context
            start_context = max(0, match.start() - 200)
            end_context = min(len(text), match.end() + 200)
            context = text[start_context:end_context]

            # Find the segment this belongs to
            matching_segment = self._find_matching_segment(segments, match.start())

            # Extract the actual matched text with some buffer
            quote_start = max(match.start() - 50, 0)
            quote_end = min(match.end() + 50, len(text))
            quote = text[quote_start:quote_end].strip()

            evidence = Evidence(
                text_quote=quote,
                context=context,
                position=match.start(),
                confidence=self._calculate_keyword_confidence(context, keyword),
                segment_type=matching_segment.segment_type if matching_segment else 'unknown',
                related_concepts=self._extract_related_concepts_from_context(context)
            )
            evidence_list.append(evidence)

        return evidence_list

    def _find_pattern_evidence(
        self,
        text: str,
        segments: List[TextSegment],
        pattern: str
    ) -> List[Evidence]:
        """Find evidence for a regex pattern"""
        evidence_list = []

        try:
            compiled_pattern = re.compile(pattern, re.IGNORECASE)
            for match in compiled_pattern.finditer(text):
                # Get context
                start_context = max(0, match.start() - 150)
                end_context = min(len(text), match.end() + 150)
                context = text[start_context:end_context]

                # Find segment
                matching_segment = self._find_matching_segment(segments, match.start())

                evidence = Evidence(
                    text_quote=match.group(),
                    context=context,
                    position=match.start(),
                    confidence=0.7,
                    segment_type=matching_segment.segment_type if matching_segment else 'unknown',
                    related_concepts=[]
                )
                evidence_list.append(evidence)

        except re.error as e:
            logger.warning(f"Invalid regex pattern '{pattern}': {e}")

        return evidence_list

    def _find_matching_segment(self, segments: List[TextSegment], position: int) -> Optional[TextSegment]:
        """Find the segment that contains the given position"""
        for segment in segments:
            if segment.start_position <= position <= segment.end_position:
                return segment
        return None

    def _map_control_to_category(self, control_id: str) -> str:
        """Map control ID to semantic analysis category"""
        if control_id.startswith('5'):
            return 'compliance_governance'
        elif control_id.startswith('6'):
            return 'human_resources'
        elif control_id.startswith('7'):
            return 'physical_security'
        elif control_id.startswith('8'):
            if any(keyword in control_id for keyword in ['8.1', '8.2', '8.3']):
                return 'access_control'
            elif any(keyword in control_id for keyword in ['8.16', '8.28']):
                return 'technical_security'
            else:
                return 'technical_security'
        else:
            return 'information_security'

    def _is_relevant_to_control(self, concept_match: ConceptMatch, control: Control) -> bool:
        """Check if a concept match is relevant to the specific control"""
        # Simple relevance check based on concept and control description
        concept_words = set(concept_match.concept.lower().split())
        control_words = set((control.title + ' ' + control.description).lower().split())

        # Calculate word overlap
        overlap = len(concept_words.intersection(control_words))
        return overlap >= 1 or concept_match.relevance_score > 0.6

    def _assess_implementation_level(
        self,
        evidence: ControlEvidence,
        document_text: str,
        control_mapping: Dict[str, Any]
    ) -> ComplianceStatus:
        """Assess the implementation level of a control"""

        # Check for explicit implementation statements
        text_lower = document_text.lower()

        # Check for full implementation indicators
        full_impl_patterns = self.implementation_patterns['fully_implemented']
        full_impl_count = sum(1 for pattern in full_impl_patterns
                             if re.search(pattern, text_lower, re.IGNORECASE))

        # Check for partial implementation indicators
        partial_impl_patterns = self.implementation_patterns['partially_implemented']
        partial_impl_count = sum(1 for pattern in partial_impl_patterns
                                if re.search(pattern, text_lower, re.IGNORECASE))

        # Check for non-implementation indicators
        no_impl_patterns = self.implementation_patterns['not_implemented']
        no_impl_count = sum(1 for pattern in no_impl_patterns
                           if re.search(pattern, text_lower, re.IGNORECASE))

        # Scoring based on evidence
        evidence_score = evidence.compliance_score

        # Determine status based on multiple factors
        if evidence_score >= 0.8 and full_impl_count > 0 and no_impl_count == 0:
            return ComplianceStatus.COMPLIANT
        elif evidence_score >= 0.4 or partial_impl_count > 0:
            return ComplianceStatus.PARTIALLY_COMPLIANT
        elif no_impl_count > 0 or evidence_score < 0.2:
            return ComplianceStatus.NON_COMPLIANT
        elif len(evidence.direct_evidence) == 0 and len(evidence.contextual_evidence) == 0:
            return ComplianceStatus.INSUFFICIENT_INFORMATION
        else:
            return ComplianceStatus.PARTIALLY_COMPLIANT

    def _identify_control_gaps(
        self,
        control: Control,
        evidence: ControlEvidence,
        implementation_level: ComplianceStatus,
        control_mapping: Dict[str, Any]
    ) -> List[GapDetail]:
        """Identify specific gaps for a control"""
        gaps = []

        required_elements = control_mapping.get('required_elements', [])

        if implementation_level == ComplianceStatus.NON_COMPLIANT:
            # Major gap - control not implemented
            gaps.append(GapDetail(
                gap_type='missing',
                description=f'Control {control.id} ({control.title}) is not implemented or lacks sufficient evidence',
                impact_level='high',
                specific_requirements=required_elements,
                recommended_actions=[
                    f'Implement {control.title.lower()} with all required elements',
                    'Develop comprehensive procedures and documentation',
                    'Assign clear responsibilities and allocate resources',
                    'Establish monitoring and review processes'
                ]
            ))

        elif implementation_level == ComplianceStatus.PARTIALLY_COMPLIANT:
            # Check for missing required elements
            missing_elements = []
            for element in required_elements:
                if not self._element_addressed(element, evidence):
                    missing_elements.append(element)

            if missing_elements:
                gaps.append(GapDetail(
                    gap_type='incomplete',
                    description=f'Missing required elements for {control.title}: {", ".join(missing_elements[:2])}',
                    impact_level='medium',
                    specific_requirements=missing_elements,
                    recommended_actions=[
                        f'Complete implementation of {", ".join(missing_elements[:2])}',
                        'Enhance existing controls to cover all requirements',
                        'Update documentation to reflect full implementation',
                        'Provide comprehensive evidence of all control aspects'
                    ]
                ))

            # Add gap for partial implementation even if some elements exist
            if len(evidence.implementation_indicators) < 3:
                gaps.append(GapDetail(
                    gap_type='insufficient',
                    description=f'Limited evidence of {control.title} implementation',
                    impact_level='medium',
                    specific_requirements=['Comprehensive implementation evidence'],
                    recommended_actions=[
                        'Strengthen control implementation',
                        'Expand coverage to all systems and processes',
                        'Document implementation more thoroughly'
                    ]
                ))

        # Check for gap indicators in the evidence
        if evidence.gap_indicators:
            gaps.append(GapDetail(
                gap_type='inadequate',
                description='Document explicitly mentions gaps or areas for improvement',
                impact_level='medium',
                specific_requirements=['Address explicitly identified gaps'],
                recommended_actions=[
                    'Review and address all identified deficiencies',
                    'Strengthen existing control measures',
                    'Improve documentation and evidence quality',
                    'Implement corrective actions'
                ]
            ))

        # Always provide at least one gap for non-compliant controls
        if not gaps and implementation_level != ComplianceStatus.COMPLIANT:
            gaps.append(GapDetail(
                gap_type='improvement',
                description=f'Control {control.id} requires enhancement to achieve full compliance',
                impact_level='low',
                specific_requirements=required_elements,
                recommended_actions=[
                    f'Review and strengthen {control.title.lower()}',
                    'Ensure all required elements are fully addressed',
                    'Improve evidence documentation'
                ]
            ))

        return gaps

    def _element_addressed(self, element: str, evidence: ControlEvidence) -> bool:
        """Check if a required element is addressed in the evidence"""
        element_lower = element.lower()

        # Check in implementation indicators
        for indicator in evidence.implementation_indicators:
            if any(word in indicator.lower() for word in element_lower.split()):
                return True

        # Check in evidence text
        for ev in evidence.direct_evidence + evidence.contextual_evidence:
            if any(word in ev.context.lower() for word in element_lower.split()):
                return True

        return False

    def _generate_control_recommendations(
        self,
        control: Control,
        gaps: List[GapDetail],
        implementation_level: ComplianceStatus
    ) -> List[Any]:  # Would be Recommendation objects
        """Generate specific recommendations for a control"""
        recommendations = []

        # Generate recommendations based on gaps
        for gap in gaps:
            recommendations.extend(gap.recommended_actions[:2])  # Limit per gap to avoid duplication

        # Add control-specific recommendations based on implementation level
        if implementation_level == ComplianceStatus.NON_COMPLIANT:
            recommendations.extend([
                f'Prioritize implementation of {control.title.lower()}',
                f'Develop comprehensive {control.title.lower()} procedures and policies',
                'Create implementation roadmap with clear milestones and timelines',
                'Assign dedicated resources and establish accountability',
                'Conduct gap assessment to identify all missing components'
            ])

        elif implementation_level == ComplianceStatus.PARTIALLY_COMPLIANT:
            recommendations.extend([
                f'Complete implementation of {control.title.lower()} requirements',
                'Address identified gaps in current implementation',
                'Enhance documentation to demonstrate full compliance',
                'Conduct internal audit to verify control effectiveness',
                'Implement continuous monitoring and improvement'
            ])

        elif implementation_level == ComplianceStatus.COMPLIANT:
            recommendations.extend([
                f'Maintain and monitor {control.title.lower()} effectiveness',
                'Perform regular reviews and updates',
                'Consider automation or optimization opportunities'
            ])

        # Add implementation guidance from the control if available
        if hasattr(control, 'implementation_guidance') and control.implementation_guidance:
            if isinstance(control.implementation_guidance, list):
                recommendations.extend(control.implementation_guidance[:2])
            elif isinstance(control.implementation_guidance, str):
                recommendations.append(control.implementation_guidance)

        # Ensure we always have at least 3 recommendations
        if len(recommendations) < 3:
            recommendations.extend([
                f'Review ISO 27001:2022 requirements for {control.id}',
                'Document current state and identify improvement areas',
                'Develop action plan with timeline for full compliance'
            ])

        # Remove duplicates while preserving order
        seen = set()
        unique_recommendations = []
        for rec in recommendations:
            if rec and rec not in seen:
                seen.add(rec)
                unique_recommendations.append(rec)

        # Return top 5 unique recommendations
        return unique_recommendations[:5]

    def _calculate_compliance_score(
        self,
        direct_evidence_count: int,
        implementation_indicators_count: int,
        gap_indicators_count: int,
        contextual_evidence_count: int
    ) -> float:
        """Calculate a compliance score based on evidence"""

        # Positive score from evidence
        positive_score = (
            direct_evidence_count * 0.4 +
            implementation_indicators_count * 0.3 +
            contextual_evidence_count * 0.1
        )

        # Negative impact from gaps
        negative_score = gap_indicators_count * 0.3

        # Calculate final score
        raw_score = positive_score - negative_score
        normalized_score = max(0.0, min(1.0, raw_score / 5.0))  # Normalize to 0-1

        return normalized_score

    def _calculate_evidence_confidence(self, evidence_list: List[Evidence]) -> float:
        """Calculate confidence level based on evidence quality"""
        if not evidence_list:
            return 0.1

        # Average confidence from individual evidence
        if np:
            avg_confidence = np.mean([ev.confidence for ev in evidence_list])
        else:
            confidences = [ev.confidence for ev in evidence_list]
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

        # Boost confidence based on evidence quantity
        quantity_boost = min(len(evidence_list) / 10.0, 0.3)

        # Boost confidence based on evidence diversity
        segment_types = set(ev.segment_type for ev in evidence_list)
        diversity_boost = len(segment_types) * 0.1

        total_confidence = avg_confidence + quantity_boost + diversity_boost
        return min(total_confidence, 1.0)

    def _calculate_assessment_confidence(
        self,
        evidence: ControlEvidence,
        implementation_level: ComplianceStatus,
        gaps_count: int
    ) -> float:
        """Calculate overall assessment confidence"""

        base_confidence = evidence.confidence_level

        # Adjust based on implementation level certainty
        if implementation_level in [ComplianceStatus.COMPLIANT, ComplianceStatus.NON_COMPLIANT]:
            certainty_boost = 0.2
        else:
            certainty_boost = 0.0

        # Penalize for high uncertainty
        uncertainty_penalty = gaps_count * 0.05

        final_confidence = base_confidence + certainty_boost - uncertainty_penalty
        return max(0.1, min(1.0, final_confidence))

    def _calculate_keyword_confidence(self, context: str, keyword: str) -> float:
        """Calculate confidence for a keyword match based on context"""
        confidence = 0.6  # Base confidence

        context_lower = context.lower()

        # Boost for implementation language
        impl_words = ['implemented', 'established', 'deployed', 'maintained', 'enforced']
        confidence += 0.1 * sum(1 for word in impl_words if word in context_lower)

        # Boost for specific details
        detail_words = ['procedure', 'process', 'documented', 'policy', 'control']
        confidence += 0.05 * sum(1 for word in detail_words if word in context_lower)

        # Penalize for negation
        negation_words = ['not', 'no', 'absence', 'lack', 'without', 'missing']
        confidence -= 0.2 * sum(1 for word in negation_words if word in context_lower)

        return max(0.1, min(1.0, confidence))

    def _extract_related_concepts_from_context(self, context: str) -> List[str]:
        """Extract related concepts from context"""
        related = []
        context_lower = context.lower()

        # Key ISO concepts to look for
        iso_concepts = [
            'risk management', 'access control', 'incident response',
            'business continuity', 'information security', 'compliance',
            'audit', 'monitoring', 'governance', 'policy'
        ]

        for concept in iso_concepts:
            if concept in context_lower:
                related.append(concept)

        return related[:3]  # Return top 3

    async def assess_control(self, text: str, control: Control, document_features: Dict[str, Any]) -> 'ControlAssessment':
        """Wrapper method for control assessment"""
        from .semantic_analyzer import ISOSemanticAnalyzer

        # Use semantic analyzer to get text segments
        analyzer = ISOSemanticAnalyzer()
        segments = analyzer.analyze_document_structure(text)

        # Perform comprehensive assessment
        return await self.assess_control_comprehensive(
            control, text, document_features, segments
        )

    async def assess_clause(self, text: str, clause_id: str, requirement: 'Requirement', document_features: Dict[str, Any]) -> 'ClauseAssessment':
        """Assess a management clause requirement"""
        from ..iso_knowledge.agent_response import ClauseAssessment, ComplianceStatus

        # Simple assessment logic - this could be enhanced further
        requirement_keywords = requirement.title.lower().split() + requirement.description.lower().split()[:10]
        text_lower = text.lower()

        evidence_count = sum(1 for keyword in requirement_keywords if keyword in text_lower and len(keyword) > 3)

        # Extract relevant text segments as evidence
        evidence_found = []
        missing_elements = []
        recommendations = []

        # Look for actual evidence in the text
        sentences = text.split('.')
        for sentence in sentences[:20]:  # Limit for performance
            sentence_lower = sentence.lower()
            if any(keyword in sentence_lower for keyword in requirement_keywords[:5]):
                if len(sentence.strip()) > 20:
                    evidence_found.append(sentence.strip()[:200] + "...")

        if evidence_count >= 2:
            status = ComplianceStatus.COMPLIANT
            if not evidence_found:
                evidence_found = [f"Found evidence of {requirement.title.lower()}"]
        elif evidence_count >= 1:
            status = ComplianceStatus.PARTIALLY_COMPLIANT
            missing_elements = ["Complete implementation documentation needed"]
            recommendations = [f"Strengthen {requirement.title.lower()} documentation"]
            if not evidence_found:
                evidence_found = [f"Partial evidence of {requirement.title.lower()}"]
        else:
            status = ComplianceStatus.NON_COMPLIANT
            missing_elements = [f"No evidence of {requirement.title.lower()}"]
            recommendations = [f"Implement {requirement.title.lower()} requirements"]

        confidence_score = min(evidence_count / 3.0, 1.0)

        return ClauseAssessment(
            clause_id=f"{clause_id}.{requirement.id.split('.')[-1]}",
            requirement_title=requirement.title,
            status=status,
            confidence_score=confidence_score,
            evidence_found=evidence_found[:3],  # Limit evidence
            missing_elements=missing_elements,
            recommendations=recommendations
        )