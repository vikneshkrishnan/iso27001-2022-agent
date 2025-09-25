"""
Enhanced Semantic Analysis for ISO 27001:2022 Document Processing
Provides sophisticated text analysis beyond simple keyword matching
"""

import re
import logging
from typing import Dict, List, Tuple, Set, Any, Optional
from collections import Counter, defaultdict
from dataclasses import dataclass
import math

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logging.warning("scikit-learn not available, using fallback analysis methods")

logger = logging.getLogger(__name__)


@dataclass
class TextSegment:
    """Represents a segment of text with context"""
    content: str
    start_position: int
    end_position: int
    segment_type: str  # header, paragraph, list_item, table, etc.
    level: int = 0  # for hierarchical content like headers
    parent_section: Optional[str] = None


@dataclass
class Evidence:
    """Represents evidence found in document"""
    text_quote: str
    context: str
    position: int
    confidence: float
    segment_type: str
    related_concepts: List[str]


@dataclass
class ConceptMatch:
    """Represents a matched concept with relevance"""
    concept: str
    relevance_score: float
    evidence_segments: List[Evidence]
    frequency: int
    context_quality: float


class ISOSemanticAnalyzer:
    """Advanced semantic analyzer for ISO document analysis"""

    def __init__(self, vector_store=None):
        """Initialize the semantic analyzer"""
        self.iso_vocabulary = self._build_iso_vocabulary()
        self.document_structure_patterns = self._build_structure_patterns()
        self.compliance_indicators = self._build_compliance_indicators()

        # Optional Pinecone vector store for enhanced semantic capabilities
        self.vector_store = vector_store
        self.use_vector_search = vector_store is not None

        if SKLEARN_AVAILABLE:
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=1000,
                stop_words='english',
                ngram_range=(1, 3),
                lowercase=True
            )
        else:
            self.tfidf_vectorizer = None

    def _build_iso_vocabulary(self) -> Dict[str, Dict[str, float]]:
        """Build comprehensive ISO 27001:2022 vocabulary with weighted importance"""
        return {
            # Core ISO Concepts (high weight)
            'information_security': {
                'information security': 1.0,
                'infosec': 0.9,
                'cyber security': 0.8,
                'cybersecurity': 0.8,
                'data protection': 0.7,
                'security management': 0.9
            },

            'isms_concepts': {
                'isms': 1.0,
                'information security management system': 1.0,
                'management system': 0.7,
                'security governance': 0.8,
                'security framework': 0.7
            },

            'risk_management': {
                'risk assessment': 1.0,
                'risk management': 1.0,
                'risk treatment': 0.9,
                'risk analysis': 0.9,
                'threat': 0.8,
                'vulnerability': 0.8,
                'risk register': 0.7,
                'risk appetite': 0.7,
                'risk mitigation': 0.8
            },

            'access_control': {
                'access control': 1.0,
                'authentication': 0.9,
                'authorization': 0.9,
                'privileged access': 0.9,
                'user access': 0.8,
                'identity management': 0.8,
                'access rights': 0.8,
                'access review': 0.7,
                'least privilege': 0.9
            },

            'incident_management': {
                'incident response': 1.0,
                'incident management': 1.0,
                'security incident': 0.9,
                'incident handling': 0.8,
                'incident reporting': 0.8,
                'security event': 0.7,
                'incident recovery': 0.8,
                'forensic': 0.7
            },

            'business_continuity': {
                'business continuity': 1.0,
                'disaster recovery': 0.9,
                'continuity planning': 0.8,
                'recovery procedures': 0.8,
                'backup': 0.7,
                'redundancy': 0.6,
                'resilience': 0.7
            },

            'compliance_governance': {
                'compliance': 1.0,
                'governance': 0.9,
                'policy': 0.8,
                'procedure': 0.8,
                'standard': 0.7,
                'guideline': 0.6,
                'control': 0.9,
                'requirement': 0.7,
                'audit': 0.8,
                'monitoring': 0.8,
                'review': 0.6
            },

            'physical_security': {
                'physical security': 1.0,
                'physical access': 0.8,
                'facility security': 0.8,
                'perimeter security': 0.8,
                'environmental protection': 0.7,
                'secure area': 0.8,
                'visitor management': 0.6,
                'equipment protection': 0.7
            },

            'technical_security': {
                'network security': 0.9,
                'endpoint security': 0.8,
                'malware protection': 0.8,
                'encryption': 0.9,
                'cryptography': 0.8,
                'secure configuration': 0.8,
                'vulnerability management': 0.9,
                'patch management': 0.8,
                'logging': 0.7,
                'monitoring': 0.8
            },

            'human_resources': {
                'security awareness': 0.9,
                'training': 0.8,
                'background check': 0.8,
                'screening': 0.8,
                'confidentiality agreement': 0.7,
                'terms of employment': 0.7,
                'disciplinary process': 0.6
            }
        }

    def _build_structure_patterns(self) -> Dict[str, List[str]]:
        """Build patterns for detecting document structure"""
        return {
            'headers': [
                r'^#+\s+(.+)$',  # Markdown headers
                r'^(\d+\.?\d*\.?\s+.+)$',  # Numbered sections
                r'^([A-Z][A-Za-z\s&-]+)$',  # Title case headers
                r'^\s*([A-Z][A-Z\s]+)\s*$',  # All caps headers
            ],
            'policy_statements': [
                r'(must|shall|should|will|required|mandatory)',
                r'(policy|procedure|standard|guideline)',
                r'(responsible for|accountable|owns)',
                r'(ensure|maintain|establish|implement)'
            ],
            'control_implementations': [
                r'(implemented|in place|established|deployed)',
                r'(regularly|annually|monthly|quarterly)',
                r'(reviewed|monitored|assessed|audited)',
                r'(documented|recorded|logged|tracked)'
            ],
            'compliance_statements': [
                r'(complies with|adherence to|accordance with)',
                r'(meets requirements|satisfies|fulfills)',
                r'(certified|accredited|approved)',
                r'(audit|assessment|review) (passed|successful)'
            ]
        }

    def _build_compliance_indicators(self) -> Dict[str, List[str]]:
        """Build indicators for compliance levels"""
        return {
            'strong_compliance': [
                'fully implemented', 'completely deployed', 'comprehensive coverage',
                'regular monitoring', 'continuous improvement', 'mature process',
                'well established', 'robust controls', 'effective implementation'
            ],
            'partial_compliance': [
                'partially implemented', 'in progress', 'being developed',
                'planned implementation', 'initial deployment', 'basic coverage',
                'limited monitoring', 'developing process'
            ],
            'non_compliance': [
                'not implemented', 'no coverage', 'missing controls',
                'gaps identified', 'requires implementation', 'not addressed',
                'absent', 'lacking', 'deficient'
            ],
            'gaps_indicators': [
                'however', 'but', 'although', 'except', 'unless',
                'needs improvement', 'requires enhancement', 'should be strengthened',
                'recommendation', 'action item', 'to be implemented'
            ]
        }

    def analyze_document_structure(self, text: str) -> List[TextSegment]:
        """Analyze document structure to identify sections and hierarchy"""
        segments = []
        lines = text.split('\n')
        current_position = 0
        current_section = None

        for i, line in enumerate(lines):
            line_stripped = line.strip()
            if not line_stripped:
                current_position += len(line) + 1
                continue

            segment_type = 'paragraph'
            level = 0

            # Check for headers
            for pattern in self.document_structure_patterns['headers']:
                if re.match(pattern, line_stripped):
                    segment_type = 'header'
                    level = self._determine_header_level(line_stripped)
                    if level <= 2:  # Main section headers
                        current_section = line_stripped
                    break

            # Check for lists
            if re.match(r'^\s*[•\-\*]\s+', line) or re.match(r'^\s*\d+\.\s+', line):
                segment_type = 'list_item'

            # Check for table-like structures
            if '|' in line and line.count('|') >= 2:
                segment_type = 'table'

            segment = TextSegment(
                content=line_stripped,
                start_position=current_position,
                end_position=current_position + len(line),
                segment_type=segment_type,
                level=level,
                parent_section=current_section
            )

            segments.append(segment)
            current_position += len(line) + 1

        return segments

    def _determine_header_level(self, text: str) -> int:
        """Determine header hierarchy level"""
        # Markdown style
        if text.startswith('#'):
            return text.count('#')

        # Numbered style
        if re.match(r'^\d+\.\s+', text):
            return 1
        elif re.match(r'^\d+\.\d+\.\s+', text):
            return 2
        elif re.match(r'^\d+\.\d+\.\d+\.\s+', text):
            return 3

        # All caps likely top level
        if text.isupper():
            return 1

        return 2

    def extract_semantic_features(self, text: str, segments: List[TextSegment]) -> Dict[str, Any]:
        """Extract semantic features from the document"""
        features = {
            'concept_matches': {},
            'compliance_indicators': {},
            'document_characteristics': {},
            'evidence_segments': []
        }

        # Analyze concept matches with context
        for category, concepts in self.iso_vocabulary.items():
            category_matches = []

            for concept, weight in concepts.items():
                matches = self._find_concept_matches(text, segments, concept, weight)
                if matches:
                    category_matches.extend(matches)

            if category_matches:
                features['concept_matches'][category] = category_matches

        # Analyze compliance indicators
        for indicator_type, indicators in self.compliance_indicators.items():
            matches = []
            for indicator in indicators:
                pattern_matches = self._find_pattern_matches(text, segments, indicator)
                matches.extend(pattern_matches)

            if matches:
                features['compliance_indicators'][indicator_type] = matches

        # Document characteristics
        features['document_characteristics'] = {
            'total_segments': len(segments),
            'header_count': len([s for s in segments if s.segment_type == 'header']),
            'list_items': len([s for s in segments if s.segment_type == 'list_item']),
            'tables': len([s for s in segments if s.segment_type == 'table']),
            'avg_segment_length': (sum(len(s.content) for s in segments) / len(segments)) if segments else 0,
            'structure_complexity': self._calculate_structure_complexity(segments)
        }

        return features

    def _find_concept_matches(self, text: str, segments: List[TextSegment],
                            concept: str, weight: float) -> List[ConceptMatch]:
        """Find matches for a specific concept with context"""
        matches = []
        text_lower = text.lower()
        concept_lower = concept.lower()

        # Find all occurrences
        pattern = re.compile(r'\b' + re.escape(concept_lower) + r'\b', re.IGNORECASE)
        occurrences = list(pattern.finditer(text_lower))

        if not occurrences:
            return matches

        # Group evidence by context
        evidence_segments = []
        for match in occurrences:
            start_pos = max(0, match.start() - 200)
            end_pos = min(len(text), match.end() + 200)
            context = text[start_pos:end_pos]

            # Find the segment this match belongs to
            matching_segment = None
            for segment in segments:
                if (segment.start_position <= match.start() <= segment.end_position):
                    matching_segment = segment
                    break

            evidence = Evidence(
                text_quote=text[match.start():match.end()],
                context=context,
                position=match.start(),
                confidence=self._calculate_context_confidence(context, concept),
                segment_type=matching_segment.segment_type if matching_segment else 'unknown',
                related_concepts=self._extract_related_concepts(context)
            )
            evidence_segments.append(evidence)

        if evidence_segments:
            # Calculate overall relevance
            relevance_score = self._calculate_relevance_score(evidence_segments, weight)
            confidences = [e.confidence for e in evidence_segments]
            context_quality = sum(confidences) / len(confidences) if confidences else 0

            concept_match = ConceptMatch(
                concept=concept,
                relevance_score=relevance_score,
                evidence_segments=evidence_segments,
                frequency=len(occurrences),
                context_quality=context_quality
            )
            matches.append(concept_match)

        return matches

    def _find_pattern_matches(self, text: str, segments: List[TextSegment],
                            pattern: str) -> List[Evidence]:
        """Find pattern matches with context"""
        evidence_list = []

        try:
            compiled_pattern = re.compile(pattern, re.IGNORECASE)
            matches = list(compiled_pattern.finditer(text))

            for match in matches:
                start_pos = max(0, match.start() - 150)
                end_pos = min(len(text), match.end() + 150)
                context = text[start_pos:end_pos]

                # Find segment
                matching_segment = None
                for segment in segments:
                    if (segment.start_position <= match.start() <= segment.end_position):
                        matching_segment = segment
                        break

                evidence = Evidence(
                    text_quote=match.group(),
                    context=context,
                    position=match.start(),
                    confidence=0.7,  # Base confidence for pattern matches
                    segment_type=matching_segment.segment_type if matching_segment else 'unknown',
                    related_concepts=[]
                )
                evidence_list.append(evidence)

        except re.error:
            # Handle regex compilation errors
            pass

        return evidence_list

    def _calculate_context_confidence(self, context: str, concept: str) -> float:
        """Calculate confidence based on context quality"""
        confidence = 0.5  # Base confidence

        context_lower = context.lower()

        # Boost confidence for compliance language
        compliance_words = ['implement', 'establish', 'maintain', 'ensure', 'comply', 'adhere']
        confidence += 0.1 * sum(1 for word in compliance_words if word in context_lower)

        # Boost for specific implementation details
        implementation_words = ['procedure', 'process', 'control', 'policy', 'documented']
        confidence += 0.1 * sum(1 for word in implementation_words if word in context_lower)

        # Penalize for vague or generic mentions
        vague_words = ['may', 'could', 'might', 'possibly', 'general']
        confidence -= 0.1 * sum(1 for word in vague_words if word in context_lower)

        return max(0.1, min(1.0, confidence))

    def _extract_related_concepts(self, context: str) -> List[str]:
        """Extract related concepts from context"""
        related = []
        context_lower = context.lower()

        # Look for related ISO concepts in the context
        for category, concepts in self.iso_vocabulary.items():
            for concept, weight in concepts.items():
                if concept.lower() in context_lower and weight > 0.6:
                    related.append(concept)

        return related[:5]  # Limit to top 5 related concepts

    def _calculate_relevance_score(self, evidence_segments: List[Evidence], weight: float) -> float:
        """Calculate overall relevance score for a concept"""
        if not evidence_segments:
            return 0.0

        # Base score from frequency and weight
        frequency_score = min(len(evidence_segments) / 10.0, 1.0)  # Normalize frequency
        weighted_score = frequency_score * weight

        # Quality boost from evidence quality
        confidences = [e.confidence for e in evidence_segments]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0
        quality_boost = avg_confidence * 0.3

        # Context diversity boost
        segment_types = set(e.segment_type for e in evidence_segments)
        diversity_boost = len(segment_types) * 0.1

        total_score = weighted_score + quality_boost + diversity_boost
        return min(total_score, 1.0)

    def _calculate_structure_complexity(self, segments: List[TextSegment]) -> float:
        """Calculate document structure complexity score"""
        if not segments:
            return 0.0

        # Count different segment types
        segment_types = set(s.segment_type for s in segments)
        type_diversity = len(segment_types) / 5.0  # Normalize by expected max types

        # Count header levels
        header_levels = set(s.level for s in segments if s.segment_type == 'header')
        level_diversity = len(header_levels) / 5.0  # Normalize by expected max levels

        # Calculate hierarchy depth
        max_level = max((s.level for s in segments if s.segment_type == 'header'), default=0)
        hierarchy_score = min(max_level / 5.0, 1.0)

        complexity = (type_diversity + level_diversity + hierarchy_score) / 3.0
        return min(complexity, 1.0)

    def calculate_tfidf_similarity(self, document_text: str, control_description: str) -> float:
        """Calculate TF-IDF similarity between document and control"""
        if not SKLEARN_AVAILABLE or not self.tfidf_vectorizer:
            raise Exception("Cannot perform ISO analysis without proper vectorization capabilities")

        try:
            # Prepare texts
            texts = [document_text, control_description]

            # Calculate TF-IDF vectors
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)

            # Calculate cosine similarity
            similarity_matrix = cosine_similarity(tfidf_matrix)

            # Return similarity between document and control
            return float(similarity_matrix[0, 1])

        except Exception as e:
            logger.error(f"TF-IDF calculation failed: {e}")
            raise Exception(f"ISO semantic analysis failed - cannot provide non-ISO response: {e}")


    def generate_analysis_summary(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary of semantic analysis"""
        summary = {
            'concept_coverage': {},
            'compliance_strength': {},
            'document_quality': {},
            'key_findings': []
        }

        # Concept coverage analysis
        for category, matches in features['concept_matches'].items():
            if matches:
                relevances = [m.relevance_score for m in matches]
                avg_relevance = sum(relevances) / len(relevances) if relevances else 0
                total_evidence = sum(len(m.evidence_segments) for m in matches)
                summary['concept_coverage'][category] = {
                    'relevance_score': avg_relevance,
                    'evidence_count': total_evidence,
                    'coverage_level': self._determine_coverage_level(avg_relevance, total_evidence)
                }

        # Compliance strength analysis
        compliance_indicators = features.get('compliance_indicators', {})
        strong_indicators = len(compliance_indicators.get('strong_compliance', []))
        partial_indicators = len(compliance_indicators.get('partial_compliance', []))
        non_compliance_indicators = len(compliance_indicators.get('non_compliance', []))

        total_indicators = strong_indicators + partial_indicators + non_compliance_indicators
        if total_indicators > 0:
            summary['compliance_strength'] = {
                'strong_ratio': strong_indicators / total_indicators,
                'partial_ratio': partial_indicators / total_indicators,
                'gaps_ratio': non_compliance_indicators / total_indicators,
                'overall_strength': (strong_indicators * 1.0 + partial_indicators * 0.5) / total_indicators
            }

        # Document quality
        chars = features['document_characteristics']
        summary['document_quality'] = {
            'structure_score': chars.get('structure_complexity', 0),
            'organization_score': min(chars.get('header_count', 0) / 10.0, 1.0),
            'detail_score': min(chars.get('avg_segment_length', 0) / 100.0, 1.0),
            'completeness_score': len(summary['concept_coverage']) / 10.0  # Normalize by expected categories
        }

        return summary

    def _determine_coverage_level(self, relevance: float, evidence_count: int) -> str:
        """Determine coverage level based on relevance and evidence"""
        if relevance >= 0.8 and evidence_count >= 5:
            return 'comprehensive'
        elif relevance >= 0.6 and evidence_count >= 3:
            return 'good'
        elif relevance >= 0.4 and evidence_count >= 1:
            return 'partial'
        else:
            return 'minimal'

    def classify_document(self, text: str, document_info: Dict[str, Any]) -> 'DocumentClassification':
        """Enhanced document classification using semantic analysis"""
        try:
            # Analyze document structure and extract features
            segments = self.analyze_document_structure(text)
            features = self.extract_semantic_features(text, segments)

            # Determine document type from structure and content
            filename = document_info.get('filename', '').lower()

            # Document type detection using enhanced analysis
            doc_type = "other"
            if any(keyword in filename for keyword in ['policy', 'policies']):
                doc_type = "policy"
            elif any(keyword in filename for keyword in ['procedure', 'process']):
                doc_type = "procedure"
            elif any(keyword in filename for keyword in ['risk', 'assessment']):
                doc_type = "risk_assessment"
            elif any(keyword in filename for keyword in ['audit', 'review']):
                doc_type = "audit_report"

            # ISO relevance using semantic features
            iso_score = features.get('iso_relevance_score', 0.5)
            if iso_score > 0.7:
                iso_relevance = "high"
            elif iso_score > 0.4:
                iso_relevance = "medium"
            else:
                iso_relevance = "low"

            # Applicable categories from semantic analysis
            applicable_categories = []
            concept_matches = features.get('concept_matches', {})
            if concept_matches.get('organizational_security', []):
                applicable_categories.append('organizational')
            if concept_matches.get('people_security', []):
                applicable_categories.append('people')
            if concept_matches.get('physical_security', []):
                applicable_categories.append('physical')
            if concept_matches.get('technological_security', []):
                applicable_categories.append('technological')

            # Import DocumentClassification and DocumentType here to avoid circular imports
            from ..iso_knowledge.agent_response import DocumentClassification, DocumentType

            return DocumentClassification(
                document_type=DocumentType(doc_type),
                confidence_score=iso_score,
                iso_relevance=iso_relevance,
                applicable_categories=applicable_categories,
                primary_focus_areas=list(concept_matches.keys())[:5]
            )

        except Exception as e:
            logger.error(f"Enhanced document classification failed: {e}")
            # Fallback classification
            from ..iso_knowledge.agent_response import DocumentClassification, DocumentType
            return DocumentClassification(
                document_type=DocumentType.OTHER,
                confidence_score=0.5,
                iso_relevance="medium",
                applicable_categories=["organizational"],
                primary_focus_areas=[]
            )

    def extract_document_features(self, text: str) -> Dict[str, Any]:
        """Extract document features for analysis"""
        segments = self.analyze_document_structure(text)
        return self.extract_semantic_features(text, segments)

    def identify_relevant_controls(self, features: Dict[str, Any]) -> List[str]:
        """Identify relevant control IDs based on document features"""
        relevant_controls = []
        document_text = features.get('raw_text', '').lower()

        # Enhanced keyword mapping with context patterns and priority scoring
        keyword_control_mapping = {
            # Authentication & Access Control (High Priority)
            'multi-factor': ['A.9.4.2', 'A.9.2.2'],
            'mfa': ['A.9.4.2', 'A.9.2.2'],
            '2fa': ['A.9.4.2'],
            'two-factor': ['A.9.4.2'],
            'biometric': ['A.9.4.2', 'A.11.1.2'],
            'privileged': ['A.9.2.2', 'A.9.2.6'],
            'admin': ['A.9.2.2'],
            'badge access': ['A.11.1.2'],
            'access card': ['A.11.1.2'],
            'authentication': ['A.9.4.2', 'A.9.1.1'],
            'single sign-on': ['A.9.4.2'],
            'sso': ['A.9.4.2'],
            'ldap': ['A.9.2.1'],
            'active directory': ['A.9.2.1'],

            # Cryptographic Controls (High Priority)
            'aes-256': ['A.8.2.2', 'A.10.1.2'],
            'aes-128': ['A.8.2.2', 'A.10.1.2'],
            'encryption': ['A.10.1.1', 'A.8.2.2', 'A.13.2.1'],
            'tls 1.3': ['A.13.2.1'],
            'tls 1.2': ['A.13.2.1'],
            'tls': ['A.13.2.1'],
            'ssl': ['A.13.2.1'],
            'https': ['A.13.2.1'],
            'key management': ['A.10.1.2'],
            'hsm': ['A.10.1.2'],
            'hardware security module': ['A.10.1.2'],
            'cryptographic': ['A.10.1.1', 'A.10.1.2'],
            'digital signature': ['A.10.1.3'],
            'certificate': ['A.10.1.3'],
            'pki': ['A.10.1.3'],

            # Incident Management (High Priority)
            '4 hours': ['A.16.1.2'],
            '72 hours': ['A.16.1.4'],
            '24 hours': ['A.16.1.2'],
            'incident': ['A.16.1.1', 'A.16.1.2'],
            'incident response': ['A.16.1.1', 'A.16.1.2'],
            'security incident': ['A.16.1.1', 'A.16.1.2'],
            'incident team': ['A.16.1.1'],
            'escalation': ['A.16.1.2'],
            'forensic': ['A.16.1.4'],
            'siem': ['A.16.1.1', 'A.12.4.1'],
            'security operations center': ['A.16.1.1'],
            'soc': ['A.16.1.1'],

            # Physical Security (Medium Priority)
            'server room': ['A.11.1.1', 'A.11.1.2'],
            'data center': ['A.11.1.1', 'A.11.1.2'],
            'physical': ['A.11.1.1', 'A.11.1.2'],
            'environmental monitoring': ['A.11.2.3'],
            'temperature': ['A.11.2.3'],
            'humidity': ['A.11.2.3'],
            '24/7': ['A.11.2.3', 'A.12.1.2'],
            'backup power': ['A.11.2.3'],
            'ups': ['A.11.2.3'],
            'cctv': ['A.11.1.2'],
            'surveillance': ['A.11.1.2'],
            'secure perimeter': ['A.11.1.1'],

            # Network Security (High Priority)
            'firewall': ['A.13.1.1', 'A.13.1.3'],
            'network': ['A.13.1.1', 'A.9.1.2'],
            'vpn': ['A.9.1.2', 'A.13.2.1'],
            'segmentation': ['A.13.1.1'],
            'intrusion detection': ['A.13.1.1'],
            'ids': ['A.13.1.1'],
            'ips': ['A.13.1.1'],
            'network monitoring': ['A.13.1.1'],
            'dmz': ['A.13.1.1'],

            # Data Protection (High Priority)
            'classification': ['A.8.2.1'],
            'data handling': ['A.8.2.3'],
            'sensitive data': ['A.8.2.1', 'A.8.2.2'],
            'personal data': ['A.8.2.1'],
            'pii': ['A.8.2.1', 'A.8.2.2'],
            'gdpr': ['A.8.2.1', 'A.8.2.2'],
            'data retention': ['A.8.2.3'],
            'data disposal': ['A.8.3.2'],

            # Vulnerability Management (Medium Priority)
            'patch': ['A.12.6.1'],
            'vulnerability': ['A.12.6.1'],
            'security update': ['A.12.6.1'],
            'vulnerability scan': ['A.12.6.1'],
            'penetration test': ['A.14.2.5'],
            'pen test': ['A.14.2.5'],

            # Supplier/Third Party (Medium Priority)
            'supplier': ['A.15.1.1', 'A.15.2.1'],
            'third-party': ['A.15.1.1', 'A.15.2.1'],
            'contractor': ['A.15.1.1'],
            'vendor': ['A.15.1.1', 'A.15.2.1'],
            'outsourcing': ['A.15.1.1'],

            # Personnel Security (Medium Priority)
            'background check': ['A.7.1.1'],
            'screening': ['A.7.1.1'],
            'security training': ['A.7.2.2'],
            'awareness': ['A.7.2.2'],
            'security clearance': ['A.7.1.1'],
            'confidentiality agreement': ['A.7.1.2'],
            'nda': ['A.7.1.2'],

            # Asset Management (Medium Priority)
            'asset': ['A.8.1.1', 'A.8.1.2'],
            'inventory': ['A.8.1.1'],
            'asset register': ['A.8.1.1'],
            'configuration management': ['A.8.1.3'],
            'cmdb': ['A.8.1.1'],

            # Business Continuity (Medium Priority)
            'backup': ['A.12.3.1'],
            'disaster recovery': ['A.17.1.2'],
            'business continuity': ['A.17.1.1'],
            'rpo': ['A.12.3.1'],
            'rto': ['A.17.1.2'],

            # Monitoring & Logging (Medium Priority)
            'log': ['A.12.4.1'],
            'logging': ['A.12.4.1'],
            'audit log': ['A.12.4.1'],
            'event monitoring': ['A.12.4.1'],
            'security monitoring': ['A.12.4.1', 'A.16.1.1'],

            # Change Management (Low Priority)
            'change management': ['A.12.1.2'],
            'change control': ['A.12.1.2'],
            'development': ['A.14.2.1'],
            'testing': ['A.14.2.8']
        }

        # Context-aware keyword matching with scoring
        keyword_scores = {}
        control_matches = {}

        for keyword, control_list in keyword_control_mapping.items():
            if keyword in document_text:
                # Calculate keyword context score
                context_score = self._calculate_keyword_context_score(document_text, keyword)

                for control in control_list:
                    if control not in control_matches:
                        control_matches[control] = []
                    control_matches[control].append({
                        'keyword': keyword,
                        'score': context_score,
                        'priority': self._get_keyword_priority(keyword)
                    })

        # Use concept matches to identify additional relevant controls
        concept_matches = features.get('concept_matches', {})

        # Enhanced concept to control mapping
        concept_control_mapping = {
            'access_control': ['A.9.1.1', 'A.9.1.2', 'A.9.2.1', 'A.9.2.2', 'A.9.4.2'],
            'technical_security': ['A.10.1.1', 'A.10.1.2', 'A.8.2.2', 'A.13.2.1'],
            'incident_management': ['A.16.1.1', 'A.16.1.2', 'A.16.1.3', 'A.16.1.4'],
            'physical_security': ['A.11.1.1', 'A.11.1.2', 'A.11.2.3'],
            'information_security': ['A.13.1.1', 'A.13.1.3', 'A.13.2.1'],
            'isms_concepts': ['5.1', '5.2', '5.3', '5.4'],
            'risk_management': ['6.1.2', '8.2'],
            'human_resources': ['A.7.1.1', 'A.7.2.2', 'A.7.3.1'],
            'compliance_governance': ['5.1', '5.2', '9.2', '9.3']
        }

        # Map concepts to controls with weighting
        for concept, control_list in concept_control_mapping.items():
            concept_data = concept_matches.get(concept, [])
            if concept_data:
                # Calculate average relevance score for the concept
                avg_relevance = sum(match.relevance_score for match in concept_data) / len(concept_data)

                for control in control_list:
                    if control not in control_matches:
                        control_matches[control] = []
                    control_matches[control].append({
                        'keyword': f'concept_{concept}',
                        'score': avg_relevance,
                        'priority': 'medium'
                    })

        # Score and rank controls
        scored_controls = []
        for control, matches in control_matches.items():
            total_score = 0
            priority_boost = 0

            for match in matches:
                total_score += match['score']
                if match['priority'] == 'high':
                    priority_boost += 0.3
                elif match['priority'] == 'medium':
                    priority_boost += 0.1

            # Final score considers both keyword matches and priority
            final_score = total_score + priority_boost + (len(matches) * 0.1)  # Multi-match bonus
            scored_controls.append((control, final_score, len(matches)))

        # Sort by score (descending) and return top controls
        scored_controls.sort(key=lambda x: (x[1], x[2]), reverse=True)

        # Return top 15 controls with highest scores
        return [control[0] for control in scored_controls[:15]]

    def _calculate_keyword_context_score(self, text: str, keyword: str) -> float:
        """Calculate context-aware score for keyword matches"""
        base_score = 0.5

        # Find all occurrences and analyze context
        keyword_pattern = re.compile(r'\b' + re.escape(keyword) + r'\b', re.IGNORECASE)
        matches = list(keyword_pattern.finditer(text))

        if not matches:
            return 0.0

        total_context_score = 0
        for match in matches:
            # Get surrounding context (100 chars each side)
            start = max(0, match.start() - 100)
            end = min(len(text), match.end() + 100)
            context = text[start:end].lower()

            context_score = base_score

            # Boost for implementation language
            impl_indicators = ['implemented', 'deployed', 'configured', 'established', 'enabled']
            context_score += 0.2 * sum(1 for indicator in impl_indicators if indicator in context)

            # Boost for policy/procedure language
            policy_indicators = ['policy', 'procedure', 'standard', 'requirement', 'must', 'shall']
            context_score += 0.15 * sum(1 for indicator in policy_indicators if indicator in context)

            # Boost for specific technical details
            technical_indicators = ['configuration', 'setting', 'parameter', 'version', 'protocol']
            context_score += 0.1 * sum(1 for indicator in technical_indicators if indicator in context)

            # Penalize for negative context
            negative_indicators = ['not', 'no', 'unable', 'failed', 'missing', 'lacking']
            context_score -= 0.2 * sum(1 for indicator in negative_indicators if indicator in context)

            total_context_score += max(0.1, min(1.0, context_score))

        return total_context_score / len(matches) if matches else 0.0

    def _get_keyword_priority(self, keyword: str) -> str:
        """Determine priority level for keywords based on security impact"""
        high_priority_keywords = {
            'mfa', 'multi-factor', '2fa', 'aes-256', 'tls 1.3', 'encryption',
            'incident', '4 hours', '72 hours', 'firewall', 'vulnerability',
            'privileged', 'biometric', 'hsm', 'siem'
        }

        medium_priority_keywords = {
            'backup', 'patch', 'training', 'awareness', 'supplier',
            'physical', 'monitoring', 'log', 'asset', 'classification'
        }

        if keyword in high_priority_keywords:
            return 'high'
        elif keyword in medium_priority_keywords:
            return 'medium'
        else:
            return 'low'

    def identify_relevant_clauses(self, features: Dict[str, Any]) -> List[str]:
        """Identify relevant clause IDs based on document features"""
        relevant_clauses = []

        # Use concept matches to identify relevant clauses
        concept_matches = features.get('concept_matches', {})

        if concept_matches.get('leadership', []):
            relevant_clauses.append('5')
        if concept_matches.get('planning', []):
            relevant_clauses.append('6')
        if concept_matches.get('support', []):
            relevant_clauses.append('7')
        if concept_matches.get('operation', []):
            relevant_clauses.append('8')
        if concept_matches.get('performance_evaluation', []):
            relevant_clauses.append('9')
        if concept_matches.get('improvement', []):
            relevant_clauses.append('10')

        return relevant_clauses

    def calculate_analysis_confidence(self, control_assessments: List, clause_assessments: List) -> float:
        """Calculate overall analysis confidence"""
        if not control_assessments and not clause_assessments:
            return 0.3

        total_confidence = 0
        count = 0

        for assessment in control_assessments:
            if hasattr(assessment, 'confidence_score'):
                total_confidence += assessment.confidence_score
                count += 1

        for assessment in clause_assessments:
            if hasattr(assessment, 'confidence_score'):
                total_confidence += assessment.confidence_score
                count += 1

        return total_confidence / count if count > 0 else 0.5

    def calculate_coverage_percentage(self, control_assessments: List, clause_assessments: List) -> float:
        """Calculate coverage percentage"""
        total_controls = 93  # Total ISO controls
        total_clauses = 7   # Management clauses 4-10

        covered_controls = len(control_assessments)
        covered_clauses = len(clause_assessments)

        total_coverage = (covered_controls + covered_clauses) / (total_controls + total_clauses)
        return min(total_coverage * 100, 100)

    async def enhanced_control_relevance_analysis(self, document_text: str, control_ids: List[str]) -> Dict[str, float]:
        """Enhanced control relevance analysis using vector search when available"""
        relevance_scores = {}

        for control_id in control_ids:
            # Base relevance using traditional methods
            base_score = self._calculate_control_relevance(document_text, control_id)

            # Enhanced relevance using vector search
            if self.use_vector_search and self.vector_store:
                try:
                    from .vector_store import SearchQuery, VectorNamespace

                    # Search for similar control implementations
                    query = SearchQuery(
                        text=f"Control {control_id} implementation",
                        namespace=VectorNamespace.CONTROLS.value,
                        top_k=3,
                        filter_metadata={"control_id": control_id},
                        min_score=0.6
                    )

                    results = await self.vector_store.search(query)

                    if results:
                        # Calculate enhanced score based on semantic similarity to successful implementations
                        vector_scores = [result.score for result in results]
                        vector_boost = sum(vector_scores) / len(vector_scores) * 0.3
                        enhanced_score = min(base_score + vector_boost, 1.0)
                        relevance_scores[control_id] = enhanced_score
                    else:
                        relevance_scores[control_id] = base_score

                except Exception as e:
                    logger.warning(f"Vector search failed for control {control_id}, using base score: {e}")
                    relevance_scores[control_id] = base_score
            else:
                relevance_scores[control_id] = base_score

        return relevance_scores

    def _calculate_control_relevance(self, document_text: str, control_id: str) -> float:
        """Calculate base control relevance using traditional methods"""
        # Simplified relevance calculation
        text_lower = document_text.lower()
        control_lower = control_id.lower()

        # Direct mention of control
        if control_lower in text_lower:
            return 0.9

        # Partial matches and related terms would be implemented here
        # This is a simplified version
        return 0.3

    async def find_similar_document_patterns(self, document_features: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Find similar document patterns using vector search"""
        if not self.use_vector_search or not self.vector_store:
            return []

        try:
            from .vector_store import SearchQuery, VectorNamespace

            # Create search text from key features
            concept_matches = document_features.get('concept_matches', {})
            search_components = []

            # Extract key concepts for search
            for category, matches in concept_matches.items():
                if isinstance(matches, list):
                    for match in matches[:3]:  # Top 3 matches per category
                        if hasattr(match, 'concept'):
                            search_components.append(match.concept)

            search_text = " ".join(search_components)

            if not search_text:
                return []

            query = SearchQuery(
                text=search_text,
                namespace=VectorNamespace.ANALYSES.value,
                top_k=5,
                min_score=0.7
            )

            results = await self.vector_store.search(query)

            similar_patterns = []
            for result in results:
                pattern = {
                    "analysis_id": result.metadata.get("analysis_id"),
                    "document_type": result.metadata.get("document_type"),
                    "similarity_score": result.score,
                    "compliance_level": result.metadata.get("compliance_overview"),
                    "common_themes": search_components[:3]  # Top 3 themes
                }
                similar_patterns.append(pattern)

            return similar_patterns

        except Exception as e:
            logger.warning(f"Failed to find similar document patterns: {e}")
            return []