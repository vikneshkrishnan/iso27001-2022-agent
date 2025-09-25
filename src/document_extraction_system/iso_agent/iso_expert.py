"""
ISO 27001:2022 Expert Agent
Core module for analyzing documents against ISO 27001:2022 requirements
"""

import os
import json
import uuid
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import asdict

try:
    import openai
    from anthropic import Anthropic
except ImportError as e:
    logging.warning(f"LLM libraries not available: {e}")
    openai = None
    Anthropic = None

from ..iso_knowledge.annex_a_controls import (
    get_all_controls, get_control_by_id, search_controls,
    get_controls_by_category, Control
)
from ..iso_knowledge.management_clauses import (
    get_all_clauses, get_clause_by_id, search_requirements,
    get_all_requirements, Clause, Requirement
)
from ..iso_knowledge.agent_response import (
    ISOAgentCard, ISOConsultationCard, ComplianceStatus, Priority,
    DocumentType, DocumentClassification, ComplianceOverview,
    CategoryScore, ControlAssessment, ClauseAssessment,
    GapAnalysis, Recommendation
)
from .semantic_analyzer import ISOSemanticAnalyzer
from .intelligent_assessor import IntelligentControlAssessor
from .enhanced_llm import EnhancedLLMProcessor
from .recommendation_engine import SmartRecommendationEngine
from .iso_validator import ISOResponseValidator, ValidationLevel
from .vector_store import PineconeVectorStore, SearchResult
from .semantic_search import SemanticSearchEngine
from .vector_manager import VectorIndexManager

logger = logging.getLogger(__name__)


class ISOExpertAgent:
    """
    Expert agent for ISO 27001:2022 analysis and consultation
    """

    def __init__(self, openai_api_key: Optional[str] = None, anthropic_api_key: Optional[str] = None):
        """Initialize the ISO expert agent with LLM capabilities"""
        self.openai_client = None
        self.anthropic_client = None

        # Initialize OpenAI if available
        if openai and (openai_api_key or os.getenv("OPENAI_API_KEY")):
            try:
                from openai import AsyncOpenAI
                self.openai_client = AsyncOpenAI(api_key=openai_api_key or os.getenv("OPENAI_API_KEY"))
                logger.info("OpenAI client initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize OpenAI client: {e}")

        # Initialize Anthropic if available
        if Anthropic and (anthropic_api_key or os.getenv("ANTHROPIC_API_KEY")):
            try:
                self.anthropic_client = Anthropic(
                    api_key=anthropic_api_key or os.getenv("ANTHROPIC_API_KEY")
                )
                logger.info("Anthropic client initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize Anthropic client: {e}")

        # Load knowledge bases
        self.controls = get_all_controls()
        self.clauses = get_all_clauses()

        # Initialize vector storage first
        self.vector_store = PineconeVectorStore()

        # Initialize enhanced analysis components with vector store
        self.semantic_analyzer = ISOSemanticAnalyzer(vector_store=self.vector_store)
        self.intelligent_assessor = IntelligentControlAssessor()
        self.enhanced_llm = EnhancedLLMProcessor(
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY")
        )
        self.recommendation_engine = SmartRecommendationEngine()
        self.validator = ISOResponseValidator(ValidationLevel.GUIDED)

        # Initialize semantic search and vector manager
        self.semantic_search = SemanticSearchEngine(self.vector_store)
        self.vector_manager = VectorIndexManager(self.vector_store)

        logger.info(f"ISO Expert Agent initialized with {len(self.controls)} controls and {len(self.clauses)} clauses")
        logger.info("Enhanced analysis components loaded: semantic analyzer, intelligent assessor, enhanced LLM, recommendation engine, ISO validator")
        logger.info("Vector storage and semantic search capabilities enabled")

    async def analyze_document(
        self,
        document_text: str,
        document_info: Dict[str, Any],
        analysis_scope: str = "comprehensive"
    ) -> ISOAgentCard:
        """
        Analyze a document for ISO 27001:2022 compliance

        Args:
            document_text: Extracted text from document
            document_info: Document metadata (filename, size, etc.)
            analysis_scope: "comprehensive", "controls_only", "clauses_only"

        Returns:
            ISOAgentCard with complete analysis results
        """
        logger.info(f"Starting ISO analysis for document: {document_info.get('filename', 'unknown')}")

        analysis_id = str(uuid.uuid4())
        timestamp = datetime.now()

        try:
            # Step 1: Classify document
            document_classification = await self._classify_document(document_text, document_info)

            # Step 2: Analyze against controls (if scope includes controls)
            control_assessments = []
            if analysis_scope in ["comprehensive", "controls_only"]:
                control_assessments = await self._analyze_controls(document_text, document_classification)

            # Step 3: Analyze against management clauses (if scope includes clauses)
            clause_assessments = []
            if analysis_scope in ["comprehensive", "clauses_only"]:
                clause_assessments = await self._analyze_clauses(document_text, document_classification)

            # Step 4: Calculate category scores
            category_scores = self._calculate_category_scores(control_assessments)

            # Step 5: Generate compliance overview
            compliance_overview = self._generate_compliance_overview(
                control_assessments, clause_assessments, category_scores
            )

            # Step 6: Perform gap analysis
            gap_analysis = self._perform_gap_analysis(control_assessments, clause_assessments)

            # Step 7: Generate recommendations
            recommendations = await self._generate_recommendations(
                gap_analysis, control_assessments, clause_assessments
            )

            # Step 8: Identify risk areas and quick wins
            risk_areas, quick_wins = self._identify_risk_areas_and_quick_wins(
                control_assessments, clause_assessments, gap_analysis
            )

            # Step 9: Create implementation roadmap
            implementation_roadmap = self._create_implementation_roadmap(recommendations)

            # Step 10: Calculate quality metrics
            analysis_confidence = self._calculate_analysis_confidence(
                control_assessments, clause_assessments
            )
            coverage_percentage = self._calculate_coverage_percentage(
                control_assessments, clause_assessments
            )

            # Step 11: Generate executive summary and next steps
            executive_summary = await self._generate_executive_summary(
                compliance_overview, gap_analysis, recommendations
            )
            next_steps = self._generate_next_steps(recommendations, quick_wins)

            # Create agent card
            agent_card = ISOAgentCard(
                analysis_id=analysis_id,
                timestamp=timestamp,
                document_info=document_info,
                document_classification=document_classification,
                compliance_overview=compliance_overview,
                category_scores=category_scores,
                control_assessments=control_assessments,
                clause_assessments=clause_assessments,
                gap_analysis=gap_analysis,
                recommendations=recommendations,
                risk_areas=risk_areas,
                quick_wins=quick_wins,
                implementation_roadmap=implementation_roadmap,
                analysis_confidence=analysis_confidence,
                coverage_percentage=coverage_percentage,
                executive_summary=executive_summary,
                next_steps=next_steps
            )

            # Validate the complete response for ISO compliance
            validation_result = self.validator.validate_response(executive_summary, "analysis_summary")
            if not validation_result.is_valid:
                raise Exception(f"ISO analysis validation failed: {'; '.join(validation_result.validation_errors)}")

            # Store analysis results in vector database for future semantic search
            try:
                indexing_success = await self.vector_manager.index_iso_agent_analysis(agent_card)
                if indexing_success:
                    logger.info(f"Analysis results successfully indexed for semantic search")
                else:
                    logger.warning(f"Failed to index analysis results - semantic search may be limited")
            except Exception as e:
                logger.warning(f"Vector indexing failed but analysis continues: {e}")

            logger.info(f"ISO analysis completed and validated successfully. Analysis ID: {analysis_id}")
            return agent_card

        except Exception as e:
            logger.error(f"Error during ISO analysis: {e}")
            raise

    async def consult(self, question: str) -> ISOConsultationCard:
        """
        Provide expert consultation on ISO 27001:2022 topics

        Args:
            question: User's question about ISO 27001:2022

        Returns:
            ISOConsultationCard with expert response
        """
        query_id = str(uuid.uuid4())
        timestamp = datetime.now()

        try:
            # Find relevant controls and clauses using both traditional and semantic search
            relevant_controls = await self._find_relevant_controls_enhanced(question)
            relevant_clauses = await self._find_relevant_clauses_enhanced(question)

            # Generate expert answer using LLM
            answer = await self._generate_expert_answer(
                question, relevant_controls, relevant_clauses
            )

            # Extract implementation guidance
            implementation_guidance = self._extract_implementation_guidance(
                relevant_controls, relevant_clauses
            )

            # Identify related topics
            related_topics = self._identify_related_topics(question, relevant_controls, relevant_clauses)

            # Calculate confidence score
            confidence_score = self._calculate_consultation_confidence(
                question, relevant_controls, relevant_clauses
            )

            # Compile sources
            sources = self._compile_sources(relevant_controls, relevant_clauses)

            # Validate consultation response for ISO compliance
            validation_result = self.validator.validate_response(answer, "consultation_response")
            if not validation_result.is_valid:
                raise Exception(f"ISO consultation validation failed: {'; '.join(validation_result.validation_errors)}")

            # Store consultation in vector database for future reference
            try:
                indexing_success = await self.vector_manager.index_consultation_response(
                    question, answer,
                    [ctrl.id for ctrl in relevant_controls],
                    [clause.id for clause in relevant_clauses],
                    confidence_score
                )
                if indexing_success:
                    logger.info(f"Consultation response successfully indexed for semantic search")
            except Exception as e:
                logger.warning(f"Failed to index consultation response: {e}")

            consultation_card = ISOConsultationCard(
                query_id=query_id,
                timestamp=timestamp,
                question=question,
                answer=answer,
                relevant_controls=[ctrl.id for ctrl in relevant_controls],
                relevant_clauses=[clause.id for clause in relevant_clauses],
                implementation_guidance=implementation_guidance,
                related_topics=related_topics,
                confidence_score=confidence_score,
                sources=sources
            )

            logger.info(f"ISO consultation completed and validated. Query ID: {query_id}")
            return consultation_card

        except Exception as e:
            logger.error(f"Error during ISO consultation: {e}")
            raise

    async def _classify_document(self, text: str, document_info: Dict[str, Any]) -> DocumentClassification:
        """Classify document type and determine ISO relevance using enhanced analysis with historical patterns"""
        try:
            # Step 1: Extract document features for pattern matching
            document_features = self.semantic_analyzer.extract_document_features(text)

            # Step 2: Find similar historical analyses for pattern enhancement
            similar_patterns = await self._find_similar_document_patterns(document_features)

            # Step 3: Use enhanced LLM service for classification with pattern context
            classification = await self.enhanced_llm.classify_document(text, document_info)

            # Step 4: Enhance classification with historical pattern insights
            if similar_patterns:
                classification = await self._enhance_classification_with_patterns(
                    classification, similar_patterns, document_features
                )

            # Step 5: Validate classification response
            classification_dict = {
                'document_type': classification.document_type.value,
                'iso_relevance': classification.iso_relevance,
                'confidence_score': classification.confidence_score,
                'applicable_categories': classification.applicable_categories,
                'primary_focus_areas': classification.primary_focus_areas
            }

            validation_result = self.validator.validate_document_classification(classification_dict)
            if not validation_result.is_valid:
                raise Exception(f"Document classification validation failed: {'; '.join(validation_result.validation_errors)}")

            logger.info(f"Document classified as {classification.document_type.value} with {len(similar_patterns)} pattern matches")
            return classification

        except Exception as e:
            logger.error(f"Enhanced document classification failed: {e}")
            raise Exception(f"ISO document classification failed - cannot provide non-ISO response: {e}")



    async def _analyze_controls(self, text: str, classification: DocumentClassification) -> List[ControlAssessment]:
        """Analyze document against applicable ISO controls using enhanced assessment with semantic search"""
        # Extract document features for semantic analysis
        document_features = self.semantic_analyzer.extract_document_features(text)

        # Step 1: Use semantic search to find similar implementations if knowledge base is available
        relevant_control_ids = await self._get_semantic_relevant_controls(text, document_features)

        # Step 2: Fallback to traditional semantic analyzer if no vector results
        if not relevant_control_ids:
            relevant_control_ids = self.semantic_analyzer.identify_relevant_controls(document_features)

        # Get control objects
        relevant_controls = [get_control_by_id(control_id) for control_id in relevant_control_ids]
        relevant_controls = [ctrl for ctrl in relevant_controls if ctrl is not None]

        # Step 3: If no semantic matches, use document classification categories
        if not relevant_controls:
            for category in classification.applicable_categories:
                relevant_controls.extend(get_controls_by_category(category))

        # Step 4: Enhanced relevance analysis using vector similarity if available
        if relevant_controls and self.vector_store:
            try:
                relevance_scores = await self.semantic_analyzer.enhanced_control_relevance_analysis(
                    text, [ctrl.id for ctrl in relevant_controls]
                )

                # Sort controls by relevance score
                relevant_controls = sorted(
                    relevant_controls,
                    key=lambda ctrl: relevance_scores.get(ctrl.id, 0.0),
                    reverse=True
                )[:15]  # Keep top 15 most relevant

            except Exception as e:
                logger.warning(f"Enhanced relevance analysis failed, using standard approach: {e}")

        # Step 5: If still no matches, use top controls by semantic relevance
        if not relevant_controls:
            all_controls = list(self.controls.values())
            relevant_controls = self._rank_controls_by_basic_relevance(
                document_features, all_controls
            )[:15]

        # Step 6: Assess each relevant control
        assessments = []
        for control in relevant_controls:
            assessment = await self.intelligent_assessor.assess_control(
                text, control, document_features
            )
            assessments.append(assessment)

        logger.info(f"Analyzed {len(assessments)} controls for document")
        return assessments


    async def _analyze_clauses(self, text: str, classification: DocumentClassification) -> List[ClauseAssessment]:
        """Analyze document against management clauses using enhanced assessment"""
        document_features = self.semantic_analyzer.extract_document_features(text)
        relevant_clauses = self.semantic_analyzer.identify_relevant_clauses(document_features)

        assessments = []
        all_requirements = get_all_requirements()

        for clause_id, requirement in all_requirements:
            if clause_id in relevant_clauses or len(relevant_clauses) == 0:
                assessment = await self.intelligent_assessor.assess_clause(
                    text, clause_id, requirement, document_features
                )
                assessments.append(assessment)

        # If no relevant clauses found, assess top priority clauses
        if not assessments and len(relevant_clauses) == 0:
            priority_clauses = ["4", "5", "6", "7", "8", "9", "10"]
            for clause_id, requirement in all_requirements:
                if clause_id.split('.')[0] in priority_clauses:
                    assessment = await self.intelligent_assessor.assess_clause(
                        text, clause_id, requirement, document_features
                    )
                    assessments.append(assessment)

        return assessments


    def _calculate_category_scores(self, control_assessments: List[ControlAssessment]) -> List[CategoryScore]:
        """Calculate scores for each control category"""
        categories = ["organizational", "people", "physical", "technological"]
        category_scores = []

        for category in categories:
            category_controls = get_controls_by_category(category)
            category_assessments = [
                assessment for assessment in control_assessments
                if any(ctrl.id == assessment.control_id for ctrl in category_controls)
            ]

            if not category_assessments:
                continue

            total_controls = len(category_controls)
            assessed_controls = len(category_assessments)

            compliant_count = sum(1 for a in category_assessments if a.status == ComplianceStatus.COMPLIANT)
            partially_compliant_count = sum(1 for a in category_assessments if a.status == ComplianceStatus.PARTIALLY_COMPLIANT)
            non_compliant_count = sum(1 for a in category_assessments if a.status == ComplianceStatus.NON_COMPLIANT)
            not_applicable_count = sum(1 for a in category_assessments if a.status == ComplianceStatus.NOT_APPLICABLE)

            # Calculate overall score (0-100)
            if assessed_controls > 0:
                score = (compliant_count * 100 + partially_compliant_count * 50) / assessed_controls
            else:
                score = 0

            # Determine maturity level
            if score >= 90:
                maturity_level = "optimizing"
            elif score >= 75:
                maturity_level = "quantified"
            elif score >= 60:
                maturity_level = "defined"
            elif score >= 40:
                maturity_level = "managed"
            else:
                maturity_level = "basic"

            category_scores.append(CategoryScore(
                category=category,
                total_controls=total_controls,
                assessed_controls=assessed_controls,
                compliant_count=compliant_count,
                partially_compliant_count=partially_compliant_count,
                non_compliant_count=non_compliant_count,
                not_applicable_count=not_applicable_count,
                overall_score=score,
                maturity_level=maturity_level
            ))

        return category_scores

    def _generate_compliance_overview(
        self,
        control_assessments: List[ControlAssessment],
        clause_assessments: List[ClauseAssessment],
        category_scores: List[CategoryScore]
    ) -> ComplianceOverview:
        """Generate compliance overview using enhanced analysis"""
        return self.recommendation_engine.generate_compliance_overview(
            control_assessments, clause_assessments, category_scores
        )

    def _perform_gap_analysis(
        self,
        control_assessments: List[ControlAssessment],
        clause_assessments: List[ClauseAssessment]
    ) -> GapAnalysis:
        """Perform gap analysis using enhanced analysis"""
        return self.recommendation_engine.perform_gap_analysis(
            control_assessments, clause_assessments
        )

    async def _generate_recommendations(
        self,
        gap_analysis: GapAnalysis,
        control_assessments: List[ControlAssessment],
        clause_assessments: List[ClauseAssessment]
    ) -> List[Recommendation]:
        """Generate recommendations using enhanced recommendation engine"""
        # Create recommendation context
        from .recommendation_engine import RecommendationContext
        context = RecommendationContext(
            organization_maturity="developing",
            document_type="policy",
            primary_focus_areas=["access_control", "incident_management"],
            available_resources="medium",
            compliance_urgency="medium",
            industry_sector="general"
        )

        return await self.recommendation_engine.generate_comprehensive_recommendations(
            control_assessments, clause_assessments, context
        )

    def _identify_risk_areas_and_quick_wins(
        self,
        control_assessments: List[ControlAssessment],
        clause_assessments: List[ClauseAssessment],
        gap_analysis: GapAnalysis
    ) -> Tuple[List[str], List[str]]:
        """Identify risk areas and quick wins using enhanced analysis"""
        return self.recommendation_engine.identify_risk_areas_and_quick_wins(
            control_assessments, clause_assessments, gap_analysis
        )

    def _create_implementation_roadmap(
        self, recommendations: List[Recommendation]
    ) -> List[Dict[str, Any]]:
        """Create implementation roadmap using enhanced recommendation engine"""
        return self.recommendation_engine.generate_implementation_roadmap(recommendations)

    def _calculate_analysis_confidence(
        self,
        control_assessments: List[ControlAssessment],
        clause_assessments: List[ClauseAssessment]
    ) -> float:
        """Calculate analysis confidence using enhanced analysis"""
        return self.semantic_analyzer.calculate_analysis_confidence(
            control_assessments, clause_assessments
        )

    def _calculate_coverage_percentage(
        self,
        control_assessments: List[ControlAssessment],
        clause_assessments: List[ClauseAssessment]
    ) -> float:
        """Calculate coverage percentage using enhanced analysis"""
        return self.semantic_analyzer.calculate_coverage_percentage(
            control_assessments, clause_assessments
        )

    async def _generate_executive_summary(
        self,
        compliance_overview: ComplianceOverview,
        gap_analysis: GapAnalysis,
        recommendations: List[Recommendation]
    ) -> str:
        """Generate executive summary using enhanced LLM service"""
        try:
            return await self.enhanced_llm.generate_executive_summary(
                compliance_overview, gap_analysis, recommendations
            )
        except Exception as e:
            logger.error(f"Enhanced LLM executive summary failed: {e}")
            raise Exception(f"ISO executive summary generation failed - cannot provide non-ISO response: {e}")

    def _generate_next_steps(
        self,
        recommendations: List[Recommendation],
        quick_wins: List[str]
    ) -> List[str]:
        """Generate next steps using enhanced analysis"""
        return self.recommendation_engine.generate_next_steps(recommendations, quick_wins)

    async def _find_relevant_controls_enhanced(self, question: str) -> List[Control]:
        """Find controls relevant to the question using both semantic and traditional search"""
        # Traditional semantic analysis
        document_features = self.semantic_analyzer.extract_document_features(question)
        traditional_control_ids = self.semantic_analyzer.identify_relevant_controls(document_features)
        traditional_controls = [get_control_by_id(control_id) for control_id in traditional_control_ids]
        traditional_controls = [ctrl for ctrl in traditional_controls if ctrl is not None]

        # Enhanced semantic search using vector database
        vector_controls = []
        try:
            similar_consultations = await self.semantic_search.search_expert_consultations(question)
            for result in similar_consultations[:3]:
                if result.iso_context.get("referenced_controls"):
                    for control_id in result.iso_context["referenced_controls"]:
                        control = get_control_by_id(control_id)
                        if control and control not in vector_controls:
                            vector_controls.append(control)

            # Also search for similar control implementations
            control_implementations = await self.semantic_search.search_control_implementations("", question)
            for result in control_implementations[:5]:
                control_id = result.result.metadata.get("control_id")
                if control_id:
                    control = get_control_by_id(control_id)
                    if control and control not in vector_controls:
                        vector_controls.append(control)

        except Exception as e:
            logger.warning(f"Vector search for controls failed, using traditional only: {e}")

        # Combine and deduplicate results, prioritizing vector search results
        combined_controls = vector_controls[:3]  # Top 3 from vector search

        # Add traditional results that aren't already included
        for ctrl in traditional_controls:
            if ctrl not in combined_controls and len(combined_controls) < 5:
                combined_controls.append(ctrl)

        # Fallback to traditional search if no results
        if not combined_controls:
            combined_controls = search_controls(question)[:5]

        return combined_controls

    async def _find_relevant_clauses_enhanced(self, question: str) -> List[Clause]:
        """Find clauses relevant to the question using both semantic and traditional search"""
        # Traditional semantic analysis
        document_features = self.semantic_analyzer.extract_document_features(question)
        traditional_clause_ids = self.semantic_analyzer.identify_relevant_clauses(document_features)
        traditional_clauses = [get_clause_by_id(clause_id) for clause_id in traditional_clause_ids]
        traditional_clauses = [clause for clause in traditional_clauses if clause is not None]

        # Enhanced semantic search using vector database
        vector_clauses = []
        try:
            similar_consultations = await self.semantic_search.search_expert_consultations(question)
            for result in similar_consultations[:3]:
                if result.iso_context.get("referenced_clauses"):
                    for clause_id in result.iso_context["referenced_clauses"]:
                        clause = get_clause_by_id(clause_id)
                        if clause and clause not in vector_clauses:
                            vector_clauses.append(clause)

        except Exception as e:
            logger.warning(f"Vector search for clauses failed, using traditional only: {e}")

        # Combine and deduplicate results
        combined_clauses = vector_clauses[:2]  # Top 2 from vector search

        # Add traditional results that aren't already included
        for clause in traditional_clauses:
            if clause not in combined_clauses and len(combined_clauses) < 3:
                combined_clauses.append(clause)

        # Fallback to traditional search if no results
        if not combined_clauses:
            relevant_requirements = search_requirements(question)
            clause_ids = list(set(clause_id for clause_id, _ in relevant_requirements))
            combined_clauses = [get_clause_by_id(clause_id) for clause_id in clause_ids if get_clause_by_id(clause_id)][:3]

        return combined_clauses

    async def search_similar_documents(self, document_content: str, document_type: str = None) -> List[Dict[str, Any]]:
        """Search for documents similar to the provided content"""
        try:
            results = await self.semantic_search.search_similar_documents(
                document_content, document_type, top_k=10
            )

            return [{
                "analysis_id": result.result.metadata.get("analysis_id"),
                "document_filename": result.result.metadata.get("document_filename"),
                "document_type": result.result.metadata.get("document_type"),
                "similarity_score": result.result.score,
                "compliance_overview": result.result.metadata.get("compliance_overview"),
                "analysis_date": result.result.metadata.get("analysis_date"),
                "relevance_explanation": result.relevance_explanation
            } for result in results]

        except Exception as e:
            logger.error(f"Failed to search similar documents: {e}")
            return []

    async def get_control_implementation_examples(self, control_id: str) -> List[Dict[str, Any]]:
        """Get examples of how a specific control has been implemented"""
        try:
            results = await self.semantic_search.search_control_implementations(control_id)

            return [{
                "analysis_id": result.result.metadata.get("analysis_id"),
                "compliance_status": result.result.metadata.get("compliance_status"),
                "implementation_approach": result.result.metadata.get("implementation_approach"),
                "evidence_strength": result.result.metadata.get("evidence_count", 0),
                "gaps_identified": result.result.metadata.get("gaps_identified", []),
                "similarity_score": result.result.score,
                "relevance_explanation": result.relevance_explanation
            } for result in results]

        except Exception as e:
            logger.error(f"Failed to get control implementation examples: {e}")
            return []

    async def _get_semantic_relevant_controls(self, document_text: str, document_features: Dict[str, Any]) -> List[str]:
        """Use semantic search to find relevant controls based on document content"""
        try:
            if not self.vector_store or not self.semantic_search:
                return []

            # Search for similar control implementations
            search_results = await self.semantic_search.find_similar_controls(
                document_text,
                top_k=20,
                min_similarity=0.6
            )

            if not search_results:
                return []

            # Extract control IDs from search results
            control_ids = []
            for result in search_results:
                control_id = result.metadata.get('control_id')
                if control_id and control_id not in control_ids:
                    control_ids.append(control_id)

            logger.info(f"Found {len(control_ids)} semantically relevant controls")
            return control_ids[:15]  # Limit to top 15

        except Exception as e:
            logger.warning(f"Semantic control search failed: {e}")
            return []

    def _rank_controls_by_basic_relevance(self, document_features: Dict[str, Any], controls: List[Control]) -> List[Control]:
        """Fallback method to rank controls by basic relevance when semantic search unavailable"""
        try:
            # Simple scoring based on concept matches
            control_scores = {}
            concept_matches = document_features.get('concept_matches', {})

            for control in controls:
                score = 0.0

                # Score based on category match
                for category, matches in concept_matches.items():
                    if control.category == category.replace('_', ' '):
                        score += len(matches) * 0.3

                # Score based on title/description keyword overlap
                control_text = f"{control.title} {control.description}".lower()
                for category, matches in concept_matches.items():
                    for match in matches[:3]:  # Top 3 matches per category
                        if hasattr(match, 'concept') and match.concept.lower() in control_text:
                            score += match.relevance_score * 0.2

                control_scores[control.id] = score

            # Sort controls by score
            sorted_controls = sorted(
                controls,
                key=lambda ctrl: control_scores.get(ctrl.id, 0.0),
                reverse=True
            )

            return sorted_controls

        except Exception as e:
            logger.warning(f"Basic control ranking failed: {e}")
            return controls

    async def _find_similar_document_patterns(self, document_features: Dict[str, Any]) -> List[SearchResult]:
        """Find similar document patterns from historical analyses"""
        try:
            if not self.semantic_search:
                return []

            return await self.semantic_search.find_similar_analyses(document_features)

        except Exception as e:
            logger.warning(f"Pattern search failed: {e}")
            return []

    async def _enhance_classification_with_patterns(self, classification: DocumentClassification,
                                                  similar_patterns: List[SearchResult],
                                                  document_features: Dict[str, Any]) -> DocumentClassification:
        """Enhance classification using insights from similar historical patterns"""
        try:
            if not similar_patterns:
                return classification

            # Aggregate insights from similar patterns
            pattern_categories = set()
            pattern_focus_areas = set()
            confidence_boosts = []

            for pattern in similar_patterns[:3]:  # Top 3 most similar
                metadata = pattern.metadata

                # Collect categories from similar documents
                categories = metadata.get("applicable_categories", [])
                if isinstance(categories, list):
                    pattern_categories.update(categories)

                # Collect focus areas
                focus_areas = metadata.get("primary_focus_areas", [])
                if isinstance(focus_areas, list):
                    pattern_focus_areas.update(focus_areas)

                # Consider similarity score for confidence adjustment
                confidence_boosts.append(pattern.score * 0.1)  # Small confidence boost

            # Enhance applicable categories
            enhanced_categories = list(set(classification.applicable_categories) | pattern_categories)

            # Enhance focus areas with pattern insights
            enhanced_focus_areas = list(set(classification.primary_focus_areas) | pattern_focus_areas)

            # Adjust confidence based on pattern matching
            confidence_adjustment = sum(confidence_boosts) / len(confidence_boosts) if confidence_boosts else 0
            enhanced_confidence = min(classification.confidence_score + confidence_adjustment, 1.0)

            # Create enhanced classification
            from ..iso_knowledge.agent_response import DocumentClassification
            enhanced_classification = DocumentClassification(
                document_type=classification.document_type,
                confidence_score=enhanced_confidence,
                iso_relevance=classification.iso_relevance,
                applicable_categories=enhanced_categories[:4],  # Limit to top 4
                primary_focus_areas=enhanced_focus_areas[:5]    # Limit to top 5
            )

            logger.info(f"Enhanced classification with {len(similar_patterns)} pattern insights")
            return enhanced_classification

        except Exception as e:
            logger.warning(f"Pattern enhancement failed: {e}")
            return classification

    async def initialize_knowledge_base(self) -> bool:
        """Initialize the vector knowledge base with ISO controls and clauses"""
        try:
            job = await self.vector_manager.bulk_index_knowledge_base()
            logger.info(f"Knowledge base indexing job started: {job.job_id}")

            # Wait for job completion (in production, this might be async)
            while job.status == "running":
                await asyncio.sleep(1)
                job = self.vector_manager.get_job_status(job.job_id)

            if job.status == "completed":
                logger.info(f"Knowledge base successfully indexed: {job.processed_documents}/{job.total_documents} documents")
                return True
            else:
                logger.error(f"Knowledge base indexing failed: {job.error_message}")
                return False

        except Exception as e:
            logger.error(f"Failed to initialize knowledge base: {e}")
            return False

    def get_vector_store_health(self) -> Dict[str, Any]:
        """Get health status of vector store"""
        return self.vector_store.health_check()

    def _find_relevant_controls(self, question: str) -> List[Control]:
        """Find controls relevant to the question using semantic analysis"""
        document_features = self.semantic_analyzer.extract_document_features(question)
        relevant_control_ids = self.semantic_analyzer.identify_relevant_controls(document_features)

        relevant_controls = [get_control_by_id(control_id) for control_id in relevant_control_ids]
        relevant_controls = [ctrl for ctrl in relevant_controls if ctrl is not None]

        # Fallback to traditional search if no semantic matches
        if not relevant_controls:
            relevant_controls = search_controls(question)

        return relevant_controls[:5]

    def _find_relevant_clauses(self, question: str) -> List[Clause]:
        """Find clauses relevant to the question using semantic analysis"""
        document_features = self.semantic_analyzer.extract_document_features(question)
        relevant_clause_ids = self.semantic_analyzer.identify_relevant_clauses(document_features)

        relevant_clauses = [get_clause_by_id(clause_id) for clause_id in relevant_clause_ids]
        relevant_clauses = [clause for clause in relevant_clauses if clause is not None]

        # Fallback to traditional search if no semantic matches
        if not relevant_clauses:
            relevant_requirements = search_requirements(question)
            clause_ids = list(set(clause_id for clause_id, _ in relevant_requirements))
            relevant_clauses = [get_clause_by_id(clause_id) for clause_id in clause_ids if get_clause_by_id(clause_id)]

        return relevant_clauses[:3]

    async def _generate_expert_answer(
        self,
        question: str,
        relevant_controls: List[Control],
        relevant_clauses: List[Clause]
    ) -> str:
        """Generate expert answer using enhanced LLM service"""
        try:
            return await self.enhanced_llm.generate_consultation_response(
                question, relevant_controls, relevant_clauses
            )
        except Exception as e:
            logger.error(f"Enhanced LLM consultation failed: {e}")
            raise Exception(f"ISO consultation failed - cannot provide non-ISO response: {e}")



    def _extract_implementation_guidance(
        self,
        relevant_controls: List[Control],
        relevant_clauses: List[Clause]
    ) -> List[str]:
        """Extract implementation guidance from relevant controls and clauses"""
        guidance = []

        for control in relevant_controls:
            guidance.extend(control.implementation_guidance[:2])

        for clause in relevant_clauses:
            for requirement in clause.requirements[:1]:
                guidance.extend(requirement.implementation_guidance[:2])

        unique_guidance = list(dict.fromkeys(guidance))
        return unique_guidance[:5]

    def _identify_related_topics(
        self,
        question: str,
        relevant_controls: List[Control],
        relevant_clauses: List[Clause]
    ) -> List[str]:
        """Identify related topics based on controls and clauses"""
        topics = set()

        for control in relevant_controls:
            topics.add(f"{control.category.title()} controls")

        for clause in relevant_clauses:
            topics.add(f"{clause.clause_type.value.replace('_', ' ').title()}")

        for control in relevant_controls:
            for related_id in control.related_controls[:2]:
                related_control = get_control_by_id(related_id)
                if related_control:
                    topics.add(f"Control {related_id} - {related_control.title}")

        return list(topics)[:5]

    def _calculate_consultation_confidence(
        self,
        question: str,
        relevant_controls: List[Control],
        relevant_clauses: List[Clause]
    ) -> float:
        """Calculate confidence in consultation response"""
        base_confidence = 0.6

        if relevant_controls:
            base_confidence += len(relevant_controls) * 0.1

        if relevant_clauses:
            base_confidence += len(relevant_clauses) * 0.15

        if len(question.split()) < 5:
            base_confidence -= 0.2

        return min(max(base_confidence, 0.3), 1.0)

    def _compile_sources(
        self,
        relevant_controls: List[Control],
        relevant_clauses: List[Clause]
    ) -> List[str]:
        """Compile source references"""
        sources = ["ISO/IEC 27001:2022 Information security management systems — Requirements"]

        for control in relevant_controls:
            sources.append(f"ISO 27001:2022 Annex A, Control {control.id}")

        for clause in relevant_clauses:
            sources.append(f"ISO 27001:2022 Clause {clause.id}")

        return list(dict.fromkeys(sources))

