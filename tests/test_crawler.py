#!/usr/bin/env python3
"""
Test script for ISO Knowledge Crawler
Tests crawler functionality with sample ISO resources
"""

import asyncio
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from document_extraction_system.iso_agent.iso_knowledge_crawler import ISOKnowledgeCrawler
from document_extraction_system.iso_agent.knowledge_cache import KnowledgeCache
from document_extraction_system.iso_agent.intelligent_assessor import IntelligentControlAssessor
from document_extraction_system.iso_knowledge.annex_a_controls import Control


async def test_basic_crawler():
    """Test basic crawler functionality"""
    print("=" * 60)
    print("Testing ISO Knowledge Crawler")
    print("=" * 60)

    # Initialize crawler and cache
    crawler = ISOKnowledgeCrawler()
    cache = KnowledgeCache()

    print("\n1. Testing crawler initialization...")
    initialized = await crawler.initialize_crawler()
    if initialized:
        print("   ✅ Crawler initialized successfully")
    else:
        print("   ❌ Failed to initialize crawler (Crawl4AI may not be installed)")
        return

    # Test control search
    print("\n2. Testing control search by keyword...")
    keyword = "multi-factor authentication"
    controls = await crawler.search_control_by_keyword(keyword)
    if controls:
        print(f"   ✅ Found {len(controls)} controls related to '{keyword}':")
        for control_id in controls[:3]:
            print(f"      - {control_id}")
    else:
        print(f"   ⚠️  No controls found for '{keyword}'")

    # Test control details crawling (using a mock/test approach)
    print("\n3. Testing control details crawling...")
    test_control_id = "A.9.4.2"
    control_knowledge = await crawler.crawl_control_details(test_control_id)
    if control_knowledge:
        print(f"   ✅ Successfully crawled details for {test_control_id}:")
        print(f"      Title: {control_knowledge.title}")
        print(f"      Confidence: {control_knowledge.confidence_score:.2f}")
    else:
        print(f"   ⚠️  Could not crawl details for {test_control_id}")

    # Test cache storage
    print("\n4. Testing cache storage...")
    if control_knowledge:
        stored = cache.store_control_knowledge(
            test_control_id,
            {
                "title": control_knowledge.title,
                "description": control_knowledge.description,
                "implementation_guidance": control_knowledge.implementation_guidance,
                "best_practices": control_knowledge.best_practices,
                "common_gaps": control_knowledge.common_gaps,
                "confidence_score": control_knowledge.confidence_score
            },
            control_knowledge.source_url
        )
        if stored:
            print("   ✅ Control knowledge stored in cache")
        else:
            print("   ❌ Failed to store in cache")

    # Test cache retrieval
    print("\n5. Testing cache retrieval...")
    cached_data = cache.get_control_details(test_control_id)
    if cached_data:
        print(f"   ✅ Retrieved control {test_control_id} from cache")
        print(f"      Title: {cached_data['title']}")
    else:
        print(f"   ⚠️  Could not retrieve from cache")

    # Test cache statistics
    print("\n6. Testing cache statistics...")
    stats = cache.get_cache_statistics()
    print(f"   📊 Cache Statistics:")
    print(f"      Total controls: {stats['total_controls']}")
    print(f"      Valid controls: {stats['valid_controls']}")
    print(f"      Expired controls: {stats['expired_controls']}")

    # Cleanup
    await crawler.cleanup()
    print("\n✅ Crawler test completed")


async def test_enhanced_assessor():
    """Test intelligent assessor with crawler enhancement"""
    print("\n" + "=" * 60)
    print("Testing Enhanced Intelligent Assessor")
    print("=" * 60)

    # Initialize assessor with crawler enabled
    assessor = IntelligentControlAssessor(enable_crawler=True)

    print("\n1. Testing assessor with crawler integration...")
    if assessor.crawler and assessor.knowledge_cache:
        print("   ✅ Assessor initialized with crawler support")
    else:
        print("   ⚠️  Assessor running without crawler (fallback mode)")

    # Test control enhancement
    print("\n2. Testing control enhancement with crawled knowledge...")
    test_control_id = "A.9.4.2"

    # Get original mapping
    original_mapping = assessor.control_mappings.get(test_control_id, {})
    original_keywords = len(original_mapping.get("keywords", []))

    # Enhance with crawled knowledge
    if assessor.crawler:
        enhanced_mapping = await assessor.enhance_control_with_crawled_knowledge(test_control_id)
        enhanced_keywords = len(enhanced_mapping.get("keywords", []))

        if enhanced_keywords > original_keywords:
            print(f"   ✅ Control mapping enhanced:")
            print(f"      Original keywords: {original_keywords}")
            print(f"      Enhanced keywords: {enhanced_keywords}")
            if enhanced_mapping.get("nist_mappings"):
                print(f"      NIST mappings: {len(enhanced_mapping['nist_mappings'])}")
            if enhanced_mapping.get("industry_examples"):
                print(f"      Industry examples: {len(enhanced_mapping['industry_examples'])}")
        else:
            print(f"   ⚠️  No enhancement achieved (may need actual web sources)")
    else:
        print("   ⚠️  Crawler not available for enhancement")

    print("\n✅ Enhanced assessor test completed")


async def test_comprehensive_workflow():
    """Test complete workflow from crawling to assessment"""
    print("\n" + "=" * 60)
    print("Testing Comprehensive Workflow")
    print("=" * 60)

    # Sample document for testing
    test_document = """
    Security Policy Document

    1. Authentication Controls
    We have implemented multi-factor authentication (MFA) for all critical systems.
    All users must use MFA when accessing sensitive resources.

    2. Encryption
    Data at rest is protected using AES-256 encryption.
    All network communications use TLS 1.3 protocol.

    3. Incident Response
    Security incidents must be reported within 4 hours of discovery.
    Our incident response team is available 24/7.
    """

    # Initialize components
    assessor = IntelligentControlAssessor(enable_crawler=True)

    print("\n1. Analyzing document with enhanced assessor...")

    # Create a test control
    test_control = Control(
        id="A.9.4.2",
        title="Secure logon procedures",
        description="Use of secure logon procedures",
        purpose="To prevent unauthorized access",
        implementation_guidance=["Implement MFA"],
        category="Access Control",
        control_type=[],
        security_properties=[],
        cybersecurity_concepts=[],
        related_controls=[]
    )

    # Perform assessment
    try:
        # Extract features
        segments = assessor.semantic_analyzer.analyze_document_structure(test_document)
        features = assessor.semantic_analyzer.extract_semantic_features(test_document, segments)
        features['raw_text'] = test_document

        # Assess control
        assessment = await assessor.assess_control_comprehensive(
            control=test_control,
            document_text=test_document,
            semantic_features=features,
            segments=segments
        )

        print(f"\n   📋 Assessment Results:")
        print(f"      Control: {assessment.control_id}")
        print(f"      Status: {assessment.status}")
        print(f"      Confidence: {assessment.confidence_score:.2%}")
        print(f"      Evidence found: {len(assessment.evidence_found)}")
        print(f"      Gaps identified: {len(assessment.gaps_identified)}")
        print(f"      Recommendations: {len(assessment.recommendations)}")

        if assessment.gaps_identified:
            print(f"\n   🔍 Identified Gaps:")
            for gap in assessment.gaps_identified[:2]:
                print(f"      - {gap}")

        if assessment.recommendations:
            print(f"\n   💡 Recommendations:")
            for rec in assessment.recommendations[:3]:
                print(f"      - {rec}")

        print("\n✅ Comprehensive workflow test completed")

    except Exception as e:
        print(f"\n❌ Error in comprehensive workflow: {e}")
        import traceback
        traceback.print_exc()


async def main():
    """Run all tests"""
    try:
        # Test 1: Basic crawler functionality
        await test_basic_crawler()

        # Test 2: Enhanced assessor
        await test_enhanced_assessor()

        # Test 3: Comprehensive workflow
        await test_comprehensive_workflow()

        print("\n" + "=" * 60)
        print("🎉 All tests completed!")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())