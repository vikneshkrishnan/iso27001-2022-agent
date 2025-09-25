"""
ISO Knowledge Crawler using Crawl4AI
Fetches and updates ISO control knowledge from authoritative sources
"""

import asyncio
import json
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import re

try:
    from crawl4ai import AsyncWebCrawler
    from crawl4ai.extraction_strategy import (
        LLMExtractionStrategy,
        JsonCssExtractionStrategy,
        JsonXPathExtractionStrategy
    )
    CRAWL4AI_AVAILABLE = True
except ImportError:
    CRAWL4AI_AVAILABLE = False
    logging.warning("Crawl4AI not available. Install with: pip install crawl4ai")

try:
    from bs4 import BeautifulSoup
    BS4_AVAILABLE = True
except ImportError:
    BS4_AVAILABLE = False
    logging.warning("BeautifulSoup not available. Install with: pip install beautifulsoup4")

logger = logging.getLogger(__name__)


@dataclass
class ControlKnowledge:
    """Represents crawled knowledge about an ISO control"""
    control_id: str
    title: str
    description: str
    implementation_guidance: List[str]
    best_practices: List[str]
    common_gaps: List[str]
    nist_mappings: List[str]
    industry_examples: Dict[str, str]
    regulatory_mappings: Dict[str, List[str]]
    last_updated: str
    source_url: str
    confidence_score: float


class ISOKnowledgeCrawler:
    """Crawler for ISO 27001 control knowledge using Crawl4AI"""

    def __init__(self, cache_dir: str = "data/iso_knowledge_cache", strict_mode: bool = True):
        """Initialize the ISO knowledge crawler"""
        self.cache_dir = cache_dir
        self.strict_mode = strict_mode  # Only return verified ISO data
        self.crawler = None
        self.sources = self._load_crawler_config()
        self.authorized_domains = self._get_authorized_domains()
        self._ensure_cache_dir()

    def _ensure_cache_dir(self):
        """Ensure cache directory exists"""
        import os
        os.makedirs(self.cache_dir, exist_ok=True)

    def _get_authorized_domains(self) -> List[str]:
        """Get list of authorized domains for ISO information"""
        return [
            "iso.org",
            "www.iso.org",
            "csrc.nist.gov",
            "www.nist.gov",
            "enisa.europa.eu",
            "cisa.gov",
            "www.cisa.gov",
            # Only official standards bodies and government sources
        ]

    def _validate_source(self, url: str) -> bool:
        """Validate that source is from an authorized domain"""
        if not self.strict_mode:
            return True

        from urllib.parse import urlparse
        try:
            domain = urlparse(url).netloc.lower()
            return any(auth_domain in domain for auth_domain in self.authorized_domains)
        except Exception:
            return False

    def _load_crawler_config(self) -> Dict[str, Any]:
        """Load crawler configuration"""
        # Default configuration if config file doesn't exist
        return {
            "iso_controls": [
                {
                    "name": "ISO 27001 Controls",
                    "url": "https://www.iso27001security.com/html/27001.html",
                    "type": "reference"
                },
                {
                    "name": "NIST Mapping",
                    "url": "https://csrc.nist.gov/projects/olir/informative-reference-catalog",
                    "type": "mapping"
                }
            ],
            "implementation_guides": [
                {
                    "name": "Implementation Guide",
                    "url": "https://www.itgovernance.co.uk/iso27001",
                    "type": "guide"
                }
            ],
            "crawl_settings": {
                "max_depth": 2,
                "rate_limit": 1,  # requests per second
                "timeout": 30,
                "cache_duration_days": 7
            }
        }

    async def initialize_crawler(self):
        """Initialize the async web crawler"""
        if not CRAWL4AI_AVAILABLE:
            logger.error("Crawl4AI not available")
            return False

        try:
            # Initialize crawler with specific browser configuration
            self.crawler = AsyncWebCrawler(
                verbose=False,
                max_concurrent_requests=5,
                headless=True,  # Ensure headless mode
                browser_type="chromium"  # Specify browser type
            )
            # Note: warmup() method removed in newer Crawl4AI versions
            logger.info("Crawl4AI initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize crawler: {e}")
            return False

    async def crawl_control_details(self, control_id: str) -> Optional[ControlKnowledge]:
        """Crawl detailed information about a specific ISO control"""
        if not self.crawler:
            logger.info(f"Crawler not available for control {control_id}, skipping crawling")
            return None

        try:
            # Check cache first
            cached_data = self._get_cached_control(control_id)
            if cached_data:
                return cached_data

            # Prepare extraction strategy for control details
            control_schema = {
                "name": "control_details",
                "baseSelector": "div.control-content",
                "fields": [
                    {"name": "title", "selector": "h2", "type": "text"},
                    {"name": "description", "selector": "p.description", "type": "text"},
                    {"name": "implementation", "selector": "ul.implementation li", "type": "list"},
                    {"name": "requirements", "selector": "ul.requirements li", "type": "list"}
                ]
            }

            extraction_strategy = JsonCssExtractionStrategy(
                schema=control_schema,
                verbose=False
            )

            # Crawl control information from sources
            control_knowledge = None
            for source in self.sources.get("iso_controls", []):
                url = f"{source['url']}#{control_id}"

                # Validate source in strict mode
                if not self._validate_source(url):
                    logger.warning(f"Skipping unauthorized source: {url}")
                    continue

                result = await self.crawler.arun(
                    url=url,
                    extraction_strategy=extraction_strategy,
                    bypass_cache=False
                )

                if result.success and result.extracted_content:
                    control_knowledge = self._parse_control_knowledge(
                        control_id,
                        result.extracted_content,
                        url
                    )
                    break

            if control_knowledge:
                self._cache_control(control_knowledge)
                return control_knowledge

        except Exception as e:
            logger.error(f"Error crawling control {control_id}: {e}")

        # Strict mode: only return verified ISO data or None
        return None

    async def crawl_nist_mapping(self, control_id: str) -> Optional[List[str]]:
        """Crawl NIST CSF mapping for an ISO control"""
        if not self.crawler:
            if not await self.initialize_crawler():
                return None

        try:
            # Use LLM extraction for unstructured mapping data
            llm_strategy = LLMExtractionStrategy(
                provider="openai/gpt-4o-mini",
                api_key=None,  # Will use environment variable
                instruction=f"""
                Extract NIST CSF mappings for ISO 27001 control {control_id}.
                Return a JSON list of NIST control IDs that map to this ISO control.
                Include the NIST function (Identify, Protect, Detect, Respond, Recover).
                Format: ["NIST_ID: Function.Category", ...]
                """
            )

            for source in self.sources.get("iso_controls", []):
                if source.get("type") == "mapping":
                    # Validate source in strict mode
                    if not self._validate_source(source["url"]):
                        logger.warning(f"Skipping unauthorized mapping source: {source['url']}")
                        continue

                    result = await self.crawler.arun(
                        url=source["url"],
                        extraction_strategy=llm_strategy
                    )

                    if result.success and result.extracted_content:
                        try:
                            mappings = json.loads(result.extracted_content)
                            return mappings if isinstance(mappings, list) else None
                        except json.JSONDecodeError:
                            logger.warning(f"Invalid JSON from NIST mapping crawl")
                            continue

        except Exception as e:
            logger.error(f"Error crawling NIST mapping for {control_id}: {e}")

        # Strict mode: return None if no verified mappings found
        return None

    async def crawl_implementation_guides(self, control_id: str) -> Optional[Dict[str, Any]]:
        """Crawl implementation guides and best practices for a control"""
        if not self.crawler:
            if not await self.initialize_crawler():
                return None  # Strict mode: return None if crawler unavailable

        implementation_data = {
            "best_practices": [],
            "common_gaps": [],
            "industry_examples": {}
        }

        try:
            # XPath strategy for structured implementation guides
            xpath_schema = {
                "name": "implementation_guide",
                "baseSelector": "//div[@class='implementation-guide']",
                "fields": [
                    {
                        "name": "best_practices",
                        "selector": "//ul[@class='best-practices']/li",
                        "type": "list"
                    },
                    {
                        "name": "common_gaps",
                        "selector": "//div[@class='common-gaps']//li",
                        "type": "list"
                    },
                    {
                        "name": "examples",
                        "selector": "//div[@class='industry-example']",
                        "type": "nested",
                        "fields": [
                            {"name": "industry", "selector": ".//h3", "type": "text"},
                            {"name": "description", "selector": ".//p", "type": "text"}
                        ]
                    }
                ]
            }

            extraction_strategy = JsonXPathExtractionStrategy(
                schema=xpath_schema,
                verbose=False
            )

            for source in self.sources.get("implementation_guides", []):
                url = f"{source['url']}/control/{control_id}"

                # Validate source in strict mode
                if not self._validate_source(url):
                    logger.warning(f"Skipping unauthorized implementation source: {url}")
                    continue

                result = await self.crawler.arun(
                    url=url,
                    extraction_strategy=extraction_strategy,
                    wait_for="css:.implementation-guide",
                    timeout=30000
                )

                if result.success and result.extracted_content:
                    try:
                        data = json.loads(result.extracted_content)
                        implementation_data["best_practices"].extend(
                            data.get("best_practices", [])
                        )
                        implementation_data["common_gaps"].extend(
                            data.get("common_gaps", [])
                        )

                        # Process industry examples
                        for example in data.get("examples", []):
                            industry = example.get("industry", "General")
                            implementation_data["industry_examples"][industry] = \
                                example.get("description", "")

                    except json.JSONDecodeError:
                        logger.warning(f"Invalid JSON from implementation guide")

        except Exception as e:
            logger.error(f"Error crawling implementation guides for {control_id}: {e}")

        # Strict mode: only return if we have actual data from authorized sources
        if (implementation_data["best_practices"] or
            implementation_data["common_gaps"] or
            implementation_data["industry_examples"]):
            return implementation_data

        return None

    async def update_control_knowledge_base(self) -> Dict[str, ControlKnowledge]:
        """Update the entire control knowledge base"""
        knowledge_base = {}

        # List of all ISO 27001:2022 Annex A controls
        control_ids = self._get_all_control_ids()

        logger.info(f"Updating knowledge base for {len(control_ids)} controls")

        for control_id in control_ids:
            try:
                # Crawl control details
                control_knowledge = await self.crawl_control_details(control_id)

                if control_knowledge:
                    # Enhance with NIST mappings
                    nist_mappings = await self.crawl_nist_mapping(control_id)
                    if nist_mappings:
                        control_knowledge.nist_mappings = nist_mappings

                    # Enhance with implementation guides
                    impl_data = await self.crawl_implementation_guides(control_id)
                    if impl_data:
                        control_knowledge.best_practices.extend(impl_data.get("best_practices", []))
                        control_knowledge.common_gaps.extend(impl_data.get("common_gaps", []))
                        control_knowledge.industry_examples.update(impl_data.get("industry_examples", {}))

                    knowledge_base[control_id] = control_knowledge

                # Rate limiting
                await asyncio.sleep(self.sources["crawl_settings"]["rate_limit"])

            except Exception as e:
                logger.error(f"Error updating knowledge for control {control_id}: {e}")

        # Save the entire knowledge base
        self._save_knowledge_base(knowledge_base)

        return knowledge_base

    async def crawl_regulatory_mappings(self, control_id: str) -> Optional[Dict[str, List[str]]]:
        """Crawl regulatory compliance mappings (GDPR, HIPAA, etc.)"""
        regulatory_mappings = {}

        try:
            # Use LLM to extract regulatory mappings from unstructured content
            llm_strategy = LLMExtractionStrategy(
                provider="openai/gpt-4o-mini",
                instruction=f"""
                Extract regulatory compliance mappings for ISO 27001 control {control_id}.
                Identify which regulations (GDPR, HIPAA, PCI DSS, SOC 2, etc.)
                map to this control and list specific requirements.
                Return as JSON: {{"regulation": ["requirement1", "requirement2"]}}
                """
            )

            # Only use authorized compliance mapping sources in strict mode
            mapping_url = "https://csrc.nist.gov/projects/olir/informative-reference-catalog"

            # Validate source in strict mode
            if not self._validate_source(mapping_url):
                logger.warning(f"Unauthorized regulatory mapping source: {mapping_url}")
                return None

            # Crawl from authorized compliance mapping sources
            result = await self.crawler.arun(
                url=mapping_url,
                extraction_strategy=llm_strategy
            )

            if result.success and result.extracted_content:
                try:
                    regulatory_mappings = json.loads(result.extracted_content)
                    return regulatory_mappings if regulatory_mappings else None
                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON from regulatory mapping crawl")
                    return None

        except Exception as e:
            logger.error(f"Error crawling regulatory mappings: {e}")

        return None

    def _parse_control_knowledge(self, control_id: str, extracted_content: str, source_url: str) -> ControlKnowledge:
        """Parse extracted content into ControlKnowledge object"""
        try:
            data = json.loads(extracted_content) if isinstance(extracted_content, str) else extracted_content

            return ControlKnowledge(
                control_id=control_id,
                title=data.get("title", f"Control {control_id}"),
                description=data.get("description", ""),
                implementation_guidance=data.get("implementation", []),
                best_practices=data.get("best_practices", []),
                common_gaps=data.get("common_gaps", []),
                nist_mappings=[],
                industry_examples={},
                regulatory_mappings={},
                last_updated=datetime.now().isoformat(),
                source_url=source_url,
                confidence_score=0.8
            )
        except Exception as e:
            logger.error(f"Error parsing control knowledge: {e}")
            return None

    def _get_all_control_ids(self) -> List[str]:
        """Get list of all ISO 27001:2022 Annex A control IDs"""
        control_ids = []

        # Annex A controls (A.5 through A.18 in ISO 27001:2022)
        control_structure = {
            "A.5": 37,   # Organizational controls
            "A.6": 5,    # People controls
            "A.7": 4,    # Physical controls
            "A.8": 34    # Technological controls
        }

        for prefix, count in control_structure.items():
            for i in range(1, count + 1):
                control_ids.append(f"{prefix}.{i}")

        return control_ids

    def _get_cached_control(self, control_id: str) -> Optional[ControlKnowledge]:
        """Get cached control knowledge if still valid"""
        import os
        cache_file = os.path.join(self.cache_dir, f"{control_id}.json")

        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r') as f:
                    data = json.load(f)

                # Check cache validity
                last_updated = datetime.fromisoformat(data["last_updated"])
                cache_duration = timedelta(days=self.sources["crawl_settings"]["cache_duration_days"])

                if datetime.now() - last_updated < cache_duration:
                    return ControlKnowledge(**data)

            except Exception as e:
                logger.warning(f"Error reading cache for {control_id}: {e}")

        return None

    def _cache_control(self, control_knowledge: ControlKnowledge):
        """Cache control knowledge to file"""
        import os
        cache_file = os.path.join(self.cache_dir, f"{control_knowledge.control_id}.json")

        try:
            with open(cache_file, 'w') as f:
                json.dump(asdict(control_knowledge), f, indent=2)
        except Exception as e:
            logger.error(f"Error caching control {control_knowledge.control_id}: {e}")

    def _save_knowledge_base(self, knowledge_base: Dict[str, ControlKnowledge]):
        """Save entire knowledge base to file"""
        import os
        kb_file = os.path.join(self.cache_dir, "knowledge_base.json")

        try:
            kb_dict = {
                control_id: asdict(knowledge)
                for control_id, knowledge in knowledge_base.items()
            }

            with open(kb_file, 'w') as f:
                json.dump(kb_dict, f, indent=2)

            logger.info(f"Saved knowledge base with {len(knowledge_base)} controls")

        except Exception as e:
            logger.error(f"Error saving knowledge base: {e}")

    async def search_control_by_keyword(self, keyword: str) -> Optional[List[str]]:
        """Search for controls related to a keyword using web search"""
        if not self.crawler:
            if not await self.initialize_crawler():
                return None

        try:
            # In strict mode, only search within authorized ISO sources
            search_urls = [
                f"https://www.iso.org/standard/27001/search?q={keyword}",
                f"https://csrc.nist.gov/projects/olir/search?q=ISO+27001+{keyword}"
            ]

            all_matches = set()

            for search_url in search_urls:
                if not self._validate_source(search_url):
                    continue

                result = await self.crawler.arun(
                    url=search_url,
                    bypass_cache=True
                )

                if result.success and result.markdown_content:
                    # Extract control IDs from search results
                    control_pattern = r'A\.\d+\.\d+'
                    matches = re.findall(control_pattern, result.markdown_content)
                    all_matches.update(matches)

            if all_matches:
                return list(all_matches)  # Return found matches from authorized sources

        except Exception as e:
            logger.error(f"Error searching controls for keyword '{keyword}': {e}")

        # Strict mode: return None if search fails or no authorized results
        return None

    async def cleanup(self):
        """Cleanup crawler resources"""
        if self.crawler:
            await self.crawler.close()


# Example usage
async def main():
    """Example of using the ISO Knowledge Crawler"""
    crawler = ISOKnowledgeCrawler()

    # Initialize crawler
    if await crawler.initialize_crawler():
        # Crawl specific control
        control_knowledge = await crawler.crawl_control_details("A.9.4.2")
        if control_knowledge:
            print(f"Control: {control_knowledge.control_id}")
            print(f"Title: {control_knowledge.title}")
            print(f"Best Practices: {control_knowledge.best_practices[:3]}")

        # Search for controls by keyword
        mfa_controls = await crawler.search_control_by_keyword("multi-factor authentication")
        print(f"Controls related to MFA: {mfa_controls}")

        # Cleanup
        await crawler.cleanup()


if __name__ == "__main__":
    asyncio.run(main())