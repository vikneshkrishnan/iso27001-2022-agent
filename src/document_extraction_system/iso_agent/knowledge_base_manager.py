"""
Knowledge Base Management System for ISO 27001:2022 Agent
Provides CLI and programmatic interfaces for managing Pinecone knowledge base
"""

import asyncio
import logging
import json
from datetime import datetime
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
import click

from .vector_store import PineconeVectorStore, VectorNamespace
from .vector_manager import VectorIndexManager, IndexingJob, IndexStats
from ..config.settings import get_settings

logger = logging.getLogger(__name__)


@dataclass
class KnowledgeBaseStatus:
    """Knowledge base health and status information"""
    is_healthy: bool
    total_documents: int
    documents_by_namespace: Dict[str, int]
    last_updated: Optional[datetime]
    storage_usage_mb: float
    initialization_required: bool
    errors: List[str]
    recommendations: List[str]


class KnowledgeBaseManager:
    """Comprehensive knowledge base management system"""

    def __init__(self):
        """Initialize knowledge base manager"""
        self.settings = get_settings()
        self.vector_store = PineconeVectorStore()
        self.vector_manager = VectorIndexManager(self.vector_store)

    async def get_status(self) -> KnowledgeBaseStatus:
        """Get comprehensive knowledge base status"""
        try:
            # Get vector store health
            health = self.vector_store.health_check()

            # Get index statistics
            try:
                stats = await self.vector_manager.get_index_statistics()
                total_docs = stats.total_documents
                docs_by_ns = stats.documents_by_namespace
                storage_mb = stats.storage_usage_mb
                last_updated = stats.last_updated
            except Exception as e:
                total_docs = 0
                docs_by_ns = {}
                storage_mb = 0.0
                last_updated = None
                logger.warning(f"Failed to get detailed stats: {e}")

            # Determine if initialization is required
            expected_namespaces = {
                VectorNamespace.KNOWLEDGE.value,
                VectorNamespace.CONTROLS.value,
                VectorNamespace.CLAUSES.value
            }

            existing_namespaces = set(docs_by_ns.keys()) if docs_by_ns else set()
            missing_namespaces = expected_namespaces - existing_namespaces

            initialization_required = (
                not health.get("operational", False) or
                total_docs < 100 or  # Expect ~200+ knowledge documents
                len(missing_namespaces) > 0
            )

            # Generate errors and recommendations
            errors = []
            recommendations = []

            if not health.get("pinecone_available", False):
                errors.append("Pinecone client not available")

            if not health.get("openai_available", False):
                errors.append("OpenAI client not available for embeddings")

            if not health.get("api_key_configured", False):
                errors.append("Pinecone API key not configured")

            if not health.get("index_available", False):
                errors.append("Pinecone index not available")

            if total_docs < 50:
                recommendations.append("Initialize knowledge base - very few documents found")
            elif total_docs < 150:
                recommendations.append("Consider refreshing knowledge base - missing content detected")

            if missing_namespaces:
                recommendations.append(f"Missing namespaces: {', '.join(missing_namespaces)}")

            if storage_mb > 80:  # Assuming 100MB limit for free tier
                recommendations.append("Consider cleanup - approaching storage limits")

            is_healthy = (
                health.get("operational", False) and
                len(errors) == 0 and
                total_docs >= 100
            )

            return KnowledgeBaseStatus(
                is_healthy=is_healthy,
                total_documents=total_docs,
                documents_by_namespace=docs_by_ns,
                last_updated=last_updated,
                storage_usage_mb=storage_mb,
                initialization_required=initialization_required,
                errors=errors,
                recommendations=recommendations
            )

        except Exception as e:
            logger.error(f"Failed to get knowledge base status: {e}")
            return KnowledgeBaseStatus(
                is_healthy=False,
                total_documents=0,
                documents_by_namespace={},
                last_updated=None,
                storage_usage_mb=0.0,
                initialization_required=True,
                errors=[f"Status check failed: {str(e)}"],
                recommendations=["Fix configuration issues before proceeding"]
            )

    async def initialize(self, force: bool = False) -> bool:
        """Initialize or refresh the knowledge base"""
        try:
            click.echo("🚀 Starting knowledge base initialization...")

            # Check if already initialized
            if not force:
                status = await self.get_status()
                if status.is_healthy and not status.initialization_required:
                    click.echo("✅ Knowledge base already initialized and healthy")
                    return True

                if status.total_documents > 0:
                    if not click.confirm(f"Knowledge base has {status.total_documents} documents. Reinitialize?"):
                        return False

            # Start bulk indexing job
            job = await self.vector_manager.bulk_index_knowledge_base()
            click.echo(f"📊 Indexing job started: {job.job_id}")

            # Monitor progress
            last_processed = 0
            while job.status == "running":
                await asyncio.sleep(2)
                job = self.vector_manager.get_job_status(job.job_id)

                if job and job.processed_documents > last_processed:
                    progress = (job.processed_documents / job.total_documents) * 100 if job.total_documents > 0 else 0
                    click.echo(f"📈 Progress: {job.processed_documents}/{job.total_documents} ({progress:.1f}%)")
                    last_processed = job.processed_documents

            # Check final status
            if job and job.status == "completed":
                click.echo(f"✅ Knowledge base initialized successfully!")
                click.echo(f"📊 Indexed {job.processed_documents}/{job.total_documents} documents")
                if job.failed_documents > 0:
                    click.echo(f"⚠️  {job.failed_documents} documents failed to index")
                return True
            else:
                error_msg = job.error_message if job else "Unknown error"
                click.echo(f"❌ Knowledge base initialization failed: {error_msg}")
                return False

        except Exception as e:
            click.echo(f"❌ Initialization failed: {e}")
            logger.error(f"Knowledge base initialization error: {e}")
            return False

    async def cleanup(self, namespace: Optional[str] = None, confirm: bool = False) -> bool:
        """Clean up knowledge base documents"""
        try:
            if namespace:
                if not confirm and not click.confirm(f"Delete all documents in namespace '{namespace}'?"):
                    return False

                success = self.vector_store.delete_namespace(namespace)
                if success:
                    click.echo(f"✅ Namespace '{namespace}' cleaned up successfully")
                else:
                    click.echo(f"❌ Failed to clean up namespace '{namespace}'")
                return success
            else:
                # Clean up all namespaces
                if not confirm and not click.confirm("Delete ALL documents from knowledge base?"):
                    return False

                namespaces = [ns.value for ns in VectorNamespace]
                success_count = 0

                for ns in namespaces:
                    if self.vector_store.delete_namespace(ns):
                        success_count += 1
                        click.echo(f"✅ Cleaned up namespace: {ns}")
                    else:
                        click.echo(f"❌ Failed to clean up namespace: {ns}")

                click.echo(f"🧹 Cleanup completed: {success_count}/{len(namespaces)} namespaces")
                return success_count == len(namespaces)

        except Exception as e:
            click.echo(f"❌ Cleanup failed: {e}")
            logger.error(f"Knowledge base cleanup error: {e}")
            return False

    async def validate(self) -> Dict[str, Any]:
        """Validate knowledge base integrity and completeness"""
        try:
            click.echo("🔍 Validating knowledge base...")

            validation_results = {
                "overall_health": False,
                "namespace_checks": {},
                "content_validation": {},
                "performance_metrics": {},
                "recommendations": []
            }

            # Get status
            status = await self.get_status()

            # Namespace validation
            expected_namespaces = [ns.value for ns in VectorNamespace]
            for ns in expected_namespaces:
                namespace_data = status.documents_by_namespace.get(ns, 0)
                # Handle both integer and dictionary formats
                if isinstance(namespace_data, dict):
                    doc_count = namespace_data.get('vector_count', 0)
                else:
                    doc_count = namespace_data if isinstance(namespace_data, int) else 0

                validation_results["namespace_checks"][ns] = {
                    "present": doc_count > 0,
                    "document_count": doc_count,
                    "expected_minimum": self._get_expected_doc_count(ns)
                }

            # Content validation (sample checks)
            try:
                from .vector_store import SearchQuery

                # Test search in knowledge namespace
                query = SearchQuery(
                    text="information security policy",
                    namespace=VectorNamespace.KNOWLEDGE.value,
                    top_k=3
                )

                results = await self.vector_store.search(query)
                validation_results["content_validation"]["search_test"] = {
                    "query_successful": True,
                    "results_count": len(results),
                    "average_score": sum(r.score for r in results) / len(results) if results else 0
                }

            except Exception as e:
                validation_results["content_validation"]["search_test"] = {
                    "query_successful": False,
                    "error": str(e)
                }

            # Performance metrics
            validation_results["performance_metrics"] = {
                "total_documents": status.total_documents,
                "storage_usage_mb": status.storage_usage_mb,
                "is_healthy": status.is_healthy,
                "initialization_required": status.initialization_required
            }

            # Generate recommendations
            if status.initialization_required:
                validation_results["recommendations"].append("Initialize knowledge base")

            if any(not check["present"] for check in validation_results["namespace_checks"].values()):
                validation_results["recommendations"].append("Some namespaces are missing content")

            validation_results["overall_health"] = (
                status.is_healthy and
                not status.initialization_required and
                len(status.errors) == 0
            )

            # Display results
            if validation_results["overall_health"]:
                click.echo("✅ Knowledge base validation passed")
            else:
                click.echo("⚠️  Knowledge base validation found issues")

            for rec in validation_results["recommendations"]:
                click.echo(f"💡 Recommendation: {rec}")

            return validation_results

        except Exception as e:
            click.echo(f"❌ Validation failed: {e}")
            logger.error(f"Knowledge base validation error: {e}")
            return {"overall_health": False, "error": str(e)}

    def _get_expected_doc_count(self, namespace: str) -> int:
        """Get expected minimum document count for namespace"""
        expected_counts = {
            VectorNamespace.KNOWLEDGE.value: 150,  # ~93 controls + clauses + requirements
            VectorNamespace.CONTROLS.value: 0,     # Populated during analysis
            VectorNamespace.CLAUSES.value: 0,      # Populated during analysis
            VectorNamespace.RECOMMENDATIONS.value: 0,  # Populated during analysis
            VectorNamespace.ANALYSES.value: 0,     # Populated during analysis
            VectorNamespace.CONSULTATIONS.value: 0,   # Populated during usage
            VectorNamespace.DOCUMENTS.value: 0     # User documents
        }
        return expected_counts.get(namespace, 0)


# CLI Interface
@click.group()
def kb():
    """Knowledge Base Management Commands"""
    pass


@kb.command()
@click.option('--json', 'output_json', is_flag=True, help='Output in JSON format')
def status(output_json):
    """Check knowledge base status"""
    async def _status():
        manager = KnowledgeBaseManager()
        status = await manager.get_status()

        if output_json:
            click.echo(json.dumps(asdict(status), indent=2, default=str))
        else:
            click.echo(f"\n📊 Knowledge Base Status")
            click.echo(f"{'='*50}")
            click.echo(f"Health: {'✅ Healthy' if status.is_healthy else '❌ Unhealthy'}")
            click.echo(f"Total Documents: {status.total_documents}")
            click.echo(f"Storage Usage: {status.storage_usage_mb:.2f} MB")
            click.echo(f"Last Updated: {status.last_updated or 'Never'}")
            click.echo(f"Initialization Required: {'Yes' if status.initialization_required else 'No'}")

            if status.documents_by_namespace:
                click.echo(f"\n📁 Documents by Namespace:")
                for ns, count in status.documents_by_namespace.items():
                    click.echo(f"  • {ns}: {count} documents")

            if status.errors:
                click.echo(f"\n❌ Errors:")
                for error in status.errors:
                    click.echo(f"  • {error}")

            if status.recommendations:
                click.echo(f"\n💡 Recommendations:")
                for rec in status.recommendations:
                    click.echo(f"  • {rec}")

    asyncio.run(_status())


@kb.command()
@click.option('--force', is_flag=True, help='Force reinitialize even if healthy')
def init(force):
    """Initialize knowledge base with ISO controls and clauses"""
    async def _init():
        manager = KnowledgeBaseManager()
        success = await manager.initialize(force=force)
        if success:
            click.echo("🎉 Knowledge base ready for use!")
        else:
            click.echo("💥 Initialization failed. Check logs for details.")

    asyncio.run(_init())


@kb.command()
@click.option('--namespace', help='Specific namespace to clean up')
@click.option('--yes', is_flag=True, help='Skip confirmation prompts')
def cleanup(namespace, yes):
    """Clean up knowledge base documents"""
    async def _cleanup():
        manager = KnowledgeBaseManager()
        success = await manager.cleanup(namespace=namespace, confirm=yes)
        if success:
            click.echo("🧹 Cleanup completed successfully!")
        else:
            click.echo("💥 Cleanup failed. Check logs for details.")

    asyncio.run(_cleanup())


@kb.command()
@click.option('--json', 'output_json', is_flag=True, help='Output in JSON format')
def validate(output_json):
    """Validate knowledge base integrity"""
    async def _validate():
        manager = KnowledgeBaseManager()
        results = await manager.validate()

        if output_json:
            click.echo(json.dumps(results, indent=2, default=str))
        else:
            overall_health = results.get("overall_health", False)
            click.echo(f"\n🔍 Validation Results")
            click.echo(f"{'='*50}")
            click.echo(f"Overall Health: {'✅ Passed' if overall_health else '❌ Failed'}")

            # Show namespace checks
            ns_checks = results.get("namespace_checks", {})
            if ns_checks:
                click.echo(f"\n📁 Namespace Checks:")
                for ns, check in ns_checks.items():
                    status = "✅" if check["present"] else "❌"
                    click.echo(f"  {status} {ns}: {check['document_count']} docs (min: {check['expected_minimum']})")

    asyncio.run(_validate())


if __name__ == "__main__":
    kb()