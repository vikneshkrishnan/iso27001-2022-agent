"""
Knowledge Cache for ISO Control Information
Manages cached data from web crawling and provides efficient access
"""

import json
import os
import sqlite3
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import logging
import hashlib

logger = logging.getLogger(__name__)


@dataclass
class CachedControlData:
    """Cached control data with metadata"""
    control_id: str
    data: Dict[str, Any]
    source_url: str
    crawled_at: str
    expires_at: str
    checksum: str


class KnowledgeCache:
    """Manages cached ISO control knowledge"""

    def __init__(self, cache_dir: str = "data/iso_knowledge_cache"):
        """Initialize the knowledge cache"""
        self.cache_dir = cache_dir
        self.db_path = os.path.join(cache_dir, "knowledge_cache.db")
        self._ensure_cache_dir()
        self._init_database()

    def _ensure_cache_dir(self):
        """Ensure cache directory exists"""
        os.makedirs(self.cache_dir, exist_ok=True)

    def _init_database(self):
        """Initialize SQLite database for cache management"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Control knowledge table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS control_knowledge (
                    control_id TEXT PRIMARY KEY,
                    title TEXT,
                    description TEXT,
                    implementation_guidance TEXT,
                    best_practices TEXT,
                    common_gaps TEXT,
                    nist_mappings TEXT,
                    industry_examples TEXT,
                    regulatory_mappings TEXT,
                    source_url TEXT,
                    crawled_at TIMESTAMP,
                    expires_at TIMESTAMP,
                    checksum TEXT,
                    confidence_score REAL
                )
            """)

            # Mapping relationships table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS control_mappings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_control TEXT,
                    target_framework TEXT,
                    target_control TEXT,
                    mapping_type TEXT,
                    confidence REAL,
                    FOREIGN KEY (source_control) REFERENCES control_knowledge(control_id)
                )
            """)

            # Implementation examples table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS implementation_examples (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    control_id TEXT,
                    industry TEXT,
                    example_description TEXT,
                    source TEXT,
                    added_at TIMESTAMP,
                    FOREIGN KEY (control_id) REFERENCES control_knowledge(control_id)
                )
            """)

            # Cache metadata table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS cache_metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT,
                    updated_at TIMESTAMP
                )
            """)

            conn.commit()

    def store_control_knowledge(self, control_id: str, knowledge_data: Dict[str, Any],
                               source_url: str, cache_duration_days: int = 7) -> bool:
        """Store control knowledge in cache"""
        try:
            now = datetime.now()
            expires_at = now + timedelta(days=cache_duration_days)

            # Calculate checksum for data integrity
            checksum = self._calculate_checksum(knowledge_data)

            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                cursor.execute("""
                    INSERT OR REPLACE INTO control_knowledge
                    (control_id, title, description, implementation_guidance,
                     best_practices, common_gaps, nist_mappings, industry_examples,
                     regulatory_mappings, source_url, crawled_at, expires_at,
                     checksum, confidence_score)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    control_id,
                    knowledge_data.get("title", ""),
                    knowledge_data.get("description", ""),
                    json.dumps(knowledge_data.get("implementation_guidance", [])),
                    json.dumps(knowledge_data.get("best_practices", [])),
                    json.dumps(knowledge_data.get("common_gaps", [])),
                    json.dumps(knowledge_data.get("nist_mappings", [])),
                    json.dumps(knowledge_data.get("industry_examples", {})),
                    json.dumps(knowledge_data.get("regulatory_mappings", {})),
                    source_url,
                    now.isoformat(),
                    expires_at.isoformat(),
                    checksum,
                    knowledge_data.get("confidence_score", 0.5)
                ))

                # Store mappings separately for efficient querying
                self._store_mappings(cursor, control_id, knowledge_data)

                # Store implementation examples
                self._store_examples(cursor, control_id, knowledge_data)

                conn.commit()

            # Also store as JSON file for backup
            self._store_json_backup(control_id, knowledge_data)

            logger.info(f"Cached knowledge for control {control_id}")
            return True

        except Exception as e:
            logger.error(f"Error storing control knowledge for {control_id}: {e}")
            return False

    def get_control_details(self, control_id: str, check_expiry: bool = True) -> Optional[Dict[str, Any]]:
        """Retrieve control details from cache"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                cursor.execute("""
                    SELECT * FROM control_knowledge
                    WHERE control_id = ?
                """, (control_id,))

                row = cursor.fetchone()

                if row:
                    # Check expiry if requested
                    if check_expiry:
                        expires_at = datetime.fromisoformat(row[11])
                        if datetime.now() > expires_at:
                            logger.info(f"Cache expired for control {control_id}")
                            return None

                    # Reconstruct knowledge data
                    knowledge_data = {
                        "control_id": row[0],
                        "title": row[1],
                        "description": row[2],
                        "implementation_guidance": json.loads(row[3]),
                        "best_practices": json.loads(row[4]),
                        "common_gaps": json.loads(row[5]),
                        "nist_mappings": json.loads(row[6]),
                        "industry_examples": json.loads(row[7]),
                        "regulatory_mappings": json.loads(row[8]),
                        "source_url": row[9],
                        "crawled_at": row[10],
                        "expires_at": row[11],
                        "checksum": row[12],
                        "confidence_score": row[13]
                    }

                    return knowledge_data

        except Exception as e:
            logger.error(f"Error retrieving control {control_id} from cache: {e}")

        return None

    def get_controls_by_keyword(self, keyword: str) -> List[Dict[str, Any]]:
        """Search cached controls by keyword"""
        results = []

        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # Search in multiple fields
                cursor.execute("""
                    SELECT control_id, title, description, confidence_score
                    FROM control_knowledge
                    WHERE title LIKE ? OR description LIKE ?
                       OR implementation_guidance LIKE ?
                       OR best_practices LIKE ?
                    ORDER BY confidence_score DESC
                """, (f"%{keyword}%", f"%{keyword}%", f"%{keyword}%", f"%{keyword}%"))

                rows = cursor.fetchall()

                for row in rows:
                    results.append({
                        "control_id": row[0],
                        "title": row[1],
                        "description": row[2],
                        "confidence_score": row[3]
                    })

        except Exception as e:
            logger.error(f"Error searching controls by keyword '{keyword}': {e}")

        return results

    def get_control_mappings(self, control_id: str, target_framework: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get framework mappings for a control"""
        mappings = []

        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                if target_framework:
                    cursor.execute("""
                        SELECT target_framework, target_control, mapping_type, confidence
                        FROM control_mappings
                        WHERE source_control = ? AND target_framework = ?
                    """, (control_id, target_framework))
                else:
                    cursor.execute("""
                        SELECT target_framework, target_control, mapping_type, confidence
                        FROM control_mappings
                        WHERE source_control = ?
                    """, (control_id,))

                rows = cursor.fetchall()

                for row in rows:
                    mappings.append({
                        "framework": row[0],
                        "control": row[1],
                        "type": row[2],
                        "confidence": row[3]
                    })

        except Exception as e:
            logger.error(f"Error retrieving mappings for control {control_id}: {e}")

        return mappings

    def get_implementation_examples(self, control_id: str, industry: Optional[str] = None) -> List[Dict[str, str]]:
        """Get implementation examples for a control"""
        examples = []

        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                if industry:
                    cursor.execute("""
                        SELECT industry, example_description, source
                        FROM implementation_examples
                        WHERE control_id = ? AND industry = ?
                    """, (control_id, industry))
                else:
                    cursor.execute("""
                        SELECT industry, example_description, source
                        FROM implementation_examples
                        WHERE control_id = ?
                    """, (control_id,))

                rows = cursor.fetchall()

                for row in rows:
                    examples.append({
                        "industry": row[0],
                        "description": row[1],
                        "source": row[2]
                    })

        except Exception as e:
            logger.error(f"Error retrieving examples for control {control_id}: {e}")

        return examples

    def update_timestamp(self, control_id: str):
        """Update the timestamp for a cached control"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                now = datetime.now()
                expires_at = now + timedelta(days=7)  # Default 7 days

                cursor.execute("""
                    UPDATE control_knowledge
                    SET crawled_at = ?, expires_at = ?
                    WHERE control_id = ?
                """, (now.isoformat(), expires_at.isoformat(), control_id))

                conn.commit()

        except Exception as e:
            logger.error(f"Error updating timestamp for control {control_id}: {e}")

    def check_cache_validity(self, control_id: str) -> bool:
        """Check if cached data is still valid"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                cursor.execute("""
                    SELECT expires_at FROM control_knowledge
                    WHERE control_id = ?
                """, (control_id,))

                row = cursor.fetchone()

                if row:
                    expires_at = datetime.fromisoformat(row[0])
                    return datetime.now() < expires_at

        except Exception as e:
            logger.error(f"Error checking cache validity for {control_id}: {e}")

        return False

    def get_cache_statistics(self) -> Dict[str, Any]:
        """Get cache statistics"""
        stats = {
            "total_controls": 0,
            "valid_controls": 0,
            "expired_controls": 0,
            "total_mappings": 0,
            "total_examples": 0
        }

        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # Total controls
                cursor.execute("SELECT COUNT(*) FROM control_knowledge")
                stats["total_controls"] = cursor.fetchone()[0]

                # Valid controls
                cursor.execute("""
                    SELECT COUNT(*) FROM control_knowledge
                    WHERE expires_at > ?
                """, (datetime.now().isoformat(),))
                stats["valid_controls"] = cursor.fetchone()[0]

                stats["expired_controls"] = stats["total_controls"] - stats["valid_controls"]

                # Total mappings
                cursor.execute("SELECT COUNT(*) FROM control_mappings")
                stats["total_mappings"] = cursor.fetchone()[0]

                # Total examples
                cursor.execute("SELECT COUNT(*) FROM implementation_examples")
                stats["total_examples"] = cursor.fetchone()[0]

        except Exception as e:
            logger.error(f"Error getting cache statistics: {e}")

        return stats

    def clear_expired(self) -> int:
        """Clear expired cache entries"""
        cleared = 0

        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # Get expired control IDs
                cursor.execute("""
                    SELECT control_id FROM control_knowledge
                    WHERE expires_at < ?
                """, (datetime.now().isoformat(),))

                expired_ids = [row[0] for row in cursor.fetchall()]

                # Delete expired entries
                for control_id in expired_ids:
                    cursor.execute("DELETE FROM control_knowledge WHERE control_id = ?", (control_id,))
                    cursor.execute("DELETE FROM control_mappings WHERE source_control = ?", (control_id,))
                    cursor.execute("DELETE FROM implementation_examples WHERE control_id = ?", (control_id,))

                conn.commit()
                cleared = len(expired_ids)

                logger.info(f"Cleared {cleared} expired cache entries")

        except Exception as e:
            logger.error(f"Error clearing expired cache: {e}")

        return cleared

    def _store_mappings(self, cursor, control_id: str, knowledge_data: Dict[str, Any]):
        """Store control mappings in separate table"""
        # Store NIST mappings
        for nist_control in knowledge_data.get("nist_mappings", []):
            cursor.execute("""
                INSERT OR REPLACE INTO control_mappings
                (source_control, target_framework, target_control, mapping_type, confidence)
                VALUES (?, ?, ?, ?, ?)
            """, (control_id, "NIST CSF", nist_control, "direct", 0.8))

        # Store regulatory mappings
        for regulation, requirements in knowledge_data.get("regulatory_mappings", {}).items():
            for requirement in requirements:
                cursor.execute("""
                    INSERT OR REPLACE INTO control_mappings
                    (source_control, target_framework, target_control, mapping_type, confidence)
                    VALUES (?, ?, ?, ?, ?)
                """, (control_id, regulation, requirement, "regulatory", 0.7))

    def _store_examples(self, cursor, control_id: str, knowledge_data: Dict[str, Any]):
        """Store implementation examples in separate table"""
        for industry, description in knowledge_data.get("industry_examples", {}).items():
            cursor.execute("""
                INSERT OR REPLACE INTO implementation_examples
                (control_id, industry, example_description, source, added_at)
                VALUES (?, ?, ?, ?, ?)
            """, (control_id, industry, description, "crawled", datetime.now().isoformat()))

    def _store_json_backup(self, control_id: str, knowledge_data: Dict[str, Any]):
        """Store JSON backup of control knowledge"""
        try:
            backup_file = os.path.join(self.cache_dir, f"{control_id}_backup.json")
            with open(backup_file, 'w') as f:
                json.dump(knowledge_data, f, indent=2)
        except Exception as e:
            logger.warning(f"Error storing JSON backup for {control_id}: {e}")

    def _calculate_checksum(self, data: Dict[str, Any]) -> str:
        """Calculate checksum for data integrity"""
        data_str = json.dumps(data, sort_keys=True)
        return hashlib.sha256(data_str.encode()).hexdigest()

    def export_knowledge_base(self, export_path: str) -> bool:
        """Export entire knowledge base to JSON"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                cursor.execute("SELECT * FROM control_knowledge")
                rows = cursor.fetchall()

                knowledge_base = {}
                for row in rows:
                    control_id = row[0]
                    knowledge_base[control_id] = {
                        "title": row[1],
                        "description": row[2],
                        "implementation_guidance": json.loads(row[3]),
                        "best_practices": json.loads(row[4]),
                        "common_gaps": json.loads(row[5]),
                        "nist_mappings": json.loads(row[6]),
                        "industry_examples": json.loads(row[7]),
                        "regulatory_mappings": json.loads(row[8]),
                        "source_url": row[9],
                        "crawled_at": row[10]
                    }

            with open(export_path, 'w') as f:
                json.dump(knowledge_base, f, indent=2)

            logger.info(f"Exported {len(knowledge_base)} controls to {export_path}")
            return True

        except Exception as e:
            logger.error(f"Error exporting knowledge base: {e}")
            return False