"""
Experiment tracker for research data persistence.

This module implements experiment tracking functionality for saving,
retrieving, and managing research experiments and their results.
"""

import logging
import json
import sqlite3
from typing import Dict, Any, List, Optional
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


class ExperimentTracker:
    """
    Experiment tracker for research data persistence.
    
    Manages experiment data storage, retrieval, and organization
    for research analysis and comparison.
    """
    
    def __init__(self, db_path: str = "experiments.db"):
        """
        Initialize experiment tracker.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self._init_database()
        logger.info(f"ExperimentTracker initialized with database: {self.db_path}")
    
    def _init_database(self) -> None:
        """Initialize the experiments database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Create experiments table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS experiments (
                        id TEXT PRIMARY KEY,
                        name TEXT NOT NULL,
                        description TEXT,
                        tags TEXT,  -- JSON array
                        results TEXT,  -- JSON object
                        compliance_mode TEXT,
                        hardware_info TEXT,  -- JSON object
                        system_state TEXT,  -- JSON object
                        timestamp TEXT,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Create index on timestamp for faster queries
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_experiments_timestamp 
                    ON experiments(timestamp)
                """)
                
                # Create index on tags for filtering (only if column exists)
                try:
                    cursor.execute("""
                        CREATE INDEX IF NOT EXISTS idx_experiments_tags 
                        ON experiments(tags)
                    """)
                except sqlite3.OperationalError:
                    # Column might not exist in existing database
                    logger.warning("Could not create tags index - column may not exist")
                
                conn.commit()
                logger.debug("Database initialized successfully")
                
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise
    
    def save_experiment(self, experiment_data: Dict[str, Any]) -> str:
        """
        Save experiment data.
        
        Args:
            experiment_data: Dictionary containing experiment information
            
        Returns:
            str: Unique experiment ID
        """
        try:
            # Generate unique experiment ID
            timestamp = datetime.now()
            experiment_id = f"exp_{timestamp.strftime('%Y%m%d_%H%M%S')}_{id(experiment_data) % 10000:04d}"
            
            # Prepare data for storage
            name = experiment_data.get('name', 'Unnamed Experiment')
            description = experiment_data.get('description')
            tags = json.dumps(experiment_data.get('tags', []))
            results = json.dumps(experiment_data.get('results', {}))
            compliance_mode = experiment_data.get('compliance_mode', 'unknown')
            hardware_info = json.dumps(experiment_data.get('hardware_info', {}))
            system_state = json.dumps(experiment_data.get('system_state', {}))
            timestamp_str = experiment_data.get('timestamp', timestamp.isoformat())
            
            # Save to database
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO experiments 
                    (id, name, description, tags, results, compliance_mode, 
                     hardware_info, system_state, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    experiment_id, name, description, tags, results,
                    compliance_mode, hardware_info, system_state, timestamp_str
                ))
                conn.commit()
            
            logger.info(f"Experiment saved: {experiment_id}")
            return experiment_id
            
        except Exception as e:
            logger.error(f"Failed to save experiment: {e}")
            raise
    
    def get_experiment(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve experiment data by ID.
        
        Args:
            experiment_id: Unique experiment identifier
            
        Returns:
            Dict containing experiment data or None if not found
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT id, name, description, tags, results, compliance_mode,
                           hardware_info, system_state, timestamp, created_at
                    FROM experiments 
                    WHERE id = ?
                """, (experiment_id,))
                
                row = cursor.fetchone()
                
                if not row:
                    return None
                
                # Parse JSON fields
                return {
                    'id': row[0],
                    'name': row[1],
                    'description': row[2],
                    'tags': json.loads(row[3]) if row[3] else [],
                    'results': json.loads(row[4]) if row[4] else {},
                    'compliance_mode': row[5],
                    'hardware_info': json.loads(row[6]) if row[6] else {},
                    'system_state': json.loads(row[7]) if row[7] else {},
                    'timestamp': row[8],
                    'created_at': row[9]
                }
                
        except Exception as e:
            logger.error(f"Failed to retrieve experiment {experiment_id}: {e}")
            return None
    
    def list_experiments(
        self, 
        limit: int = 50, 
        offset: int = 0,
        tag_filter: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        List experiments with optional filtering.
        
        Args:
            limit: Maximum number of experiments to return
            offset: Number of experiments to skip
            tag_filter: List of tags to filter by
            
        Returns:
            List of experiment summaries
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Build query with optional tag filtering
                query = """
                    SELECT id, name, description, tags, compliance_mode, timestamp, created_at
                    FROM experiments
                """
                params = []
                
                if tag_filter:
                    # Simple tag filtering (could be improved with proper JSON queries)
                    tag_conditions = []
                    for tag in tag_filter:
                        tag_conditions.append("tags LIKE ?")
                        params.append(f'%"{tag}"%')
                    
                    query += " WHERE " + " OR ".join(tag_conditions)
                
                query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
                params.extend([limit, offset])
                
                cursor.execute(query, params)
                rows = cursor.fetchall()
                
                experiments = []
                for row in rows:
                    experiments.append({
                        'id': row[0],
                        'name': row[1],
                        'description': row[2],
                        'tags': json.loads(row[3]) if row[3] else [],
                        'compliance_mode': row[4],
                        'timestamp': row[5],
                        'created_at': row[6]
                    })
                
                return experiments
                
        except Exception as e:
            logger.error(f"Failed to list experiments: {e}")
            return []
    
    def delete_experiment(self, experiment_id: str) -> bool:
        """
        Delete an experiment.
        
        Args:
            experiment_id: Unique experiment identifier
            
        Returns:
            bool: True if deletion successful
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM experiments WHERE id = ?", (experiment_id,))
                
                if cursor.rowcount > 0:
                    conn.commit()
                    logger.info(f"Experiment deleted: {experiment_id}")
                    return True
                else:
                    logger.warning(f"Experiment not found for deletion: {experiment_id}")
                    return False
                    
        except Exception as e:
            logger.error(f"Failed to delete experiment {experiment_id}: {e}")
            return False
    
    def get_experiment_count(self) -> int:
        """Get total number of experiments."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM experiments")
                return cursor.fetchone()[0]
                
        except Exception as e:
            logger.error(f"Failed to get experiment count: {e}")
            return 0
    
    def get_tags_summary(self) -> Dict[str, int]:
        """Get summary of all tags and their usage counts."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT tags FROM experiments WHERE tags IS NOT NULL")
                rows = cursor.fetchall()
                
                tag_counts = {}
                for row in rows:
                    try:
                        tags = json.loads(row[0])
                        for tag in tags:
                            tag_counts[tag] = tag_counts.get(tag, 0) + 1
                    except (json.JSONDecodeError, TypeError):
                        continue
                
                return tag_counts
                
        except Exception as e:
            logger.error(f"Failed to get tags summary: {e}")
            return {}
    
    def cleanup_old_experiments(self, days_old: int = 30) -> int:
        """
        Clean up experiments older than specified days.
        
        Args:
            days_old: Number of days after which to delete experiments
            
        Returns:
            int: Number of experiments deleted
        """
        try:
            cutoff_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
            cutoff_date = cutoff_date.replace(day=cutoff_date.day - days_old)
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    DELETE FROM experiments 
                    WHERE created_at < ?
                """, (cutoff_date.isoformat(),))
                
                deleted_count = cursor.rowcount
                conn.commit()
                
                logger.info(f"Cleaned up {deleted_count} old experiments")
                return deleted_count
                
        except Exception as e:
            logger.error(f"Failed to cleanup old experiments: {e}")
            return 0
    
    def close(self) -> None:
        """
        Close the experiment tracker and clean up resources.
        
        Note: SQLite connections are automatically closed when using context managers,
        so this method is mainly for consistency with other components.
        """
        logger.info("ExperimentTracker closed")
        # No explicit cleanup needed for SQLite with context managers