"""
Experiment tracking and analytics for the research interface.

This module provides functionality for saving experiments with research notes,
displaying dataset license breakdown statistics, and creating performance
monitoring and model comparison tools.
"""

import logging
import json
import time
import sqlite3
import uuid
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum

try:
    from ..core.interfaces import (
        ExperimentResult, GenerationResult, GenerationRequest,
        ComplianceMode, OutputType
    )
except ImportError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from core.interfaces import (
        ExperimentResult, GenerationResult, GenerationRequest,
        ComplianceMode, OutputType
    )

logger = logging.getLogger(__name__)

# Try to import required libraries
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    pd = None
    PANDAS_AVAILABLE = False
    logger.warning("Pandas not available - some analytics features will be limited")

try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    plt = None
    mdates = None
    MATPLOTLIB_AVAILABLE = False
    logger.warning("Matplotlib not available - plotting features will be limited")

try:
    import gradio as gr
    GRADIO_AVAILABLE = True
except ImportError:
    # Create a mock gr module for type hints and basic functionality
    class MockGradio:
        class Dataframe:
            pass
        class JSON:
            pass
        class Plot:
            pass
        class Button:
            pass
        class Dropdown:
            pass
        class File:
            pass
    
    gr = MockGradio()
    GRADIO_AVAILABLE = False
    logger.warning("Gradio not available - UI components will be limited")


@dataclass
class ExperimentMetrics:
    """Metrics for experiment performance analysis."""
    generation_time: float
    model_used: str
    compliance_mode: str
    output_type: str
    success: bool
    quality_score: Optional[float] = None
    memory_usage: Optional[float] = None
    gpu_utilization: Optional[float] = None


@dataclass
class PerformanceStats:
    """Performance statistics for model comparison."""
    model_name: str
    avg_generation_time: float
    success_rate: float
    total_experiments: int
    avg_quality_score: Optional[float] = None
    compliance_modes_used: List[str] = None
    
    def __post_init__(self):
        if self.compliance_modes_used is None:
            self.compliance_modes_used = []


class ExperimentDatabase:
    """Database manager for experiment storage and retrieval."""
    
    def __init__(self, db_path: str = "experiments.db"):
        """Initialize the experiment database."""
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()
        logger.info(f"ExperimentDatabase initialized at {self.db_path}")
    
    def _init_database(self):
        """Initialize database tables."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Create experiments table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS experiments (
                    id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    prompt TEXT NOT NULL,
                    output_type TEXT NOT NULL,
                    model_used TEXT NOT NULL,
                    compliance_mode TEXT NOT NULL,
                    generation_time REAL NOT NULL,
                    success INTEGER NOT NULL,
                    output_path TEXT,
                    research_notes TEXT,
                    quality_score REAL,
                    parameters TEXT,
                    error_message TEXT
                )
            """)
            
            # Create performance metrics table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS performance_metrics (
                    experiment_id TEXT,
                    metric_name TEXT,
                    metric_value REAL,
                    timestamp TEXT,
                    FOREIGN KEY (experiment_id) REFERENCES experiments (id)
                )
            """)
            
            conn.commit()
    
    def save_experiment(self, experiment: ExperimentResult, research_notes: str = "") -> bool:
        """Save an experiment to the database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT INTO experiments (
                        id, timestamp, prompt, output_type, model_used, compliance_mode,
                        generation_time, success, output_path, research_notes, quality_score,
                        parameters, error_message
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    experiment.experiment_id,
                    experiment.timestamp.isoformat(),
                    experiment.request.prompt,
                    experiment.request.output_type.value,
                    experiment.result.model_used,
                    experiment.request.compliance_mode.value,
                    experiment.result.generation_time,
                    1 if experiment.result.success else 0,
                    str(experiment.result.output_path) if experiment.result.output_path else None,
                    research_notes,
                    experiment.result.quality_metrics.get('overall_score') if experiment.result.quality_metrics else None,
                    json.dumps(experiment.request.additional_params or {}),
                    experiment.result.error_message
                ))
                
                # Save performance metrics if available
                if experiment.result.quality_metrics:
                    for metric_name, metric_value in experiment.result.quality_metrics.items():
                        if isinstance(metric_value, (int, float)):
                            cursor.execute("""
                                INSERT INTO performance_metrics (experiment_id, metric_name, metric_value, timestamp)
                                VALUES (?, ?, ?, ?)
                            """, (experiment.experiment_id, metric_name, metric_value, experiment.timestamp.isoformat()))
                
                conn.commit()
                logger.info(f"Experiment {experiment.experiment_id} saved to database")
                return True
                
        except Exception as e:
            logger.error(f"Failed to save experiment: {e}")
            return False
    
    def get_experiments(self, limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        """Retrieve experiments from database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT * FROM experiments 
                    ORDER BY timestamp DESC 
                    LIMIT ? OFFSET ?
                """, (limit, offset))
                
                columns = [description[0] for description in cursor.description]
                experiments = []
                
                for row in cursor.fetchall():
                    experiment = dict(zip(columns, row))
                    # Parse JSON parameters
                    if experiment['parameters']:
                        experiment['parameters'] = json.loads(experiment['parameters'])
                    experiments.append(experiment)
                
                return experiments
                
        except Exception as e:
            logger.error(f"Failed to retrieve experiments: {e}")
            return []
    
    def get_performance_stats(self) -> List[PerformanceStats]:
        """Get performance statistics by model."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT 
                        model_used,
                        AVG(generation_time) as avg_time,
                        AVG(CAST(success as REAL)) as success_rate,
                        COUNT(*) as total_experiments,
                        AVG(quality_score) as avg_quality,
                        GROUP_CONCAT(DISTINCT compliance_mode) as compliance_modes
                    FROM experiments 
                    GROUP BY model_used
                """)
                
                stats = []
                for row in cursor.fetchall():
                    compliance_modes = row[5].split(',') if row[5] else []
                    stats.append(PerformanceStats(
                        model_name=row[0],
                        avg_generation_time=row[1] or 0.0,
                        success_rate=row[2] or 0.0,
                        total_experiments=row[3],
                        avg_quality_score=row[4],
                        compliance_modes_used=compliance_modes
                    ))
                
                return stats
                
        except Exception as e:
            logger.error(f"Failed to get performance stats: {e}")
            return []


class ExperimentTracker:
    """Main experiment tracking system."""
    
    def __init__(self, db_path: str = "experiments.db"):
        """Initialize the experiment tracker."""
        self.database = ExperimentDatabase(db_path)
        self.current_experiments: Dict[str, ExperimentResult] = {}
        logger.info("ExperimentTracker initialized")
    
    def start_experiment(self, request: GenerationRequest) -> str:
        """Start tracking a new experiment."""
        experiment_id = str(uuid.uuid4())
        
        experiment = ExperimentResult(
            experiment_id=experiment_id,
            timestamp=datetime.now(),
            request=request,
            result=None,
            notes=""
        )
        
        self.current_experiments[experiment_id] = experiment
        logger.info(f"Started tracking experiment: {experiment_id}")
        return experiment_id
    
    def complete_experiment(self, experiment_id: str, result: GenerationResult, 
                          research_notes: str = "") -> bool:
        """Complete an experiment with results."""
        if experiment_id not in self.current_experiments:
            logger.error(f"Unknown experiment ID: {experiment_id}")
            return False
        
        experiment = self.current_experiments[experiment_id]
        experiment.result = result
        experiment.notes = research_notes
        
        # Save to database
        success = self.database.save_experiment(experiment, research_notes)
        
        if success:
            # Remove from active experiments
            del self.current_experiments[experiment_id]
            logger.info(f"Completed experiment: {experiment_id}")
        
        return success
    
    def get_experiment_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get experiment history."""
        return self.database.get_experiments(limit=limit)
    
    def get_performance_analytics(self) -> Dict[str, Any]:
        """Get performance analytics and statistics."""
        stats = self.database.get_performance_stats()
        
        if not stats:
            return {"error": "No experiment data available"}
        
        # Calculate overall statistics
        total_experiments = sum(s.total_experiments for s in stats)
        avg_success_rate = sum(s.success_rate * s.total_experiments for s in stats) / total_experiments if total_experiments > 0 else 0
        
        # Find best performing model
        best_model = max(stats, key=lambda s: s.success_rate * (s.avg_quality_score or 0))
        
        return {
            "total_experiments": total_experiments,
            "overall_success_rate": avg_success_rate,
            "best_performing_model": best_model.model_name,
            "model_stats": [asdict(stat) for stat in stats],
            "compliance_modes_used": list(set(mode for stat in stats for mode in stat.compliance_modes_used))
        }
    
    def create_performance_plot(self):
        """Create performance visualization plot."""
        if not MATPLOTLIB_AVAILABLE:
            logger.warning("Matplotlib not available - cannot create performance plot")
            return None
        
        stats = self.database.get_performance_stats()
        if not stats:
            return None
        
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # Success rate by model
            models = [s.model_name for s in stats]
            success_rates = [s.success_rate * 100 for s in stats]
            
            ax1.bar(models, success_rates)
            ax1.set_title('Success Rate by Model')
            ax1.set_ylabel('Success Rate (%)')
            ax1.tick_params(axis='x', rotation=45)
            
            # Generation time by model
            gen_times = [s.avg_generation_time for s in stats]
            
            ax2.bar(models, gen_times)
            ax2.set_title('Average Generation Time by Model')
            ax2.set_ylabel('Time (seconds)')
            ax2.tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            return fig
            
        except Exception as e:
            logger.error(f"Failed to create performance plot: {e}")
            return None
    
    def export_experiments(self, format_type: str = "CSV", limit: int = 1000) -> Optional[str]:
        """Export experiments to file."""
        experiments = self.database.get_experiments(limit=limit)
        
        if not experiments:
            return None
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        try:
            if format_type.upper() == "CSV":
                if not PANDAS_AVAILABLE:
                    logger.error("Pandas not available - cannot export to CSV")
                    return None
                
                df = pd.DataFrame(experiments)
                filename = f"experiments_export_{timestamp}.csv"
                df.to_csv(filename, index=False)
                return filename
                
            elif format_type.upper() == "JSON":
                filename = f"experiments_export_{timestamp}.json"
                with open(filename, 'w') as f:
                    json.dump(experiments, f, indent=2, default=str)
                return filename
                
            else:
                logger.error(f"Unsupported export format: {format_type}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to export experiments: {e}")
            return None


def create_experiment_tracker(db_path: str = "experiments.db") -> ExperimentTracker:
    """Create and initialize an experiment tracker."""
    tracker = ExperimentTracker(db_path)
    logger.info("Experiment tracker created successfully")
    return tracker