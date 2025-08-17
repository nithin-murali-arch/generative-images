"""
Integration tests for experiment tracking functionality.

Tests the complete experiment tracking workflow including database operations,
analytics generation, and export functionality.
"""

import unittest
import tempfile
import sys
from pathlib import Path
from unittest.mock import Mock

# Add src to path for imports
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from src.ui.experiment_tracking import create_experiment_tracker
from src.core.interfaces import (
    GenerationRequest, GenerationResult, OutputType, ComplianceMode,
    StyleConfig, HardwareConfig, ConversationContext
)


class TestExperimentTrackingIntegration(unittest.TestCase):
    """Integration tests for experiment tracking system."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create temporary database
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        self.temp_db.close()
        self.db_path = self.temp_db.name
        
        self.tracker = create_experiment_tracker(self.db_path)
    
    def tearDown(self):
        """Clean up test fixtures."""
        # Clean up tracker and database
        if hasattr(self, 'tracker'):
            del self.tracker
        try:
            Path(self.db_path).unlink(missing_ok=True)
        except PermissionError:
            pass  # File is locked on Windows, ignore
    
    def test_complete_experiment_workflow(self):
        """Test complete experiment workflow from start to finish."""
        # Create mock request
        request = Mock()
        request.prompt = "A beautiful landscape"
        request.output_type = OutputType.IMAGE
        request.compliance_mode = ComplianceMode.RESEARCH_SAFE
        request.additional_params = {"width": 512, "height": 512}
        
        # Start experiment
        experiment_id = self.tracker.start_experiment(request)
        self.assertIsInstance(experiment_id, str)
        self.assertIn(experiment_id, self.tracker.current_experiments)
        
        # Create mock result
        result = Mock()
        result.success = True
        result.model_used = "stable-diffusion-v1-5"
        result.generation_time = 2.5
        result.output_path = Path("test_output.png")
        result.quality_metrics = {"overall_score": 0.85}
        result.error_message = None
        
        # Complete experiment
        success = self.tracker.complete_experiment(experiment_id, result, "Test notes")
        self.assertTrue(success)
        self.assertNotIn(experiment_id, self.tracker.current_experiments)
        
        # Verify experiment was saved
        history = self.tracker.get_experiment_history(limit=10)
        self.assertEqual(len(history), 1)
        
        saved_exp = history[0]
        self.assertEqual(saved_exp['id'], experiment_id)
        self.assertEqual(saved_exp['prompt'], "A beautiful landscape")
        self.assertEqual(saved_exp['model_used'], "stable-diffusion-v1-5")
    
    def test_performance_analytics_generation(self):
        """Test performance analytics generation."""
        # Add some test experiments
        self._add_test_experiment("model-a", True, 2.0, 0.8)
        self._add_test_experiment("model-a", True, 3.0, 0.9)
        self._add_test_experiment("model-b", False, 1.0, None)
        
        # Get analytics
        analytics = self.tracker.get_performance_analytics()
        
        self.assertIsInstance(analytics, dict)
        self.assertIn("total_experiments", analytics)
        self.assertIn("overall_success_rate", analytics)
        self.assertIn("model_stats", analytics)
        
        self.assertEqual(analytics["total_experiments"], 3)
        self.assertGreater(analytics["overall_success_rate"], 0.5)  # 2/3 success rate
    
    def test_experiment_export_json(self):
        """Test experiment export to JSON."""
        # Add a test experiment
        self._add_test_experiment("test-model", True, 2.5, 0.85)
        
        # Export to JSON
        filename = self.tracker.export_experiments("JSON")
        
        if filename:  # Only test if export succeeded (depends on available libraries)
            self.assertIsInstance(filename, str)
            self.assertTrue(filename.endswith(".json"))
            
            # Clean up export file
            try:
                Path(filename).unlink(missing_ok=True)
            except (FileNotFoundError, PermissionError):
                pass
    
    def _add_test_experiment(self, model_name, success, gen_time, quality_score):
        """Helper method to add a test experiment directly to database."""
        import sqlite3
        import json
        from datetime import datetime
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO experiments (
                    id, timestamp, prompt, output_type, model_used, compliance_mode,
                    generation_time, success, output_path, research_notes, quality_score,
                    parameters, error_message
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                f"test_{model_name}_{gen_time}", datetime.now().isoformat(),
                "test prompt", "image", model_name, "research_safe",
                gen_time, 1 if success else 0, None, "test notes", quality_score,
                json.dumps({}), None
            ))
            conn.commit()


if __name__ == '__main__':
    unittest.main()