"""
Unit tests for the experiment tracking module.

Tests the experiment tracking functionality including experiment saving,
performance analytics, and data export capabilities.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import sys
import tempfile
import sqlite3
import json
from pathlib import Path
from datetime import datetime

# Add src to path for imports
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from src.ui.experiment_tracking import (
    ExperimentTracker, ExperimentDatabase, ExperimentMetrics, PerformanceStats,
    create_experiment_tracker
)
from src.core.interfaces import (
    ExperimentResult, GenerationResult, GenerationRequest,
    ComplianceMode, OutputType
)


class TestExperimentMetrics(unittest.TestCase):
    """Test the ExperimentMetrics dataclass."""
    
    def test_experiment_metrics_creation(self):
        """Test ExperimentMetrics creation."""
        metrics = ExperimentMetrics(
            generation_time=2.5,
            model_used="stable-diffusion-v1-5",
            compliance_mode="research_safe",
            output_type="image",
            success=True,
            quality_score=0.85,
            memory_usage=4096.0,
            gpu_utilization=75.0
        )
        
        self.assertEqual(metrics.generation_time, 2.5)
        self.assertEqual(metrics.model_used, "stable-diffusion-v1-5")
        self.assertEqual(metrics.compliance_mode, "research_safe")
        self.assertEqual(metrics.output_type, "image")
        self.assertTrue(metrics.success)
        self.assertEqual(metrics.quality_score, 0.85)
        self.assertEqual(metrics.memory_usage, 4096.0)
        self.assertEqual(metrics.gpu_utilization, 75.0)


class TestPerformanceStats(unittest.TestCase):
    """Test the PerformanceStats dataclass."""
    
    def test_performance_stats_creation(self):
        """Test PerformanceStats creation."""
        stats = PerformanceStats(
            model_name="test-model",
            avg_generation_time=2.5,
            success_rate=0.95,
            total_experiments=100,
            avg_quality_score=0.85,
            compliance_modes_used=["research_safe", "open_only"]
        )
        
        self.assertEqual(stats.model_name, "test-model")
        self.assertEqual(stats.avg_generation_time, 2.5)
        self.assertEqual(stats.success_rate, 0.95)
        self.assertEqual(stats.total_experiments, 100)
        self.assertEqual(stats.avg_quality_score, 0.85)
        self.assertEqual(stats.compliance_modes_used, ["research_safe", "open_only"])
    
    def test_performance_stats_default_compliance_modes(self):
        """Test PerformanceStats with default compliance modes."""
        stats = PerformanceStats(
            model_name="test-model",
            avg_generation_time=2.5,
            success_rate=0.95,
            total_experiments=100
        )
        
        self.assertEqual(stats.compliance_modes_used, [])


class TestExperimentDatabase(unittest.TestCase):
    """Test the ExperimentDatabase class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create temporary database for testing
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        self.temp_db.close()
        self.db_path = self.temp_db.name
        
        self.database = ExperimentDatabase(self.db_path)
    
    def tearDown(self):
        """Clean up test fixtures."""
        # Close database connection properly
        if hasattr(self, 'database'):
            del self.database
        # Try to remove file, ignore if locked
        try:
            Path(self.db_path).unlink(missing_ok=True)
        except PermissionError:
            pass  # File is locked, ignore
    
    def test_database_initialization(self):
        """Test database initialization."""
        self.assertTrue(Path(self.db_path).exists())
        
        # Check that tables were created
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Check experiments table
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='experiments'")
            self.assertIsNotNone(cursor.fetchone())
            
            # Check performance_metrics table
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='performance_metrics'")
            self.assertIsNotNone(cursor.fetchone())
    
    def test_save_experiment_success(self):
        """Test successful experiment saving."""
        # Create mock experiment
        request = Mock()
        request.prompt = "test prompt"
        request.output_type = OutputType.IMAGE
        request.compliance_mode = ComplianceMode.RESEARCH_SAFE
        request.additional_params = {"width": 512, "height": 512}
        
        result = Mock()
        result.model_used = "stable-diffusion-v1-5"
        result.generation_time = 2.5
        result.success = True
        result.output_path = Path("test_output.png")
        result.quality_metrics = {"overall_score": 0.85, "aesthetic_score": 0.9}
        result.error_message = None
        
        experiment = Mock()
        experiment.experiment_id = "test_exp_001"
        experiment.timestamp = "2024-01-01T12:00:00"
        experiment.generation_request = request
        experiment.generation_result = result
        
        success = self.database.save_experiment(experiment, "Test research notes")
        
        self.assertTrue(success)
        
        # Verify experiment was saved
        experiments = self.database.get_experiments()
        self.assertEqual(len(experiments), 1)
        
        saved_exp = experiments[0]
        self.assertEqual(saved_exp['id'], "test_exp_001")
        self.assertEqual(saved_exp['prompt'], "test prompt")
        self.assertEqual(saved_exp['model_used'], "stable-diffusion-v1-5")
        self.assertEqual(saved_exp['research_notes'], "Test research notes")
    
    def test_save_experiment_with_metrics(self):
        """Test saving experiment with performance metrics."""
        # Create mock experiment with quality metrics
        request = Mock()
        request.prompt = "test prompt"
        request.output_type = OutputType.IMAGE
        request.compliance_mode = ComplianceMode.RESEARCH_SAFE
        request.additional_params = {}
        
        result = Mock()
        result.model_used = "stable-diffusion-v1-5"
        result.generation_time = 2.5
        result.success = True
        result.output_path = None
        result.quality_metrics = {
            "overall_score": 0.85,
            "aesthetic_score": 0.9,
            "technical_score": 0.8
        }
        result.error_message = None
        
        experiment = Mock()
        experiment.experiment_id = "test_exp_002"
        experiment.timestamp = "2024-01-01T12:00:00"
        experiment.generation_request = request
        experiment.generation_result = result
        
        success = self.database.save_experiment(experiment)
        
        self.assertTrue(success)
        
        # Check that performance metrics were saved
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM performance_metrics WHERE experiment_id = ?", ("test_exp_002",))
            metrics = cursor.fetchall()
            
            self.assertEqual(len(metrics), 3)  # Three metrics saved
    
    def test_get_experiments(self):
        """Test retrieving experiments."""
        # Save a test experiment first
        self._save_test_experiment("exp_001", "Test prompt 1")
        self._save_test_experiment("exp_002", "Test prompt 2")
        
        experiments = self.database.get_experiments(limit=10)
        
        self.assertEqual(len(experiments), 2)
        self.assertIsInstance(experiments[0], dict)
        self.assertIn('id', experiments[0])
        self.assertIn('prompt', experiments[0])
    
    def test_get_experiments_with_limit(self):
        """Test retrieving experiments with limit."""
        # Save multiple test experiments
        for i in range(5):
            self._save_test_experiment(f"exp_{i:03d}", f"Test prompt {i}")
        
        experiments = self.database.get_experiments(limit=3)
        
        self.assertEqual(len(experiments), 3)
    
    def test_get_performance_stats(self):
        """Test getting performance statistics."""
        # Save experiments with different models
        self._save_test_experiment("exp_001", "Test 1", model="model-a", success=True, quality=0.8)
        self._save_test_experiment("exp_002", "Test 2", model="model-a", success=True, quality=0.9)
        self._save_test_experiment("exp_003", "Test 3", model="model-b", success=False, quality=None)
        
        stats = self.database.get_performance_stats()
        
        self.assertEqual(len(stats), 2)  # Two different models
        
        # Find stats for model-a
        model_a_stats = next((s for s in stats if s.model_name == "model-a"), None)
        self.assertIsNotNone(model_a_stats)
        self.assertEqual(model_a_stats.total_experiments, 2)
        self.assertEqual(model_a_stats.success_rate, 1.0)  # Both successful
        self.assertAlmostEqual(model_a_stats.avg_quality_score, 0.85, places=2)  # Average of 0.8 and 0.9
    
    def _save_test_experiment(self, exp_id, prompt, model="test-model", success=True, quality=0.8):
        """Helper method to save a test experiment."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO experiments (
                    id, timestamp, prompt, output_type, model_used, compliance_mode,
                    generation_time, success, output_path, research_notes, quality_score,
                    parameters, error_message
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                exp_id, "2024-01-01T12:00:00", prompt, "image", model, "research_safe",
                2.5, 1 if success else 0, None, "", quality, "{}", None
            ))
            conn.commit()


class TestExperimentTracker(unittest.TestCase):
    """Test the ExperimentTracker class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create temporary database for testing
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        self.temp_db.close()
        self.db_path = self.temp_db.name
        
        self.tracker = ExperimentTracker(self.db_path)
    
    def tearDown(self):
        """Clean up test fixtures."""
        # Close database connection properly
        if hasattr(self, 'tracker'):
            del self.tracker
        # Try to remove file, ignore if locked
        try:
            Path(self.db_path).unlink(missing_ok=True)
        except PermissionError:
            pass  # File is locked, ignore
    
    def test_tracker_initialization(self):
        """Test tracker initialization."""
        self.assertIsInstance(self.tracker.database, ExperimentDatabase)
        self.assertEqual(self.tracker.current_experiments, {})
    
    def test_start_experiment(self):
        """Test starting a new experiment."""
        request = Mock()
        request.prompt = "test prompt"
        
        experiment_id = self.tracker.start_experiment(request)
        
        self.assertIsInstance(experiment_id, str)
        self.assertIn(experiment_id, self.tracker.current_experiments)
        
        experiment = self.tracker.current_experiments[experiment_id]
        self.assertEqual(experiment.request, request)
        self.assertIsNone(experiment.result)
    
    def test_complete_experiment_success(self):
        """Test completing an experiment successfully."""
        # Start an experiment
        request = Mock()
        experiment_id = self.tracker.start_experiment(request)
        
        # Complete it
        result = Mock()
        result.success = True
        
        success = self.tracker.complete_experiment(experiment_id, result, "Test notes")
        
        self.assertTrue(success)
        self.assertNotIn(experiment_id, self.tracker.current_experiments)
    
    def test_complete_experiment_unknown_id(self):
        """Test completing an experiment with unknown ID."""
        result = Mock()
        
        success = self.tracker.complete_experiment("unknown_id", result)
        
        self.assertFalse(success)
    
    def test_get_experiment_history(self):
        """Test getting experiment history."""
        # Mock the database method
        self.tracker.database.get_experiments = Mock(return_value=[
            {"id": "exp_001", "prompt": "test 1"},
            {"id": "exp_002", "prompt": "test 2"}
        ])
        
        history = self.tracker.get_experiment_history()
        
        self.assertEqual(len(history), 2)
        self.tracker.database.get_experiments.assert_called_once_with(limit=100)
    
    def test_get_performance_analytics_no_data(self):
        """Test getting performance analytics with no data."""
        # Mock empty stats
        self.tracker.database.get_performance_stats = Mock(return_value=[])
        
        analytics = self.tracker.get_performance_analytics()
        
        self.assertIn("error", analytics)
        self.assertIn("No experiment data", analytics["error"])
    
    def test_get_performance_analytics_with_data(self):
        """Test getting performance analytics with data."""
        # Mock stats
        mock_stats = [
            PerformanceStats(
                model_name="model-a",
                avg_generation_time=2.5,
                success_rate=0.9,
                total_experiments=10,
                avg_quality_score=0.8,
                compliance_modes_used=["research_safe"]
            ),
            PerformanceStats(
                model_name="model-b",
                avg_generation_time=3.0,
                success_rate=0.8,
                total_experiments=5,
                avg_quality_score=0.7,
                compliance_modes_used=["open_only"]
            )
        ]
        
        self.tracker.database.get_performance_stats = Mock(return_value=mock_stats)
        
        analytics = self.tracker.get_performance_analytics()
        
        self.assertEqual(analytics["total_experiments"], 15)
        self.assertAlmostEqual(analytics["overall_success_rate"], 0.8667, places=3)
        self.assertEqual(analytics["best_performing_model"], "model-a")
        self.assertEqual(len(analytics["model_stats"]), 2)
    
    @patch('src.ui.experiment_tracking.MATPLOTLIB_AVAILABLE', False)
    def test_create_performance_plot_no_matplotlib(self):
        """Test creating performance plot without matplotlib."""
        plot = self.tracker.create_performance_plot()
        
        self.assertIsNone(plot)
    
    @patch('src.ui.experiment_tracking.MATPLOTLIB_AVAILABLE', True)
    @patch('src.ui.experiment_tracking.plt')
    def test_create_performance_plot_with_matplotlib(self, mock_plt):
        """Test creating performance plot with matplotlib."""
        # Mock stats
        mock_stats = [
            PerformanceStats("model-a", 2.5, 0.9, 10),
            PerformanceStats("model-b", 3.0, 0.8, 5)
        ]
        
        self.tracker.database.get_performance_stats = Mock(return_value=mock_stats)
        
        # Mock matplotlib
        mock_fig = Mock()
        mock_ax1 = Mock()
        mock_ax2 = Mock()
        mock_plt.subplots.return_value = (mock_fig, (mock_ax1, mock_ax2))
        
        plot = self.tracker.create_performance_plot()
        
        self.assertEqual(plot, mock_fig)
        mock_plt.subplots.assert_called_once_with(1, 2, figsize=(12, 5))
    
    @patch('src.ui.experiment_tracking.PANDAS_AVAILABLE', False)
    def test_export_experiments_csv_no_pandas(self):
        """Test exporting to CSV without pandas."""
        result = self.tracker.export_experiments("CSV")
        
        self.assertIsNone(result)
    
    @patch('src.ui.experiment_tracking.PANDAS_AVAILABLE', True)
    @patch('src.ui.experiment_tracking.pd')
    def test_export_experiments_csv_with_pandas(self, mock_pd):
        """Test exporting to CSV with pandas."""
        # Mock experiments data
        mock_experiments = [
            {"id": "exp_001", "prompt": "test 1"},
            {"id": "exp_002", "prompt": "test 2"}
        ]
        
        self.tracker.database.get_experiments = Mock(return_value=mock_experiments)
        
        # Mock pandas DataFrame
        mock_df = Mock()
        mock_pd.DataFrame.return_value = mock_df
        
        result = self.tracker.export_experiments("CSV")
        
        self.assertIsInstance(result, str)
        self.assertTrue(result.endswith(".csv"))
        mock_pd.DataFrame.assert_called_once_with(mock_experiments)
        mock_df.to_csv.assert_called_once()
    
    def test_export_experiments_json(self):
        """Test exporting to JSON."""
        # Mock experiments data
        mock_experiments = [
            {"id": "exp_001", "prompt": "test 1"},
            {"id": "exp_002", "prompt": "test 2"}
        ]
        
        self.tracker.database.get_experiments = Mock(return_value=mock_experiments)
        
        with patch('builtins.open', unittest.mock.mock_open()) as mock_file:
            result = self.tracker.export_experiments("JSON")
            
            self.assertIsInstance(result, str)
            self.assertTrue(result.endswith(".json"))
            mock_file.assert_called_once()
    
    def test_export_experiments_unsupported_format(self):
        """Test exporting with unsupported format."""
        result = self.tracker.export_experiments("UNSUPPORTED")
        
        self.assertIsNone(result)
    
    def test_export_experiments_no_data(self):
        """Test exporting with no experiment data."""
        self.tracker.database.get_experiments = Mock(return_value=[])
        
        result = self.tracker.export_experiments("JSON")
        
        self.assertIsNone(result)


class TestUtilityFunctions(unittest.TestCase):
    """Test utility functions for experiment tracking."""
    
    def test_create_experiment_tracker(self):
        """Test creating experiment tracker."""
        with tempfile.NamedTemporaryFile(delete=False, suffix='.db') as temp_db:
            db_path = temp_db.name
        
        try:
            tracker = create_experiment_tracker(db_path)
            
            self.assertIsInstance(tracker, ExperimentTracker)
            self.assertTrue(Path(db_path).exists())
        finally:
            try:
                Path(db_path).unlink(missing_ok=True)
            except PermissionError:
                pass  # File is locked, ignore
    
    def test_create_experiment_tracker_default_path(self):
        """Test creating experiment tracker with default path."""
        tracker = create_experiment_tracker()
        
        self.assertIsInstance(tracker, ExperimentTracker)
        
        # Clean up default database file
        try:
            Path("experiments.db").unlink(missing_ok=True)
        except PermissionError:
            pass  # File is locked, ignore


if __name__ == '__main__':
    unittest.main()