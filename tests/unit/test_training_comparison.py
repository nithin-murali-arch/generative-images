"""
Unit tests for training result comparison and analysis.

Tests comprehensive comparison tools for LoRA training results,
performance analysis, and quality assessment functionality.
"""

import pytest
import tempfile
import statistics
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta

from src.pipelines.training_comparison import (
    TrainingComparator, ComparisonResult, ComparisonType, ComparisonMetric,
    QualityMetrics, PerformanceProfile
)
from src.pipelines.training_monitor import TrainingSession, TrainingMetrics, TrainingMonitor
from src.pipelines.lora_training import LoRAConfig


class TestQualityMetrics:
    """Test quality metrics data structure."""
    
    def test_quality_metrics_initialization(self):
        """Test quality metrics initialization."""
        metrics = QualityMetrics()
        
        assert metrics.convergence_rate == 0.0
        assert metrics.final_loss == 0.0
        assert metrics.loss_stability == 0.0
        assert metrics.overfitting_score == 0.0
        assert metrics.training_efficiency == 0.0
        assert metrics.memory_efficiency == 0.0
        assert metrics.overall_score == 0.0
    
    def test_quality_metrics_with_values(self):
        """Test quality metrics with values."""
        metrics = QualityMetrics(
            convergence_rate=0.05,
            final_loss=0.1,
            loss_stability=0.8,
            overfitting_score=0.7,
            training_efficiency=0.6,
            memory_efficiency=0.9,
            overall_score=0.75
        )
        
        assert metrics.convergence_rate == 0.05
        assert metrics.final_loss == 0.1
        assert metrics.loss_stability == 0.8
        assert metrics.overfitting_score == 0.7
        assert metrics.training_efficiency == 0.6
        assert metrics.memory_efficiency == 0.9
        assert metrics.overall_score == 0.75


class TestPerformanceProfile:
    """Test performance profile data structure."""
    
    def test_performance_profile_initialization(self):
        """Test performance profile initialization."""
        config = LoRAConfig(rank=16, alpha=32.0)
        metrics = QualityMetrics(overall_score=0.8)
        
        profile = PerformanceProfile(
            session_id="test_session",
            config=config,
            quality_metrics=metrics,
            training_time=3600.0,
            peak_memory_mb=8000.0
        )
        
        assert profile.session_id == "test_session"
        assert profile.config == config
        assert profile.quality_metrics == metrics
        assert profile.training_time == 3600.0
        assert profile.peak_memory_mb == 8000.0
        assert profile.convergence_step is None
        assert profile.best_validation_loss is None
        assert profile.training_stable is True
        assert profile.notes == []


class TestComparisonResult:
    """Test comparison result data structure."""
    
    def test_comparison_result_initialization(self):
        """Test comparison result initialization."""
        result = ComparisonResult(
            comparison_id="test_comparison",
            comparison_type=ComparisonType.HYPERPARAMETER,
            sessions=["session1", "session2"],
            metrics={"final_loss": {"session1": 0.1, "session2": 0.2}},
            rankings={"final_loss": ["session1", "session2"]},
            summary={"best_overall": "session1"},
            recommendations=["Use session1 configuration"]
        )
        
        assert result.comparison_id == "test_comparison"
        assert result.comparison_type == ComparisonType.HYPERPARAMETER
        assert result.sessions == ["session1", "session2"]
        assert result.metrics["final_loss"]["session1"] == 0.1
        assert result.rankings["final_loss"] == ["session1", "session2"]
        assert result.summary["best_overall"] == "session1"
        assert len(result.recommendations) == 1


class TestTrainingComparator:
    """Test training comparator functionality."""
    
    @pytest.fixture
    def temp_checkpoint_dir(self):
        """Create temporary checkpoint directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)
    
    @pytest.fixture
    def mock_monitor(self, temp_checkpoint_dir):
        """Create mock training monitor."""
        return TrainingMonitor(checkpoint_dir=temp_checkpoint_dir)
    
    @pytest.fixture
    def comparator(self, mock_monitor):
        """Create training comparator."""
        return TrainingComparator(monitor=mock_monitor)
    
    @pytest.fixture
    def sample_sessions(self):
        """Create sample training sessions."""
        config1 = LoRAConfig(rank=16, alpha=32.0, learning_rate=1e-4)
        config2 = LoRAConfig(rank=32, alpha=64.0, learning_rate=5e-5)
        
        # Create training metrics
        metrics1 = TrainingMetrics(
            train_losses=[1.0, 0.8, 0.6, 0.4, 0.2],
            validation_losses=[0.9, 0.7, 0.5, 0.35, 0.25],
            learning_rates=[1e-4, 9e-5, 8e-5, 7e-5, 6e-5],
            step_times=[1.0, 1.1, 1.0, 0.9, 1.0],
            memory_usage=[4000, 4100, 4000, 3900, 4000]
        )
        
        metrics2 = TrainingMetrics(
            train_losses=[1.2, 0.9, 0.7, 0.5, 0.3],
            validation_losses=[1.1, 0.8, 0.6, 0.45, 0.35],
            learning_rates=[5e-5, 4.5e-5, 4e-5, 3.5e-5, 3e-5],
            step_times=[1.5, 1.6, 1.4, 1.3, 1.4],
            memory_usage=[6000, 6200, 6100, 6000, 6050]
        )
        
        session1 = TrainingSession(
            session_id="session1",
            start_time=datetime.now() - timedelta(hours=2),
            end_time=datetime.now() - timedelta(hours=1),
            config=config1,
            dataset_name="dataset1",
            base_model="stable-diffusion-v1-5",
            status="completed",
            metrics=metrics1
        )
        
        session2 = TrainingSession(
            session_id="session2",
            start_time=datetime.now() - timedelta(hours=3),
            end_time=datetime.now() - timedelta(hours=1, minutes=30),
            config=config2,
            dataset_name="dataset1",
            base_model="stable-diffusion-v1-5",
            status="completed",
            metrics=metrics2
        )
        
        return {"session1": session1, "session2": session2}
    
    def test_comparator_initialization(self, mock_monitor):
        """Test comparator initialization."""
        comparator = TrainingComparator(monitor=mock_monitor)
        
        assert comparator.monitor == mock_monitor
        assert comparator.comparison_cache == {}
        assert isinstance(comparator.quality_weights, dict)
        assert 'convergence_rate' in comparator.quality_weights
        assert 'final_loss' in comparator.quality_weights
    
    def test_calculate_session_metrics_basic(self, comparator):
        """Test basic session metrics calculation."""
        # Create session with minimal data
        metrics = TrainingMetrics(train_losses=[1.0, 0.5, 0.2])
        session = TrainingSession(
            session_id="test_session",
            start_time=datetime.now(),
            metrics=metrics
        )
        
        quality_metrics = comparator._calculate_session_metrics(session)
        
        assert isinstance(quality_metrics, QualityMetrics)
        assert quality_metrics.final_loss == 0.2
        assert quality_metrics.overall_score >= 0
    
    def test_calculate_session_metrics_comprehensive(self, comparator):
        """Test comprehensive session metrics calculation."""
        # Create session with comprehensive data
        metrics = TrainingMetrics(
            train_losses=[1.0, 0.8, 0.6, 0.4, 0.2, 0.15, 0.12, 0.1, 0.09, 0.08, 0.07],
            validation_losses=[0.9, 0.7, 0.5, 0.35, 0.25, 0.2, 0.18, 0.15, 0.14, 0.13],
            step_times=[1.0, 1.1, 1.0, 0.9, 1.0, 1.1, 1.0, 0.9, 1.0, 1.1],
            memory_usage=[4000, 4100, 4000, 3900, 4000, 4050, 4000, 3950, 4000, 4025]
        )
        
        session = TrainingSession(
            session_id="test_session",
            start_time=datetime.now(),
            metrics=metrics
        )
        
        quality_metrics = comparator._calculate_session_metrics(session)
        
        assert quality_metrics.final_loss == 0.07
        assert quality_metrics.convergence_rate > 0  # Should show convergence
        assert quality_metrics.loss_stability > 0  # Should have some stability
        assert quality_metrics.overfitting_score > 0  # Should have overfitting score
        assert quality_metrics.training_efficiency > 0  # Should have efficiency
        assert quality_metrics.memory_efficiency > 0  # Should have memory efficiency
        assert quality_metrics.overall_score > 0  # Should have overall score
    
    def test_extract_metric_value(self, comparator):
        """Test metric value extraction."""
        metrics = QualityMetrics(
            final_loss=0.1,
            convergence_rate=0.05,
            loss_stability=0.8,
            overall_score=0.75
        )
        
        assert comparator._extract_metric_value(metrics, ComparisonMetric.FINAL_LOSS) == 0.1
        assert comparator._extract_metric_value(metrics, ComparisonMetric.CONVERGENCE_SPEED) == 0.05
        assert comparator._extract_metric_value(metrics, ComparisonMetric.STABILITY) == 0.8
        assert comparator._extract_metric_value(metrics, ComparisonMetric.QUALITY_SCORE) == 0.75
    
    def test_rank_sessions_by_metric(self, comparator):
        """Test session ranking by metric."""
        metric_values = {"session1": 0.1, "session2": 0.2, "session3": 0.05}
        
        # For final loss (lower is better)
        ranking = comparator._rank_sessions_by_metric(metric_values, ComparisonMetric.FINAL_LOSS)
        assert ranking == ["session3", "session1", "session2"]
        
        # For quality score (higher is better)
        ranking = comparator._rank_sessions_by_metric(metric_values, ComparisonMetric.QUALITY_SCORE)
        assert ranking == ["session2", "session1", "session3"]
    
    def test_compare_sessions_success(self, comparator, sample_sessions):
        """Test successful session comparison."""
        # Mock the monitor to return sample sessions
        def mock_load_session(session_id):
            return sample_sessions.get(session_id)
        
        comparator.monitor.load_session = mock_load_session
        
        result = comparator.compare_sessions(
            ["session1", "session2"],
            ComparisonType.HYPERPARAMETER
        )
        
        assert isinstance(result, ComparisonResult)
        assert result.comparison_type == ComparisonType.HYPERPARAMETER
        assert len(result.sessions) == 2
        assert "session1" in result.sessions
        assert "session2" in result.sessions
        assert len(result.metrics) > 0
        assert len(result.rankings) > 0
        assert isinstance(result.summary, dict)
        assert isinstance(result.recommendations, list)
    
    def test_compare_sessions_insufficient_sessions(self, comparator):
        """Test comparison with insufficient sessions."""
        # Mock monitor to return None (session not found)
        comparator.monitor.load_session = Mock(return_value=None)
        
        with pytest.raises(ValueError, match="Need at least 2 valid sessions"):
            comparator.compare_sessions(["session1", "session2"])
    
    def test_analyze_hyperparameter_impact(self, comparator, sample_sessions):
        """Test hyperparameter impact analysis."""
        # Mock the monitor
        def mock_load_session(session_id):
            return sample_sessions.get(session_id)
        
        comparator.monitor.load_session = mock_load_session
        
        result = comparator.analyze_hyperparameter_impact(["session1", "session2"])
        
        assert isinstance(result, dict)
        assert 'parameter_analysis' in result
        assert 'overall_recommendations' in result
        assert 'best_session' in result
        assert 'sessions_analyzed' in result
        assert result['sessions_analyzed'] == 2
    
    def test_analyze_hyperparameter_impact_insufficient_data(self, comparator):
        """Test hyperparameter analysis with insufficient data."""
        # Mock monitor to return session without config
        session_no_config = TrainingSession(
            session_id="test_session",
            start_time=datetime.now(),
            config=None  # No configuration
        )
        
        comparator.monitor.load_session = Mock(return_value=session_no_config)
        
        result = comparator.analyze_hyperparameter_impact(["session1"])
        
        assert "error" in result
        assert "Need at least 2 sessions" in result["error"]
    
    def test_generate_performance_report(self, comparator, sample_sessions):
        """Test performance report generation."""
        # Mock the monitor
        def mock_load_session(session_id):
            return sample_sessions.get(session_id)
        
        comparator.monitor.load_session = mock_load_session
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "test_report.md"
            
            result_path = comparator.generate_performance_report(
                ["session1", "session2"],
                output_path
            )
            
            assert result_path == output_path
            assert output_path.exists()
            
            # Check report content
            content = output_path.read_text()
            assert "Training Performance Report" in content
            assert "session1" in content
            assert "session2" in content
            assert "Executive Summary" in content
            assert "Individual Session Analysis" in content
    
    @patch('src.pipelines.training_comparison.MATPLOTLIB_AVAILABLE', False)
    def test_create_comparison_visualization_no_matplotlib(self, comparator):
        """Test visualization creation without matplotlib."""
        comparison_result = ComparisonResult(
            comparison_id="test",
            comparison_type=ComparisonType.HYPERPARAMETER,
            sessions=["session1"],
            metrics={},
            rankings={},
            summary={},
            recommendations=[]
        )
        
        result = comparator.create_comparison_visualization(comparison_result)
        assert result is None
    
    @patch('src.pipelines.training_comparison.MATPLOTLIB_AVAILABLE', True)
    @patch('matplotlib.pyplot.subplots')
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    def test_create_comparison_visualization_with_matplotlib(self, mock_close, mock_savefig, 
                                                           mock_subplots, comparator):
        """Test visualization creation with matplotlib."""
        # Mock matplotlib components
        mock_fig = Mock()
        mock_axes = [[Mock(), Mock()], [Mock(), Mock()]]
        mock_subplots.return_value = (mock_fig, mock_axes)
        
        comparison_result = ComparisonResult(
            comparison_id="test",
            comparison_type=ComparisonType.HYPERPARAMETER,
            sessions=["session1", "session2"],
            metrics={
                "final_loss": {"session1": 0.1, "session2": 0.2},
                "quality_score": {"session1": 0.8, "session2": 0.7}
            },
            rankings={},
            summary={},
            recommendations=[]
        )
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "test_viz.png"
            
            result = comparator.create_comparison_visualization(comparison_result, output_path)
            
            assert result == output_path
            assert mock_subplots.called
            assert mock_savefig.called
            assert mock_close.called
    
    def test_get_best_configuration(self, comparator, sample_sessions):
        """Test best configuration identification."""
        # Mock the monitor
        def mock_load_session(session_id):
            return sample_sessions.get(session_id)
        
        comparator.monitor.load_session = mock_load_session
        
        result = comparator.get_best_configuration(["session1", "session2"], "overall")
        
        assert isinstance(result, dict)
        assert 'best_session_id' in result
        assert 'best_config' in result
        assert 'optimization_target' in result
        assert 'score' in result
        assert 'analysis' in result
        assert 'comparison_summary' in result
        assert result['optimization_target'] == "overall"
    
    def test_get_best_configuration_no_sessions(self, comparator):
        """Test best configuration with no valid sessions."""
        comparator.monitor.load_session = Mock(return_value=None)
        
        result = comparator.get_best_configuration(["nonexistent"])
        
        assert "error" in result
        assert "No valid sessions" in result["error"]
    
    def test_simple_correlation(self, comparator):
        """Test simple correlation calculation."""
        x_values = [1, 2, 3, 4, 5]
        y_values = [2, 4, 6, 8, 10]  # Perfect positive correlation
        
        correlation = comparator._simple_correlation(x_values, y_values)
        assert abs(correlation - 1.0) < 0.01  # Should be close to 1.0
        
        # Test negative correlation
        y_negative = [10, 8, 6, 4, 2]
        correlation_neg = comparator._simple_correlation(x_values, y_negative)
        assert abs(correlation_neg - (-1.0)) < 0.01  # Should be close to -1.0
        
        # Test no correlation
        y_random = [3, 7, 2, 9, 1]
        correlation_none = comparator._simple_correlation(x_values, y_random)
        assert abs(correlation_none) < 0.5  # Should be low correlation
    
    def test_find_optimal_parameter_range(self, comparator):
        """Test optimal parameter range finding."""
        param_values = [1, 2, 3, 4, 5]
        quality_scores = [0.5, 0.8, 0.9, 0.7, 0.6]  # Peak at param=3
        
        optimal_range = comparator._find_optimal_parameter_range(param_values, quality_scores)
        
        assert isinstance(optimal_range, tuple)
        assert len(optimal_range) == 2
        assert optimal_range[0] <= optimal_range[1]
        
        # Should include the best performing values
        top_params = [param for param, score in zip(param_values, quality_scores) 
                     if score >= 0.7]  # Top performers
        assert optimal_range[0] <= min(top_params)
        assert optimal_range[1] >= max(top_params)
    
    def test_get_parameter_recommendation(self, comparator):
        """Test parameter recommendation generation."""
        # High positive correlation
        rec_high_pos = comparator._get_parameter_recommendation("rank", 0.8, (16, 32))
        assert "Higher rank values improve quality" in rec_high_pos
        
        # High negative correlation
        rec_high_neg = comparator._get_parameter_recommendation("learning_rate", -0.7, (1e-5, 1e-4))
        assert "Lower learning_rate values improve quality" in rec_high_neg
        
        # Low correlation
        rec_low = comparator._get_parameter_recommendation("batch_size", 0.05, (1, 4))
        assert "little correlation" in rec_low
    
    def test_create_performance_profile(self, comparator, sample_sessions):
        """Test performance profile creation."""
        session = sample_sessions["session1"]
        metrics = comparator._calculate_session_metrics(session)
        
        profile = comparator._create_performance_profile(session, metrics)
        
        assert isinstance(profile, PerformanceProfile)
        assert profile.session_id == session.session_id
        assert profile.config == session.config
        assert profile.quality_metrics == metrics
        assert profile.training_time > 0  # Should calculate training time
        assert profile.peak_memory_mb > 0  # Should calculate peak memory
        assert isinstance(profile.notes, list)
    
    def test_analyze_winning_configuration(self, comparator):
        """Test winning configuration analysis."""
        config1 = LoRAConfig(rank=16, alpha=32.0, learning_rate=1e-4)
        config2 = LoRAConfig(rank=32, alpha=64.0, learning_rate=5e-5)
        
        metrics1 = QualityMetrics(overall_score=0.9, convergence_rate=0.05)
        metrics2 = QualityMetrics(overall_score=0.7, convergence_rate=0.02)
        
        best_session_data = {
            'session_id': 'session1',
            'config': config1,
            'metrics': metrics1
        }
        
        all_sessions_data = [
            best_session_data,
            {
                'session_id': 'session2',
                'config': config2,
                'metrics': metrics2
            }
        ]
        
        analysis = comparator._analyze_winning_configuration(
            best_session_data, all_sessions_data, "overall"
        )
        
        assert isinstance(analysis, dict)
        assert 'winning_factors' in analysis
        assert 'key_differences' in analysis
        assert 'configuration_strengths' in analysis
        assert isinstance(analysis['winning_factors'], list)
        assert isinstance(analysis['key_differences'], dict)
        assert isinstance(analysis['configuration_strengths'], list)
    
    def test_calculate_improvement_over_average(self, comparator):
        """Test improvement calculation over average."""
        metrics1 = QualityMetrics(overall_score=0.9)
        metrics2 = QualityMetrics(overall_score=0.7)
        metrics3 = QualityMetrics(overall_score=0.6)
        
        best_session_data = {'metrics': metrics1}
        all_sessions_data = [
            best_session_data,
            {'metrics': metrics2},
            {'metrics': metrics3}
        ]
        
        improvement = comparator._calculate_improvement_over_average(
            best_session_data, all_sessions_data, "overall_score"
        )
        
        assert isinstance(improvement, float)
        assert improvement > 0  # Should show improvement
        
        # Calculate expected improvement
        # Average of others: (0.7 + 0.6) / 2 = 0.65
        # Improvement: (0.9 - 0.65) / 0.65 * 100 ≈ 38.46%
        expected = ((0.9 - 0.65) / 0.65) * 100
        assert abs(improvement - expected) < 1.0
    
    def test_generate_hyperparameter_recommendations(self, comparator):
        """Test hyperparameter recommendation generation."""
        param_analysis = {
            'rank': {
                'correlation_with_quality': 0.8,
                'recommendation': 'Higher rank values improve quality'
            },
            'learning_rate': {
                'correlation_with_quality': -0.6,
                'recommendation': 'Lower learning_rate values improve quality'
            },
            'batch_size': {
                'correlation_with_quality': 0.1,
                'recommendation': 'batch_size shows little correlation'
            }
        }
        
        recommendations = comparator._generate_hyperparameter_recommendations(param_analysis)
        
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0
        
        # Should include high impact parameters
        rec_text = ' '.join(recommendations)
        assert 'rank' in rec_text  # High impact parameter
        assert 'learning_rate' in rec_text  # High impact parameter
    
    def test_generate_performance_insights(self, comparator):
        """Test performance insights generation."""
        config1 = LoRAConfig()
        config2 = LoRAConfig()
        
        metrics1 = QualityMetrics(overall_score=0.9)
        metrics2 = QualityMetrics(overall_score=0.5)  # Large difference
        
        profile1 = PerformanceProfile("session1", config1, metrics1, 1000, 4000)
        profile2 = PerformanceProfile("session2", config2, metrics2, 3000, 8000)  # Different times/memory
        
        session1 = Mock()
        session2 = Mock()
        
        sessions_data = [(session1, profile1), (session2, profile2)]
        
        insights = comparator._generate_performance_insights(sessions_data)
        
        assert isinstance(insights, list)
        assert len(insights) > 0
        
        # Should detect significant variations
        insights_text = ' '.join(insights)
        assert 'variation' in insights_text or 'vary' in insights_text
    
    def test_generate_performance_recommendations(self, comparator):
        """Test performance recommendations generation."""
        config1 = LoRAConfig()
        
        metrics1 = QualityMetrics(overall_score=0.9, final_loss=0.05)
        profile1 = PerformanceProfile("session1", config1, metrics1, 1000, 4000)
        
        session1 = Mock()
        session1.session_id = "session1"
        
        sessions_data = [(session1, profile1)]
        
        recommendations = comparator._generate_performance_recommendations(sessions_data)
        
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0
        
        # Should recommend using the best configuration
        rec_text = ' '.join(recommendations)
        assert 'session1' in rec_text


class TestIntegration:
    """Integration tests for training comparison system."""
    
    def test_full_comparison_workflow(self):
        """Test complete comparison workflow."""
        with tempfile.TemporaryDirectory() as temp_dir:
            monitor = TrainingMonitor(checkpoint_dir=Path(temp_dir))
            comparator = TrainingComparator(monitor=monitor)
            
            # Create sample configurations
            config1 = LoRAConfig(rank=16, alpha=32.0, learning_rate=1e-4)
            config2 = LoRAConfig(rank=32, alpha=64.0, learning_rate=5e-5)
            
            # Start and complete training sessions
            session1 = monitor.start_session("session1", config1, "dataset1", "model1")
            
            # Simulate training progress for session1
            for i in range(10):
                progress = Mock()
                progress.epoch = 0
                progress.step = i
                progress.loss = 1.0 - i * 0.1
                progress.learning_rate = 1e-4
                progress.elapsed_time = i * 10
                progress.memory_usage_mb = 4000 + i * 10
                progress.validation_loss = 0.9 - i * 0.08 if i % 3 == 0 else None
                
                monitor.update_progress(progress)
            
            monitor.stop_session(final_result={"success": True})
            
            # Start second session
            session2 = monitor.start_session("session2", config2, "dataset1", "model1")
            
            # Simulate training progress for session2
            for i in range(10):
                progress = Mock()
                progress.epoch = 0
                progress.step = i
                progress.loss = 1.2 - i * 0.08
                progress.learning_rate = 5e-5
                progress.elapsed_time = i * 15
                progress.memory_usage_mb = 6000 + i * 20
                progress.validation_loss = 1.0 - i * 0.07 if i % 3 == 0 else None
                
                monitor.update_progress(progress)
            
            monitor.stop_session(final_result={"success": True})
            
            # Compare sessions
            comparison_result = comparator.compare_sessions(
                ["session1", "session2"],
                ComparisonType.HYPERPARAMETER
            )
            
            # Verify comparison result
            assert comparison_result.comparison_type == ComparisonType.HYPERPARAMETER
            assert len(comparison_result.sessions) == 2
            assert len(comparison_result.metrics) > 0
            assert len(comparison_result.recommendations) > 0
            
            # Analyze hyperparameter impact
            hp_analysis = comparator.analyze_hyperparameter_impact(["session1", "session2"])
            assert 'parameter_analysis' in hp_analysis
            assert hp_analysis['sessions_analyzed'] == 2
            
            # Generate performance report
            report_path = comparator.generate_performance_report(["session1", "session2"])
            assert report_path.exists()
            
            # Get best configuration
            best_config = comparator.get_best_configuration(["session1", "session2"])
            assert 'best_session_id' in best_config
            assert best_config['best_session_id'] in ["session1", "session2"]