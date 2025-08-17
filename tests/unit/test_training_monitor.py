"""
Unit tests for training progress monitoring and checkpointing.

Tests comprehensive training progress monitoring, checkpointing,
and metrics tracking functionality.
"""

import pytest
import tempfile
import json
import time
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta

from src.pipelines.training_monitor import (
    TrainingMonitor, TrainingSession, TrainingMetrics, Checkpoint,
    CheckpointType, TrainingProgress
)
from src.pipelines.lora_training import LoRAConfig


class TestTrainingMetrics:
    """Test training metrics data structure."""
    
    def test_metrics_initialization(self):
        """Test training metrics initialization."""
        metrics = TrainingMetrics()
        
        assert metrics.train_losses == []
        assert metrics.validation_losses == []
        assert metrics.loss_timestamps == []
        assert metrics.learning_rates == []
        assert metrics.step_times == []
        assert metrics.memory_usage == []
        assert metrics.gradient_norms == []
        assert metrics.gpu_utilization == []
        assert metrics.gpu_temperature == []
    
    def test_metrics_with_data(self):
        """Test training metrics with data."""
        now = datetime.now()
        
        metrics = TrainingMetrics(
            train_losses=[0.5, 0.4, 0.3],
            validation_losses=[0.45, 0.35],
            loss_timestamps=[now, now, now],
            learning_rates=[1e-4, 9e-5, 8e-5],
            step_times=[1.2, 1.1, 1.0],
            memory_usage=[4000, 4100, 4200]
        )
        
        assert len(metrics.train_losses) == 3
        assert len(metrics.validation_losses) == 2
        assert len(metrics.loss_timestamps) == 3
        assert metrics.train_losses[0] == 0.5
        assert metrics.validation_losses[0] == 0.45


class TestTrainingSession:
    """Test training session data structure."""
    
    def test_session_initialization(self):
        """Test training session initialization."""
        session_id = "test_session_123"
        start_time = datetime.now()
        
        session = TrainingSession(
            session_id=session_id,
            start_time=start_time
        )
        
        assert session.session_id == session_id
        assert session.start_time == start_time
        assert session.end_time is None
        assert session.config is None
        assert session.dataset_name == ""
        assert session.base_model == ""
        assert session.status == "running"
        assert session.checkpoints == []
        assert isinstance(session.metrics, TrainingMetrics)
        assert session.final_result is None
        assert session.error_message is None
    
    def test_session_with_config(self):
        """Test training session with configuration."""
        config = LoRAConfig(rank=16, alpha=32.0)
        
        session = TrainingSession(
            session_id="test_session",
            start_time=datetime.now(),
            config=config,
            dataset_name="test_dataset",
            base_model="stable-diffusion-v1-5"
        )
        
        assert session.config == config
        assert session.dataset_name == "test_dataset"
        assert session.base_model == "stable-diffusion-v1-5"


class TestCheckpoint:
    """Test checkpoint data structure."""
    
    def test_checkpoint_creation(self):
        """Test checkpoint creation."""
        checkpoint_id = "test_checkpoint_123"
        timestamp = datetime.now()
        model_path = Path("/path/to/model.pt")
        config_path = Path("/path/to/config.json")
        
        checkpoint = Checkpoint(
            checkpoint_id=checkpoint_id,
            checkpoint_type=CheckpointType.REGULAR,
            timestamp=timestamp,
            epoch=5,
            step=100,
            loss=0.3,
            validation_loss=0.25,
            learning_rate=1e-4,
            model_path=model_path,
            config_path=config_path
        )
        
        assert checkpoint.checkpoint_id == checkpoint_id
        assert checkpoint.checkpoint_type == CheckpointType.REGULAR
        assert checkpoint.timestamp == timestamp
        assert checkpoint.epoch == 5
        assert checkpoint.step == 100
        assert checkpoint.loss == 0.3
        assert checkpoint.validation_loss == 0.25
        assert checkpoint.learning_rate == 1e-4
        assert checkpoint.model_path == model_path
        assert checkpoint.config_path == config_path
        assert checkpoint.metadata == {}


class TestTrainingMonitor:
    """Test training monitor functionality."""
    
    @pytest.fixture
    def temp_checkpoint_dir(self):
        """Create temporary checkpoint directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)
    
    @pytest.fixture
    def monitor(self, temp_checkpoint_dir):
        """Create training monitor with temporary directory."""
        return TrainingMonitor(checkpoint_dir=temp_checkpoint_dir)
    
    @pytest.fixture
    def sample_config(self):
        """Create sample LoRA configuration."""
        return LoRAConfig(
            rank=16,
            alpha=32.0,
            learning_rate=1e-4,
            batch_size=2,
            num_epochs=10
        )
    
    @pytest.fixture
    def sample_progress(self):
        """Create sample training progress."""
        return TrainingProgress(
            epoch=2,
            step=50,
            total_steps=200,
            loss=0.4,
            learning_rate=9e-5,
            elapsed_time=1800.0,
            estimated_time_remaining=5400.0,
            memory_usage_mb=6000.0,
            validation_loss=0.35
        )
    
    def test_monitor_initialization(self, temp_checkpoint_dir):
        """Test monitor initialization."""
        monitor = TrainingMonitor(checkpoint_dir=temp_checkpoint_dir)
        
        assert monitor.checkpoint_dir == temp_checkpoint_dir
        assert temp_checkpoint_dir.exists()
        assert monitor.current_session is None
        assert not monitor.monitoring_active
        assert monitor.monitor_thread is None
        assert monitor.progress_callbacks == []
        assert monitor.checkpoint_callbacks == []
        assert monitor.auto_checkpoint_steps == 500
        assert monitor.keep_best_checkpoints == 3
        assert monitor.keep_recent_checkpoints == 5
    
    def test_start_session(self, monitor, sample_config):
        """Test starting a training session."""
        session_id = "test_session_123"
        dataset_name = "test_dataset"
        base_model = "stable-diffusion-v1-5"
        
        session = monitor.start_session(session_id, sample_config, dataset_name, base_model)
        
        assert isinstance(session, TrainingSession)
        assert session.session_id == session_id
        assert session.config == sample_config
        assert session.dataset_name == dataset_name
        assert session.base_model == base_model
        assert session.status == "running"
        assert monitor.current_session == session
        assert monitor.monitoring_active is True
        assert monitor.monitor_thread is not None
        
        # Check session directory created
        session_dir = monitor.checkpoint_dir / session_id
        assert session_dir.exists()
        
        # Clean up
        monitor.stop_session()
    
    def test_start_session_stops_previous(self, monitor, sample_config):
        """Test starting session stops previous session."""
        # Start first session
        session1 = monitor.start_session("session1", sample_config, "dataset1", "model1")
        assert monitor.current_session.session_id == "session1"
        
        # Start second session
        session2 = monitor.start_session("session2", sample_config, "dataset2", "model2")
        assert monitor.current_session.session_id == "session2"
        
        # Clean up
        monitor.stop_session()
    
    def test_update_progress(self, monitor, sample_config, sample_progress):
        """Test updating training progress."""
        # Start session
        monitor.start_session("test_session", sample_config, "test_dataset", "test_model")
        
        # Update progress
        monitor.update_progress(sample_progress)
        
        # Check metrics updated
        metrics = monitor.current_session.metrics
        assert len(metrics.train_losses) == 1
        assert metrics.train_losses[0] == sample_progress.loss
        assert len(metrics.learning_rates) == 1
        assert metrics.learning_rates[0] == sample_progress.learning_rate
        assert len(metrics.validation_losses) == 1
        assert metrics.validation_losses[0] == sample_progress.validation_loss
        assert len(metrics.memory_usage) == 1
        assert metrics.memory_usage[0] == sample_progress.memory_usage_mb
        
        # Clean up
        monitor.stop_session()
    
    def test_update_progress_no_session(self, monitor, sample_progress):
        """Test updating progress with no active session."""
        # Should not raise error
        monitor.update_progress(sample_progress)
        assert monitor.current_session is None
    
    def test_create_checkpoint(self, monitor, sample_config, sample_progress):
        """Test creating a checkpoint."""
        # Start session
        monitor.start_session("test_session", sample_config, "test_dataset", "test_model")
        
        # Create checkpoint
        checkpoint = monitor.create_checkpoint(CheckpointType.REGULAR, sample_progress)
        
        assert isinstance(checkpoint, Checkpoint)
        assert checkpoint.checkpoint_type == CheckpointType.REGULAR
        assert checkpoint.epoch == sample_progress.epoch
        assert checkpoint.step == sample_progress.step
        assert checkpoint.loss == sample_progress.loss
        assert checkpoint.validation_loss == sample_progress.validation_loss
        assert checkpoint.learning_rate == sample_progress.learning_rate
        
        # Check files created
        assert checkpoint.model_path.exists()
        assert checkpoint.config_path.exists()
        
        # Check added to session
        assert len(monitor.current_session.checkpoints) == 1
        assert monitor.current_session.checkpoints[0] == checkpoint
        
        # Clean up
        monitor.stop_session()
    
    def test_create_checkpoint_no_session(self, monitor, sample_progress):
        """Test creating checkpoint with no active session."""
        with pytest.raises(RuntimeError, match="No active training session"):
            monitor.create_checkpoint(CheckpointType.REGULAR, sample_progress)
    
    def test_stop_session(self, monitor, sample_config):
        """Test stopping a training session."""
        # Start session
        session = monitor.start_session("test_session", sample_config, "test_dataset", "test_model")
        assert monitor.monitoring_active is True
        
        # Stop session
        final_result = {"final_loss": 0.1, "success": True}
        monitor.stop_session(final_result=final_result)
        
        assert monitor.monitoring_active is False
        assert monitor.current_session is None
        assert session.end_time is not None
        assert session.final_result == final_result
        assert session.status == "completed"
    
    def test_stop_session_with_error(self, monitor, sample_config):
        """Test stopping session with error."""
        # Start session
        session = monitor.start_session("test_session", sample_config, "test_dataset", "test_model")
        
        # Stop with error
        error_message = "Training failed due to memory error"
        monitor.stop_session(error_message=error_message)
        
        assert session.error_message == error_message
        assert session.status == "failed"
    
    def test_auto_checkpointing(self, monitor, sample_config):
        """Test automatic checkpointing."""
        # Start session
        monitor.start_session("test_session", sample_config, "test_dataset", "test_model")
        monitor.auto_checkpoint_steps = 10  # Checkpoint every 10 steps
        
        # Update progress to trigger checkpoint
        progress = TrainingProgress(step=10, loss=0.5)
        monitor.update_progress(progress)
        
        # Should create checkpoint
        assert len(monitor.current_session.checkpoints) == 1
        assert monitor.current_session.checkpoints[0].checkpoint_type == CheckpointType.REGULAR
        
        # Clean up
        monitor.stop_session()
    
    def test_best_loss_checkpointing(self, monitor, sample_config):
        """Test best validation loss checkpointing."""
        # Start session
        monitor.start_session("test_session", sample_config, "test_dataset", "test_model")
        
        # Update with validation loss
        progress1 = TrainingProgress(step=1, loss=0.5, validation_loss=0.4)
        monitor.update_progress(progress1)
        
        # Update with better validation loss
        progress2 = TrainingProgress(step=2, loss=0.4, validation_loss=0.3)
        monitor.update_progress(progress2)
        
        # Should create best loss checkpoint
        best_checkpoints = [cp for cp in monitor.current_session.checkpoints 
                          if cp.checkpoint_type == CheckpointType.BEST_LOSS]
        assert len(best_checkpoints) >= 1
        
        # Clean up
        monitor.stop_session()
    
    def test_progress_callbacks(self, monitor, sample_config, sample_progress):
        """Test progress callbacks."""
        callback_called = False
        received_progress = None
        
        def progress_callback(progress):
            nonlocal callback_called, received_progress
            callback_called = True
            received_progress = progress
        
        # Add callback
        monitor.add_progress_callback(progress_callback)
        
        # Start session and update progress
        monitor.start_session("test_session", sample_config, "test_dataset", "test_model")
        monitor.update_progress(sample_progress)
        
        # Check callback called
        assert callback_called
        assert received_progress == sample_progress
        
        # Clean up
        monitor.stop_session()
    
    def test_checkpoint_callbacks(self, monitor, sample_config, sample_progress):
        """Test checkpoint callbacks."""
        callback_called = False
        received_checkpoint = None
        
        def checkpoint_callback(checkpoint):
            nonlocal callback_called, received_checkpoint
            callback_called = True
            received_checkpoint = checkpoint
        
        # Add callback
        monitor.add_checkpoint_callback(checkpoint_callback)
        
        # Start session and create checkpoint
        monitor.start_session("test_session", sample_config, "test_dataset", "test_model")
        checkpoint = monitor.create_checkpoint(CheckpointType.MANUAL, sample_progress)
        
        # Check callback called
        assert callback_called
        assert received_checkpoint == checkpoint
        
        # Clean up
        monitor.stop_session()
    
    def test_save_and_load_session(self, monitor, sample_config):
        """Test saving and loading session information."""
        session_id = "test_session_save_load"
        
        # Start session and add some data
        session = monitor.start_session(session_id, sample_config, "test_dataset", "test_model")
        
        # Add some progress
        progress = TrainingProgress(step=10, loss=0.5, validation_loss=0.4)
        monitor.update_progress(progress)
        
        # Create checkpoint
        checkpoint = monitor.create_checkpoint(CheckpointType.REGULAR, progress)
        
        # Stop session
        monitor.stop_session(final_result={"success": True})
        
        # Load session
        loaded_session = monitor.load_session(session_id)
        
        assert loaded_session is not None
        assert loaded_session.session_id == session_id
        assert loaded_session.dataset_name == "test_dataset"
        assert loaded_session.base_model == "test_model"
        assert loaded_session.status == "completed"
        assert loaded_session.final_result == {"success": True}
        
        # Check metrics loaded
        assert len(loaded_session.metrics.train_losses) == 1
        assert loaded_session.metrics.train_losses[0] == 0.5
        
        # Check checkpoints loaded
        assert len(loaded_session.checkpoints) == 1
        assert loaded_session.checkpoints[0].step == 10
    
    def test_load_nonexistent_session(self, monitor):
        """Test loading nonexistent session."""
        loaded_session = monitor.load_session("nonexistent_session")
        assert loaded_session is None
    
    def test_list_sessions(self, monitor, sample_config):
        """Test listing training sessions."""
        # Initially no sessions
        sessions = monitor.list_sessions()
        assert sessions == []
        
        # Create sessions
        monitor.start_session("session1", sample_config, "dataset1", "model1")
        monitor.stop_session()
        
        monitor.start_session("session2", sample_config, "dataset2", "model2")
        monitor.stop_session()
        
        # List sessions
        sessions = monitor.list_sessions()
        assert len(sessions) == 2
        assert "session1" in sessions
        assert "session2" in sessions
    
    def test_get_session_summary(self, monitor, sample_config):
        """Test getting session summary."""
        session_id = "test_session_summary"
        
        # Create session with data
        monitor.start_session(session_id, sample_config, "test_dataset", "test_model")
        
        # Add progress
        progress = TrainingProgress(step=10, loss=0.3, validation_loss=0.25)
        monitor.update_progress(progress)
        
        # Stop session
        monitor.stop_session(final_result={"success": True})
        
        # Get summary
        summary = monitor.get_session_summary(session_id)
        
        assert summary is not None
        assert summary['session_id'] == session_id
        assert summary['dataset_name'] == "test_dataset"
        assert summary['base_model'] == "test_model"
        assert summary['status'] == "completed"
        assert summary['final_loss'] == 0.3
        assert summary['best_validation_loss'] == 0.25
        assert summary['duration_seconds'] is not None
        assert summary['error_message'] is None
    
    def test_get_session_summary_nonexistent(self, monitor):
        """Test getting summary for nonexistent session."""
        summary = monitor.get_session_summary("nonexistent")
        assert summary is None
    
    @patch('src.pipelines.training_monitor.MATPLOTLIB_AVAILABLE', False)
    def test_create_progress_plot_no_matplotlib(self, monitor, sample_config):
        """Test creating progress plot without matplotlib."""
        monitor.start_session("test_session", sample_config, "test_dataset", "test_model")
        monitor.stop_session()
        
        plot_path = monitor.create_progress_plot("test_session")
        assert plot_path is None
    
    @patch('src.pipelines.training_monitor.MATPLOTLIB_AVAILABLE', True)
    @patch('matplotlib.pyplot.subplots')
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    def test_create_progress_plot_with_matplotlib(self, mock_close, mock_savefig, mock_subplots, 
                                                 monitor, sample_config):
        """Test creating progress plot with matplotlib."""
        # Mock matplotlib components
        mock_fig = Mock()
        mock_axes = [[Mock(), Mock()], [Mock(), Mock()]]
        mock_subplots.return_value = (mock_fig, mock_axes)
        
        # Create session with data
        monitor.start_session("test_session", sample_config, "test_dataset", "test_model")
        
        # Add some progress data
        for i in range(5):
            progress = TrainingProgress(step=i, loss=0.5 - i*0.1, validation_loss=0.4 - i*0.08)
            monitor.update_progress(progress)
        
        monitor.stop_session()
        
        # Create plot
        plot_path = monitor.create_progress_plot("test_session")
        
        assert plot_path is not None
        assert mock_subplots.called
        assert mock_savefig.called
        assert mock_close.called
    
    def test_checkpoint_cleanup(self, monitor, sample_config):
        """Test checkpoint cleanup."""
        monitor.keep_recent_checkpoints = 2  # Keep only 2 recent checkpoints
        
        # Start session
        monitor.start_session("test_session", sample_config, "test_dataset", "test_model")
        
        # Create multiple regular checkpoints
        for i in range(5):
            progress = TrainingProgress(step=i*10, loss=0.5)
            monitor.create_checkpoint(CheckpointType.REGULAR, progress)
        
        # Should only keep recent checkpoints
        regular_checkpoints = [cp for cp in monitor.current_session.checkpoints 
                             if cp.checkpoint_type == CheckpointType.REGULAR]
        assert len(regular_checkpoints) <= monitor.keep_recent_checkpoints
        
        # Clean up
        monitor.stop_session()
    
    def test_monitoring_loop_error_handling(self, monitor, sample_config):
        """Test monitoring loop error handling."""
        # Start session
        monitor.start_session("test_session", sample_config, "test_dataset", "test_model")
        
        # Mock system metrics collection to raise error
        with patch.object(monitor, '_collect_system_metrics', side_effect=Exception("Test error")):
            # Should not crash the monitoring loop
            time.sleep(0.1)  # Let monitoring loop run briefly
        
        # Should still be monitoring
        assert monitor.monitoring_active
        
        # Clean up
        monitor.stop_session()
    
    def test_training_anomaly_detection(self, monitor, sample_config):
        """Test training anomaly detection."""
        # Start session
        monitor.start_session("test_session", sample_config, "test_dataset", "test_model")
        
        # Add progress with exploding loss
        for i in range(10):
            loss = 0.1 if i < 5 else 20.0  # Loss explosion after step 5
            progress = TrainingProgress(step=i, loss=loss)
            monitor.update_progress(progress)
        
        # Check anomaly detection (would log warnings)
        monitor._check_training_anomalies()
        
        # Clean up
        monitor.stop_session()


class TestIntegration:
    """Integration tests for training monitor."""
    
    def test_full_monitoring_session(self):
        """Test complete monitoring session."""
        with tempfile.TemporaryDirectory() as temp_dir:
            monitor = TrainingMonitor(checkpoint_dir=Path(temp_dir))
            config = LoRAConfig(rank=16, num_epochs=5)
            
            # Start session
            session = monitor.start_session("integration_test", config, "test_dataset", "test_model")
            
            # Simulate training progress
            for epoch in range(2):
                for step in range(10):
                    global_step = epoch * 10 + step
                    loss = 1.0 - global_step * 0.05  # Decreasing loss
                    val_loss = loss * 0.9 if step % 5 == 0 else None
                    
                    progress = TrainingProgress(
                        epoch=epoch,
                        step=global_step,
                        total_steps=20,
                        loss=max(0.1, loss),
                        learning_rate=1e-4 * (0.95 ** global_step),
                        elapsed_time=global_step * 2.0,
                        memory_usage_mb=4000 + global_step * 10,
                        validation_loss=val_loss
                    )
                    
                    monitor.update_progress(progress)
                    
                    # Create manual checkpoint at epoch end
                    if step == 9:
                        monitor.create_checkpoint(CheckpointType.EPOCH_END, progress)
            
            # Stop session
            final_result = {"success": True, "final_loss": 0.1}
            monitor.stop_session(final_result=final_result)
            
            # Verify session data
            assert session.status == "completed"
            assert session.final_result == final_result
            assert len(session.metrics.train_losses) == 20
            assert len(session.checkpoints) > 0
            
            # Test session persistence
            loaded_session = monitor.load_session("integration_test")
            assert loaded_session is not None
            assert loaded_session.session_id == "integration_test"
            assert len(loaded_session.metrics.train_losses) == 20
            
            # Test session summary
            summary = monitor.get_session_summary("integration_test")
            assert summary['status'] == "completed"
            assert summary['final_loss'] is not None
            assert summary['duration_seconds'] is not None