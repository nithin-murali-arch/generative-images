"""
Training Progress Monitoring and Checkpointing System

This module provides comprehensive training progress monitoring, checkpointing,
and real-time metrics tracking for LoRA training sessions.
"""

import logging
import time
import json
import threading
from typing import Dict, Any, Optional, List, Callable
from pathlib import Path
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum

from .lora_training import TrainingProgress, LoRAConfig

logger = logging.getLogger(__name__)

# Try to import optional dependencies
try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.animation import FuncAnimation
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logger.warning("Matplotlib not available - plotting disabled")

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    logger.warning("Weights & Biases not available - advanced tracking disabled")


class CheckpointType(Enum):
    """Types of training checkpoints."""
    REGULAR = "regular"          # Regular interval checkpoint
    BEST_LOSS = "best_loss"      # Best validation loss checkpoint
    EPOCH_END = "epoch_end"      # End of epoch checkpoint
    MANUAL = "manual"            # Manual checkpoint
    EMERGENCY = "emergency"      # Emergency checkpoint (on error/stop)


@dataclass
class Checkpoint:
    """Training checkpoint information."""
    checkpoint_id: str
    checkpoint_type: CheckpointType
    timestamp: datetime
    epoch: int
    step: int
    loss: float
    validation_loss: Optional[float]
    learning_rate: float
    model_path: Path
    config_path: Path
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainingMetrics:
    """Comprehensive training metrics."""
    # Loss metrics
    train_losses: List[float] = field(default_factory=list)
    validation_losses: List[float] = field(default_factory=list)
    loss_timestamps: List[datetime] = field(default_factory=list)
    
    # Learning rate tracking
    learning_rates: List[float] = field(default_factory=list)
    
    # Performance metrics
    step_times: List[float] = field(default_factory=list)
    memory_usage: List[float] = field(default_factory=list)
    
    # Quality metrics
    gradient_norms: List[float] = field(default_factory=list)
    
    # Hardware metrics
    gpu_utilization: List[float] = field(default_factory=list)
    gpu_temperature: List[float] = field(default_factory=list)


@dataclass
class TrainingSession:
    """Complete training session information."""
    session_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    config: Optional[LoRAConfig] = None
    dataset_name: str = ""
    base_model: str = ""
    status: str = "running"  # running, completed, failed, stopped
    checkpoints: List[Checkpoint] = field(default_factory=list)
    metrics: TrainingMetrics = field(default_factory=TrainingMetrics)
    final_result: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None


class TrainingMonitor:
    """
    Comprehensive training progress monitoring and checkpointing system.
    
    Features:
    - Real-time progress tracking
    - Automatic checkpointing
    - Metrics visualization
    - Performance analysis
    - Recovery from interruptions
    """
    
    def __init__(self, checkpoint_dir: Path = Path("checkpoints")):
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Current session
        self.current_session: Optional[TrainingSession] = None
        self.monitoring_active = False
        self.monitor_thread: Optional[threading.Thread] = None
        
        # Callbacks
        self.progress_callbacks: List[Callable] = []
        self.checkpoint_callbacks: List[Callable] = []
        
        # Checkpointing settings
        self.auto_checkpoint_steps = 500
        self.keep_best_checkpoints = 3
        self.keep_recent_checkpoints = 5
        
        # Performance tracking
        self.last_step_time = time.time()
        self.best_validation_loss = float('inf')
        
        logger.info("TrainingMonitor initialized")
    
    def start_session(self, session_id: str, config: LoRAConfig,
                     dataset_name: str, base_model: str) -> TrainingSession:
        """
        Start a new training session.
        
        Args:
            session_id: Unique identifier for the session
            config: LoRA training configuration
            dataset_name: Name of the dataset being used
            base_model: Base model being trained
            
        Returns:
            TrainingSession object
        """
        if self.current_session and self.current_session.status == "running":
            logger.warning("Stopping previous session before starting new one")
            self.stop_session()
        
        self.current_session = TrainingSession(
            session_id=session_id,
            start_time=datetime.now(),
            config=config,
            dataset_name=dataset_name,
            base_model=base_model
        )
        
        # Create session directory
        session_dir = self.checkpoint_dir / session_id
        session_dir.mkdir(parents=True, exist_ok=True)
        
        # Save initial session info
        self._save_session_info()
        
        # Start monitoring
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        
        logger.info(f"Training session '{session_id}' started")
        return self.current_session
    
    def update_progress(self, progress: TrainingProgress) -> None:
        """
        Update training progress.
        
        Args:
            progress: Current training progress
        """
        if not self.current_session:
            return
        
        current_time = datetime.now()
        
        # Update metrics
        self.current_session.metrics.train_losses.append(progress.loss)
        self.current_session.metrics.loss_timestamps.append(current_time)
        self.current_session.metrics.learning_rates.append(progress.learning_rate)
        
        if progress.validation_loss is not None:
            self.current_session.metrics.validation_losses.append(progress.validation_loss)
        
        # Calculate step time
        step_time = time.time() - self.last_step_time
        self.current_session.metrics.step_times.append(step_time)
        self.last_step_time = time.time()
        
        # Add memory usage
        self.current_session.metrics.memory_usage.append(progress.memory_usage_mb)
        
        # Check for automatic checkpointing
        if progress.step % self.auto_checkpoint_steps == 0:
            self.create_checkpoint(CheckpointType.REGULAR, progress)
        
        # Check for best validation loss
        if (progress.validation_loss is not None and 
            progress.validation_loss < self.best_validation_loss):
            self.best_validation_loss = progress.validation_loss
            self.create_checkpoint(CheckpointType.BEST_LOSS, progress)
        
        # Call progress callbacks
        for callback in self.progress_callbacks:
            try:
                callback(progress)
            except Exception as e:
                logger.warning(f"Progress callback failed: {e}")
        
        # Save session info periodically
        if progress.step % 100 == 0:
            self._save_session_info()
    
    def create_checkpoint(self, checkpoint_type: CheckpointType,
                         progress: TrainingProgress,
                         model_state: Optional[Dict] = None) -> Checkpoint:
        """
        Create a training checkpoint.
        
        Args:
            checkpoint_type: Type of checkpoint
            progress: Current training progress
            model_state: Optional model state to save
            
        Returns:
            Checkpoint object
        """
        if not self.current_session:
            raise RuntimeError("No active training session")
        
        checkpoint_id = f"{self.current_session.session_id}_step_{progress.step}_{checkpoint_type.value}"
        timestamp = datetime.now()
        
        # Create checkpoint directory
        checkpoint_dir = self.checkpoint_dir / self.current_session.session_id / checkpoint_id
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model state
        model_path = checkpoint_dir / "model_state.pt"
        config_path = checkpoint_dir / "config.json"
        
        if model_state:
            try:
                import torch
                torch.save(model_state, model_path)
            except Exception as e:
                logger.warning(f"Failed to save model state: {e}")
                # Create empty file for testing
                model_path.touch()
        else:
            # Create empty file for testing
            model_path.touch()
        
        # Save configuration
        with open(config_path, 'w') as f:
            json.dump({
                'lora_config': asdict(self.current_session.config),
                'progress': asdict(progress),
                'checkpoint_info': {
                    'type': checkpoint_type.value,
                    'timestamp': timestamp.isoformat(),
                    'session_id': self.current_session.session_id
                }
            }, f, indent=2, default=str)
        
        # Create checkpoint object
        checkpoint = Checkpoint(
            checkpoint_id=checkpoint_id,
            checkpoint_type=checkpoint_type,
            timestamp=timestamp,
            epoch=progress.epoch,
            step=progress.step,
            loss=progress.loss,
            validation_loss=progress.validation_loss,
            learning_rate=progress.learning_rate,
            model_path=model_path,
            config_path=config_path,
            metadata={
                'memory_usage_mb': progress.memory_usage_mb,
                'elapsed_time': progress.elapsed_time,
                'estimated_time_remaining': progress.estimated_time_remaining
            }
        )
        
        # Add to session
        self.current_session.checkpoints.append(checkpoint)
        
        # Clean up old checkpoints
        self._cleanup_checkpoints()
        
        # Call checkpoint callbacks
        for callback in self.checkpoint_callbacks:
            try:
                callback(checkpoint)
            except Exception as e:
                logger.warning(f"Checkpoint callback failed: {e}")
        
        logger.info(f"Checkpoint created: {checkpoint_id}")
        return checkpoint
    
    def stop_session(self, final_result: Optional[Dict[str, Any]] = None,
                    error_message: Optional[str] = None) -> None:
        """
        Stop the current training session.
        
        Args:
            final_result: Final training result
            error_message: Error message if session failed
        """
        if not self.current_session:
            return
        
        # Stop monitoring
        self.monitoring_active = False
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5.0)
        
        # Update session
        self.current_session.end_time = datetime.now()
        self.current_session.final_result = final_result
        self.current_session.error_message = error_message
        
        if error_message:
            self.current_session.status = "failed"
        elif final_result:
            self.current_session.status = "completed"
        else:
            self.current_session.status = "stopped"
        
        # Save final session info
        self._save_session_info()
        
        logger.info(f"Training session stopped: {self.current_session.status}")
        self.current_session = None
    
    def load_session(self, session_id: str) -> Optional[TrainingSession]:
        """
        Load a previous training session.
        
        Args:
            session_id: Session ID to load
            
        Returns:
            TrainingSession if found, None otherwise
        """
        session_file = self.checkpoint_dir / session_id / "session_info.json"
        
        if not session_file.exists():
            return None
        
        try:
            with open(session_file, 'r') as f:
                session_data = json.load(f)
            
            # Reconstruct session object
            session = TrainingSession(
                session_id=session_data['session_id'],
                start_time=datetime.fromisoformat(session_data['start_time']),
                end_time=datetime.fromisoformat(session_data['end_time']) if session_data.get('end_time') else None,
                dataset_name=session_data.get('dataset_name', ''),
                base_model=session_data.get('base_model', ''),
                status=session_data.get('status', 'unknown'),
                final_result=session_data.get('final_result'),
                error_message=session_data.get('error_message')
            )
            
            # Load config if available
            if 'config' in session_data:
                session.config = LoRAConfig(**session_data['config'])
            
            # Load checkpoints
            session.checkpoints = self._load_checkpoints(session_id)
            
            # Load metrics
            session.metrics = self._load_metrics(session_id)
            
            return session
            
        except Exception as e:
            logger.error(f"Failed to load session {session_id}: {e}")
            return None
    
    def list_sessions(self) -> List[str]:
        """List all available training sessions."""
        sessions = []
        
        if not self.checkpoint_dir.exists():
            return sessions
        
        for item in self.checkpoint_dir.iterdir():
            if item.is_dir() and (item / "session_info.json").exists():
                sessions.append(item.name)
        
        return sorted(sessions, reverse=True)  # Most recent first
    
    def get_session_summary(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get summary information for a session."""
        session = self.load_session(session_id)
        
        if not session:
            return None
        
        duration = None
        if session.end_time:
            duration = (session.end_time - session.start_time).total_seconds()
        
        return {
            'session_id': session.session_id,
            'start_time': session.start_time.isoformat(),
            'end_time': session.end_time.isoformat() if session.end_time else None,
            'duration_seconds': duration,
            'status': session.status,
            'dataset_name': session.dataset_name,
            'base_model': session.base_model,
            'num_checkpoints': len(session.checkpoints),
            'final_loss': session.metrics.train_losses[-1] if session.metrics.train_losses else None,
            'best_validation_loss': min(session.metrics.validation_losses) if session.metrics.validation_losses else None,
            'error_message': session.error_message
        }
    
    def create_progress_plot(self, session_id: str, output_path: Optional[Path] = None) -> Optional[Path]:
        """
        Create training progress plot.
        
        Args:
            session_id: Session to plot
            output_path: Optional output path for plot
            
        Returns:
            Path to saved plot if successful
        """
        if not MATPLOTLIB_AVAILABLE:
            logger.warning("Matplotlib not available - cannot create plots")
            return None
        
        session = self.load_session(session_id)
        if not session:
            return None
        
        # Create plot
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Training Progress - {session_id}', fontsize=16)
        
        # Training loss
        if session.metrics.train_losses:
            ax1.plot(session.metrics.train_losses, label='Training Loss', color='blue')
            if session.metrics.validation_losses:
                # Align validation losses with training steps
                val_steps = [i * len(session.metrics.train_losses) // len(session.metrics.validation_losses) 
                           for i in range(len(session.metrics.validation_losses))]
                ax1.plot(val_steps, session.metrics.validation_losses, 
                        label='Validation Loss', color='red', linestyle='--')
            ax1.set_xlabel('Step')
            ax1.set_ylabel('Loss')
            ax1.set_title('Training Loss')
            ax1.legend()
            ax1.grid(True)
        
        # Learning rate
        if session.metrics.learning_rates:
            ax2.plot(session.metrics.learning_rates, color='green')
            ax2.set_xlabel('Step')
            ax2.set_ylabel('Learning Rate')
            ax2.set_title('Learning Rate Schedule')
            ax2.grid(True)
        
        # Memory usage
        if session.metrics.memory_usage:
            ax3.plot(session.metrics.memory_usage, color='orange')
            ax3.set_xlabel('Step')
            ax3.set_ylabel('Memory Usage (MB)')
            ax3.set_title('GPU Memory Usage')
            ax3.grid(True)
        
        # Step times
        if session.metrics.step_times:
            ax4.plot(session.metrics.step_times, color='purple')
            ax4.set_xlabel('Step')
            ax4.set_ylabel('Time (seconds)')
            ax4.set_title('Step Duration')
            ax4.grid(True)
        
        plt.tight_layout()
        
        # Save plot
        if output_path is None:
            output_path = self.checkpoint_dir / session_id / "training_progress.png"
        
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Training progress plot saved to: {output_path}")
        return output_path
    
    def add_progress_callback(self, callback: Callable) -> None:
        """Add a progress update callback."""
        self.progress_callbacks.append(callback)
    
    def add_checkpoint_callback(self, callback: Callable) -> None:
        """Add a checkpoint creation callback."""
        self.checkpoint_callbacks.append(callback)
    
    def _monitoring_loop(self) -> None:
        """Background monitoring loop."""
        while self.monitoring_active:
            try:
                # Monitor system resources
                self._collect_system_metrics()
                
                # Check for anomalies
                self._check_training_anomalies()
                
                time.sleep(10)  # Monitor every 10 seconds
                
            except Exception as e:
                logger.warning(f"Monitoring loop error: {e}")
    
    def _collect_system_metrics(self) -> None:
        """Collect system performance metrics."""
        if not self.current_session:
            return
        
        try:
            # Try to get GPU metrics
            import torch
            if torch.cuda.is_available():
                gpu_util = torch.cuda.utilization()
                self.current_session.metrics.gpu_utilization.append(gpu_util)
                
                # GPU temperature would require additional libraries like nvidia-ml-py
                # For now, we'll skip this
                
        except Exception as e:
            logger.debug(f"Failed to collect GPU metrics: {e}")
    
    def _check_training_anomalies(self) -> None:
        """Check for training anomalies and issues."""
        if not self.current_session or not self.current_session.metrics.train_losses:
            return
        
        recent_losses = self.current_session.metrics.train_losses[-10:]
        
        # Check for loss explosion
        if len(recent_losses) >= 5:
            recent_avg = sum(recent_losses[-5:]) / 5
            if recent_avg > 10.0:  # Arbitrary threshold
                logger.warning("Potential loss explosion detected")
        
        # Check for loss stagnation
        if len(recent_losses) >= 10:
            loss_variance = sum((x - sum(recent_losses)/len(recent_losses))**2 for x in recent_losses) / len(recent_losses)
            if loss_variance < 1e-6:  # Very low variance
                logger.warning("Training loss appears to have stagnated")
    
    def _save_session_info(self) -> None:
        """Save current session information to disk."""
        if not self.current_session:
            return
        
        session_dir = self.checkpoint_dir / self.current_session.session_id
        session_file = session_dir / "session_info.json"
        
        session_data = {
            'session_id': self.current_session.session_id,
            'start_time': self.current_session.start_time.isoformat(),
            'end_time': self.current_session.end_time.isoformat() if self.current_session.end_time else None,
            'status': self.current_session.status,
            'dataset_name': self.current_session.dataset_name,
            'base_model': self.current_session.base_model,
            'final_result': self.current_session.final_result,
            'error_message': self.current_session.error_message
        }
        
        if self.current_session.config:
            session_data['config'] = asdict(self.current_session.config)
        
        with open(session_file, 'w') as f:
            json.dump(session_data, f, indent=2, default=str)
        
        # Save metrics separately
        self._save_metrics()
    
    def _save_metrics(self) -> None:
        """Save training metrics to disk."""
        if not self.current_session:
            return
        
        session_dir = self.checkpoint_dir / self.current_session.session_id
        metrics_file = session_dir / "metrics.json"
        
        metrics_data = asdict(self.current_session.metrics)
        
        with open(metrics_file, 'w') as f:
            json.dump(metrics_data, f, indent=2, default=str)
    
    def _load_checkpoints(self, session_id: str) -> List[Checkpoint]:
        """Load checkpoints for a session."""
        checkpoints = []
        session_dir = self.checkpoint_dir / session_id
        
        if not session_dir.exists():
            return checkpoints
        
        for checkpoint_dir in session_dir.iterdir():
            if checkpoint_dir.is_dir() and (checkpoint_dir / "config.json").exists():
                try:
                    with open(checkpoint_dir / "config.json", 'r') as f:
                        checkpoint_data = json.load(f)
                    
                    checkpoint_info = checkpoint_data.get('checkpoint_info', {})
                    progress_data = checkpoint_data.get('progress', {})
                    
                    checkpoint = Checkpoint(
                        checkpoint_id=checkpoint_dir.name,
                        checkpoint_type=CheckpointType(checkpoint_info.get('type', 'regular')),
                        timestamp=datetime.fromisoformat(checkpoint_info.get('timestamp', datetime.now().isoformat())),
                        epoch=progress_data.get('epoch', 0),
                        step=progress_data.get('step', 0),
                        loss=progress_data.get('loss', 0.0),
                        validation_loss=progress_data.get('validation_loss'),
                        learning_rate=progress_data.get('learning_rate', 0.0),
                        model_path=checkpoint_dir / "model_state.pt",
                        config_path=checkpoint_dir / "config.json",
                        metadata=checkpoint_info
                    )
                    
                    checkpoints.append(checkpoint)
                    
                except Exception as e:
                    logger.warning(f"Failed to load checkpoint {checkpoint_dir.name}: {e}")
        
        return sorted(checkpoints, key=lambda x: x.step)
    
    def _load_metrics(self, session_id: str) -> TrainingMetrics:
        """Load training metrics for a session."""
        metrics_file = self.checkpoint_dir / session_id / "metrics.json"
        
        if not metrics_file.exists():
            return TrainingMetrics()
        
        try:
            with open(metrics_file, 'r') as f:
                metrics_data = json.load(f)
            
            # Convert timestamp strings back to datetime objects
            if 'loss_timestamps' in metrics_data:
                metrics_data['loss_timestamps'] = [
                    datetime.fromisoformat(ts) for ts in metrics_data['loss_timestamps']
                ]
            
            return TrainingMetrics(**metrics_data)
            
        except Exception as e:
            logger.warning(f"Failed to load metrics for session {session_id}: {e}")
            return TrainingMetrics()
    
    def _cleanup_checkpoints(self) -> None:
        """Clean up old checkpoints based on retention policy."""
        if not self.current_session:
            return
        
        # Separate checkpoints by type
        regular_checkpoints = [cp for cp in self.current_session.checkpoints 
                             if cp.checkpoint_type == CheckpointType.REGULAR]
        best_checkpoints = [cp for cp in self.current_session.checkpoints 
                          if cp.checkpoint_type == CheckpointType.BEST_LOSS]
        
        # Keep only recent regular checkpoints
        if len(regular_checkpoints) > self.keep_recent_checkpoints:
            to_remove = regular_checkpoints[:-self.keep_recent_checkpoints]
            for checkpoint in to_remove:
                self._remove_checkpoint(checkpoint)
        
        # Keep only best checkpoints
        if len(best_checkpoints) > self.keep_best_checkpoints:
            # Sort by validation loss and keep the best
            best_checkpoints.sort(key=lambda x: x.validation_loss or float('inf'))
            to_remove = best_checkpoints[self.keep_best_checkpoints:]
            for checkpoint in to_remove:
                self._remove_checkpoint(checkpoint)
    
    def _remove_checkpoint(self, checkpoint: Checkpoint) -> None:
        """Remove a checkpoint from disk and session."""
        try:
            # Remove checkpoint directory
            checkpoint_dir = checkpoint.model_path.parent
            if checkpoint_dir.exists():
                import shutil
                shutil.rmtree(checkpoint_dir)
            
            # Remove from session
            if checkpoint in self.current_session.checkpoints:
                self.current_session.checkpoints.remove(checkpoint)
            
            logger.debug(f"Removed checkpoint: {checkpoint.checkpoint_id}")
            
        except Exception as e:
            logger.warning(f"Failed to remove checkpoint {checkpoint.checkpoint_id}: {e}")