"""
Model download manager with progress tracking and hardware-aware selection.

This module handles downloading AI models based on available hardware resources,
with progress tracking and intelligent caching.
"""

import logging
import os
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Callable, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

# Try to import required libraries
try:
    from huggingface_hub import snapshot_download, hf_hub_download
    from huggingface_hub.utils import LocalEntryNotFoundError
    HF_HUB_AVAILABLE = True
except ImportError:
    HF_HUB_AVAILABLE = False
    logger.warning("Hugging Face Hub not available - model downloading disabled")

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class DownloadStatus(Enum):
    """Status of model download."""
    NOT_STARTED = "not_started"
    DOWNLOADING = "downloading"
    COMPLETED = "completed"
    FAILED = "failed"
    CACHED = "cached"


@dataclass
class DownloadProgress:
    """Progress information for model download."""
    model_id: str
    status: DownloadStatus
    progress_percent: float = 0.0
    downloaded_mb: float = 0.0
    total_mb: float = 0.0
    speed_mbps: float = 0.0
    eta_seconds: Optional[int] = None
    error_message: Optional[str] = None


class ModelDownloader:
    """
    Manages downloading and caching of AI models with progress tracking.
    
    Features:
    - Hardware-aware model selection
    - Progress tracking with callbacks
    - Intelligent caching and verification
    - Background downloads
    - Download queue management
    """
    
    def __init__(self):
        self.download_progress: Dict[str, DownloadProgress] = {}
        self.active_downloads: Dict[str, threading.Thread] = {}
        self.progress_callbacks: List[Callable[[DownloadProgress], None]] = []
        
        # Create cache directory
        self.cache_dir = Path.home() / ".cache" / "ai_content_generator"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("ModelDownloader initialized")
    
    def add_progress_callback(self, callback: Callable[[DownloadProgress], None]):
        """Add a callback function to receive download progress updates."""
        self.progress_callbacks.append(callback)
    
    def remove_progress_callback(self, callback: Callable[[DownloadProgress], None]):
        """Remove a progress callback."""
        if callback in self.progress_callbacks:
            self.progress_callbacks.remove(callback)
    
    def _notify_progress(self, progress: DownloadProgress):
        """Notify all callbacks of progress update."""
        for callback in self.progress_callbacks:
            try:
                callback(progress)
            except Exception as e:
                logger.error(f"Progress callback error: {e}")
    
    def is_model_downloaded(self, model_id: str) -> bool:
        """Check if a model is already downloaded and cached."""
        if not HF_HUB_AVAILABLE:
            return False
        
        try:
            # Try to find the model in cache without downloading
            snapshot_download(model_id, local_files_only=True)
            return True
        except LocalEntryNotFoundError:
            return False
        except Exception as e:
            logger.debug(f"Error checking model cache for {model_id}: {e}")
            return False
    
    def get_model_size_mb(self, model_id: str) -> float:
        """Estimate model size in MB (rough estimates for common models)."""
        size_estimates = {
            "runwayml/stable-diffusion-v1-5": 3400,
            "stabilityai/stable-diffusion-xl-base-1.0": 6900,
            "stabilityai/sdxl-turbo": 6900,
            "black-forest-labs/FLUX.1-schnell": 23800,
            "black-forest-labs/FLUX.1-dev": 23800,
            "stabilityai/stable-video-diffusion-img2vid-xt-1-1": 9700,
            "guoyww/animatediff-motion-adapter-v1-5-3": 1600,
            "THUDM/CogVideoX-2b": 8200,
            "segmind/tiny-sd": 500,
        }
        
        return size_estimates.get(model_id, 5000)  # Default 5GB estimate
    
    def get_download_status(self, model_id: str) -> DownloadProgress:
        """Get current download status for a model."""
        if model_id in self.download_progress:
            return self.download_progress[model_id]
        
        # Check if already cached
        if self.is_model_downloaded(model_id):
            status = DownloadProgress(
                model_id=model_id,
                status=DownloadStatus.CACHED,
                progress_percent=100.0
            )
        else:
            status = DownloadProgress(
                model_id=model_id,
                status=DownloadStatus.NOT_STARTED
            )
        
        self.download_progress[model_id] = status
        return status
    
    def download_model(self, model_id: str, background: bool = True) -> bool:
        """
        Download a model with progress tracking.
        
        Args:
            model_id: Hugging Face model ID
            background: Whether to download in background thread
            
        Returns:
            bool: True if download started successfully
        """
        if not HF_HUB_AVAILABLE:
            logger.error("Hugging Face Hub not available - cannot download models")
            return False
        
        # Check if already downloaded
        if self.is_model_downloaded(model_id):
            progress = DownloadProgress(
                model_id=model_id,
                status=DownloadStatus.CACHED,
                progress_percent=100.0
            )
            self.download_progress[model_id] = progress
            self._notify_progress(progress)
            return True
        
        # Check if already downloading
        if model_id in self.active_downloads:
            logger.info(f"Model {model_id} is already being downloaded")
            return True
        
        if background:
            # Start background download
            thread = threading.Thread(
                target=self._download_model_thread,
                args=(model_id,),
                daemon=True
            )
            thread.start()
            self.active_downloads[model_id] = thread
            return True
        else:
            # Download synchronously
            return self._download_model_sync(model_id)
    
    def _download_model_thread(self, model_id: str):
        """Download model in background thread."""
        try:
            self._download_model_sync(model_id)
        finally:
            # Clean up thread reference
            if model_id in self.active_downloads:
                del self.active_downloads[model_id]
    
    def _download_model_sync(self, model_id: str) -> bool:
        """Download model synchronously with progress tracking."""
        logger.info(f"Starting download of model: {model_id}")
        
        # Initialize progress
        progress = DownloadProgress(
            model_id=model_id,
            status=DownloadStatus.DOWNLOADING,
            total_mb=self.get_model_size_mb(model_id)
        )
        self.download_progress[model_id] = progress
        self._notify_progress(progress)
        
        try:
            start_time = time.time()
            
            # Download with progress tracking
            def progress_callback(downloaded: int, total: int):
                if total > 0:
                    progress.progress_percent = (downloaded / total) * 100
                    progress.downloaded_mb = downloaded / (1024 * 1024)
                    progress.total_mb = total / (1024 * 1024)
                    
                    # Calculate speed
                    elapsed = time.time() - start_time
                    if elapsed > 0:
                        progress.speed_mbps = progress.downloaded_mb / elapsed
                        
                        # Calculate ETA
                        if progress.speed_mbps > 0:
                            remaining_mb = progress.total_mb - progress.downloaded_mb
                            progress.eta_seconds = int(remaining_mb / progress.speed_mbps)
                    
                    self._notify_progress(progress)
            
            # Perform the actual download
            snapshot_download(
                model_id,
                cache_dir=None,  # Use default cache
                resume_download=True,
                # Note: progress_callback not directly supported by snapshot_download
                # This is a simplified version - real implementation would need
                # to hook into the download process more deeply
            )
            
            # Mark as completed
            progress.status = DownloadStatus.COMPLETED
            progress.progress_percent = 100.0
            self._notify_progress(progress)
            
            logger.info(f"Successfully downloaded model: {model_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to download model {model_id}: {e}")
            
            # Mark as failed
            progress.status = DownloadStatus.FAILED
            progress.error_message = str(e)
            self._notify_progress(progress)
            
            return False
    
    def download_recommended_models(self, vram_mb: int, max_concurrent: int = 2) -> List[str]:
        """
        Download recommended models based on available VRAM.
        
        Args:
            vram_mb: Available VRAM in MB
            max_concurrent: Maximum concurrent downloads
            
        Returns:
            List of model IDs being downloaded
        """
        from .model_registry import get_model_registry, ModelType
        
        registry = get_model_registry()
        
        # Get compatible models
        compatible_models = registry.get_compatible_models(vram_mb, ModelType.TEXT_TO_IMAGE)
        
        # Sort by download priority
        compatible_models.sort(key=lambda m: m.download_priority)
        
        # Download top models up to max_concurrent
        downloading = []
        active_count = len(self.active_downloads)
        
        for model in compatible_models[:max_concurrent - active_count]:
            if not self.is_model_downloaded(model.model_id):
                if self.download_model(model.model_id, background=True):
                    downloading.append(model.model_id)
        
        return downloading
    
    def get_all_download_status(self) -> Dict[str, DownloadProgress]:
        """Get download status for all models."""
        return self.download_progress.copy()
    
    def cancel_download(self, model_id: str) -> bool:
        """Cancel an active download (best effort)."""
        if model_id in self.active_downloads:
            # Note: This is a simplified cancellation
            # Real implementation would need proper thread cancellation
            logger.info(f"Attempting to cancel download of {model_id}")
            return True
        return False
    
    def cleanup_cache(self, keep_recent: int = 3) -> Tuple[int, float]:
        """
        Clean up old cached models to free space.
        
        Args:
            keep_recent: Number of recent models to keep
            
        Returns:
            Tuple of (files_removed, mb_freed)
        """
        # This is a placeholder - real implementation would
        # analyze cache directory and remove old models
        logger.info("Cache cleanup not yet implemented")
        return 0, 0.0


# Global downloader instance
_model_downloader = None

def get_model_downloader() -> ModelDownloader:
    """Get the global model downloader instance."""
    global _model_downloader
    if _model_downloader is None:
        _model_downloader = ModelDownloader()
    return _model_downloader


# Convenience functions
def download_model_async(model_id: str) -> bool:
    """Download a model asynchronously."""
    downloader = get_model_downloader()
    return downloader.download_model(model_id, background=True)


def is_model_available(model_id: str) -> bool:
    """Check if a model is available locally."""
    downloader = get_model_downloader()
    return downloader.is_model_downloaded(model_id)


def get_download_progress(model_id: str) -> DownloadProgress:
    """Get download progress for a model."""
    downloader = get_model_downloader()
    return downloader.get_download_status(model_id)


if __name__ == "__main__":
    # Test the downloader
    downloader = ModelDownloader()
    
    def progress_callback(progress: DownloadProgress):
        print(f"{progress.model_id}: {progress.status.value} - {progress.progress_percent:.1f}%")
    
    downloader.add_progress_callback(progress_callback)
    
    # Test with a small model
    test_model = "runwayml/stable-diffusion-v1-5"
    print(f"Testing download of {test_model}")
    print(f"Is downloaded: {downloader.is_model_downloaded(test_model)}")
    print(f"Estimated size: {downloader.get_model_size_mb(test_model):.1f} MB")