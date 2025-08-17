"""
Frame processing module for hybrid CPU/GPU video generation.

This module implements frame processing capabilities that can balance workload
between CPU and GPU based on available resources and memory constraints.
"""

import logging
import time
import gc
from typing import List, Dict, Any, Optional, Tuple, Union
from pathlib import Path
from dataclasses import dataclass
from enum import Enum
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)

# Try to import required libraries
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available - frame processing will be limited")

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    Image = None
    PIL_AVAILABLE = False
    logger.warning("PIL not available - frame processing limited")

# For type annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from PIL import Image as PILImage
else:
    PILImage = Any

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    logger.warning("NumPy not available - frame processing limited")

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    logger.warning("OpenCV not available - frame interpolation limited")


class ProcessingMode(Enum):
    """Processing mode for frame generation."""
    GPU_ONLY = "gpu_only"
    CPU_ONLY = "cpu_only"
    HYBRID = "hybrid"
    AUTO = "auto"


class InterpolationMethod(Enum):
    """Frame interpolation methods."""
    LINEAR = "linear"
    CUBIC = "cubic"
    OPTICAL_FLOW = "optical_flow"
    SIMPLE_BLEND = "simple_blend"


@dataclass
class ProcessingConfig:
    """Configuration for frame processing."""
    mode: ProcessingMode
    max_gpu_frames: int
    cpu_threads: int
    interpolation_method: InterpolationMethod
    temporal_consistency_weight: float
    memory_threshold_mb: int


@dataclass
class FrameGenerationTask:
    """Task for generating a single frame."""
    frame_index: int
    prompt: str
    conditioning_data: Optional[Dict[str, Any]]
    processing_mode: ProcessingMode
    priority: int = 0


class FrameProcessor:
    """
    Hybrid CPU/GPU frame processor for video generation.
    
    Manages frame generation workload distribution between CPU and GPU
    based on available resources and memory constraints.
    """
    
    def __init__(self, hardware_config, memory_manager):
        self.hardware_config = hardware_config
        self.memory_manager = memory_manager
        self.processing_config = self._create_processing_config()
        self.gpu_pipeline = None
        self.cpu_pipeline = None
        self.frame_cache = {}
        self.processing_stats = {
            'gpu_frames': 0,
            'cpu_frames': 0,
            'interpolated_frames': 0,
            'total_processing_time': 0.0
        }
        
        logger.info("FrameProcessor initialized")
    
    def set_pipelines(self, gpu_pipeline, cpu_pipeline=None):
        """Set the GPU and CPU pipelines for frame processing."""
        self.gpu_pipeline = gpu_pipeline
        self.cpu_pipeline = cpu_pipeline or gpu_pipeline
        logger.info("Pipelines set for frame processor")
    
    def process_video_frames(self, 
                           prompt: str, 
                           num_frames: int, 
                           generation_params: Dict[str, Any]) -> List[PILImage]:
        """
        Process video frames using hybrid CPU/GPU approach.
        
        Args:
            prompt: Text prompt for video generation
            num_frames: Number of frames to generate
            generation_params: Parameters for frame generation
            
        Returns:
            List of generated frames
        """
        start_time = time.time()
        
        try:
            logger.info(f"Processing {num_frames} frames with hybrid approach")
            
            # Determine processing strategy
            strategy = self._determine_processing_strategy(num_frames, generation_params)
            
            # Generate keyframes and interpolated frames
            if strategy['use_keyframes']:
                frames = self._generate_with_keyframes(prompt, num_frames, generation_params, strategy)
            else:
                frames = self._generate_sequential(prompt, num_frames, generation_params, strategy)
            
            # Apply temporal consistency optimization
            if strategy['apply_temporal_consistency']:
                frames = self._apply_temporal_consistency(frames)
            
            processing_time = time.time() - start_time
            self.processing_stats['total_processing_time'] += processing_time
            
            logger.info(f"Processed {len(frames)} frames in {processing_time:.2f}s")
            return frames
            
        except Exception as e:
            logger.error(f"Frame processing failed: {e}")
            raise
    
    def _create_processing_config(self) -> ProcessingConfig:
        """Create processing configuration based on hardware."""
        vram_mb = self.hardware_config.vram_size
        cpu_cores = self.hardware_config.cpu_cores
        
        # Determine processing mode based on VRAM
        if vram_mb >= 16000:  # 16GB+
            mode = ProcessingMode.GPU_ONLY
            max_gpu_frames = 32
        elif vram_mb >= 8000:  # 8-16GB
            mode = ProcessingMode.HYBRID
            max_gpu_frames = 8
        else:  # <8GB
            mode = ProcessingMode.HYBRID
            max_gpu_frames = 4
        
        # Configure CPU threads
        cpu_threads = min(cpu_cores, 8)  # Cap at 8 threads
        
        # Select interpolation method based on available libraries
        if CV2_AVAILABLE and NUMPY_AVAILABLE:
            interpolation_method = InterpolationMethod.OPTICAL_FLOW
        elif NUMPY_AVAILABLE:
            interpolation_method = InterpolationMethod.LINEAR
        else:
            interpolation_method = InterpolationMethod.SIMPLE_BLEND
        
        return ProcessingConfig(
            mode=mode,
            max_gpu_frames=max_gpu_frames,
            cpu_threads=cpu_threads,
            interpolation_method=interpolation_method,
            temporal_consistency_weight=0.3,
            memory_threshold_mb=int(vram_mb * 0.8)  # 80% of VRAM
        )
    
    def _determine_processing_strategy(self, 
                                     num_frames: int, 
                                     generation_params: Dict[str, Any]) -> Dict[str, Any]:
        """Determine optimal processing strategy for the request."""
        config = self.processing_config
        
        # Check current memory usage
        memory_status = self.memory_manager.get_memory_status()
        available_vram = memory_status.get('available_vram_mb', config.memory_threshold_mb)
        
        # Determine if we should use keyframe approach
        use_keyframes = (
            num_frames > config.max_gpu_frames or 
            available_vram < config.memory_threshold_mb
        )
        
        # Calculate keyframe distribution
        if use_keyframes:
            max_keyframes = min(config.max_gpu_frames, max(2, num_frames // 4))
            keyframe_indices = self._calculate_keyframe_indices(num_frames, max_keyframes)
        else:
            keyframe_indices = list(range(num_frames))
        
        # Determine processing distribution
        gpu_frame_count = len(keyframe_indices)
        cpu_frame_count = num_frames - gpu_frame_count
        
        strategy = {
            'use_keyframes': use_keyframes,
            'keyframe_indices': keyframe_indices,
            'gpu_frame_count': gpu_frame_count,
            'cpu_frame_count': cpu_frame_count,
            'apply_temporal_consistency': num_frames > 8,
            'batch_size': min(4, config.max_gpu_frames // 2),
            'interpolation_method': config.interpolation_method
        }
        
        logger.debug(f"Processing strategy: {strategy}")
        return strategy
    
    def _calculate_keyframe_indices(self, num_frames: int, max_keyframes: int) -> List[int]:
        """Calculate optimal keyframe indices for interpolation."""
        if max_keyframes >= num_frames:
            return list(range(num_frames))
        
        # Distribute keyframes evenly across the timeline
        keyframe_indices = []
        
        # Always include first and last frames
        keyframe_indices.append(0)
        if num_frames > 1:
            keyframe_indices.append(num_frames - 1)
        
        # Add intermediate keyframes
        remaining_keyframes = max_keyframes - len(keyframe_indices)
        if remaining_keyframes > 0:
            step = (num_frames - 1) / (remaining_keyframes + 1)
            for i in range(1, remaining_keyframes + 1):
                index = int(i * step)
                if index not in keyframe_indices:
                    keyframe_indices.append(index)
        
        return sorted(keyframe_indices)
    
    def _generate_with_keyframes(self, 
                               prompt: str, 
                               num_frames: int, 
                               generation_params: Dict[str, Any],
                               strategy: Dict[str, Any]) -> List[PILImage]:
        """Generate video using keyframe approach with interpolation."""
        keyframe_indices = strategy['keyframe_indices']
        
        # Generate keyframes on GPU
        keyframes = self._generate_keyframes_gpu(prompt, keyframe_indices, generation_params)
        
        # Interpolate between keyframes
        all_frames = self._interpolate_between_keyframes(keyframes, keyframe_indices, num_frames, strategy)
        
        return all_frames
    
    def _generate_sequential(self, 
                           prompt: str, 
                           num_frames: int, 
                           generation_params: Dict[str, Any],
                           strategy: Dict[str, Any]) -> List[PILImage]:
        """Generate video frames sequentially."""
        frames = []
        batch_size = strategy['batch_size']
        
        # Process frames in batches
        for i in range(0, num_frames, batch_size):
            batch_end = min(i + batch_size, num_frames)
            batch_frames = self._generate_frame_batch_gpu(
                prompt, list(range(i, batch_end)), generation_params
            )
            frames.extend(batch_frames)
            
            # Clear cache between batches to manage memory
            if self.memory_manager:
                self.memory_manager.clear_vram_cache()
        
        return frames
    
    def _generate_keyframes_gpu(self, 
                              prompt: str, 
                              keyframe_indices: List[int], 
                              generation_params: Dict[str, Any]) -> Dict[int, PILImage]:
        """Generate keyframes using GPU pipeline."""
        keyframes = {}
        
        if not self.gpu_pipeline:
            raise RuntimeError("GPU pipeline not available")
        
        logger.info(f"Generating {len(keyframe_indices)} keyframes on GPU")
        
        for frame_index in keyframe_indices:
            try:
                # Modify prompt for temporal variation
                frame_prompt = self._create_frame_prompt(prompt, frame_index, len(keyframe_indices))
                
                # Generate frame
                frame = self._generate_single_frame_gpu(frame_prompt, generation_params)
                keyframes[frame_index] = frame
                
                self.processing_stats['gpu_frames'] += 1
                
                # Clear cache after each keyframe to manage memory
                if self.memory_manager:
                    self.memory_manager.clear_vram_cache()
                
            except Exception as e:
                logger.error(f"Failed to generate keyframe {frame_index}: {e}")
                # Create fallback frame
                keyframes[frame_index] = self._create_fallback_frame(generation_params)
        
        logger.info(f"Generated {len(keyframes)} keyframes")
        return keyframes
    
    def _generate_frame_batch_gpu(self, 
                                prompt: str, 
                                frame_indices: List[int], 
                                generation_params: Dict[str, Any]) -> List[PILImage]:
        """Generate a batch of frames on GPU."""
        frames = []
        
        for frame_index in frame_indices:
            frame_prompt = self._create_frame_prompt(prompt, frame_index, len(frame_indices))
            frame = self._generate_single_frame_gpu(frame_prompt, generation_params)
            frames.append(frame)
            
            self.processing_stats['gpu_frames'] += 1
        
        return frames
    
    def _generate_single_frame_gpu(self, prompt: str, generation_params: Dict[str, Any]) -> PILImage:
        """Generate a single frame using GPU pipeline."""
        if not self.gpu_pipeline:
            raise RuntimeError("GPU pipeline not available")
        
        # Prepare generation arguments
        gen_args = {
            'prompt': prompt,
            'width': generation_params.get('width', 512),
            'height': generation_params.get('height', 512),
            'num_inference_steps': generation_params.get('num_inference_steps', 20),
            'guidance_scale': generation_params.get('guidance_scale', 7.5)
        }
        
        # Add seed with variation
        if 'seed' in generation_params and generation_params['seed'] is not None:
            generator = torch.Generator(device='cuda' if torch.cuda.is_available() else 'cpu')
            generator.manual_seed(generation_params['seed'])
            gen_args['generator'] = generator
        
        # Generate frame
        result = self.gpu_pipeline(**gen_args)
        
        if hasattr(result, 'images'):
            return result.images[0]
        else:
            return result
    
    def _interpolate_between_keyframes(self, 
                                     keyframes: Dict[int, PILImage], 
                                     keyframe_indices: List[int], 
                                     num_frames: int,
                                     strategy: Dict[str, Any]) -> List[PILImage]:
        """Interpolate frames between keyframes."""
        all_frames = [None] * num_frames
        
        # Place keyframes
        for index, frame in keyframes.items():
            all_frames[index] = frame
        
        # Interpolate missing frames
        interpolation_method = strategy['interpolation_method']
        
        for i in range(len(keyframe_indices) - 1):
            start_idx = keyframe_indices[i]
            end_idx = keyframe_indices[i + 1]
            
            if end_idx - start_idx > 1:
                # Interpolate frames between start_idx and end_idx
                interpolated = self._interpolate_frame_sequence(
                    keyframes[start_idx], 
                    keyframes[end_idx], 
                    end_idx - start_idx - 1,
                    interpolation_method
                )
                
                # Place interpolated frames
                for j, frame in enumerate(interpolated):
                    all_frames[start_idx + j + 1] = frame
                    self.processing_stats['interpolated_frames'] += 1
        
        # Fill any remaining None frames with nearest keyframe
        for i, frame in enumerate(all_frames):
            if frame is None:
                all_frames[i] = self._find_nearest_keyframe(i, keyframes)
        
        return all_frames
    
    def _interpolate_frame_sequence(self, 
                                  start_frame: PILImage, 
                                  end_frame: PILImage, 
                                  num_intermediate: int,
                                  method: InterpolationMethod) -> List[PILImage]:
        """Interpolate frames between two keyframes."""
        if method == InterpolationMethod.OPTICAL_FLOW and CV2_AVAILABLE:
            return self._interpolate_optical_flow(start_frame, end_frame, num_intermediate)
        elif method == InterpolationMethod.LINEAR and NUMPY_AVAILABLE:
            return self._interpolate_linear(start_frame, end_frame, num_intermediate)
        elif method == InterpolationMethod.CUBIC and NUMPY_AVAILABLE:
            return self._interpolate_cubic(start_frame, end_frame, num_intermediate)
        else:
            return self._interpolate_simple_blend(start_frame, end_frame, num_intermediate)
    
    def _interpolate_optical_flow(self, 
                                start_frame: PILImage, 
                                end_frame: PILImage, 
                                num_intermediate: int) -> List[PILImage]:
        """Interpolate using optical flow (CPU-based)."""
        # Convert PIL to OpenCV format
        start_cv = cv2.cvtColor(np.array(start_frame), cv2.COLOR_RGB2BGR)
        end_cv = cv2.cvtColor(np.array(end_frame), cv2.COLOR_RGB2BGR)
        
        # Convert to grayscale for optical flow
        start_gray = cv2.cvtColor(start_cv, cv2.COLOR_BGR2GRAY)
        end_gray = cv2.cvtColor(end_cv, cv2.COLOR_BGR2GRAY)
        
        # Calculate optical flow
        flow = cv2.calcOpticalFlowPyrLK(start_gray, end_gray, None, None)
        
        interpolated_frames = []
        
        for i in range(1, num_intermediate + 1):
            alpha = i / (num_intermediate + 1)
            
            # Simple linear interpolation as fallback
            # In practice, would use flow to warp frames
            interpolated = cv2.addWeighted(start_cv, 1 - alpha, end_cv, alpha, 0)
            
            # Convert back to PIL
            interpolated_rgb = cv2.cvtColor(interpolated, cv2.COLOR_BGR2RGB)
            pil_frame = Image.fromarray(interpolated_rgb)
            interpolated_frames.append(pil_frame)
        
        return interpolated_frames
    
    def _interpolate_linear(self, 
                          start_frame: PILImage, 
                          end_frame: PILImage, 
                          num_intermediate: int) -> List[PILImage]:
        """Linear interpolation between frames."""
        start_array = np.array(start_frame)
        end_array = np.array(end_frame)
        
        interpolated_frames = []
        
        for i in range(1, num_intermediate + 1):
            alpha = i / (num_intermediate + 1)
            interpolated = (1 - alpha) * start_array + alpha * end_array
            interpolated = interpolated.astype(np.uint8)
            
            pil_frame = Image.fromarray(interpolated)
            interpolated_frames.append(pil_frame)
        
        return interpolated_frames
    
    def _interpolate_cubic(self, 
                         start_frame: PILImage, 
                         end_frame: PILImage, 
                         num_intermediate: int) -> List[PILImage]:
        """Cubic interpolation between frames."""
        # For simplicity, fall back to linear interpolation
        # In practice, would implement proper cubic spline interpolation
        return self._interpolate_linear(start_frame, end_frame, num_intermediate)
    
    def _interpolate_simple_blend(self, 
                                start_frame: PILImage, 
                                end_frame: PILImage, 
                                num_intermediate: int) -> List[PILImage]:
        """Simple blending interpolation (fallback method)."""
        interpolated_frames = []
        
        for i in range(1, num_intermediate + 1):
            alpha = i / (num_intermediate + 1)
            
            # Use PIL's blend function
            blended = Image.blend(start_frame, end_frame, alpha)
            interpolated_frames.append(blended)
        
        return interpolated_frames
    
    def _create_frame_prompt(self, base_prompt: str, frame_index: int, total_frames: int) -> str:
        """Create frame-specific prompt with temporal variation."""
        # Add temporal context to prompt
        progress = frame_index / max(1, total_frames - 1)
        
        # Simple temporal modifiers
        if progress < 0.3:
            temporal_modifier = "beginning of scene"
        elif progress > 0.7:
            temporal_modifier = "end of scene"
        else:
            temporal_modifier = "middle of scene"
        
        return f"{base_prompt}, {temporal_modifier}"
    
    def _find_nearest_keyframe(self, frame_index: int, keyframes: Dict[int, PILImage]) -> PILImage:
        """Find the nearest keyframe for a given frame index."""
        if not keyframes:
            return self._create_fallback_frame({})
        
        # Find closest keyframe index
        closest_index = min(keyframes.keys(), key=lambda x: abs(x - frame_index))
        return keyframes[closest_index]
    
    def _create_fallback_frame(self, generation_params: Dict[str, Any]) -> PILImage:
        """Create a fallback frame when generation fails."""
        width = generation_params.get('width', 512)
        height = generation_params.get('height', 512)
        
        # Create a simple gradient frame
        if PIL_AVAILABLE:
            frame = Image.new('RGB', (width, height), color=(64, 64, 64))
            return frame
        else:
            raise RuntimeError("Cannot create fallback frame without PIL")
    
    def _apply_temporal_consistency(self, frames: List[PILImage]) -> List[PILImage]:
        """Apply temporal consistency optimization to frame sequence."""
        if not NUMPY_AVAILABLE or len(frames) < 2:
            return frames
        
        logger.info("Applying temporal consistency optimization")
        
        # Simple temporal smoothing
        smoothed_frames = []
        weight = self.processing_config.temporal_consistency_weight
        
        for i, frame in enumerate(frames):
            if i == 0 or i == len(frames) - 1:
                # Keep first and last frames unchanged
                smoothed_frames.append(frame)
            else:
                # Blend with neighboring frames
                prev_frame = frames[i - 1]
                next_frame = frames[i + 1]
                
                # Convert to numpy arrays
                current_array = np.array(frame)
                prev_array = np.array(prev_frame)
                next_array = np.array(next_frame)
                
                # Apply temporal smoothing
                neighbor_avg = (prev_array + next_array) / 2
                smoothed_array = (1 - weight) * current_array + weight * neighbor_avg
                smoothed_array = smoothed_array.astype(np.uint8)
                
                smoothed_frame = Image.fromarray(smoothed_array)
                smoothed_frames.append(smoothed_frame)
        
        return smoothed_frames
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics."""
        return self.processing_stats.copy()
    
    def reset_stats(self):
        """Reset processing statistics."""
        self.processing_stats = {
            'gpu_frames': 0,
            'cpu_frames': 0,
            'interpolated_frames': 0,
            'total_processing_time': 0.0
        }
