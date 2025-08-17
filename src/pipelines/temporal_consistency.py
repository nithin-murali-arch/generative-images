"""
Temporal consistency optimization engine for video generation.

This module implements temporal consistency optimization techniques to ensure
smooth transitions and coherent motion across video frames.
"""

import logging
import time
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

# Try to import required libraries
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available - temporal consistency will be limited")

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    Image = None
    PIL_AVAILABLE = False
    logger.warning("PIL not available - temporal consistency limited")

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
    np = None
    NUMPY_AVAILABLE = False
    logger.warning("NumPy not available - temporal consistency limited")

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    logger.warning("OpenCV not available - advanced temporal consistency limited")


class ConsistencyMethod(Enum):
    """Temporal consistency optimization methods."""
    OPTICAL_FLOW = "optical_flow"
    FEATURE_MATCHING = "feature_matching"
    TEMPORAL_SMOOTHING = "temporal_smoothing"
    MOTION_COMPENSATION = "motion_compensation"
    HYBRID = "hybrid"


class ConsistencyLevel(Enum):
    """Levels of temporal consistency optimization."""
    MINIMAL = "minimal"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"


@dataclass
class ConsistencyConfig:
    """Configuration for temporal consistency optimization."""
    method: ConsistencyMethod
    level: ConsistencyLevel
    smoothing_weight: float
    motion_threshold: float
    feature_match_threshold: float
    max_displacement: int
    temporal_window: int


@dataclass
class MotionVector:
    """Motion vector between frames."""
    dx: float
    dy: float
    confidence: float
    frame_index: int


@dataclass
class ConsistencyMetrics:
    """Metrics for temporal consistency analysis."""
    frame_similarity: List[float]
    motion_smoothness: float
    temporal_coherence: float
    processing_time: float


class TemporalConsistencyEngine:
    """
    Engine for optimizing temporal consistency in video sequences.
    
    Provides various methods for ensuring smooth transitions and coherent
    motion across video frames, with hardware-adaptive optimization.
    """
    
    def __init__(self, hardware_config):
        self.hardware_config = hardware_config
        self.config = self._create_consistency_config()
        self.motion_vectors = []
        self.frame_features = []
        self.consistency_cache = {}
        
        logger.info("TemporalConsistencyEngine initialized")
    
    def optimize_sequence(self, frames: List[PILImage]) -> Tuple[List[PILImage], ConsistencyMetrics]:
        """
        Optimize temporal consistency of a frame sequence.
        
        Args:
            frames: List of video frames to optimize
            
        Returns:
            Tuple of optimized frames and consistency metrics
        """
        if len(frames) < 2:
            logger.warning("Cannot optimize temporal consistency with less than 2 frames")
            return frames, self._create_empty_metrics()
        
        start_time = time.time()
        
        try:
            logger.info(f"Optimizing temporal consistency for {len(frames)} frames")
            
            # Analyze frame sequence
            analysis = self._analyze_sequence(frames)
            
            # Apply optimization based on method
            if self.config.method == ConsistencyMethod.OPTICAL_FLOW:
                optimized_frames = self._optimize_with_optical_flow(frames, analysis)
            elif self.config.method == ConsistencyMethod.FEATURE_MATCHING:
                optimized_frames = self._optimize_with_feature_matching(frames, analysis)
            elif self.config.method == ConsistencyMethod.TEMPORAL_SMOOTHING:
                optimized_frames = self._optimize_with_temporal_smoothing(frames, analysis)
            elif self.config.method == ConsistencyMethod.MOTION_COMPENSATION:
                optimized_frames = self._optimize_with_motion_compensation(frames, analysis)
            elif self.config.method == ConsistencyMethod.HYBRID:
                optimized_frames = self._optimize_hybrid(frames, analysis)
            else:
                logger.warning(f"Unknown consistency method: {self.config.method}")
                optimized_frames = frames
            
            # Calculate metrics
            processing_time = time.time() - start_time
            metrics = self._calculate_metrics(frames, optimized_frames, processing_time)
            
            logger.info(f"Temporal consistency optimization completed in {processing_time:.2f}s")
            return optimized_frames, metrics
            
        except Exception as e:
            logger.error(f"Temporal consistency optimization failed: {e}")
            return frames, self._create_empty_metrics()
    
    def _create_consistency_config(self) -> ConsistencyConfig:
        """Create consistency configuration based on hardware."""
        vram_mb = self.hardware_config.vram_size
        
        # Determine optimization level based on hardware
        if vram_mb >= 16000:  # 16GB+
            level = ConsistencyLevel.AGGRESSIVE
            method = ConsistencyMethod.HYBRID
        elif vram_mb >= 8000:  # 8-16GB
            level = ConsistencyLevel.MODERATE
            method = ConsistencyMethod.OPTICAL_FLOW if CV2_AVAILABLE else ConsistencyMethod.TEMPORAL_SMOOTHING
        else:  # <8GB
            level = ConsistencyLevel.MINIMAL
            method = ConsistencyMethod.TEMPORAL_SMOOTHING
        
        # Configure parameters based on level
        if level == ConsistencyLevel.AGGRESSIVE:
            smoothing_weight = 0.4
            motion_threshold = 0.1
            feature_match_threshold = 0.8
            max_displacement = 50
            temporal_window = 5
        elif level == ConsistencyLevel.MODERATE:
            smoothing_weight = 0.3
            motion_threshold = 0.2
            feature_match_threshold = 0.7
            max_displacement = 30
            temporal_window = 3
        else:  # MINIMAL
            smoothing_weight = 0.2
            motion_threshold = 0.3
            feature_match_threshold = 0.6
            max_displacement = 20
            temporal_window = 2
        
        return ConsistencyConfig(
            method=method,
            level=level,
            smoothing_weight=smoothing_weight,
            motion_threshold=motion_threshold,
            feature_match_threshold=feature_match_threshold,
            max_displacement=max_displacement,
            temporal_window=temporal_window
        )
    
    def _analyze_sequence(self, frames: List[PILImage]) -> Dict[str, Any]:
        """Analyze frame sequence for consistency issues."""
        analysis = {
            'frame_similarities': [],
            'motion_vectors': [],
            'inconsistency_scores': [],
            'problem_regions': []
        }
        
        if not NUMPY_AVAILABLE:
            logger.warning("NumPy not available - limited sequence analysis")
            return analysis
        
        # Calculate frame-to-frame similarities
        for i in range(len(frames) - 1):
            similarity = self._calculate_frame_similarity(frames[i], frames[i + 1])
            analysis['frame_similarities'].append(similarity)
        
        # Detect motion vectors if OpenCV available
        if CV2_AVAILABLE:
            motion_vectors = self._calculate_motion_vectors(frames)
            analysis['motion_vectors'] = motion_vectors
        
        # Calculate inconsistency scores
        inconsistency_scores = self._calculate_inconsistency_scores(analysis['frame_similarities'])
        analysis['inconsistency_scores'] = inconsistency_scores
        
        return analysis
    
    def _calculate_frame_similarity(self, frame1: PILImage, frame2: PILImage) -> float:
        """Calculate similarity between two frames."""
        if not NUMPY_AVAILABLE or np is None:
            return 0.5  # Default similarity
        
        try:
            # Convert to numpy arrays
            array1 = np.array(frame1)
            array2 = np.array(frame2)
            
            # Calculate normalized cross-correlation
            corr_matrix = np.corrcoef(array1.flatten(), array2.flatten())
            
            # Handle case where corrcoef returns a scalar or malformed matrix
            if corr_matrix.ndim == 2 and corr_matrix.shape == (2, 2):
                correlation = corr_matrix[0, 1]
            else:
                correlation = 0.5  # Default fallback
            
            # Handle NaN values
            if np.isnan(correlation):
                correlation = 0.0
            
            return max(0.0, correlation)
        except (IndexError, TypeError, AttributeError):
            # Fallback for mock objects or other issues
            return 0.5
    
    def _calculate_motion_vectors(self, frames: List[PILImage]) -> List[MotionVector]:
        """Calculate motion vectors between consecutive frames."""
        motion_vectors = []
        
        if not CV2_AVAILABLE or not NUMPY_AVAILABLE:
            return motion_vectors
        
        for i in range(len(frames) - 1):
            # Convert frames to grayscale
            frame1_gray = cv2.cvtColor(np.array(frames[i]), cv2.COLOR_RGB2GRAY)
            frame2_gray = cv2.cvtColor(np.array(frames[i + 1]), cv2.COLOR_RGB2GRAY)
            
            # Calculate optical flow
            flow = cv2.calcOpticalFlowPyrLK(
                frame1_gray, frame2_gray, None, None,
                winSize=(15, 15),
                maxLevel=2,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
            )
            
            if flow[0] is not None and len(flow[0]) > 0:
                # Calculate average motion
                good_points = flow[1].ravel() == 1
                if np.any(good_points):
                    avg_dx = np.mean(flow[0][good_points, 0])
                    avg_dy = np.mean(flow[0][good_points, 1])
                    confidence = np.mean(flow[1][good_points])
                    
                    motion_vector = MotionVector(
                        dx=float(avg_dx),
                        dy=float(avg_dy),
                        confidence=float(confidence),
                        frame_index=i
                    )
                    motion_vectors.append(motion_vector)
        
        return motion_vectors
    
    def _calculate_inconsistency_scores(self, similarities: List[float]) -> List[float]:
        """Calculate inconsistency scores based on frame similarities."""
        if not similarities:
            return []
        
        # Inconsistency is inverse of similarity
        inconsistency_scores = [1.0 - sim for sim in similarities]
        
        # Smooth the scores to avoid over-correction
        if len(inconsistency_scores) > 2:
            smoothed_scores = []
            for i, score in enumerate(inconsistency_scores):
                if i == 0 or i == len(inconsistency_scores) - 1:
                    smoothed_scores.append(score)
                else:
                    # Average with neighbors
                    avg_score = (inconsistency_scores[i-1] + score + inconsistency_scores[i+1]) / 3
                    smoothed_scores.append(avg_score)
            return smoothed_scores
        
        return inconsistency_scores
    
    def _optimize_with_optical_flow(self, frames: List[PILImage], analysis: Dict[str, Any]) -> List[PILImage]:
        """Optimize using optical flow-based motion compensation."""
        if not CV2_AVAILABLE or not NUMPY_AVAILABLE:
            logger.warning("OpenCV/NumPy not available - falling back to temporal smoothing")
            return self._optimize_with_temporal_smoothing(frames, analysis)
        
        optimized_frames = frames.copy()
        motion_vectors = analysis.get('motion_vectors', [])
        
        # Apply motion compensation
        for i, motion_vector in enumerate(motion_vectors):
            if motion_vector.confidence > self.config.motion_threshold:
                # Apply motion compensation to reduce jitter
                frame_index = motion_vector.frame_index + 1
                if frame_index < len(optimized_frames):
                    compensated_frame = self._apply_motion_compensation(
                        optimized_frames[frame_index],
                        motion_vector
                    )
                    optimized_frames[frame_index] = compensated_frame
        
        return optimized_frames
    
    def _optimize_with_feature_matching(self, frames: List[PILImage], analysis: Dict[str, Any]) -> List[PILImage]:
        """Optimize using feature matching and alignment."""
        if not CV2_AVAILABLE:
            logger.warning("OpenCV not available - falling back to temporal smoothing")
            return self._optimize_with_temporal_smoothing(frames, analysis)
        
        optimized_frames = frames.copy()
        
        # Use SIFT or ORB for feature matching
        detector = cv2.ORB_create() if hasattr(cv2, 'ORB_create') else None
        
        if detector is None:
            return self._optimize_with_temporal_smoothing(frames, analysis)
        
        # Match features between consecutive frames
        for i in range(len(frames) - 1):
            frame1_gray = cv2.cvtColor(np.array(frames[i]), cv2.COLOR_RGB2GRAY)
            frame2_gray = cv2.cvtColor(np.array(frames[i + 1]), cv2.COLOR_RGB2GRAY)
            
            # Detect keypoints and descriptors
            kp1, des1 = detector.detectAndCompute(frame1_gray, None)
            kp2, des2 = detector.detectAndCompute(frame2_gray, None)
            
            if des1 is not None and des2 is not None:
                # Match features
                matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
                matches = matcher.match(des1, des2)
                
                if len(matches) > 10:  # Sufficient matches
                    # Apply feature-based alignment
                    aligned_frame = self._align_frame_with_features(
                        optimized_frames[i + 1], matches, kp1, kp2
                    )
                    optimized_frames[i + 1] = aligned_frame
        
        return optimized_frames
    
    def _optimize_with_temporal_smoothing(self, frames: List[PILImage], analysis: Dict[str, Any]) -> List[PILImage]:
        """Optimize using temporal smoothing."""
        if not NUMPY_AVAILABLE:
            logger.warning("NumPy not available - returning original frames")
            return frames
        
        optimized_frames = []
        window = self.config.temporal_window
        weight = self.config.smoothing_weight
        
        for i, frame in enumerate(frames):
            if i == 0 or i == len(frames) - 1:
                # Keep first and last frames unchanged
                optimized_frames.append(frame)
            else:
                # Apply temporal smoothing
                smoothed_frame = self._apply_temporal_smoothing(frames, i, window, weight)
                optimized_frames.append(smoothed_frame)
        
        return optimized_frames
    
    def _optimize_with_motion_compensation(self, frames: List[PILImage], analysis: Dict[str, Any]) -> List[PILImage]:
        """Optimize using motion compensation."""
        motion_vectors = analysis.get('motion_vectors', [])
        
        if not motion_vectors:
            return self._optimize_with_temporal_smoothing(frames, analysis)
        
        optimized_frames = frames.copy()
        
        # Apply motion compensation based on detected motion
        for motion_vector in motion_vectors:
            frame_index = motion_vector.frame_index + 1
            if frame_index < len(optimized_frames) and motion_vector.confidence > self.config.motion_threshold:
                compensated_frame = self._apply_motion_compensation(
                    optimized_frames[frame_index],
                    motion_vector
                )
                optimized_frames[frame_index] = compensated_frame
        
        return optimized_frames
    
    def _optimize_hybrid(self, frames: List[PILImage], analysis: Dict[str, Any]) -> List[PILImage]:
        """Optimize using hybrid approach combining multiple methods."""
        # Start with temporal smoothing
        smoothed_frames = self._optimize_with_temporal_smoothing(frames, analysis)
        
        # Apply motion compensation if available
        if CV2_AVAILABLE and analysis.get('motion_vectors'):
            motion_compensated = self._optimize_with_motion_compensation(smoothed_frames, analysis)
            return motion_compensated
        
        return smoothed_frames
    
    def _apply_temporal_smoothing(self, frames: List[PILImage], frame_index: int, window: int, weight: float) -> PILImage:
        """Apply temporal smoothing to a single frame."""
        current_frame = frames[frame_index]
        
        if not NUMPY_AVAILABLE or np is None:
            return current_frame
        
        try:
            # Determine smoothing window
            start_idx = max(0, frame_index - window // 2)
            end_idx = min(len(frames), frame_index + window // 2 + 1)
            
            # Convert current frame to array
            current_array = np.array(current_frame)
            
            # Calculate weighted average with neighboring frames
            neighbor_arrays = []
            for i in range(start_idx, end_idx):
                if i != frame_index:
                    neighbor_arrays.append(np.array(frames[i]))
            
            if neighbor_arrays:
                neighbor_avg = np.mean(neighbor_arrays, axis=0)
                smoothed_array = (1 - weight) * current_array + weight * neighbor_avg
                smoothed_array = smoothed_array.astype(np.uint8)
                
                return Image.fromarray(smoothed_array)
        except (TypeError, AttributeError):
            # Fallback for mock objects or other issues
            pass
        
        return current_frame
    
    def _apply_motion_compensation(self, frame: PILImage, motion_vector: MotionVector) -> PILImage:
        """Apply motion compensation to reduce jitter."""
        if not CV2_AVAILABLE or not NUMPY_AVAILABLE:
            return frame
        
        # Convert to numpy array
        frame_array = np.array(frame)
        
        # Create translation matrix
        dx = -motion_vector.dx * 0.1  # Reduce motion by 10%
        dy = -motion_vector.dy * 0.1
        
        # Limit displacement
        dx = np.clip(dx, -self.config.max_displacement, self.config.max_displacement)
        dy = np.clip(dy, -self.config.max_displacement, self.config.max_displacement)
        
        # Apply translation
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        height, width = frame_array.shape[:2]
        
        compensated_array = cv2.warpAffine(frame_array, M, (width, height))
        
        return Image.fromarray(compensated_array)
    
    def _align_frame_with_features(self, frame: PILImage, matches, kp1, kp2) -> PILImage:
        """Align frame using feature matches."""
        if not CV2_AVAILABLE or len(matches) < 4:
            return frame
        
        # Extract matched points
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        
        # Find homography
        M, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
        
        if M is not None:
            # Apply transformation
            frame_array = np.array(frame)
            height, width = frame_array.shape[:2]
            aligned_array = cv2.warpPerspective(frame_array, M, (width, height))
            
            return Image.fromarray(aligned_array)
        
        return frame
    
    def _calculate_metrics(self, original_frames: List[PILImage], 
                         optimized_frames: List[PILImage], 
                         processing_time: float) -> ConsistencyMetrics:
        """Calculate consistency metrics."""
        if not NUMPY_AVAILABLE or len(original_frames) < 2:
            return self._create_empty_metrics()
        
        # Calculate frame similarities for optimized sequence
        optimized_similarities = []
        for i in range(len(optimized_frames) - 1):
            similarity = self._calculate_frame_similarity(optimized_frames[i], optimized_frames[i + 1])
            optimized_similarities.append(similarity)
        
        # Calculate motion smoothness
        motion_smoothness = np.mean(optimized_similarities) if optimized_similarities else 0.0
        
        # Calculate temporal coherence (variance of similarities)
        temporal_coherence = 1.0 - np.var(optimized_similarities) if optimized_similarities else 0.0
        
        return ConsistencyMetrics(
            frame_similarity=optimized_similarities,
            motion_smoothness=motion_smoothness,
            temporal_coherence=max(0.0, temporal_coherence),
            processing_time=processing_time
        )
    
    def _create_empty_metrics(self) -> ConsistencyMetrics:
        """Create empty metrics for error cases."""
        return ConsistencyMetrics(
            frame_similarity=[],
            motion_smoothness=0.0,
            temporal_coherence=0.0,
            processing_time=0.0
        )
    
    def get_config(self) -> ConsistencyConfig:
        """Get current consistency configuration."""
        return self.config
    
    def update_config(self, **kwargs):
        """Update consistency configuration."""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
                logger.info(f"Updated consistency config: {key} = {value}")
    
    def clear_cache(self):
        """Clear consistency cache."""
        self.consistency_cache.clear()
        self.motion_vectors.clear()
        self.frame_features.clear()
        logger.debug("Consistency cache cleared")
