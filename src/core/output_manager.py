"""
Output Management System

This module manages all generated outputs (images, videos) with proper organization,
metadata tracking, and UI integration.
"""

import logging
import json
import shutil
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class OutputType(Enum):
    """Types of generated outputs."""
    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"
    TEXT = "text"


@dataclass
class OutputMetadata:
    """Metadata for generated outputs."""
    output_id: str
    output_type: OutputType
    file_path: Path
    thumbnail_path: Optional[Path]
    prompt: str
    model_used: str
    generation_time: float
    parameters: Dict[str, Any]
    quality_metrics: Dict[str, Any]
    compliance_mode: str
    created_at: datetime
    file_size_bytes: int
    resolution: Optional[Tuple[int, int]] = None
    duration_seconds: Optional[float] = None
    tags: List[str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []


class OutputManager:
    """
    Manages generated outputs with proper organization and metadata tracking.
    
    Organizes outputs in structured directories:
    outputs/
    ├── images/
    │   ├── 2024/01/15/
    │   └── thumbnails/
    ├── videos/
    │   ├── 2024/01/15/
    │   └── thumbnails/
    └── metadata/
        └── outputs.json
    """
    
    def __init__(self, base_output_dir: str = "outputs"):
        """
        Initialize the output manager.
        
        Args:
            base_output_dir: Base directory for all outputs
        """
        self.base_dir = Path(base_output_dir)
        self.images_dir = self.base_dir / "images"
        self.videos_dir = self.base_dir / "videos"
        self.thumbnails_dir = self.base_dir / "thumbnails"
        self.metadata_dir = self.base_dir / "metadata"
        self.metadata_file = self.metadata_dir / "outputs.json"
        
        # Create directory structure
        self._create_directories()
        
        # Load existing metadata
        self.outputs_metadata: Dict[str, OutputMetadata] = {}
        self._load_metadata()
        
        logger.info(f"OutputManager initialized with base directory: {self.base_dir}")
    
    def _create_directories(self):
        """Create the output directory structure."""
        directories = [
            self.base_dir,
            self.images_dir,
            self.videos_dir,
            self.thumbnails_dir,
            self.metadata_dir
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
        
        logger.debug("Output directories created")
    
    def _load_metadata(self):
        """Load existing output metadata."""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    data = json.load(f)
                
                for output_id, metadata_dict in data.items():
                    # Convert datetime string back to datetime object
                    metadata_dict['created_at'] = datetime.fromisoformat(metadata_dict['created_at'])
                    
                    # Convert paths back to Path objects
                    metadata_dict['file_path'] = Path(metadata_dict['file_path'])
                    if metadata_dict.get('thumbnail_path'):
                        metadata_dict['thumbnail_path'] = Path(metadata_dict['thumbnail_path'])
                    
                    # Convert output_type back to enum
                    metadata_dict['output_type'] = OutputType(metadata_dict['output_type'])
                    
                    self.outputs_metadata[output_id] = OutputMetadata(**metadata_dict)
                
                logger.info(f"Loaded metadata for {len(self.outputs_metadata)} outputs")
                
            except Exception as e:
                logger.error(f"Failed to load metadata: {e}")
                self.outputs_metadata = {}
    
    def _save_metadata(self):
        """Save output metadata to file."""
        try:
            # Convert metadata to serializable format
            serializable_data = {}
            for output_id, metadata in self.outputs_metadata.items():
                metadata_dict = asdict(metadata)
                
                # Convert datetime to string
                metadata_dict['created_at'] = metadata.created_at.isoformat()
                
                # Convert paths to strings
                metadata_dict['file_path'] = str(metadata.file_path)
                if metadata.thumbnail_path:
                    metadata_dict['thumbnail_path'] = str(metadata.thumbnail_path)
                
                # Convert enum to string
                metadata_dict['output_type'] = metadata.output_type.value
                
                serializable_data[output_id] = metadata_dict
            
            with open(self.metadata_file, 'w') as f:
                json.dump(serializable_data, f, indent=2)
            
            logger.debug("Metadata saved successfully")
            
        except Exception as e:
            logger.error(f"Failed to save metadata: {e}")
    
    def save_output(self, 
                   file_path: Path, 
                   output_type: OutputType,
                   prompt: str,
                   model_used: str,
                   generation_time: float,
                   parameters: Dict[str, Any],
                   quality_metrics: Dict[str, Any] = None,
                   compliance_mode: str = "research_safe") -> str:
        """
        Save a generated output with metadata.
        
        Args:
            file_path: Path to the generated file
            output_type: Type of output (image/video)
            prompt: Generation prompt
            model_used: Model that generated the output
            generation_time: Time taken to generate
            parameters: Generation parameters
            quality_metrics: Quality metrics if available
            compliance_mode: Compliance mode used
            
        Returns:
            str: Output ID for tracking
        """
        try:
            # Generate unique output ID
            timestamp = datetime.now()
            output_id = f"{output_type.value}_{timestamp.strftime('%Y%m%d_%H%M%S')}_{hash(prompt) % 10000:04d}"
            
            # Create date-based subdirectory
            date_subdir = timestamp.strftime('%Y/%m/%d')
            
            if output_type == OutputType.IMAGE:
                target_dir = self.images_dir / date_subdir
            elif output_type == OutputType.VIDEO:
                target_dir = self.videos_dir / date_subdir
            else:
                target_dir = self.base_dir / output_type.value / date_subdir
            
            target_dir.mkdir(parents=True, exist_ok=True)
            
            # Copy file to organized location
            file_extension = file_path.suffix
            target_file = target_dir / f"{output_id}{file_extension}"
            
            if file_path.exists():
                shutil.copy2(file_path, target_file)
                file_size = target_file.stat().st_size
            else:
                logger.warning(f"Source file does not exist: {file_path}")
                file_size = 0
            
            # Generate thumbnail if it's an image or video
            thumbnail_path = None
            resolution = None
            duration = None
            
            if output_type == OutputType.IMAGE:
                thumbnail_path, resolution = self._create_image_thumbnail(target_file, output_id)
            elif output_type == OutputType.VIDEO:
                thumbnail_path, resolution, duration = self._create_video_thumbnail(target_file, output_id)
            
            # Create metadata
            metadata = OutputMetadata(
                output_id=output_id,
                output_type=output_type,
                file_path=target_file,
                thumbnail_path=thumbnail_path,
                prompt=prompt,
                model_used=model_used,
                generation_time=generation_time,
                parameters=parameters,
                quality_metrics=quality_metrics or {},
                compliance_mode=compliance_mode,
                created_at=timestamp,
                file_size_bytes=file_size,
                resolution=resolution,
                duration_seconds=duration,
                tags=self._extract_tags_from_prompt(prompt)
            )
            
            # Store metadata
            self.outputs_metadata[output_id] = metadata
            self._save_metadata()
            
            logger.info(f"Output saved: {output_id} -> {target_file}")
            return output_id
            
        except Exception as e:
            logger.error(f"Failed to save output: {e}")
            return ""
    
    def _create_image_thumbnail(self, image_path: Path, output_id: str) -> Tuple[Optional[Path], Optional[Tuple[int, int]]]:
        """Create thumbnail for image."""
        try:
            from PIL import Image
            
            with Image.open(image_path) as img:
                # Get original resolution
                resolution = img.size
                
                # Create thumbnail
                thumbnail_size = (256, 256)
                img.thumbnail(thumbnail_size, Image.Resampling.LANCZOS)
                
                # Save thumbnail
                thumbnail_path = self.thumbnails_dir / f"{output_id}_thumb.jpg"
                img.save(thumbnail_path, "JPEG", quality=85)
                
                return thumbnail_path, resolution
                
        except Exception as e:
            logger.warning(f"Failed to create image thumbnail: {e}")
            return None, None
    
    def _create_video_thumbnail(self, video_path: Path, output_id: str) -> Tuple[Optional[Path], Optional[Tuple[int, int]], Optional[float]]:
        """Create thumbnail for video."""
        try:
            import cv2
            
            # Open video
            cap = cv2.VideoCapture(str(video_path))
            
            # Get video properties
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps if fps > 0 else None
            
            resolution = (width, height)
            
            # Extract middle frame for thumbnail
            middle_frame = frame_count // 2
            cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame)
            
            ret, frame = cap.read()
            if ret:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Create thumbnail using PIL
                from PIL import Image
                img = Image.fromarray(frame_rgb)
                img.thumbnail((256, 256), Image.Resampling.LANCZOS)
                
                # Save thumbnail
                thumbnail_path = self.thumbnails_dir / f"{output_id}_thumb.jpg"
                img.save(thumbnail_path, "JPEG", quality=85)
                
                cap.release()
                return thumbnail_path, resolution, duration
            
            cap.release()
            return None, resolution, duration
            
        except Exception as e:
            logger.warning(f"Failed to create video thumbnail: {e}")
            return None, None, None
    
    def _extract_tags_from_prompt(self, prompt: str) -> List[str]:
        """Extract tags from prompt for categorization."""
        # Simple tag extraction based on common keywords
        tag_keywords = {
            'landscape': ['mountain', 'forest', 'ocean', 'lake', 'valley', 'desert', 'landscape'],
            'portrait': ['person', 'face', 'portrait', 'character', 'human'],
            'animal': ['cat', 'dog', 'bird', 'animal', 'wildlife'],
            'abstract': ['abstract', 'geometric', 'pattern', 'artistic'],
            'nature': ['tree', 'flower', 'plant', 'nature', 'garden'],
            'architecture': ['building', 'house', 'city', 'architecture', 'urban'],
            'fantasy': ['dragon', 'magic', 'fantasy', 'mythical', 'fairy'],
            'sci-fi': ['robot', 'space', 'futuristic', 'sci-fi', 'technology']
        }
        
        prompt_lower = prompt.lower()
        tags = []
        
        for tag, keywords in tag_keywords.items():
            if any(keyword in prompt_lower for keyword in keywords):
                tags.append(tag)
        
        return tags
    
    def get_outputs(self, 
                   output_type: Optional[OutputType] = None,
                   limit: int = 50,
                   offset: int = 0,
                   tags: Optional[List[str]] = None) -> List[OutputMetadata]:
        """
        Get outputs with optional filtering.
        
        Args:
            output_type: Filter by output type
            limit: Maximum number of outputs to return
            offset: Number of outputs to skip
            tags: Filter by tags
            
        Returns:
            List of output metadata
        """
        outputs = list(self.outputs_metadata.values())
        
        # Filter by output type
        if output_type:
            outputs = [o for o in outputs if o.output_type == output_type]
        
        # Filter by tags
        if tags:
            outputs = [o for o in outputs if any(tag in o.tags for tag in tags)]
        
        # Sort by creation date (newest first)
        outputs.sort(key=lambda x: x.created_at, reverse=True)
        
        # Apply pagination
        return outputs[offset:offset + limit]
    
    def get_output_by_id(self, output_id: str) -> Optional[OutputMetadata]:
        """Get output metadata by ID."""
        return self.outputs_metadata.get(output_id)
    
    def delete_output(self, output_id: str) -> bool:
        """Delete an output and its files."""
        try:
            metadata = self.outputs_metadata.get(output_id)
            if not metadata:
                logger.warning(f"Output not found: {output_id}")
                return False
            
            # Delete files
            if metadata.file_path.exists():
                metadata.file_path.unlink()
            
            if metadata.thumbnail_path and metadata.thumbnail_path.exists():
                metadata.thumbnail_path.unlink()
            
            # Remove from metadata
            del self.outputs_metadata[output_id]
            self._save_metadata()
            
            logger.info(f"Output deleted: {output_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete output {output_id}: {e}")
            return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get output statistics."""
        total_outputs = len(self.outputs_metadata)
        
        if total_outputs == 0:
            return {
                'total_outputs': 0,
                'by_type': {},
                'total_size_mb': 0,
                'avg_generation_time': 0
            }
        
        # Count by type
        by_type = {}
        total_size = 0
        total_time = 0
        
        for metadata in self.outputs_metadata.values():
            output_type = metadata.output_type.value
            by_type[output_type] = by_type.get(output_type, 0) + 1
            total_size += metadata.file_size_bytes
            total_time += metadata.generation_time
        
        return {
            'total_outputs': total_outputs,
            'by_type': by_type,
            'total_size_mb': total_size / (1024 * 1024),
            'avg_generation_time': total_time / total_outputs,
            'most_used_models': self._get_most_used_models(),
            'popular_tags': self._get_popular_tags()
        }
    
    def _get_most_used_models(self) -> Dict[str, int]:
        """Get most frequently used models."""
        model_counts = {}
        for metadata in self.outputs_metadata.values():
            model = metadata.model_used
            model_counts[model] = model_counts.get(model, 0) + 1
        
        # Sort by count and return top 5
        sorted_models = sorted(model_counts.items(), key=lambda x: x[1], reverse=True)
        return dict(sorted_models[:5])
    
    def _get_popular_tags(self) -> Dict[str, int]:
        """Get most popular tags."""
        tag_counts = {}
        for metadata in self.outputs_metadata.values():
            for tag in metadata.tags:
                tag_counts[tag] = tag_counts.get(tag, 0) + 1
        
        # Sort by count and return top 10
        sorted_tags = sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)
        return dict(sorted_tags[:10])
    
    def cleanup_old_outputs(self, days_old: int = 30) -> int:
        """Clean up outputs older than specified days."""
        cutoff_date = datetime.now() - timedelta(days=days_old)
        deleted_count = 0
        
        outputs_to_delete = []
        for output_id, metadata in self.outputs_metadata.items():
            if metadata.created_at < cutoff_date:
                outputs_to_delete.append(output_id)
        
        for output_id in outputs_to_delete:
            if self.delete_output(output_id):
                deleted_count += 1
        
        logger.info(f"Cleaned up {deleted_count} old outputs")
        return deleted_count


# Global output manager instance
_output_manager = None


def get_output_manager() -> OutputManager:
    """Get the global output manager instance."""
    global _output_manager
    if _output_manager is None:
        _output_manager = OutputManager()
    return _output_manager