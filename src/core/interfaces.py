"""
Core interfaces and abstract base classes for the Academic Multimodal LLM Experiment System.

This module defines the fundamental interfaces that all components must implement,
ensuring consistent behavior across the system while maintaining flexibility for
different implementations and hardware configurations.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
import datetime


class OutputType(Enum):
    """Types of content that can be generated."""
    IMAGE = "image"
    VIDEO = "video"
    TEXT = "text"
    MULTIMODAL = "multimodal"


class LicenseType(Enum):
    """Copyright license classifications for training data."""
    PUBLIC_DOMAIN = "public_domain"
    CREATIVE_COMMONS = "creative_commons"
    FAIR_USE_RESEARCH = "fair_use_research"
    COPYRIGHTED = "copyrighted"
    UNKNOWN = "unknown"


class ComplianceMode(Enum):
    """Copyright compliance modes for dataset selection."""
    OPEN_SOURCE_ONLY = "open_only"  # PD + CC content only
    RESEARCH_SAFE = "research_safe"  # PD + CC + Fair Use
    FULL_DATASET = "full_dataset"   # All content (research only)


@dataclass
class HardwareConfig:
    """Hardware configuration and capabilities."""
    vram_size: int  # VRAM in MB
    gpu_model: str
    cpu_cores: int
    ram_size: int  # RAM in MB
    cuda_available: bool
    optimization_level: str  # "aggressive", "balanced", "minimal"


@dataclass
class StyleConfig:
    """Style and generation parameters."""
    style_name: Optional[str] = None
    lora_path: Optional[Path] = None
    controlnet_config: Optional[Dict[str, Any]] = None
    generation_params: Dict[str, Any] = None


@dataclass
class ConversationContext:
    """Context for maintaining conversation state."""
    conversation_id: str
    history: List[Dict[str, Any]]
    current_mode: ComplianceMode
    user_preferences: Dict[str, Any]


@dataclass
class GenerationRequest:
    """Request for content generation."""
    prompt: str
    output_type: OutputType
    style_config: StyleConfig
    compliance_mode: ComplianceMode
    hardware_constraints: HardwareConfig
    context: ConversationContext
    negative_prompt: Optional[str] = None
    additional_params: Dict[str, Any] = None


@dataclass
class ContentItem:
    """Represents a piece of training content with copyright information."""
    url: str
    local_path: Path
    license_type: LicenseType
    attribution: str
    metadata: Dict[str, Any]
    copyright_status: str
    research_safe: bool


@dataclass
class GenerationResult:
    """Result of a content generation operation."""
    success: bool
    output_path: Optional[Path]
    generation_time: float
    model_used: str
    error_message: Optional[str] = None
    quality_metrics: Dict[str, float] = None
    compliance_info: Dict[str, Any] = None
    metadata: Dict[str, Any] = None


@dataclass
class ExperimentResult:
    """Complete experiment result with metadata."""
    experiment_id: str
    timestamp: datetime.datetime
    request: GenerationRequest
    result: GenerationResult
    notes: Optional[str] = None


class ILLMController(ABC):
    """Interface for the central LLM router and controller."""
    
    @abstractmethod
    def parse_request(self, prompt: str, context: ConversationContext) -> GenerationRequest:
        """Parse user input and create a structured generation request."""
        pass
    
    @abstractmethod
    def route_request(self, request: GenerationRequest) -> str:
        """Determine which pipeline should handle the request."""
        pass
    
    @abstractmethod
    def coordinate_workflow(self, steps: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Coordinate multi-step generation workflows."""
        pass
    
    @abstractmethod
    def manage_context(self, conversation_id: str) -> ConversationContext:
        """Manage conversation context and memory."""
        pass


class IGenerationPipeline(ABC):
    """Interface for content generation pipelines."""
    
    @abstractmethod
    def initialize(self, hardware_config: HardwareConfig) -> bool:
        """Initialize the pipeline with hardware-specific optimizations."""
        pass
    
    @abstractmethod
    def generate(self, request: GenerationRequest) -> GenerationResult:
        """Generate content based on the request."""
        pass
    
    @abstractmethod
    def optimize_for_hardware(self, hardware_config: HardwareConfig) -> None:
        """Apply hardware-specific optimizations."""
        pass
    
    @abstractmethod
    def cleanup(self) -> None:
        """Clean up resources and clear memory."""
        pass


class IMemoryManager(ABC):
    """Interface for memory and VRAM management."""
    
    @abstractmethod
    def optimize_model_loading(self, model_name: str) -> Dict[str, Any]:
        """Optimize model loading for available hardware."""
        pass
    
    @abstractmethod
    def manage_model_switching(self, current_model: str, next_model: str) -> None:
        """Efficiently switch between models."""
        pass
    
    @abstractmethod
    def clear_vram_cache(self) -> None:
        """Clear VRAM cache to free memory."""
        pass
    
    @abstractmethod
    def get_memory_status(self) -> Dict[str, Any]:
        """Get current memory usage statistics."""
        pass


class IDataManager(ABC):
    """Interface for copyright-aware data management."""
    
    @abstractmethod
    def collect_data(self, sources: List[str], license_filter: LicenseType) -> List[ContentItem]:
        """Collect data from sources with copyright classification."""
        pass
    
    @abstractmethod
    def organize_by_license(self, content_items: List[ContentItem]) -> Dict[LicenseType, List[ContentItem]]:
        """Organize content by license type."""
        pass
    
    @abstractmethod
    def create_training_dataset(self, compliance_mode: ComplianceMode) -> Dict[str, Any]:
        """Create training dataset based on compliance mode."""
        pass
    
    @abstractmethod
    def validate_compliance(self, content_item: ContentItem, compliance_mode: ComplianceMode) -> bool:
        """Validate if content item complies with the specified mode."""
        pass


class IHardwareDetector(ABC):
    """Interface for hardware detection and configuration."""
    
    @abstractmethod
    def detect_hardware(self) -> HardwareConfig:
        """Detect available hardware and create configuration."""
        pass
    
    @abstractmethod
    def get_optimization_strategy(self, hardware_config: HardwareConfig) -> str:
        """Determine optimal strategy for given hardware."""
        pass
    
    @abstractmethod
    def validate_requirements(self, hardware_config: HardwareConfig, requirements: Dict[str, Any]) -> bool:
        """Validate if hardware meets minimum requirements."""
        pass


class IExperimentTracker(ABC):
    """Interface for experiment tracking and management."""
    
    @abstractmethod
    def start_experiment(self, request: GenerationRequest) -> str:
        """Start a new experiment and return experiment ID."""
        pass
    
    @abstractmethod
    def log_result(self, experiment_id: str, result: GenerationResult) -> None:
        """Log experiment result."""
        pass
    
    @abstractmethod
    def save_experiment(self, experiment: ExperimentResult, notes: Optional[str] = None) -> None:
        """Save complete experiment with notes."""
        pass
    
    @abstractmethod
    def get_experiment_history(self, filters: Dict[str, Any] = None) -> List[ExperimentResult]:
        """Retrieve experiment history with optional filtering."""
        pass


class IComplianceEngine(ABC):
    """Interface for ethics and compliance enforcement."""
    
    @abstractmethod
    def classify_license(self, url: str, metadata: Dict[str, Any]) -> LicenseType:
        """Classify content license based on source and metadata."""
        pass
    
    @abstractmethod
    def validate_request(self, request: GenerationRequest) -> bool:
        """Validate if generation request complies with ethics guidelines."""
        pass
    
    @abstractmethod
    def get_attribution_info(self, content_items: List[ContentItem]) -> Dict[str, str]:
        """Generate attribution information for content."""
        pass
    
    @abstractmethod
    def audit_compliance(self, experiment: ExperimentResult) -> Dict[str, Any]:
        """Audit experiment for compliance violations."""
        pass


# System-wide exceptions
class SystemError(Exception):
    """Base class for system errors."""
    pass


class MemoryError(SystemError):
    """VRAM or RAM exhaustion error."""
    pass


class ModelLoadError(SystemError):
    """Model loading or switching failure."""
    pass


class ComplianceError(SystemError):
    """Copyright or ethics violation error."""
    pass


class GenerationError(SystemError):
    """Content generation failure."""
    pass


class HardwareError(SystemError):
    """Hardware detection or optimization error."""
    pass