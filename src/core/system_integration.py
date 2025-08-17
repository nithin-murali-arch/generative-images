"""
System Integration Module - Connects all components into a working system.

This module implements the integration layer that connects the LLM controller,
generation pipelines, hardware detection, and UI components into a cohesive
working system with proper request flow and error handling.
"""

import logging
import time
from typing import Dict, Any, Optional, List
from pathlib import Path
from dataclasses import asdict

try:
    # Try relative imports first (when used as module)
    from .interfaces import (
        GenerationRequest, GenerationResult, OutputType, ComplianceMode,
        HardwareConfig, ConversationContext, SystemError
    )
    from .llm_controller import LLMController
    from .generation_workflow import GenerationWorkflow, WorkflowManager
    from ..pipelines.image_generation import ImageGenerationPipeline
    from ..pipelines.video_generation import VideoGenerationPipeline
    from ..hardware.detector import HardwareDetector
    from ..hardware.memory_manager import MemoryManager
    from ..data.experiment_tracker import ExperimentTracker
except ImportError:
    # Fallback to absolute imports (when used as script)
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    
    from core.interfaces import (
        GenerationRequest, GenerationResult, OutputType, ComplianceMode,
        HardwareConfig, ConversationContext, SystemError
    )
    from core.llm_controller import LLMController
    from core.generation_workflow import GenerationWorkflow, WorkflowManager
    from pipelines.image_generation import ImageGenerationPipeline
    from pipelines.video_generation import VideoGenerationPipeline
    from hardware.detector import HardwareDetector
    from hardware.memory_manager import MemoryManager
    from data.experiment_tracker import ExperimentTracker

logger = logging.getLogger(__name__)


class SystemIntegration:
    """
    Main system integration class that coordinates all components.
    
    This class implements the request flow from UI → LLM Controller → Generation Pipeline
    and manages the integration between hardware detection, pipeline optimizations,
    and experiment tracking.
    """
    
    def __init__(self):
        """Initialize the system integration layer."""
        # Core components
        self.hardware_detector: Optional[HardwareDetector] = None
        self.hardware_config: Optional[HardwareConfig] = None
        self.memory_manager: Optional[MemoryManager] = None
        self.llm_controller: Optional[LLMController] = None
        
        # Generation pipelines
        self.image_pipeline: Optional[ImageGenerationPipeline] = None
        self.video_pipeline: Optional[VideoGenerationPipeline] = None
        
        # Data management
        self.experiment_tracker: Optional[ExperimentTracker] = None
        
        # Workflow management
        self.generation_workflow: Optional[GenerationWorkflow] = None
        self.workflow_manager: Optional[WorkflowManager] = None
        
        # System state
        self.is_initialized = False
        self.active_generations: Dict[str, Dict[str, Any]] = {}
        
        logger.info("SystemIntegration created")
    
    def initialize(self, config: Dict[str, Any]) -> bool:
        """
        Initialize all system components with proper integration.
        
        Args:
            config: System configuration dictionary
            
        Returns:
            bool: True if initialization successful
        """
        try:
            logger.info("Initializing system integration...")
            
            # Step 1: Initialize hardware detection
            if not self._initialize_hardware_detection(config):
                return False
            
            # Step 2: Initialize core components
            if not self._initialize_core_components(config):
                return False
            
            # Step 3: Initialize generation pipelines
            if not self._initialize_generation_pipelines():
                return False
            
            # Step 4: Initialize data management
            if not self._initialize_data_management(config):
                return False
            
            # Step 5: Initialize workflow management
            if not self._initialize_workflow_management(config):
                return False
            
            # Step 6: Connect components
            if not self._connect_components():
                return False
            
            self.is_initialized = True
            logger.info("System integration initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"System integration initialization failed: {e}")
            return False
    
    def _initialize_hardware_detection(self, config: Dict[str, Any]) -> bool:
        """Initialize hardware detection and configuration."""
        try:
            logger.info("Initializing hardware detection...")
            
            # Create hardware detector
            self.hardware_detector = HardwareDetector()
            
            # Detect hardware configuration
            if config.get("auto_detect_hardware", True):
                self.hardware_config = self.hardware_detector.detect_hardware()
                logger.info(f"Hardware detected: {self.hardware_config.gpu_model} "
                           f"({self.hardware_config.vram_size}MB VRAM)")
            else:
                # Use configuration from file
                hw_config = config.get("hardware_config")
                if hw_config:
                    self.hardware_config = HardwareConfig(**hw_config)
                    logger.info("Using hardware configuration from file")
                else:
                    # Fallback to detection
                    self.hardware_config = self.hardware_detector.detect_hardware()
                    logger.info("No hardware config found, falling back to detection")
            
            # Initialize memory manager
            self.memory_manager = MemoryManager(self.hardware_config)
            
            return True
            
        except Exception as e:
            logger.error(f"Hardware detection initialization failed: {e}")
            return False
    
    def _initialize_core_components(self, config: Dict[str, Any]) -> bool:
        """Initialize core system components."""
        try:
            logger.info("Initializing core components...")
            
            # Initialize LLM Controller
            self.llm_controller = LLMController(self.hardware_config)
            logger.info("LLM Controller initialized")
            
            return True
            
        except Exception as e:
            logger.error(f"Core components initialization failed: {e}")
            return False
    
    def _initialize_generation_pipelines(self) -> bool:
        """Initialize generation pipelines with hardware optimization."""
        try:
            logger.info("Initializing generation pipelines...")
            
            # Initialize Image Generation Pipeline
            self.image_pipeline = ImageGenerationPipeline()
            if not self.image_pipeline.initialize(self.hardware_config):
                logger.error("Failed to initialize image generation pipeline")
                return False
            
            # Initialize Video Generation Pipeline
            self.video_pipeline = VideoGenerationPipeline()
            if not self.video_pipeline.initialize(self.hardware_config):
                logger.error("Failed to initialize video generation pipeline")
                return False
            
            logger.info("Generation pipelines initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Generation pipelines initialization failed: {e}")
            return False
    
    def _initialize_data_management(self, config: Dict[str, Any]) -> bool:
        """Initialize data management components."""
        try:
            logger.info("Initializing data management...")
            
            # Initialize Experiment Tracker
            experiments_dir = Path(config.get("experiments_dir", "experiments"))
            experiments_dir.mkdir(parents=True, exist_ok=True)
            
            self.experiment_tracker = ExperimentTracker(
                db_path=experiments_dir / "experiments.db"
            )
            
            logger.info("Data management initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Data management initialization failed: {e}")
            return False
    
    def _initialize_workflow_management(self, config: Dict[str, Any]) -> bool:
        """Initialize workflow management components."""
        try:
            logger.info("Initializing workflow management...")
            
            # Initialize Generation Workflow
            self.generation_workflow = GenerationWorkflow(self)
            
            # Initialize Workflow Manager
            max_concurrent = config.get("max_concurrent_generations", 1)
            self.workflow_manager = WorkflowManager(self, max_concurrent)
            
            logger.info("Workflow management initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Workflow management initialization failed: {e}")
            return False
    
    def _connect_components(self) -> bool:
        """Connect all components and establish request flow."""
        try:
            logger.info("Connecting system components...")
            
            # Verify all components are initialized
            if not all([
                self.hardware_config,
                self.llm_controller,
                self.image_pipeline,
                self.video_pipeline,
                self.experiment_tracker
            ]):
                logger.error("Not all components are initialized")
                return False
            
            # Apply hardware optimizations to pipelines
            self.image_pipeline.optimize_for_hardware(self.hardware_config)
            self.video_pipeline.optimize_for_hardware(self.hardware_config)
            
            logger.info("System components connected successfully")
            return True
            
        except Exception as e:
            logger.error(f"Component connection failed: {e}")
            return False
    
    def execute_complete_generation_workflow(self, 
                                           prompt: str, 
                                           conversation_id: str = None,
                                           compliance_mode: ComplianceMode = ComplianceMode.RESEARCH_SAFE,
                                           additional_params: Optional[Dict[str, Any]] = None,
                                           progress_callback: Optional[callable] = None) -> GenerationResult:
        """
        Execute the complete generation workflow using the new workflow engine.
        
        This is the new primary method for generation that uses the complete workflow.
        
        Args:
            prompt: User's text prompt
            conversation_id: Optional conversation ID for context
            compliance_mode: Copyright compliance mode
            additional_params: Additional parameters for generation
            progress_callback: Optional callback for progress updates
            
        Returns:
            GenerationResult: Complete generation result
        """
        if not self.is_initialized or not self.generation_workflow:
            return GenerationResult(
                success=False,
                output_path=None,
                generation_time=0.0,
                model_used="none",
                error_message="System not initialized or workflow engine not available"
            )
        
        return self.generation_workflow.execute_complete_workflow(
            prompt=prompt,
            conversation_id=conversation_id,
            compliance_mode=compliance_mode,
            additional_params=additional_params,
            progress_callback=progress_callback
        )
    
    def process_generation_request(self, prompt: str, conversation_id: str, 
                                 additional_params: Optional[Dict[str, Any]] = None) -> GenerationResult:
        """
        Process a complete generation request through the system.
        
        This implements the full request flow:
        UI → LLM Controller → Generation Pipeline → Result
        
        Args:
            prompt: User's text prompt
            conversation_id: Conversation identifier for context
            additional_params: Additional parameters for generation
            
        Returns:
            GenerationResult: Complete generation result
        """
        if not self.is_initialized:
            return GenerationResult(
                success=False,
                output_path=None,
                generation_time=0.0,
                model_used="none",
                error_message="System not initialized"
            )
        
        start_time = time.time()
        generation_id = f"gen_{int(time.time() * 1000)}"
        
        try:
            logger.info(f"Processing generation request {generation_id}: '{prompt[:50]}...'")
            
            # Step 1: Get or create conversation context
            context = self.llm_controller.manage_context(conversation_id)
            
            # Step 2: Parse request using LLM Controller
            request = self.llm_controller.parse_request(prompt, context)
            
            # Apply additional parameters if provided
            if additional_params:
                if 'compliance_mode' in additional_params:
                    request.compliance_mode = ComplianceMode(additional_params['compliance_mode'])
                    context.current_mode = request.compliance_mode
                
                # Merge additional params
                if not request.additional_params:
                    request.additional_params = {}
                request.additional_params.update(additional_params)
            
            # Step 3: Route request to appropriate pipeline
            pipeline_name = self.llm_controller.route_request(request)
            
            # Track active generation
            self.active_generations[generation_id] = {
                'request': request,
                'pipeline': pipeline_name,
                'start_time': start_time,
                'status': 'processing'
            }
            
            # Step 4: Execute generation
            result = self._execute_generation(request, pipeline_name)
            
            # Step 5: Track experiment if successful
            if result.success and self.experiment_tracker:
                self._track_experiment(generation_id, request, result)
            
            # Update active generations
            self.active_generations[generation_id]['status'] = 'completed' if result.success else 'failed'
            self.active_generations[generation_id]['result'] = result
            
            logger.info(f"Generation request {generation_id} completed: {'success' if result.success else 'failed'}")
            
            return result
            
        except Exception as e:
            generation_time = time.time() - start_time
            logger.error(f"Generation request {generation_id} failed: {e}")
            
            # Update active generations
            if generation_id in self.active_generations:
                self.active_generations[generation_id]['status'] = 'failed'
                self.active_generations[generation_id]['error'] = str(e)
            
            return GenerationResult(
                success=False,
                output_path=None,
                generation_time=generation_time,
                model_used="unknown",
                error_message=str(e)
            )
    
    def _execute_generation(self, request: GenerationRequest, pipeline_name: str) -> GenerationResult:
        """Execute generation using the appropriate pipeline."""
        logger.info(f"Executing generation using {pipeline_name} pipeline")
        
        try:
            if pipeline_name == "image":
                return self.image_pipeline.generate(request)
            elif pipeline_name == "video":
                return self.video_pipeline.generate(request)
            elif pipeline_name == "multimodal":
                # For multimodal, coordinate both pipelines
                return self._execute_multimodal_generation(request)
            else:
                raise SystemError(f"Unknown pipeline: {pipeline_name}")
                
        except Exception as e:
            logger.error(f"Generation execution failed: {e}")
            raise
    
    def _execute_multimodal_generation(self, request: GenerationRequest) -> GenerationResult:
        """Execute multimodal generation coordinating multiple pipelines."""
        logger.info("Executing multimodal generation")
        
        try:
            # Create workflow steps for multimodal generation
            workflow_steps = [
                {
                    'type': 'generate',
                    'pipeline': 'image',
                    'prompt': request.prompt,
                    'params': request.style_config.generation_params if request.style_config else {},
                    'dependencies': []
                },
                {
                    'type': 'generate',
                    'pipeline': 'video',
                    'prompt': request.prompt,
                    'params': request.style_config.generation_params if request.style_config else {},
                    'dependencies': ['image']  # Video depends on image
                }
            ]
            
            # Coordinate workflow using LLM Controller
            workflow_result = self.llm_controller.coordinate_workflow(workflow_steps)
            
            # For now, execute image generation first
            image_result = self.image_pipeline.generate(request)
            
            if image_result.success:
                # Use image result as conditioning for video
                video_request = request
                if not video_request.additional_params:
                    video_request.additional_params = {}
                video_request.additional_params['conditioning_image'] = image_result.output_path
                
                video_result = self.video_pipeline.generate(video_request)
                
                # Return the video result as primary output
                return video_result
            else:
                return image_result
                
        except Exception as e:
            logger.error(f"Multimodal generation failed: {e}")
            raise
    
    def _track_experiment(self, generation_id: str, request: GenerationRequest, result: GenerationResult) -> None:
        """Track successful generation as an experiment."""
        try:
            experiment_data = {
                'generation_id': generation_id,
                'prompt': request.prompt,
                'output_type': request.output_type.value,
                'compliance_mode': request.compliance_mode.value,
                'model_used': result.model_used,
                'generation_time': result.generation_time,
                'output_path': str(result.output_path) if result.output_path else None,
                'quality_metrics': result.quality_metrics or {},
                'hardware_config': asdict(self.hardware_config),
                'style_config': asdict(request.style_config) if request.style_config else None
            }
            
            self.experiment_tracker.save_experiment(
                experiment_name=f"Generation_{generation_id}",
                experiment_data=experiment_data,
                tags=['auto_generated', request.output_type.value]
            )
            
            logger.debug(f"Experiment tracked for generation {generation_id}")
            
        except Exception as e:
            logger.warning(f"Failed to track experiment for generation {generation_id}: {e}")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        status = {
            'initialized': self.is_initialized,
            'hardware': asdict(self.hardware_config) if self.hardware_config else None,
            'active_generations': len(self.active_generations),
            'pipelines': {
                'image': {
                    'initialized': self.image_pipeline is not None and self.image_pipeline.is_initialized,
                    'current_model': self.image_pipeline.current_model if self.image_pipeline else None,
                    'available_models': self.image_pipeline.get_available_models() if self.image_pipeline else []
                },
                'video': {
                    'initialized': self.video_pipeline is not None and self.video_pipeline.is_initialized,
                    'current_model': self.video_pipeline.current_model if self.video_pipeline else None,
                    'available_models': self.video_pipeline.get_available_models() if self.video_pipeline else []
                }
            }
        }
        
        # Add memory status if available
        if self.memory_manager:
            try:
                memory_status = self.memory_manager.get_memory_status()
                status['memory'] = memory_status
            except Exception as e:
                logger.warning(f"Failed to get memory status: {e}")
                status['memory'] = {'error': str(e)}
        
        return status
    
    def switch_model(self, pipeline_type: str, model_name: str) -> bool:
        """
        Switch model in specified pipeline.
        
        Args:
            pipeline_type: Type of pipeline ('image' or 'video')
            model_name: Name of model to switch to
            
        Returns:
            bool: True if switch successful
        """
        try:
            if pipeline_type == 'image' and self.image_pipeline:
                return self.image_pipeline.switch_model(model_name)
            elif pipeline_type == 'video' and self.video_pipeline:
                return self.video_pipeline.switch_model(model_name)
            else:
                logger.error(f"Unknown pipeline type or pipeline not initialized: {pipeline_type}")
                return False
                
        except Exception as e:
            logger.error(f"Model switch failed: {e}")
            return False
    
    def clear_memory_cache(self) -> bool:
        """Clear VRAM and system memory caches."""
        try:
            logger.info("Clearing memory caches...")
            
            if self.memory_manager:
                self.memory_manager.clear_vram_cache()
            
            if self.image_pipeline:
                self.image_pipeline.cleanup()
            
            if self.video_pipeline:
                self.video_pipeline.cleanup()
            
            logger.info("Memory caches cleared successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to clear memory caches: {e}")
            return False
    
    def get_generation_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent generation history."""
        try:
            if not self.experiment_tracker:
                return []
            
            return self.experiment_tracker.get_recent_experiments(limit)
            
        except Exception as e:
            logger.error(f"Failed to get generation history: {e}")
            return []
    
    def cleanup(self) -> None:
        """Clean up all system resources."""
        logger.info("Cleaning up system integration...")
        
        try:
            # Clean up pipelines
            if self.image_pipeline:
                self.image_pipeline.cleanup()
            
            if self.video_pipeline:
                self.video_pipeline.cleanup()
            
            # Clean up LLM controller
            if self.llm_controller:
                for conv_id in list(self.llm_controller.conversations.keys()):
                    self.llm_controller.cleanup_conversation(conv_id)
            
            # Close experiment tracker
            if self.experiment_tracker:
                self.experiment_tracker.close()
            
            # Clear memory
            if self.memory_manager:
                self.memory_manager.clear_vram_cache()
            
            # Clear active generations
            self.active_generations.clear()
            
            self.is_initialized = False
            logger.info("System integration cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during system cleanup: {e}")