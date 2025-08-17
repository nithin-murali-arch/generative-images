"""
Research-focused user interface using Gradio for the Academic Multimodal LLM Experiment System.

This module provides a comprehensive interface for researchers to interact with the system,
including generation controls, copyright compliance management, and experiment tracking.
"""

import logging
import time
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
from dataclasses import dataclass
from enum import Enum

from .compliance_controls import ComplianceController, create_compliance_controller

try:
    from ..core.interfaces import (
        ComplianceMode, OutputType, GenerationRequest, StyleConfig, 
        ConversationContext, HardwareConfig, ExperimentResult
    )
except ImportError:
    # Fallback for when running as script or in tests
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from core.interfaces import (
        ComplianceMode, OutputType, GenerationRequest, StyleConfig, 
        ConversationContext, HardwareConfig, ExperimentResult
    )

logger = logging.getLogger(__name__)

# Try to import Gradio
try:
    import gradio as gr
    GRADIO_AVAILABLE = True
except ImportError:
    # Create a mock gr module for type hints and basic functionality
    class MockGradio:
        class Blocks:
            pass
        class themes:
            @staticmethod
            def Soft():
                return None
    
    gr = MockGradio()
    GRADIO_AVAILABLE = False
    logger.warning("Gradio not available - UI will be limited")

# Try to import PIL for image handling
try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    # Create a mock Image class for type hints
    class Image:
        class Image:
            pass
    PIL_AVAILABLE = False
    logger.warning("PIL not available - image handling limited")


class InterfaceTheme(Enum):
    """Available interface themes."""
    DEFAULT = "default"
    SOFT = "soft"
    MONOCHROME = "monochrome"


@dataclass
class UIState:
    """Current state of the user interface."""
    current_compliance_mode: ComplianceMode = ComplianceMode.RESEARCH_SAFE
    current_model: Optional[str] = None
    experiment_id: Optional[str] = None
    generation_history: List[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.generation_history is None:
            self.generation_history = []


class ResearchInterface:
    """
    Main research interface for the Academic Multimodal LLM Experiment System.
    
    Provides a tabbed Gradio interface with image generation, video generation,
    copyright compliance controls, and experiment tracking capabilities.
    """
    
    def __init__(self, system_controller=None, experiment_tracker=None, compliance_engine=None):
        """
        Initialize the research interface.
        
        Args:
            system_controller: Main system controller for generation requests
            experiment_tracker: Experiment tracking system
            compliance_engine: Copyright compliance engine
        """
        self.system_controller = system_controller
        self.experiment_tracker = experiment_tracker
        self.compliance_engine = compliance_engine
        
        # Initialize compliance controller
        self.compliance_controller = create_compliance_controller(
            compliance_engine=compliance_engine
        )
        
        self.ui_state = UIState()
        self.gradio_app = None
        self.is_initialized = False
        
        logger.info("ResearchInterface created")
    
    def initialize(self) -> bool:
        """
        Initialize the Gradio interface.
        
        Returns:
            bool: True if initialization successful
        """
        if not GRADIO_AVAILABLE:
            logger.error("Gradio not available - cannot initialize interface")
            return False
        
        try:
            logger.info("Initializing Gradio research interface")
            
            # Create the main Gradio interface
            self.gradio_app = self._create_gradio_interface()
            
            self.is_initialized = True
            logger.info("Research interface initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize research interface: {e}")
            return False
    
    def launch(self, share: bool = False, server_name: str = "127.0.0.1", server_port: int = 7860) -> None:
        """
        Launch the Gradio interface.
        
        Args:
            share: Whether to create a public link
            server_name: Server hostname
            server_port: Server port
        """
        if not self.is_initialized:
            logger.error("Interface not initialized - call initialize() first")
            return
        
        logger.info(f"Launching research interface on {server_name}:{server_port}")
        
        try:
            self.gradio_app.launch(
                share=share,
                server_name=server_name,
                server_port=server_port,
                show_error=True,
                quiet=False
            )
        except Exception as e:
            logger.error(f"Failed to launch interface: {e}")
    
    def _create_gradio_interface(self):
        """Create the main Gradio interface with tabs."""
        with gr.Blocks(
            title="Academic Multimodal LLM Research System",
            theme=gr.themes.Soft(),
            css=self._get_custom_css()
        ) as interface:
            
            # Header
            gr.Markdown("""
            # Academic Multimodal LLM Research System
            
            A research-focused platform for ethical experimentation with AI-powered image and video generation.
            """)
            
            # Global compliance mode selector
            with gr.Row():
                compliance_mode = gr.Dropdown(
                    choices=[mode.value for mode in ComplianceMode],
                    value=ComplianceMode.RESEARCH_SAFE.value,
                    label="Copyright Compliance Mode",
                    info="Controls which training data is used for generation"
                )
                
                current_model_display = gr.Textbox(
                    value="No model loaded",
                    label="Current Model",
                    interactive=False
                )
            
            # Main tabbed interface
            with gr.Tabs():
                
                # Image Generation Tab
                with gr.TabItem("Image Generation"):
                    image_components = self._create_image_generation_tab()
                
                # Video Generation Tab  
                with gr.TabItem("Video Generation"):
                    video_components = self._create_video_generation_tab()
                
                # Generated Outputs Gallery Tab
                with gr.TabItem("Generated Outputs"):
                    outputs_components = self._create_outputs_gallery_tab()
                
                # Experiment Tracking Tab
                with gr.TabItem("Experiment Tracking"):
                    experiment_components = self._create_experiment_tracking_tab()
                
                # Copyright Compliance Tab
                with gr.TabItem("Copyright & Compliance"):
                    compliance_components = self._create_compliance_tab()
                
                # System Status Tab
                with gr.TabItem("System Status"):
                    status_components = self._create_status_tab()
                
                # Generated Outputs Tab
                with gr.TabItem("Generated Outputs"):
                    outputs_components = self._create_outputs_tab()
            
            # Set up event handlers
            self._setup_event_handlers(
                compliance_mode, current_model_display,
                image_components, video_components, outputs_components,
                experiment_components, compliance_components, status_components
            )
        
        return interface
    
    def _create_image_generation_tab(self) -> Dict[str, Any]:
        """Create the image generation tab components."""
        components = {}
        
        with gr.Row():
            with gr.Column(scale=1):
                # Input controls
                components['prompt'] = gr.Textbox(
                    label="Prompt",
                    placeholder="Describe the image you want to generate...",
                    lines=3
                )
                
                components['negative_prompt'] = gr.Textbox(
                    label="Negative Prompt (Optional)",
                    placeholder="What to avoid in the image...",
                    lines=2
                )
                
                # Quality and Resolution Controls
                with gr.Accordion("Quality & Resolution Settings", open=True):
                    components['quality_preset'] = gr.Dropdown(
                        choices=[
                            "Draft (Fast, 10 steps)",
                            "Standard (Balanced, 20 steps)", 
                            "High Quality (Slow, 30 steps)",
                            "Ultra Quality (Very Slow, 50 steps)",
                            "Custom"
                        ],
                        value="Standard (Balanced, 20 steps)",
                        label="Quality Preset"
                    )
                    
                    components['resolution_preset'] = gr.Dropdown(
                        choices=[
                            "512x512 (Standard, Fast)",
                            "768x768 (High Quality)", 
                            "1024x1024 (Ultra High, Slow)",
                            "512x768 (Portrait)",
                            "768x512 (Landscape)",
                            "1024x768 (HD Landscape)",
                            "768x1024 (HD Portrait)",
                            "Custom"
                        ],
                        value="512x512 (Standard, Fast)",
                        label="Resolution Preset"
                    )
                    
                    with gr.Row():
                        components['width'] = gr.Slider(
                            minimum=256, maximum=1536, value=512, step=64,
                            label="Width"
                        )
                        components['height'] = gr.Slider(
                            minimum=256, maximum=1536, value=512, step=64,
                            label="Height"
                        )
                    
                    with gr.Row():
                        components['steps'] = gr.Slider(
                            minimum=1, maximum=100, value=20, step=1,
                            label="Inference Steps"
                        )
                        components['guidance_scale'] = gr.Slider(
                            minimum=1.0, maximum=20.0, value=7.5, step=0.5,
                            label="Guidance Scale"
                        )
                
                # Advanced GPU Settings
                with gr.Accordion("GPU & Performance Settings", open=False):
                    components['use_gpu'] = gr.Checkbox(
                        label="Force GPU Usage",
                        value=True,
                        info="Uncheck to use CPU (slower but uses less VRAM)"
                    )
                    
                    components['precision'] = gr.Dropdown(
                        choices=[
                            "float16 (Faster, Less VRAM)",
                            "float32 (Slower, More VRAM, Better Quality)"
                        ],
                        value="float16 (Faster, Less VRAM)",
                        label="Precision"
                    )
                    
                    components['memory_optimization'] = gr.Dropdown(
                        choices=[
                            "None (Fastest, Most VRAM)",
                            "Attention Slicing (Balanced)",
                            "CPU Offloading (Slowest, Least VRAM)"
                        ],
                        value="Attention Slicing (Balanced)",
                        label="Memory Optimization"
                    )
                    
                    components['batch_size'] = gr.Slider(
                        minimum=1, maximum=4, value=1, step=1,
                        label="Batch Size (Multiple Images)"
                    )
                
                components['seed'] = gr.Number(
                    label="Seed (Optional)",
                    value=None,
                    precision=0
                )
                
                # Model selection
                components['model_selector'] = gr.Dropdown(
                    choices=["stable-diffusion-v1-5", "sdxl-turbo", "flux.1-schnell"],
                    value="stable-diffusion-v1-5",
                    label="Image Model"
                )
                
                # Generation button
                components['generate_btn'] = gr.Button(
                    "Generate Image",
                    variant="primary",
                    size="lg"
                )
            
            with gr.Column(scale=1):
                # Output display
                components['output_image'] = gr.Image(
                    label="Generated Image",
                    type="pil"
                )
                
                # Generation info
                components['generation_info'] = gr.JSON(
                    label="Generation Information",
                    visible=False
                )
                
                # Progress and status
                components['progress'] = gr.Progress()
                components['status'] = gr.Textbox(
                    label="Status",
                    value="Ready",
                    interactive=False
                )
        
        # Research notes section
        with gr.Row():
            components['research_notes'] = gr.Textbox(
                label="Research Notes",
                placeholder="Add notes about this generation for your research...",
                lines=3
            )
            
            components['save_experiment_btn'] = gr.Button(
                "Save Experiment",
                variant="secondary"
            )
        
        return components
    
    def _create_video_generation_tab(self) -> Dict[str, Any]:
        """Create the video generation tab components."""
        components = {}
        
        with gr.Row():
            with gr.Column(scale=1):
                # Input controls
                components['prompt'] = gr.Textbox(
                    label="Prompt",
                    placeholder="Describe the video you want to generate...",
                    lines=3
                )
                
                components['conditioning_image'] = gr.Image(
                    label="Conditioning Image (Optional)",
                    type="pil"
                )
                
                with gr.Row():
                    components['width'] = gr.Slider(
                        minimum=256, maximum=1024, value=512, step=64,
                        label="Width"
                    )
                    components['height'] = gr.Slider(
                        minimum=256, maximum=1024, value=512, step=64,
                        label="Height"
                    )
                
                with gr.Row():
                    components['num_frames'] = gr.Slider(
                        minimum=4, maximum=25, value=14, step=1,
                        label="Number of Frames"
                    )
                    components['fps'] = gr.Slider(
                        minimum=1, maximum=30, value=7, step=1,
                        label="FPS"
                    )
                
                with gr.Row():
                    components['steps'] = gr.Slider(
                        minimum=1, maximum=50, value=25, step=1,
                        label="Inference Steps"
                    )
                    components['guidance_scale'] = gr.Slider(
                        minimum=1.0, maximum=20.0, value=7.5, step=0.5,
                        label="Guidance Scale"
                    )
                
                # Motion controls
                components['motion_bucket_id'] = gr.Slider(
                    minimum=1, maximum=255, value=127, step=1,
                    label="Motion Intensity"
                )
                
                components['seed'] = gr.Number(
                    label="Seed (Optional)",
                    value=None,
                    precision=0
                )
                
                # Model selection
                components['model_selector'] = gr.Dropdown(
                    choices=["stable-video-diffusion", "animatediff", "i2vgen-xl"],
                    value="stable-video-diffusion",
                    label="Video Model"
                )
                
                # Generation button
                components['generate_btn'] = gr.Button(
                    "Generate Video",
                    variant="primary",
                    size="lg"
                )
            
            with gr.Column(scale=1):
                # Output display
                components['output_video'] = gr.Video(
                    label="Generated Video"
                )
                
                # Generation info
                components['generation_info'] = gr.JSON(
                    label="Generation Information",
                    visible=False
                )
                
                # Progress and status
                components['progress'] = gr.Progress()
                components['status'] = gr.Textbox(
                    label="Status",
                    value="Ready",
                    interactive=False
                )
        
        # Research notes section
        with gr.Row():
            components['research_notes'] = gr.Textbox(
                label="Research Notes",
                placeholder="Add notes about this video generation for your research...",
                lines=3
            )
            
            components['save_experiment_btn'] = gr.Button(
                "Save Experiment",
                variant="secondary"
            )
        
        return components
    
    def _create_experiment_tracking_tab(self) -> Dict[str, Any]:
        """Create the experiment tracking tab components."""
        components = {}
        
        with gr.Row():
            with gr.Column(scale=2):
                # Experiment history
                components['experiment_history'] = gr.Dataframe(
                    headers=["ID", "Timestamp", "Type", "Model", "Prompt", "Status"],
                    label="Experiment History",
                    interactive=False
                )
                
                # Refresh button
                components['refresh_btn'] = gr.Button(
                    "Refresh History",
                    variant="secondary"
                )
            
            with gr.Column(scale=1):
                # Experiment details
                components['experiment_details'] = gr.JSON(
                    label="Experiment Details",
                    visible=True
                )
                
                # Performance metrics
                components['performance_metrics'] = gr.Plot(
                    label="Performance Metrics"
                )
        
        # Export controls
        with gr.Row():
            components['export_format'] = gr.Dropdown(
                choices=["CSV", "JSON", "PDF Report"],
                value="CSV",
                label="Export Format"
            )
            
            components['export_btn'] = gr.Button(
                "Export Experiments",
                variant="secondary"
            )
            
            components['download_file'] = gr.File(
                label="Download",
                visible=False
            )
        
        return components
    
    def _create_compliance_tab(self) -> Dict[str, Any]:
        """Create the copyright compliance tab components."""
        components = {}
        
        # Create compliance info panel
        self.compliance_controller.create_compliance_info_panel()
        
        # Get compliance components from controller
        compliance_components = self.compliance_controller.create_compliance_components()
        components.update(compliance_components)
        
        # Additional components for compliance management
        with gr.Row():
            with gr.Column():
                # Use components from compliance controller
                if 'dataset_stats' in compliance_components:
                    components['dataset_stats'] = compliance_components['dataset_stats']
                
                if 'model_licenses' in compliance_components:
                    components['model_licenses'] = compliance_components['model_licenses']
            
            with gr.Column():
                if 'attribution_info' in compliance_components:
                    components['attribution_info'] = compliance_components['attribution_info']
                
                if 'compliance_check_btn' in compliance_components:
                    components['compliance_check_btn'] = compliance_components['compliance_check_btn']
                
                if 'compliance_results' in compliance_components:
                    components['compliance_results'] = compliance_components['compliance_results']
        
        return components
    
    def _create_status_tab(self) -> Dict[str, Any]:
        """Create the system status tab components."""
        components = {}
        
        with gr.Row():
            with gr.Column():
                # Hardware status
                components['hardware_status'] = gr.JSON(
                    label="Hardware Status",
                    value={
                        "gpu_model": "Unknown",
                        "vram_total": 0,
                        "vram_used": 0,
                        "vram_free": 0,
                        "cpu_usage": 0,
                        "ram_usage": 0
                    }
                )
                
                # Model status
                components['model_status'] = gr.JSON(
                    label="Loaded Models",
                    value={}
                )
            
            with gr.Column():
                # Performance monitoring
                components['performance_plot'] = gr.Plot(
                    label="System Performance"
                )
                
                # System logs
                components['system_logs'] = gr.Textbox(
                    label="System Logs",
                    lines=10,
                    interactive=False,
                    max_lines=100
                )
        
        # Control buttons
        with gr.Row():
            components['refresh_status_btn'] = gr.Button(
                "Refresh Status",
                variant="secondary"
            )
            
            components['clear_cache_btn'] = gr.Button(
                "Clear VRAM Cache",
                variant="secondary"
            )
            
            components['restart_system_btn'] = gr.Button(
                "Restart System",
                variant="primary"
            )
        
        return components
    
    def _setup_event_handlers(self, compliance_mode, current_model_display, 
                            image_components, video_components, outputs_components,
                            experiment_components, compliance_components, status_components):
        """Set up event handlers for all interface components."""
        
        # Global compliance mode change
        compliance_mode.change(
            fn=self._on_compliance_mode_change,
            inputs=[compliance_mode],
            outputs=[current_model_display, compliance_components['dataset_stats']]
        )
        
        # Quality preset change handler
        image_components['quality_preset'].change(
            fn=self._on_quality_preset_change,
            inputs=[image_components['quality_preset']],
            outputs=[image_components['steps'], image_components['guidance_scale']]
        )
        
        # Resolution preset change handler
        image_components['resolution_preset'].change(
            fn=self._on_resolution_preset_change,
            inputs=[image_components['resolution_preset']],
            outputs=[image_components['width'], image_components['height']]
        )
        
        # Image generation
        image_components['generate_btn'].click(
            fn=self._generate_image_with_progress,
            inputs=[
                image_components['prompt'],
                image_components['negative_prompt'],
                image_components['width'],
                image_components['height'],
                image_components['steps'],
                image_components['guidance_scale'],
                image_components['seed'],
                image_components['model_selector'],
                image_components['use_gpu'],
                image_components['precision'],
                image_components['memory_optimization'],
                image_components['batch_size'],
                compliance_mode
            ],
            outputs=[
                image_components['output_image'],
                image_components['generation_info'],
                image_components['status']
            ]
        )
        
        # Video generation
        video_components['generate_btn'].click(
            fn=self._generate_video_with_progress,
            inputs=[
                video_components['prompt'],
                video_components['conditioning_image'],
                video_components['width'],
                video_components['height'],
                video_components['num_frames'],
                video_components['fps'],
                video_components['steps'],
                video_components['guidance_scale'],
                video_components['motion_bucket_id'],
                video_components['seed'],
                video_components['model_selector'],
                compliance_mode
            ],
            outputs=[
                video_components['output_video'],
                video_components['generation_info'],
                video_components['status']
            ]
        )
        
        # Experiment saving
        image_components['save_experiment_btn'].click(
            fn=self._save_experiment,
            inputs=[image_components['research_notes']],
            outputs=[image_components['status']]
        )
        
        video_components['save_experiment_btn'].click(
            fn=self._save_experiment,
            inputs=[video_components['research_notes']],
            outputs=[video_components['status']]
        )
        
        # Experiment tracking
        experiment_components['refresh_btn'].click(
            fn=self._refresh_experiment_history,
            outputs=[experiment_components['experiment_history']]
        )
        
        # Outputs gallery
        outputs_components['refresh_gallery_btn'].click(
            fn=self._refresh_outputs_gallery,
            inputs=[
                outputs_components['output_type_filter'],
                outputs_components['tag_filter']
            ],
            outputs=[
                outputs_components['outputs_gallery'],
                outputs_components['output_stats']
            ]
        )
        
        outputs_components['output_type_filter'].change(
            fn=self._refresh_outputs_gallery,
            inputs=[
                outputs_components['output_type_filter'],
                outputs_components['tag_filter']
            ],
            outputs=[
                outputs_components['outputs_gallery'],
                outputs_components['output_stats']
            ]
        )
        
        # Status monitoring
        status_components['refresh_status_btn'].click(
            fn=self._refresh_system_status,
            outputs=[
                status_components['hardware_status'],
                status_components['model_status'],
                status_components['system_logs']
            ]
        )
        
        status_components['clear_cache_btn'].click(
            fn=self._clear_vram_cache,
            outputs=[status_components['hardware_status']]
        )
    
    def _get_custom_css(self) -> str:
        """Get custom CSS for the interface."""
        return """
        .gradio-container {
            max-width: 1200px !important;
        }
        
        .tab-nav {
            background: linear-gradient(90deg, #f0f0f0, #e0e0e0);
        }
        
        .compliance-warning {
            background-color: #fff3cd;
            border: 1px solid #ffeaa7;
            border-radius: 4px;
            padding: 10px;
            margin: 10px 0;
        }
        
        .status-ready {
            color: #28a745;
        }
        
        .status-generating {
            color: #ffc107;
        }
        
        .status-error {
            color: #dc3545;
        }
        """
    
    # Event handler methods (placeholder implementations)
    
    def _on_compliance_mode_change(self, mode: str) -> Tuple[str, Dict[str, int]]:
        """Handle compliance mode change."""
        # Update compliance controller
        success, message = self.compliance_controller.set_compliance_mode(mode)
        
        if success:
            self.ui_state.current_compliance_mode = ComplianceMode(mode)
            logger.info(f"Compliance mode changed to: {mode}")
            
            # Get updated dataset stats from controller
            dataset_stats = self.compliance_controller.get_dataset_stats()
            
            # Get available models for this mode
            available_models = self.compliance_controller.get_available_models()
            model_display = f"Compliance mode: {mode} | Available models: {len(available_models)}"
            
            return model_display, dataset_stats
        else:
            logger.error(f"Failed to change compliance mode: {message}")
            return f"Error: {message}", self.compliance_controller.get_dataset_stats()
    
    def _generate_image(self, prompt: str, negative_prompt: str, width: int, height: int,
                       steps: int, guidance_scale: float, seed: Optional[int], 
                       model: str, use_gpu: bool, precision: str, memory_optimization: str,
                       batch_size: int, compliance_mode: str) -> Tuple[Optional[Any], Dict[str, Any], str]:
        """Handle image generation request."""
        try:
            logger.info(f"Generating image with prompt: {prompt[:50]}...")
            
            # Validate request with compliance controller
            is_valid, validation_message = self.compliance_controller.validate_generation_request(model, prompt)
            if not is_valid:
                logger.warning(f"Generation request failed validation: {validation_message}")
                return None, {}, f"Compliance validation failed: {validation_message}"
            
            # Check if model is available in current compliance mode
            available_models = self.compliance_controller.get_available_models()
            if model not in available_models:
                logger.warning(f"Model {model} not available in {compliance_mode} mode")
                return None, {}, f"Model {model} not available in {compliance_mode} mode. Available models: {', '.join(available_models)}"
            
            # Use real system controller if available
            if self.system_controller and hasattr(self.system_controller, 'execute_complete_generation_workflow'):
                # Prepare additional parameters with GPU and quality settings
                additional_params = {
                    'model_name': model,
                    'width': width,
                    'height': height,
                    'num_inference_steps': steps,
                    'guidance_scale': guidance_scale,
                    'negative_prompt': negative_prompt if negative_prompt else None,
                    'seed': seed if seed is not None else None,
                    'num_images_per_prompt': batch_size,
                    'force_gpu_usage': use_gpu,
                    'precision': 'float16' if 'float16' in precision else 'float32',
                    'memory_optimization': memory_optimization,
                    'force_real_generation': True  # Force real generation instead of mock
                }
                
                # Execute generation using the complete workflow
                result = self.system_controller.execute_complete_generation_workflow(
                    prompt=prompt,
                    conversation_id=f"ui_session_{int(time.time())}",
                    compliance_mode=ComplianceMode(compliance_mode),
                    additional_params=additional_params
                )
                
                if result.success:
                    # Save output using output manager
                    parameters = {
                        "width": width,
                        "height": height,
                        "steps": steps,
                        "guidance_scale": guidance_scale,
                        "seed": seed,
                        "use_gpu": use_gpu,
                        "precision": precision,
                        "memory_optimization": memory_optimization,
                        "batch_size": batch_size
                    }
                    
                    output_id = self._save_generation_output(
                        result, prompt, result.model_used, result.generation_time, 
                        parameters, "image"
                    )
                    
                    # Load the generated image
                    image = None
                    if result.output_path and PIL_AVAILABLE:
                        try:
                            from PIL import Image as PILImage
                            image = PILImage.open(result.output_path)
                        except Exception as e:
                            logger.warning(f"Failed to load generated image: {e}")
                    
                    # Get model license info for compliance tracking
                    model_info = self.compliance_controller.get_model_license_info(model)
                    
                    generation_info = {
                        "output_id": output_id,
                        "model": result.model_used,
                        "model_license": model_info.license_type if model_info else "Unknown",
                        "prompt": prompt,
                        "parameters": parameters,
                        "compliance_mode": compliance_mode,
                        "compliance_validation": validation_message,
                        "attribution_required": model_info.attribution_required if model_info else False,
                        "generation_time": result.generation_time,
                        "output_path": str(result.output_path) if result.output_path else None,
                        "quality_metrics": result.quality_metrics or {}
                    }
                    
                    return image, generation_info, f"Image generated successfully in {result.generation_time:.2f}s (ID: {output_id})"
                else:
                    return None, {}, f"Generation failed: {result.error_message}"
            
            else:
                # Fallback to mock generation for testing
                logger.warning("System controller not available, using mock generation")
                
                if PIL_AVAILABLE:
                    # Create a placeholder image
                    import PIL.Image
                    image = PIL.Image.new('RGB', (width, height), color='lightblue')
                else:
                    image = None
                
                # Get model license info for compliance tracking
                model_info = self.compliance_controller.get_model_license_info(model)
                
                generation_info = {
                    "model": model,
                    "model_license": model_info.license_type if model_info else "Unknown",
                    "prompt": prompt,
                    "parameters": {
                        "width": width,
                        "height": height,
                        "steps": steps,
                        "guidance_scale": guidance_scale,
                        "seed": seed
                    },
                    "compliance_mode": compliance_mode,
                    "compliance_validation": validation_message,
                    "attribution_required": model_info.attribution_required if model_info else False,
                    "generation_time": 2.5,
                    "mock_generation": True
                }
                
                return image, generation_info, "Mock image generated successfully (system controller not available)"
            
        except Exception as e:
            logger.error(f"Image generation failed: {e}")
            return None, {}, f"Generation failed: {str(e)}"
    
    def _generate_video(self, prompt: str, conditioning_image: Optional[Any], 
                       width: int, height: int, num_frames: int, fps: int,
                       steps: int, guidance_scale: float, motion_bucket_id: int,
                       seed: Optional[int], model: str, compliance_mode: str) -> Tuple[Optional[str], Dict[str, Any], str]:
        """Handle video generation request."""
        try:
            logger.info(f"Generating video with prompt: {prompt[:50]}...")
            
            # Validate request with compliance controller
            is_valid, validation_message = self.compliance_controller.validate_generation_request(model, prompt)
            if not is_valid:
                logger.warning(f"Video generation request failed validation: {validation_message}")
                return None, {}, f"Compliance validation failed: {validation_message}"
            
            # Check if model is available in current compliance mode
            available_models = self.compliance_controller.get_available_models()
            if model not in available_models:
                logger.warning(f"Model {model} not available in {compliance_mode} mode")
                return None, {}, f"Model {model} not available in {compliance_mode} mode. Available models: {', '.join(available_models)}"
            
            # Use real system controller if available
            if self.system_controller and hasattr(self.system_controller, 'execute_complete_generation_workflow'):
                # Prepare additional parameters
                additional_params = {
                    'model_name': model,
                    'width': width,
                    'height': height,
                    'num_frames': num_frames,
                    'fps': fps,
                    'num_inference_steps': steps,
                    'guidance_scale': guidance_scale,
                    'motion_bucket_id': motion_bucket_id,
                    'seed': seed if seed is not None else None,
                    'output_type': 'video'
                }
                
                # Add conditioning image if provided
                if conditioning_image is not None:
                    additional_params['conditioning_image'] = conditioning_image
                
                # Execute generation using the complete workflow
                result = self.system_controller.execute_complete_generation_workflow(
                    prompt=prompt,
                    conversation_id=f"ui_session_{int(time.time())}",
                    compliance_mode=ComplianceMode(compliance_mode),
                    additional_params=additional_params
                )
                
                if result.success:
                    # Save output using output manager
                    parameters = {
                        "width": width,
                        "height": height,
                        "num_frames": num_frames,
                        "fps": fps,
                        "steps": steps,
                        "guidance_scale": guidance_scale,
                        "motion_bucket_id": motion_bucket_id,
                        "seed": seed
                    }
                    
                    output_id = self._save_generation_output(
                        result, prompt, result.model_used, result.generation_time, 
                        parameters, "video"
                    )
                    
                    # Get model license info for compliance tracking
                    model_info = self.compliance_controller.get_model_license_info(model)
                    
                    generation_info = {
                        "output_id": output_id,
                        "model": result.model_used,
                        "model_license": model_info.license_type if model_info else "Unknown",
                        "prompt": prompt,
                        "parameters": parameters,
                        "compliance_mode": compliance_mode,
                        "compliance_validation": validation_message,
                        "attribution_required": model_info.attribution_required if model_info else False,
                        "generation_time": result.generation_time,
                        "output_path": str(result.output_path) if result.output_path else None,
                        "quality_metrics": result.quality_metrics or {}
                    }
                    
                    return str(result.output_path) if result.output_path else None, generation_info, f"Video generated successfully in {result.generation_time:.2f}s (ID: {output_id})"
                else:
                    return None, {}, f"Generation failed: {result.error_message}"
            
            else:
                # Fallback to mock generation for testing
                logger.warning("System controller not available, using mock video generation")
                
                # Get model license info for compliance tracking
                model_info = self.compliance_controller.get_model_license_info(model)
                
                generation_info = {
                    "model": model,
                    "model_license": model_info.license_type if model_info else "Unknown",
                    "prompt": prompt,
                    "parameters": {
                        "width": width,
                        "height": height,
                        "num_frames": num_frames,
                        "fps": fps,
                        "steps": steps,
                        "guidance_scale": guidance_scale,
                        "motion_bucket_id": motion_bucket_id,
                        "seed": seed
                    },
                    "compliance_mode": compliance_mode,
                    "compliance_validation": validation_message,
                    "attribution_required": model_info.attribution_required if model_info else False,
                    "generation_time": 15.0,
                    "mock_generation": True
                }
                
                return None, generation_info, "Mock video generation completed (system controller not available)"
            
        except Exception as e:
            logger.error(f"Video generation failed: {e}")
            return None, {}, f"Generation failed: {str(e)}"
            
            # Check if model is available in current compliance mode
            available_models = self.compliance_controller.get_available_models()
            if model not in available_models:
                logger.warning(f"Model {model} not available in {compliance_mode} mode")
                return None, {}, f"Model {model} not available in {compliance_mode} mode. Available models: {', '.join(available_models)}"
            
            # Create mock result for now
            video_path = None  # Would be actual video path
            
            # Get model license info for compliance tracking
            model_info = self.compliance_controller.get_model_license_info(model)
            
            generation_info = {
                "model": model,
                "model_license": model_info.license_type if model_info else "Unknown",
                "prompt": prompt,
                "parameters": {
                    "width": width,
                    "height": height,
                    "num_frames": num_frames,
                    "fps": fps,
                    "steps": steps,
                    "guidance_scale": guidance_scale,
                    "motion_bucket_id": motion_bucket_id,
                    "seed": seed
                },
                "compliance_mode": compliance_mode,
                "compliance_validation": validation_message,
                "attribution_required": model_info.attribution_required if model_info else False,
                "generation_time": 45.2
            }
            
            return video_path, generation_info, "Video generated successfully"
    
    def _save_experiment(self, research_notes: str) -> str:
        """Save experiment with research notes."""
        try:
            if self.experiment_tracker:
                # This would save the current experiment
                logger.info("Experiment saved with research notes")
                return "Experiment saved successfully"
            else:
                return "Experiment tracker not available"
        except Exception as e:
            logger.error(f"Failed to save experiment: {e}")
            return f"Failed to save experiment: {str(e)}"
    
    def _refresh_experiment_history(self) -> List[List[str]]:
        """Refresh experiment history display."""
        try:
            if self.experiment_tracker:
                experiments = self.experiment_tracker.get_experiment_history(limit=50)
                
                # Convert to table format
                table_data = []
                for exp in experiments:
                    table_data.append([
                        exp.get('id', '')[:8],  # Short ID
                        exp.get('timestamp', ''),
                        exp.get('output_type', ''),
                        exp.get('model_used', ''),
                        exp.get('prompt', '')[:50] + '...' if len(exp.get('prompt', '')) > 50 else exp.get('prompt', ''),
                        'Success' if exp.get('success') else 'Failed'
                    ])
                
                return table_data
            else:
                return [["No experiment tracker available", "", "", "", "", ""]]
                
        except Exception as e:
            logger.error(f"Failed to refresh experiment history: {e}")
            return [["Error loading experiments", "", "", "", "", ""]]
            
            return video_path, generation_info, "Video generated successfully"
    
    def _refresh_experiment_history(self) -> List[List[str]]:
        """Refresh experiment history display."""
        try:
            # Mock data for now
            return [
                ["exp_001", "2024-01-15 10:30", "Image", "SD 1.5", "A cat in space", "Complete"],
                ["exp_002", "2024-01-15 11:15", "Video", "SVD", "Ocean waves", "Complete"],
                ["exp_003", "2024-01-15 12:00", "Image", "SDXL", "Mountain landscape", "Failed"]
            ]
        except Exception as e:
            logger.error(f"Failed to refresh experiment history: {e}")
            return []
    
    def _refresh_system_status(self) -> Tuple[Dict[str, Any], Dict[str, Any], str]:
        """Refresh system status information."""
        try:
            hardware_status = {
                "gpu_model": "RTX 3070",
                "vram_total": 8192,
                "vram_used": 2048,
                "vram_free": 6144,
                "cpu_usage": 25.5,
                "ram_usage": 45.2
            }
            
            model_status = {
                "image_pipeline": "stable-diffusion-v1-5",
                "video_pipeline": "stable-video-diffusion",
                "llm_controller": "llama-3.1-8b"
            }
            
            logs = "System initialized successfully\nModels loaded\nReady for generation"
            
            return hardware_status, model_status, logs
            
        except Exception as e:
            logger.error(f"Failed to refresh system status: {e}")
            return {}, {}, f"Status refresh failed: {str(e)}"
    
    def _clear_vram_cache(self) -> Dict[str, Any]:
        """Clear VRAM cache and return updated status."""
        try:
            logger.info("Clearing VRAM cache")
            # Implementation would clear actual cache
            return {
                "gpu_model": "RTX 3070",
                "vram_total": 8192,
                "vram_used": 512,  # Reduced after clearing
                "vram_free": 7680,
                "cpu_usage": 15.2,
                "ram_usage": 35.1
            }
        except Exception as e:
            logger.error(f"Failed to clear VRAM cache: {e}")
            return {}


# Mock classes for when dependencies aren't available
class MockImage:
    """Mock image class when PIL isn't available."""
    def __init__(self, width=512, height=512):
        self.size = (width, height)


# Utility functions for interface creation
def create_research_interface(system_controller=None, experiment_tracker=None, compliance_engine=None) -> ResearchInterface:
    """
    Create and initialize a research interface.
    
    Args:
        system_controller: Main system controller
        experiment_tracker: Experiment tracking system
        compliance_engine: Copyright compliance engine
        
    Returns:
        ResearchInterface: Initialized interface
    """
    interface = ResearchInterface(system_controller, experiment_tracker, compliance_engine)
    
    if interface.initialize():
        logger.info("Research interface created and initialized successfully")
        return interface
    else:
        logger.error("Failed to initialize research interface")
        return None


def launch_research_interface(interface: ResearchInterface, **kwargs) -> None:
    """
    Launch a research interface.
    
    Args:
        interface: ResearchInterface to launch
        **kwargs: Additional arguments for launch
    """
    if interface and interface.is_initialized:
        interface.launch(**kwargs)
    else:
        logger.error("Cannot launch uninitialized interface")    
    
def _generate_image_with_progress(self, prompt: str, negative_prompt: str, width: int, height: int,
                                    steps: int, guidance_scale: float, seed: Optional[int], 
                                    model: str, use_gpu: bool, precision: str, memory_optimization: str,
                                    batch_size: int, compliance_mode: str) -> Tuple[Optional[Any], Dict[str, Any], str]:
        """Handle image generation with progress updates."""
        # Create a progress callback that updates the UI status
        progress_updates = []
        
        def progress_callback(progress_info):
            """Callback to track progress updates."""
            progress_updates.append(progress_info)
            # In a real implementation, this would update the UI in real-time
            logger.info(f"Generation progress: {progress_info.completed_steps}/{progress_info.total_steps}")
        
        # Update status to show generation is starting
        yield None, {}, "Starting image generation..."
        
        # Call the actual generation method
        result = self._generate_image(
            prompt, negative_prompt, width, height, steps, 
            guidance_scale, seed, model, use_gpu, precision, 
            memory_optimization, batch_size, compliance_mode
        )
        
        # Return the final result
        yield result
    
    def _generate_video_with_progress(self, prompt: str, conditioning_image: Optional[Any], 
                                    width: int, height: int, num_frames: int, fps: int,
                                    steps: int, guidance_scale: float, motion_bucket_id: int,
                                    seed: Optional[int], model: str, compliance_mode: str) -> Tuple[Optional[str], Dict[str, Any], str]:
        """Handle video generation with progress updates."""
        # Create a progress callback that updates the UI status
        progress_updates = []
        
        def progress_callback(progress_info):
            """Callback to track progress updates."""
            progress_updates.append(progress_info)
            # In a real implementation, this would update the UI in real-time
            logger.info(f"Generation progress: {progress_info.completed_steps}/{progress_info.total_steps}")
        
        # Update status to show generation is starting
        yield None, {}, "Starting video generation..."
        
        # Call the actual generation method
        result = self._generate_video(
            prompt, conditioning_image, width, height, num_frames, fps,
            steps, guidance_scale, motion_bucket_id, seed, model, compliance_mode
        )
        
        # Return the final result
        yield result
    
    def _refresh_system_status(self) -> Tuple[Dict[str, Any], Dict[str, Any], str]:
        """Refresh system status information."""
        try:
            if self.system_controller and hasattr(self.system_controller, 'get_system_status'):
                status = self.system_controller.get_system_status()
                
                # Extract hardware status
                hardware_status = status.get('hardware', {})
                
                # Extract model status
                model_status = status.get('pipelines', {})
                
                # Get recent logs (mock for now)
                logs = "System status refreshed successfully\nAll components operational"
                
                return hardware_status, model_status, logs
            else:
                # Mock status for testing
                hardware_status = {
                    "gpu_model": "Mock GPU",
                    "vram_total": 8192,
                    "vram_used": 2048,
                    "vram_free": 6144,
                    "cpu_usage": 25,
                    "ram_usage": 45
                }
                
                model_status = {
                    "image": {"initialized": True, "current_model": "stable-diffusion-v1-5"},
                    "video": {"initialized": True, "current_model": "stable-video-diffusion"}
                }
                
                logs = "Mock system status (system controller not available)"
                
                return hardware_status, model_status, logs
                
        except Exception as e:
            logger.error(f"Failed to refresh system status: {e}")
            return {}, {}, f"Error refreshing status: {str(e)}"
    
    def _clear_vram_cache(self) -> Dict[str, Any]:
        """Clear VRAM cache and return updated hardware status."""
        try:
            if self.system_controller and hasattr(self.system_controller, 'clear_memory_cache'):
                success = self.system_controller.clear_memory_cache()
                if success:
                    logger.info("VRAM cache cleared successfully")
                    # Return updated hardware status
                    status = self.system_controller.get_system_status()
                    return status.get('hardware', {})
                else:
                    logger.warning("Failed to clear VRAM cache")
                    return {"error": "Failed to clear VRAM cache"}
            else:
                logger.warning("System controller not available for cache clearing")
                return {"error": "System controller not available"}
                
        except Exception as e:
            logger.error(f"Error clearing VRAM cache: {e}")
            return {"error": f"Error clearing cache: {str(e)}"}
    
    def _refresh_experiment_history(self) -> List[List[str]]:
        """Refresh experiment history display."""
        try:
            if self.system_controller and hasattr(self.system_controller, 'get_generation_history'):
                history = self.system_controller.get_generation_history(limit=50)
                
                # Convert to format expected by Gradio Dataframe
                formatted_history = []
                for exp in history:
                    formatted_history.append([
                        exp.get('generation_id', 'Unknown'),
                        exp.get('timestamp', 'Unknown'),
                        exp.get('output_type', 'Unknown'),
                        exp.get('model_used', 'Unknown'),
                        exp.get('prompt', '')[:50] + '...' if len(exp.get('prompt', '')) > 50 else exp.get('prompt', ''),
                        'Success' if exp.get('success', False) else 'Failed'
                    ])
                
                return formatted_history
            else:
                # Mock history for testing
                return [
                    ["exp_001", "2024-01-01 12:00:00", "image", "stable-diffusion-v1-5", "A beautiful landscape...", "Success"],
                    ["exp_002", "2024-01-01 12:05:00", "video", "stable-video-diffusion", "A flowing river...", "Success"],
                    ["exp_003", "2024-01-01 12:10:00", "image", "sdxl-turbo", "Abstract art piece...", "Failed"]
                ]
                
        except Exception as e:
            logger.error(f"Failed to refresh experiment history: {e}")
            return [["Error", str(e), "", "", "", ""]]
    
    def _save_experiment(self, research_notes: str) -> str:
        """Save current experiment with research notes."""
        try:
            if self.experiment_tracker:
                # In a real implementation, this would save the current generation
                # with the provided research notes
                experiment_id = f"exp_{int(time.time())}"
                
                # Mock experiment data
                experiment_data = {
                    'research_notes': research_notes,
                    'timestamp': time.time(),
                    'ui_session': True
                }
                
                # Save experiment (mock implementation)
                logger.info(f"Saving experiment {experiment_id} with notes: {research_notes[:50]}...")
                
                return f"Experiment {experiment_id} saved successfully"
            else:
                return "Experiment tracker not available"
                
        except Exception as e:
            logger.error(f"Failed to save experiment: {e}")
            return f"Error saving experiment: {str(e)}"    def _on
_quality_preset_change(self, quality_preset: str) -> Tuple[int, float]:
        """Handle quality preset changes."""
        if "Draft" in quality_preset:
            return 10, 5.0
        elif "Standard" in quality_preset:
            return 20, 7.5
        elif "High Quality" in quality_preset:
            return 30, 9.0
        elif "Ultra Quality" in quality_preset:
            return 50, 12.0
        else:  # Custom
            return 20, 7.5  # Keep current values
    
    def _on_resolution_preset_change(self, resolution_preset: str) -> Tuple[int, int]:
        """Handle resolution preset changes."""
        if "512x512" in resolution_preset:
            return 512, 512
        elif "768x768" in resolution_preset:
            return 768, 768
        elif "1024x1024" in resolution_preset:
            return 1024, 1024
        elif "512x768" in resolution_preset:
            return 512, 768
        elif "768x512" in resolution_preset:
            return 768, 512
        elif "1024x768" in resolution_preset:
            return 1024, 768
        elif "768x1024" in resolution_preset:
            return 768, 1024
        else:  # Custom
            return 512, 512  # Keep current values  
  
    def _update_gpu_monitor(self) -> Dict[str, Any]:
        """Update GPU monitoring information."""
        try:
            from src.core.gpu_optimizer import get_gpu_optimizer
            
            gpu_optimizer = get_gpu_optimizer()
            gpu_usage = gpu_optimizer.monitor_gpu_usage()
            
            if gpu_usage.get('gpu_available'):
                return {
                    "GPU": gpu_usage.get('gpu_name', 'Unknown'),
                    "VRAM Used": f"{gpu_usage.get('allocated_mb', 0):.1f} MB",
                    "VRAM Total": f"{gpu_usage.get('total_mb', 0):.1f} MB",
                    "Utilization": f"{gpu_usage.get('utilization_percent', 0):.1f}%",
                    "Status": "Available"
                }
            else:
                return {
                    "Status": "GPU not available",
                    "Message": "Using CPU for generation"
                }
                
        except Exception as e:
            return {
                "Status": "Error",
                "Message": f"Failed to get GPU info: {str(e)}"
            } 
   
    def _create_outputs_gallery_tab(self) -> Dict[str, Any]:
        """Create the generated outputs gallery tab."""
        components = {}
        
        with gr.Row():
            with gr.Column(scale=1):
                # Filter controls
                components['output_type_filter'] = gr.Dropdown(
                    choices=["All", "Images", "Videos"],
                    value="All",
                    label="Output Type Filter"
                )
                
                components['tag_filter'] = gr.Dropdown(
                    choices=["All", "landscape", "portrait", "abstract", "nature", "architecture"],
                    value="All",
                    label="Tag Filter",
                    multiselect=True
                )
                
                components['refresh_gallery_btn'] = gr.Button(
                    "Refresh Gallery",
                    variant="secondary"
                )
                
                # Statistics
                components['output_stats'] = gr.JSON(
                    label="Output Statistics",
                    value={"total_outputs": 0, "total_size_mb": 0}
                )
            
            with gr.Column(scale=3):
                # Gallery display
                components['outputs_gallery'] = gr.Gallery(
                    label="Generated Outputs",
                    show_label=True,
                    elem_id="outputs_gallery",
                    columns=3,
                    rows=2,
                    height="auto"
                )
        
        # Output details section
        with gr.Row():
            with gr.Column():
                components['selected_output_info'] = gr.JSON(
                    label="Selected Output Details",
                    visible=False
                )
                
                components['delete_output_btn'] = gr.Button(
                    "Delete Selected Output",
                    variant="stop",
                    visible=False
                )
        
        return components    

    def _refresh_outputs_gallery(self, output_type_filter: str, tag_filter: List[str]) -> Tuple[List[str], Dict[str, Any]]:
        """Refresh the outputs gallery with current filters."""
        try:
            from src.core.output_manager import get_output_manager
            
            output_manager = get_output_manager()
            
            # Convert filter to output type
            filter_type = None
            if output_type_filter == "Images":
                from src.core.output_manager import OutputType
                filter_type = OutputType.IMAGE
            elif output_type_filter == "Videos":
                from src.core.output_manager import OutputType
                filter_type = OutputType.VIDEO
            
            # Convert tag filter
            tags = tag_filter if tag_filter and "All" not in tag_filter else None
            
            # Get outputs
            outputs = output_manager.get_outputs(
                output_type=filter_type,
                limit=50,
                tags=tags
            )
            
            # Prepare gallery items (use thumbnails if available, otherwise original files)
            gallery_items = []
            for output in outputs:
                if output.thumbnail_path and output.thumbnail_path.exists():
                    gallery_items.append(str(output.thumbnail_path))
                elif output.file_path.exists():
                    gallery_items.append(str(output.file_path))
            
            # Get statistics
            stats = output_manager.get_statistics()
            
            return gallery_items, stats
            
        except Exception as e:
            logger.error(f"Failed to refresh outputs gallery: {e}")
            return [], {"error": str(e)}
    
    def _save_generation_output(self, result, prompt: str, model_used: str, generation_time: float, 
                              parameters: Dict[str, Any], output_type: str = "image") -> str:
        """Save generation output using the output manager."""
        try:
            from src.core.output_manager import get_output_manager, OutputType
            
            if not result or not hasattr(result, 'output_path') or not result.output_path:
                return ""
            
            output_manager = get_output_manager()
            
            # Convert output type
            if output_type == "video":
                output_type_enum = OutputType.VIDEO
            else:
                output_type_enum = OutputType.IMAGE
            
            # Save output
            output_id = output_manager.save_output(
                file_path=Path(result.output_path),
                output_type=output_type_enum,
                prompt=prompt,
                model_used=model_used,
                generation_time=generation_time,
                parameters=parameters,
                quality_metrics=result.quality_metrics or {},
                compliance_mode=getattr(result, 'compliance_mode', 'research_safe')
            )
            
            logger.info(f"Output saved with ID: {output_id}")
            return output_id
            
        except Exception as e:
            logger.error(f"Failed to save generation output: {e}")
            return ""   
 
    def _create_outputs_tab(self) -> Dict[str, Any]:
        """Create the generated outputs tab components."""
        components = {}
        
        with gr.Row():
            with gr.Column(scale=2):
                # Output filters
                with gr.Row():
                    components['output_type_filter'] = gr.Dropdown(
                        choices=["All", "Images", "Videos"],
                        value="All",
                        label="Output Type"
                    )
                    
                    components['tag_filter'] = gr.Dropdown(
                        choices=["All", "landscape", "portrait", "abstract", "nature"],
                        value="All",
                        label="Tag Filter",
                        multiselect=True
                    )
                    
                    components['refresh_outputs_btn'] = gr.Button(
                        "Refresh",
                        variant="secondary"
                    )
                
                # Output gallery
                components['output_gallery'] = gr.Gallery(
                    label="Generated Outputs",
                    show_label=True,
                    elem_id="output_gallery",
                    columns=4,
                    rows=3,
                    height="auto"
                )
                
                # Output list (detailed view)
                components['output_list'] = gr.Dataframe(
                    headers=["ID", "Type", "Prompt", "Model", "Time", "Size", "Created"],
                    label="Output Details",
                    interactive=False,
                    wrap=True
                )
            
            with gr.Column(scale=1):
                # Selected output details
                components['selected_output_info'] = gr.JSON(
                    label="Selected Output Info",
                    visible=True
                )
                
                # Output statistics
                components['output_stats'] = gr.JSON(
                    label="Output Statistics",
                    value={
                        "total_outputs": 0,
                        "total_size_mb": 0,
                        "avg_generation_time": 0
                    }
                )
                
                # Output actions
                with gr.Column():
                    components['download_output_btn'] = gr.Button(
                        "Download Selected",
                        variant="secondary"
                    )
                    
                    components['delete_output_btn'] = gr.Button(
                        "Delete Selected",
                        variant="secondary"
                    )
                    
                    components['cleanup_old_btn'] = gr.Button(
                        "Cleanup Old Outputs",
                        variant="secondary"
                    )
        
        return components
    
    def _refresh_outputs_display(self, output_type_filter: str, tag_filter: list) -> Tuple[list, list, Dict[str, Any]]:
        """Refresh the outputs display with current filters."""
        try:
            from src.core.output_manager import get_output_manager
            
            output_manager = get_output_manager()
            
            # Convert filter to OutputType
            filter_type = None
            if output_type_filter == "Images":
                from src.core.output_manager import OutputType
                filter_type = OutputType.IMAGE
            elif output_type_filter == "Videos":
                from src.core.output_manager import OutputType
                filter_type = OutputType.VIDEO
            
            # Get filtered outputs
            outputs = output_manager.get_outputs(
                output_type=filter_type,
                limit=50,
                tags=tag_filter if tag_filter and "All" not in tag_filter else None
            )
            
            # Prepare gallery data (thumbnails)
            gallery_data = []
            for output in outputs:
                if output.thumbnail_path and output.thumbnail_path.exists():
                    gallery_data.append((str(output.thumbnail_path), output.prompt[:50] + "..."))
                elif output.file_path.exists():
                    gallery_data.append((str(output.file_path), output.prompt[:50] + "..."))
            
            # Prepare list data
            list_data = []
            for output in outputs:
                list_data.append([
                    output.output_id,
                    output.output_type.value.title(),
                    output.prompt[:50] + "..." if len(output.prompt) > 50 else output.prompt,
                    output.model_used,
                    f"{output.generation_time:.2f}s",
                    f"{output.file_size_bytes / (1024*1024):.1f}MB",
                    output.created_at.strftime("%Y-%m-%d %H:%M")
                ])
            
            # Get statistics
            stats = output_manager.get_statistics()
            
            return gallery_data, list_data, stats
            
        except Exception as e:
            logger.error(f"Failed to refresh outputs display: {e}")
            return [], [], {"error": str(e)}
    
    def _get_output_info(self, selected_output_id: str) -> Dict[str, Any]:
        """Get detailed information for selected output."""
        try:
            from src.core.output_manager import get_output_manager
            
            output_manager = get_output_manager()
            metadata = output_manager.get_output_by_id(selected_output_id)
            
            if metadata:
                return {
                    "output_id": metadata.output_id,
                    "type": metadata.output_type.value,
                    "prompt": metadata.prompt,
                    "model_used": metadata.model_used,
                    "generation_time": f"{metadata.generation_time:.2f}s",
                    "file_path": str(metadata.file_path),
                    "file_size": f"{metadata.file_size_bytes / (1024*1024):.2f}MB",
                    "resolution": metadata.resolution,
                    "duration": f"{metadata.duration_seconds:.2f}s" if metadata.duration_seconds else None,
                    "parameters": metadata.parameters,
                    "quality_metrics": metadata.quality_metrics,
                    "compliance_mode": metadata.compliance_mode,
                    "created_at": metadata.created_at.isoformat(),
                    "tags": metadata.tags
                }
            else:
                return {"error": "Output not found"}
                
        except Exception as e:
            logger.error(f"Failed to get output info: {e}")
            return {"error": str(e)}