"""
Modern UI for generative images and videos with easy/advanced mode toggle.

This module provides a clean, user-friendly interface with two modes:
- Easy Mode: Simple prompt input with sensible defaults
- Advanced Mode: Full control over all generation parameters
"""

import logging
import time
from typing import Dict, Any, Optional, List, Tuple, Union
from pathlib import Path
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

# Try to import Gradio
try:
    import gradio as gr
    GRADIO_AVAILABLE = True
except ImportError:
    gr = None
    GRADIO_AVAILABLE = False
    logger.warning("Gradio not available - UI will be limited")

# Try to import PIL for image handling
try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    Image = None
    PIL_AVAILABLE = False
    logger.warning("PIL not available - image handling limited")


class UIMode(Enum):
    """UI complexity modes."""
    EASY = "easy"
    ADVANCED = "advanced"


@dataclass
class GenerationSettings:
    """Settings for content generation."""
    # Basic settings
    prompt: str = ""
    negative_prompt: str = ""
    
    # Image settings
    width: int = 512
    height: int = 512
    steps: int = 20
    guidance_scale: float = 7.5
    seed: Optional[int] = None
    
    # Video settings
    num_frames: int = 14
    fps: int = 7
    motion_intensity: float = 127
    
    # Advanced settings
    model_name: str = "stable-diffusion-v1-5"
    batch_size: int = 1
    use_gpu: bool = True
    precision: str = "float16"
    memory_optimization: str = "balanced"


class ModernInterface:
    """
    Modern, clean interface for AI content generation.
    
    Features:
    - Easy/Advanced mode toggle
    - Real-time model switching
    - Optimized for both beginners and power users
    """
    
    def __init__(self, system_controller=None):
        """Initialize the modern interface."""
        self.system_controller = system_controller
        self.current_mode = UIMode.EASY
        self.settings = GenerationSettings()
        self.gradio_app = None
        self.is_initialized = False
        
        # Initialize system integration
        try:
            from .ui_integration import get_system_integration
            self.system_integration = get_system_integration()
        except (ImportError, RuntimeError) as e:
            logger.warning(f"System integration not available: {e}")
            self.system_integration = None
        
        logger.info("ModernInterface created")
    
    def initialize(self) -> bool:
        """Initialize the Gradio interface."""
        if not GRADIO_AVAILABLE:
            logger.error("Gradio not available - cannot initialize interface")
            return False
        
        try:
            logger.info("Initializing modern interface")
            self.gradio_app = self._create_interface()
            
            # Only set up the load event if gradio_app was created successfully
            if self.gradio_app is not None:
                self.is_initialized = True
                logger.info("Modern interface initialized successfully")
                return True
            else:
                logger.error("Failed to create Gradio interface")
                return False
        except Exception as e:
            logger.error(f"Failed to initialize modern interface: {e}")
            return False
    
    def launch(self, share: bool = False, server_name: str = "127.0.0.1", server_port: int = 7860) -> None:
        """Launch the interface."""
        if not self.is_initialized:
            logger.error("Interface not initialized")
            return
        
        logger.info(f"Launching modern interface on {server_name}:{server_port}")
        
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
    
    def _create_interface(self):
        """Create the main Gradio interface."""
        with gr.Blocks(
            title="AI Content Generator",
            theme=gr.themes.Soft(),
            css=self._get_custom_css()
        ) as interface:
            
            # Header with mode toggle and hardware info
            with gr.Row():
                with gr.Column(scale=2):
                    gr.Markdown("# 🎨 AI Content Generator")
                    
                with gr.Column(scale=1, min_width=200):
                    mode_toggle = gr.Radio(
                        choices=["Easy", "Advanced"],
                        value="Easy",
                        label="Mode",
                        info="Switch between simple and advanced controls"
                    )
                    
                with gr.Column(scale=1, min_width=200):
                    hardware_info = gr.Textbox(
                        label="💻 Hardware",
                        value="Detecting...",
                        interactive=False,
                        max_lines=2
                    )
            
            # Main content area
            with gr.Tabs() as main_tabs:
                
                # Image Generation Tab
                with gr.TabItem("🖼️ Images"):
                    image_components = self._create_image_tab()
                
                # Video Generation Tab
                with gr.TabItem("🎬 Videos"):
                    video_components = self._create_video_tab()
                
                # Gallery Tab
                with gr.TabItem("🖼️ Gallery"):
                    gallery_components = self._create_gallery_tab()
            
            # Setup event handlers
            self._setup_event_handlers(
                mode_toggle, image_components, video_components, gallery_components, hardware_info
            )
        
        return interface
    
    def _create_image_tab(self) -> Dict[str, Any]:
        """Create the image generation tab."""
        components = {}
        
        with gr.Row():
            # Left column - Controls
            with gr.Column(scale=1):
                # Prompt input (always visible)
                components['prompt'] = gr.Textbox(
                    label="✨ Describe your image",
                    placeholder="A beautiful sunset over mountains...",
                    lines=3,
                    max_lines=5
                )
                
                # Easy mode controls
                with gr.Group(visible=True) as easy_controls:
                    components['easy_controls'] = easy_controls
                    
                    components['style_preset'] = gr.Dropdown(
                        choices=[
                            "Photorealistic",
                            "Digital Art", 
                            "Anime/Manga",
                            "Oil Painting",
                            "Watercolor",
                            "Sketch",
                            "3D Render",
                            "Vintage Photo"
                        ],
                        value="Photorealistic",
                        label="🎨 Style",
                        info="Choose the artistic style"
                    )
                    
                    components['quality_preset'] = gr.Dropdown(
                        choices=[
                            "Fast (10 steps)",
                            "Balanced (20 steps)",
                            "High Quality (30 steps)",
                            "Ultra (50 steps)"
                        ],
                        value="Balanced (20 steps)",
                        label="⚡ Quality",
                        info="Higher quality = slower generation"
                    )
                    
                    components['size_preset'] = gr.Dropdown(
                        choices=[
                            "Square (512×512)",
                            "Portrait (512×768)",
                            "Landscape (768×512)",
                            "HD Square (1024×1024)",
                            "HD Portrait (768×1024)",
                            "HD Landscape (1024×768)"
                        ],
                        value="Square (512×512)",
                        label="📐 Size",
                        info="Larger sizes need more VRAM"
                    )
                
                # Negative prompt (always exists but hidden in easy mode)
                components['negative_prompt'] = gr.Textbox(
                    label="🚫 Negative Prompt",
                    placeholder="What to avoid in the image...",
                    lines=2,
                    visible=False
                )
                
                # Advanced mode controls
                with gr.Group(visible=False) as advanced_controls:
                    components['advanced_controls'] = advanced_controls
                    
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
                            label="Steps"
                        )
                        components['guidance_scale'] = gr.Slider(
                            minimum=1.0, maximum=20.0, value=7.5, step=0.5,
                            label="Guidance Scale"
                        )
                    
                    components['model_selector'] = gr.Dropdown(
                        choices=[],  # Will be populated dynamically
                        value=None,
                        label="🤖 Model",
                        info="Models are filtered based on your hardware"
                    )
                    
                    # Model download section
                    with gr.Row():
                        components['download_btn'] = gr.Button(
                            "📥 Download Selected Model",
                            variant="secondary",
                            size="sm",
                            visible=False
                        )
                        components['download_progress'] = gr.Textbox(
                            label="Download Progress",
                            value="",
                            interactive=False,
                            visible=False,
                            max_lines=1
                        )
                    
                    with gr.Accordion("⚙️ Performance Settings", open=False):
                        components['batch_size'] = gr.Slider(
                            minimum=1, maximum=4, value=1, step=1,
                            label="Batch Size (Multiple Images)"
                        )
                        
                        components['precision'] = gr.Dropdown(
                            choices=["float16 (Faster)", "float32 (Better Quality)"],
                            value="float16 (Faster)",
                            label="Precision"
                        )
                        
                        components['memory_optimization'] = gr.Dropdown(
                            choices=[
                                "None (Fastest, Most VRAM)",
                                "Balanced (Recommended)",
                                "Aggressive (Slowest, Least VRAM)"
                            ],
                            value="Balanced (Recommended)",
                            label="Memory Optimization"
                        )
                
                # Seed control (both modes)
                components['seed'] = gr.Number(
                    label="🎲 Seed (Optional)",
                    value=None,
                    precision=0,
                    info="Use same seed for reproducible results"
                )
                
                # Generate button
                components['generate_btn'] = gr.Button(
                    "🎨 Generate Image",
                    variant="primary",
                    size="lg"
                )
            
            # Right column - Output
            with gr.Column(scale=1):
                components['output_image'] = gr.Image(
                    label="Generated Image",
                    type="pil",
                    height=400
                )
                
                # Generation info
                with gr.Accordion("ℹ️ Generation Info", open=False):
                    components['generation_info'] = gr.JSON(
                        label="Details"
                    )
                
                # Status
                components['status'] = gr.Textbox(
                    label="Status",
                    value="Ready to generate",
                    interactive=False,
                    max_lines=2
                )
        
        return components
    
    def _create_video_tab(self) -> Dict[str, Any]:
        """Create the video generation tab."""
        components = {}
        
        with gr.Row():
            # Left column - Controls
            with gr.Column(scale=1):
                # Prompt input
                components['prompt'] = gr.Textbox(
                    label="🎬 Describe your video",
                    placeholder="A cat walking through a garden...",
                    lines=3,
                    max_lines=5
                )
                
                # Conditioning image
                components['conditioning_image'] = gr.Image(
                    label="📸 Starting Image (Optional)",
                    type="pil"
                )
                
                # Easy mode controls
                with gr.Group(visible=True) as easy_controls:
                    components['easy_controls'] = easy_controls
                    
                    components['video_length'] = gr.Dropdown(
                        choices=[
                            "Short (1-2 seconds, 8 frames)",
                            "Medium (2-3 seconds, 14 frames)",
                            "Long (3-4 seconds, 25 frames)"
                        ],
                        value="Medium (2-3 seconds, 14 frames)",
                        label="⏱️ Length",
                        info="Longer videos need more VRAM and time"
                    )
                    
                    components['motion_preset'] = gr.Dropdown(
                        choices=[
                            "Subtle Motion",
                            "Moderate Motion", 
                            "Dynamic Motion",
                            "High Action"
                        ],
                        value="Moderate Motion",
                        label="🏃 Motion Level"
                    )
                
                # Advanced mode controls
                with gr.Group(visible=False) as advanced_controls:
                    components['advanced_controls'] = advanced_controls
                    
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
                            label="Frames"
                        )
                        components['fps'] = gr.Slider(
                            minimum=1, maximum=30, value=7, step=1,
                            label="FPS"
                        )
                    
                    components['motion_bucket_id'] = gr.Slider(
                        minimum=1, maximum=255, value=127, step=1,
                        label="Motion Intensity"
                    )
                    
                    components['model_selector'] = gr.Dropdown(
                        choices=[],  # Will be populated dynamically
                        value=None,
                        label="🤖 Model",
                        info="Models are filtered based on your hardware"
                    )
                    
                    # Model download section
                    with gr.Row():
                        components['download_btn'] = gr.Button(
                            "📥 Download Selected Model",
                            variant="secondary",
                            size="sm",
                            visible=False
                        )
                        components['download_progress'] = gr.Textbox(
                            label="Download Progress",
                            value="",
                            interactive=False,
                            visible=False,
                            max_lines=1
                        )
                
                # Generate button
                components['generate_btn'] = gr.Button(
                    "🎬 Generate Video",
                    variant="primary",
                    size="lg"
                )
            
            # Right column - Output
            with gr.Column(scale=1):
                components['output_video'] = gr.Video(
                    label="Generated Video",
                    height=400
                )
                
                # Generation info
                with gr.Accordion("ℹ️ Generation Info", open=False):
                    components['generation_info'] = gr.JSON(
                        label="Details"
                    )
                
                # Status
                components['status'] = gr.Textbox(
                    label="Status",
                    value="Ready to generate",
                    interactive=False,
                    max_lines=2
                )
        
        return components
    
    def _create_gallery_tab(self) -> Dict[str, Any]:
        """Create the gallery tab."""
        components = {}
        
        with gr.Row():
            with gr.Column(scale=3):
                components['gallery'] = gr.Gallery(
                    label="Generated Content",
                    show_label=False,
                    elem_id="gallery",
                    columns=3,
                    rows=2,
                    height="auto"
                )
            
            with gr.Column(scale=1):
                components['filter_type'] = gr.Dropdown(
                    choices=["All", "Images", "Videos"],
                    value="All",
                    label="Filter by Type"
                )
                
                components['refresh_btn'] = gr.Button(
                    "🔄 Refresh Gallery",
                    variant="secondary"
                )
                
                components['clear_btn'] = gr.Button(
                    "🗑️ Clear Gallery",
                    variant="secondary"
                )
        
        return components
    
    def _setup_event_handlers(self, mode_toggle, image_components, video_components, gallery_components, hardware_info):
        """Setup event handlers for the interface."""
        
        # Initialize hardware info and models on load
        if self.gradio_app is not None:
            self.gradio_app.load(
                fn=self._initialize_interface,
                outputs=[
                    hardware_info,
                    image_components['model_selector'],
                    video_components['model_selector']
                ]
            )
        
        # Mode toggle
        mode_toggle.change(
            fn=self._toggle_mode,
            inputs=[mode_toggle],
            outputs=[
                image_components['easy_controls'],
                image_components['advanced_controls'],
                image_components['negative_prompt'],
                video_components['easy_controls'],
                video_components['advanced_controls']
            ]
        )
        
        # Model selection change handlers
        image_components['model_selector'].change(
            fn=self._on_model_select,
            inputs=[image_components['model_selector']],
            outputs=[
                image_components['download_btn'],
                image_components['download_progress']
            ]
        )
        
        video_components['model_selector'].change(
            fn=self._on_model_select,
            inputs=[video_components['model_selector']],
            outputs=[
                video_components['download_btn'],
                video_components['download_progress']
            ]
        )
        
        # Download button handlers
        image_components['download_btn'].click(
            fn=self._download_model,
            inputs=[image_components['model_selector']],
            outputs=[image_components['download_progress']]
        )
        
        video_components['download_btn'].click(
            fn=self._download_model,
            inputs=[video_components['model_selector']],
            outputs=[video_components['download_progress']]
        )
        
        # Image generation
        image_components['generate_btn'].click(
            fn=self._generate_image,
            inputs=[
                image_components['prompt'],
                image_components['negative_prompt'],
                image_components['style_preset'],
                image_components['quality_preset'],
                image_components['size_preset'],
                image_components['seed']
            ],
            outputs=[
                image_components['output_image'],
                image_components['generation_info'],
                image_components['status']
            ]
        )
        
        # Video generation
        video_components['generate_btn'].click(
            fn=self._generate_video,
            inputs=[
                video_components['prompt'],
                video_components['conditioning_image'],
                video_components['video_length'],
                video_components['motion_preset']
            ],
            outputs=[
                video_components['output_video'],
                video_components['generation_info'],
                video_components['status']
            ]
        )
        
        # Gallery refresh
        gallery_components['refresh_btn'].click(
            fn=self._refresh_gallery,
            inputs=[gallery_components['filter_type']],
            outputs=[gallery_components['gallery']]
        )
    
    def _initialize_interface(self) -> Tuple[str, gr.Dropdown, gr.Dropdown]:
        """Initialize the interface with hardware info and available models."""
        try:
            # Get hardware info
            if self.system_integration:
                hw_info = self.system_integration.get_hardware_info()
                hardware_text = f"🖥️ {hw_info['gpu_model']}\n💾 {hw_info['vram_total']}MB VRAM"
                
                # Get available models
                image_models = self.system_integration.get_available_models("image")
                video_models = self.system_integration.get_available_models("video")
                
                # Format for dropdown
                image_choices = [(f"{info['name']} - {info['description']}", key) 
                               for key, info in image_models.items()]
                video_choices = [(f"{info['name']} - {info['description']}", key) 
                               for key, info in video_models.items()]
                
                # Auto-download recommended models
                self.system_integration.auto_download_recommended_models()
                
            else:
                hardware_text = "🖥️ Hardware detection unavailable"
                image_choices = [("Stable Diffusion 1.5 - Mock Mode", "sd15")]
                video_choices = [("Stable Video Diffusion - Mock Mode", "svd")]
            
            return (
                hardware_text,
                gr.Dropdown(choices=image_choices, value=image_choices[0][1] if image_choices else None),
                gr.Dropdown(choices=video_choices, value=video_choices[0][1] if video_choices else None)
            )
            
        except Exception as e:
            logger.error(f"Failed to initialize interface: {e}")
            return (
                "❌ Hardware detection failed",
                gr.Dropdown(choices=[("Error loading models", "error")]),
                gr.Dropdown(choices=[("Error loading models", "error")])
            )
    
    def _toggle_mode(self, mode: str) -> Tuple[gr.Group, gr.Group, gr.Textbox, gr.Group, gr.Group]:
        """Toggle between easy and advanced modes."""
        is_easy = mode == "Easy"
        
        return (
            gr.Group(visible=is_easy),       # Image easy controls
            gr.Group(visible=not is_easy),   # Image advanced controls
            gr.update(visible=not is_easy),  # Negative prompt (only in advanced mode)
            gr.Group(visible=is_easy),       # Video easy controls
            gr.Group(visible=not is_easy)    # Video advanced controls
        )
    
    def _on_model_select(self, model_key: str) -> Tuple[gr.Button, gr.Textbox]:
        """Handle model selection change."""
        if not model_key or not self.system_integration:
            return gr.Button(visible=False), gr.Textbox(visible=False)
        
        try:
            # Get model info
            image_models = self.system_integration.get_available_models("image")
            video_models = self.system_integration.get_available_models("video")
            all_models = {**image_models, **video_models}
            
            if model_key in all_models:
                model_info = all_models[model_key]
                
                if model_info['is_downloaded']:
                    return (
                        gr.Button(visible=False),
                        gr.Textbox(value="✅ Model ready to use", visible=True)
                    )
                elif model_info['can_run']:
                    return (
                        gr.Button(visible=True),
                        gr.Textbox(value=f"📥 Ready to download ({model_info['download_size_gb']:.1f}GB)", visible=True)
                    )
                else:
                    return (
                        gr.Button(visible=False),
                        gr.Textbox(value=f"❌ Requires {model_info['vram_mb']}MB VRAM", visible=True)
                    )
            
        except Exception as e:
            logger.error(f"Error in model selection: {e}")
        
        return gr.Button(visible=False), gr.Textbox(visible=False)
    
    def _download_model(self, model_key: str) -> str:
        """Download the selected model."""
        if not model_key or not self.system_integration:
            return "❌ No model selected"
        
        try:
            # Get model info
            image_models = self.system_integration.get_available_models("image")
            video_models = self.system_integration.get_available_models("video")
            all_models = {**image_models, **video_models}
            
            if model_key in all_models:
                model_info = all_models[model_key]
                model_id = model_info['model_id']
                
                if self.system_integration.download_model(model_id):
                    return f"📥 Downloading {model_info['name']}... This may take several minutes."
                else:
                    return "❌ Failed to start download"
            
            return "❌ Model not found"
            
        except Exception as e:
            logger.error(f"Download failed: {e}")
            return f"❌ Download error: {str(e)}"
    
    def _generate_image(self, prompt: str, negative_prompt: str, style_preset: str, 
                       quality_preset: str, size_preset: str, seed: Optional[int]) -> Tuple[Any, Dict, str]:
        """Generate an image with the given parameters."""
        try:
            if not prompt.strip():
                return None, {}, "❌ Please enter a prompt"
            
            # Parse presets
            steps = self._parse_quality_preset(quality_preset)
            width, height = self._parse_size_preset(size_preset)
            
            # Add style to prompt if needed
            styled_prompt = self._apply_style_preset(prompt, style_preset)
            
            # Prepare generation parameters
            generation_params = {
                "negative_prompt": negative_prompt,
                "steps": steps,
                "width": width,
                "height": height,
                "seed": seed,
                "style": style_preset
            }
            
            # Use system integration if available
            if self.system_integration:
                return self.system_integration.generate_image(styled_prompt, **generation_params)
            else:
                # Fallback mock generation
                generation_info = {
                    "prompt": styled_prompt,
                    "negative_prompt": negative_prompt,
                    "steps": steps,
                    "size": f"{width}x{height}",
                    "seed": seed,
                    "model": "stable-diffusion-v1-5",
                    "generation_time": "2.3s"
                }
                
                return None, generation_info, "✅ Mock generation completed!"
            
        except Exception as e:
            logger.error(f"Image generation failed: {e}")
            return None, {}, f"❌ Generation failed: {str(e)}"
    
    def _generate_video(self, prompt: str, conditioning_image: Any, 
                       video_length: str, motion_preset: str) -> Tuple[Any, Dict, str]:
        """Generate a video with the given parameters."""
        try:
            if not prompt.strip():
                return None, {}, "❌ Please enter a prompt"
            
            # Parse presets
            num_frames = self._parse_video_length(video_length)
            motion_intensity = self._parse_motion_preset(motion_preset)
            
            # Prepare generation parameters
            generation_params = {
                "num_frames": num_frames,
                "motion_intensity": motion_intensity,
                "video_length": video_length,
                "motion_preset": motion_preset
            }
            
            # Use system integration if available
            if self.system_integration:
                return self.system_integration.generate_video(prompt, conditioning_image, **generation_params)
            else:
                # Fallback mock generation
                generation_info = {
                    "prompt": prompt,
                    "frames": num_frames,
                    "motion_intensity": motion_intensity,
                    "has_conditioning_image": conditioning_image is not None,
                    "model": "stable-video-diffusion",
                    "generation_time": "45.2s"
                }
                
                return None, generation_info, "✅ Mock generation completed!"
            
        except Exception as e:
            logger.error(f"Video generation failed: {e}")
            return None, {}, f"❌ Generation failed: {str(e)}"
    
    def _refresh_gallery(self, filter_type: str) -> List:
        """Refresh the gallery with filtered content."""
        # Mock gallery content for now
        return []
    
    def _parse_quality_preset(self, preset: str) -> int:
        """Parse quality preset to steps."""
        if "10 steps" in preset:
            return 10
        elif "20 steps" in preset:
            return 20
        elif "30 steps" in preset:
            return 30
        elif "50 steps" in preset:
            return 50
        return 20
    
    def _parse_size_preset(self, preset: str) -> Tuple[int, int]:
        """Parse size preset to width, height."""
        size_map = {
            "Square (512×512)": (512, 512),
            "Portrait (512×768)": (512, 768),
            "Landscape (768×512)": (768, 512),
            "HD Square (1024×1024)": (1024, 1024),
            "HD Portrait (768×1024)": (768, 1024),
            "HD Landscape (1024×768)": (1024, 768)
        }
        return size_map.get(preset, (512, 512))
    
    def _parse_video_length(self, preset: str) -> int:
        """Parse video length preset to frame count."""
        if "8 frames" in preset:
            return 8
        elif "14 frames" in preset:
            return 14
        elif "25 frames" in preset:
            return 25
        return 14
    
    def _parse_motion_preset(self, preset: str) -> int:
        """Parse motion preset to intensity value."""
        motion_map = {
            "Subtle Motion": 50,
            "Moderate Motion": 127,
            "Dynamic Motion": 180,
            "High Action": 220
        }
        return motion_map.get(preset, 127)
    
    def _apply_style_preset(self, prompt: str, style: str) -> str:
        """Apply style preset to prompt."""
        style_suffixes = {
            "Photorealistic": ", photorealistic, high quality, detailed",
            "Digital Art": ", digital art, concept art, trending on artstation",
            "Anime/Manga": ", anime style, manga style, cel shading",
            "Oil Painting": ", oil painting, classical art, fine art",
            "Watercolor": ", watercolor painting, soft colors, artistic",
            "Sketch": ", pencil sketch, line art, black and white",
            "3D Render": ", 3d render, octane render, unreal engine",
            "Vintage Photo": ", vintage photograph, film grain, retro"
        }
        
        suffix = style_suffixes.get(style, "")
        return prompt + suffix
    
    def _get_custom_css(self) -> str:
        """Get custom CSS for the interface."""
        return """
        .gradio-container {
            max-width: 1400px !important;
        }
        
        #gallery {
            min-height: 400px;
        }
        
        .generate-btn {
            background: linear-gradient(45deg, #667eea 0%, #764ba2 100%);
            border: none;
            color: white;
            font-weight: bold;
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


# Create a simple launcher function
def create_modern_interface(system_controller=None):
    """Create and return a modern interface instance."""
    return ModernInterface(system_controller)


if __name__ == "__main__":
    # For testing the interface standalone
    interface = ModernInterface()
    if interface.initialize():
        interface.launch()