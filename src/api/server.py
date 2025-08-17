"""
FastAPI server foundation for Academic Multimodal LLM Experiment System.

This module implements the REST API server with endpoints for image/video generation,
model management, and experiment tracking with proper parameter validation and
progress tracking.
"""

import logging
import asyncio
import uuid
from typing import Dict, Any, Optional, List
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel, Field, field_validator
import uvicorn

from ..core.llm_controller import LLMController
from ..core.interfaces import (
    GenerationRequest, ComplianceMode, OutputType, 
    HardwareConfig, StyleConfig, ConversationContext
)
from ..pipelines.image_generation import ImageGenerationPipeline
from ..pipelines.video_generation import VideoGenerationPipeline
from ..hardware.detector import HardwareDetector
from ..data.experiment_tracker import ExperimentTracker

logger = logging.getLogger(__name__)

# Pydantic models for API request/response validation
class ImageGenerationRequest(BaseModel):
    """Request model for image generation."""
    prompt: str = Field(..., min_length=1, max_length=2000, description="Text prompt for image generation")
    negative_prompt: Optional[str] = Field(None, max_length=2000, description="Negative prompt to avoid certain elements")
    width: int = Field(512, ge=256, le=2048, description="Image width in pixels")
    height: int = Field(512, ge=256, le=2048, description="Image height in pixels")
    num_inference_steps: int = Field(20, ge=1, le=100, description="Number of denoising steps")
    guidance_scale: float = Field(7.5, ge=1.0, le=20.0, description="Guidance scale for prompt adherence")
    num_images: int = Field(1, ge=1, le=4, description="Number of images to generate")
    seed: Optional[int] = Field(None, ge=0, description="Random seed for reproducibility")
    model_name: Optional[str] = Field(None, description="Specific model to use")
    compliance_mode: str = Field("research_safe", description="Copyright compliance mode")
    
    @field_validator('compliance_mode')
    @classmethod
    def validate_compliance_mode(cls, v):
        valid_modes = ['open_source_only', 'research_safe', 'full_dataset']
        if v not in valid_modes:
            raise ValueError(f'compliance_mode must be one of {valid_modes}')
        return v


class VideoGenerationRequest(BaseModel):
    """Request model for video generation."""
    prompt: str = Field(..., min_length=1, max_length=2000, description="Text prompt for video generation")
    negative_prompt: Optional[str] = Field(None, max_length=2000, description="Negative prompt to avoid certain elements")
    width: int = Field(512, ge=256, le=1024, description="Video width in pixels")
    height: int = Field(512, ge=256, le=1024, description="Video height in pixels")
    num_frames: int = Field(14, ge=4, le=25, description="Number of frames to generate")
    fps: int = Field(7, ge=1, le=30, description="Frames per second")
    num_inference_steps: int = Field(25, ge=1, le=100, description="Number of denoising steps")
    guidance_scale: float = Field(7.5, ge=1.0, le=20.0, description="Guidance scale for prompt adherence")
    seed: Optional[int] = Field(None, ge=0, description="Random seed for reproducibility")
    model_name: Optional[str] = Field(None, description="Specific model to use")
    compliance_mode: str = Field("research_safe", description="Copyright compliance mode")
    
    @field_validator('compliance_mode')
    @classmethod
    def validate_compliance_mode(cls, v):
        valid_modes = ['open_source_only', 'research_safe', 'full_dataset']
        if v not in valid_modes:
            raise ValueError(f'compliance_mode must be one of {valid_modes}')
        return v


class GenerationResponse(BaseModel):
    """Response model for generation requests."""
    task_id: str = Field(..., description="Unique task identifier")
    status: str = Field(..., description="Task status")
    message: str = Field(..., description="Status message")
    estimated_time: Optional[float] = Field(None, description="Estimated completion time in seconds")


class GenerationResult(BaseModel):
    """Response model for completed generation tasks."""
    task_id: str = Field(..., description="Task identifier")
    success: bool = Field(..., description="Whether generation was successful")
    output_path: Optional[str] = Field(None, description="Path to generated content")
    generation_time: float = Field(..., description="Time taken for generation in seconds")
    model_used: str = Field(..., description="Model used for generation")
    quality_metrics: Optional[Dict[str, Any]] = Field(None, description="Quality metrics")
    error_message: Optional[str] = Field(None, description="Error message if failed")


class ModelStatus(BaseModel):
    """Response model for model status."""
    current_model: Optional[str] = Field(None, description="Currently loaded model")
    available_models: List[str] = Field(..., description="List of available models")
    vram_usage: Dict[str, Any] = Field(..., description="VRAM usage information")
    hardware_info: Dict[str, Any] = Field(..., description="Hardware information")


class ExperimentSaveRequest(BaseModel):
    """Request model for saving experiments."""
    experiment_name: str = Field(..., min_length=1, max_length=200, description="Name for the experiment")
    description: Optional[str] = Field(None, max_length=1000, description="Experiment description")
    tags: List[str] = Field(default_factory=list, description="Tags for categorization")
    results: Dict[str, Any] = Field(..., description="Experiment results and metadata")


# Import shared dependencies
from .dependencies import APIState, api_state, get_api_state

# FastAPI app instance
app = FastAPI(
    title="Academic Multimodal LLM Experiment System API",
    description="REST API for multimodal AI generation with copyright compliance",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include model management router
from .model_management import router as model_router
app.include_router(model_router)





# Health check endpoint
@app.get("/health", response_model=Dict[str, str])
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
    }


# Image generation endpoint
@app.post("/generate/image", response_model=GenerationResponse)
async def generate_image(
    request: ImageGenerationRequest,
    background_tasks: BackgroundTasks,
    state: APIState = Depends(get_api_state)
):
    """
    Generate image from text prompt.
    
    This endpoint accepts a text prompt and generation parameters,
    then starts an asynchronous image generation task.
    """
    try:
        # Generate unique task ID
        task_id = str(uuid.uuid4())
        
        # Create conversation context
        conversation_id = f"api_session_{int(datetime.now().timestamp())}"
        context = state.llm_controller.manage_context(conversation_id)
        
        # Set compliance mode
        compliance_mode = ComplianceMode(request.compliance_mode)
        context.current_mode = compliance_mode
        
        # Create style config
        style_config = StyleConfig(
            style_name=None,
            lora_path=None,
            controlnet_config=None,
            generation_params={
                'negative_prompt': request.negative_prompt,
                'width': request.width,
                'height': request.height,
                'num_inference_steps': request.num_inference_steps,
                'guidance_scale': request.guidance_scale,
                'num_images_per_prompt': request.num_images,
                'seed': request.seed
            }
        )
        
        # Create generation request
        gen_request = GenerationRequest(
            prompt=request.prompt,
            output_type=OutputType.IMAGE,
            style_config=style_config,
            compliance_mode=compliance_mode,
            hardware_constraints=state.hardware_config,
            context=context,
            additional_params={'model_name': request.model_name} if request.model_name else {}
        )
        
        # Store task info
        api_state.active_tasks[task_id] = {
            'status': 'queued',
            'created_at': datetime.now(),
            'request': gen_request,
            'type': 'image'
        }
        
        # Start background task
        background_tasks.add_task(
            _process_image_generation,
            task_id,
            gen_request,
            state
        )
        
        # Estimate completion time based on hardware
        estimated_time = _estimate_generation_time('image', state.hardware_config)
        
        logger.info(f"Started image generation task {task_id}")
        
        return GenerationResponse(
            task_id=task_id,
            status="queued",
            message="Image generation task started",
            estimated_time=estimated_time
        )
        
    except Exception as e:
        logger.error(f"Error starting image generation: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start image generation: {str(e)}"
        )


# Video generation endpoint
@app.post("/generate/video", response_model=GenerationResponse)
async def generate_video(
    request: VideoGenerationRequest,
    background_tasks: BackgroundTasks,
    state: APIState = Depends(get_api_state)
):
    """
    Generate video from text prompt.
    
    This endpoint accepts a text prompt and generation parameters,
    then starts an asynchronous video generation task with progress tracking.
    """
    try:
        # Generate unique task ID
        task_id = str(uuid.uuid4())
        
        # Create conversation context
        conversation_id = f"api_session_{int(datetime.now().timestamp())}"
        context = state.llm_controller.manage_context(conversation_id)
        
        # Set compliance mode
        compliance_mode = ComplianceMode(request.compliance_mode)
        context.current_mode = compliance_mode
        
        # Create style config
        style_config = StyleConfig(
            style_name=None,
            lora_path=None,
            controlnet_config=None,
            generation_params={
                'negative_prompt': request.negative_prompt,
                'width': request.width,
                'height': request.height,
                'num_frames': request.num_frames,
                'fps': request.fps,
                'num_inference_steps': request.num_inference_steps,
                'guidance_scale': request.guidance_scale,
                'seed': request.seed
            }
        )
        
        # Create generation request
        gen_request = GenerationRequest(
            prompt=request.prompt,
            output_type=OutputType.VIDEO,
            style_config=style_config,
            compliance_mode=compliance_mode,
            hardware_constraints=state.hardware_config,
            context=context,
            additional_params={'model_name': request.model_name} if request.model_name else {}
        )
        
        # Store task info
        api_state.active_tasks[task_id] = {
            'status': 'queued',
            'created_at': datetime.now(),
            'request': gen_request,
            'type': 'video',
            'progress': 0.0
        }
        
        # Start background task
        background_tasks.add_task(
            _process_video_generation,
            task_id,
            gen_request,
            state
        )
        
        # Estimate completion time based on hardware
        estimated_time = _estimate_generation_time('video', state.hardware_config)
        
        logger.info(f"Started video generation task {task_id}")
        
        return GenerationResponse(
            task_id=task_id,
            status="queued",
            message="Video generation task started",
            estimated_time=estimated_time
        )
        
    except Exception as e:
        logger.error(f"Error starting video generation: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start video generation: {str(e)}"
        )


# Task status endpoint
@app.get("/tasks/{task_id}", response_model=Dict[str, Any])
async def get_task_status(task_id: str):
    """Get status of a generation task."""
    if task_id not in api_state.active_tasks:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Task not found"
        )
    
    task_info = api_state.active_tasks[task_id]
    
    response = {
        'task_id': task_id,
        'status': task_info['status'],
        'created_at': task_info['created_at'].isoformat(),
        'type': task_info['type']
    }
    
    # Add progress for video tasks
    if 'progress' in task_info:
        response['progress'] = task_info['progress']
    
    # Add result info if completed
    if task_info['status'] == 'completed' and 'result' in task_info:
        response['result'] = task_info['result']
    
    # Add error info if failed
    if task_info['status'] == 'failed' and 'error' in task_info:
        response['error'] = task_info['error']
    
    return response


# Download generated content
@app.get("/download/{task_id}")
async def download_result(task_id: str):
    """Download generated content file."""
    if task_id not in api_state.active_tasks:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Task not found"
        )
    
    task_info = api_state.active_tasks[task_id]
    
    if task_info['status'] != 'completed':
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Task not completed"
        )
    
    if 'result' not in task_info or not task_info['result'].get('output_path'):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Generated file not found"
        )
    
    output_path = Path(task_info['result']['output_path'])
    
    if not output_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Generated file no longer exists"
        )
    
    return FileResponse(
        path=output_path,
        filename=output_path.name,
        media_type='application/octet-stream'
    )


# Background task functions
async def _process_image_generation(task_id: str, request: GenerationRequest, state: APIState):
    """Background task for image generation."""
    try:
        # Update task status
        api_state.active_tasks[task_id]['status'] = 'processing'
        
        logger.info(f"Processing image generation task {task_id}")
        
        # Generate image
        result = state.image_pipeline.generate(request)
        
        # Update task with result
        api_state.active_tasks[task_id]['status'] = 'completed' if result.success else 'failed'
        api_state.active_tasks[task_id]['result'] = {
            'success': result.success,
            'output_path': str(result.output_path) if result.output_path else None,
            'generation_time': result.generation_time,
            'model_used': result.model_used,
            'quality_metrics': result.quality_metrics,
            'error_message': result.error_message
        }
        
        logger.info(f"Completed image generation task {task_id}: {'success' if result.success else 'failed'}")
        
    except Exception as e:
        logger.error(f"Error in image generation task {task_id}: {e}")
        api_state.active_tasks[task_id]['status'] = 'failed'
        api_state.active_tasks[task_id]['error'] = str(e)


async def _process_video_generation(task_id: str, request: GenerationRequest, state: APIState):
    """Background task for video generation with progress tracking."""
    try:
        # Update task status
        api_state.active_tasks[task_id]['status'] = 'processing'
        api_state.active_tasks[task_id]['progress'] = 0.0
        
        logger.info(f"Processing video generation task {task_id}")
        
        # Generate video (this would need progress callback integration)
        result = state.video_pipeline.generate(request)
        
        # Update progress to 100%
        api_state.active_tasks[task_id]['progress'] = 100.0
        
        # Update task with result
        api_state.active_tasks[task_id]['status'] = 'completed' if result.success else 'failed'
        api_state.active_tasks[task_id]['result'] = {
            'success': result.success,
            'output_path': str(result.output_path) if result.output_path else None,
            'generation_time': result.generation_time,
            'model_used': result.model_used,
            'quality_metrics': result.quality_metrics,
            'error_message': result.error_message
        }
        
        logger.info(f"Completed video generation task {task_id}: {'success' if result.success else 'failed'}")
        
    except Exception as e:
        logger.error(f"Error in video generation task {task_id}: {e}")
        api_state.active_tasks[task_id]['status'] = 'failed'
        api_state.active_tasks[task_id]['error'] = str(e)


def _estimate_generation_time(generation_type: str, hardware_config: HardwareConfig) -> float:
    """Estimate generation time based on type and hardware."""
    base_times = {
        'image': 30.0,  # seconds
        'video': 300.0  # 5 minutes
    }
    
    base_time = base_times.get(generation_type, 30.0)
    
    # Adjust for hardware
    if hardware_config.vram_size < 6000:
        base_time *= 2.0  # Slower on limited VRAM
    elif hardware_config.vram_size > 16000:
        base_time *= 0.5  # Faster on high-end hardware
    
    return base_time


# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize the API server on startup."""
    logger.info("Starting Academic Multimodal LLM Experiment System API")
    # Initialization will happen on first request via dependency


# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources on shutdown."""
    logger.info("Shutting down API server")
    
    if api_state.image_pipeline:
        api_state.image_pipeline.cleanup()
    
    if api_state.video_pipeline:
        api_state.video_pipeline.cleanup()


# Main function for running the server
def run_server(host: str = "127.0.0.1", port: int = 8000, reload: bool = False):
    """Run the FastAPI server."""
    uvicorn.run(
        "src.api.server:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info"
    )


if __name__ == "__main__":
    run_server()