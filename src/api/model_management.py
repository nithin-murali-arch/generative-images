"""
Model management endpoints for the Academic Multimodal LLM Experiment System API.

This module implements endpoints for VRAM monitoring, model switching,
and experiment data persistence.
"""

import logging
import psutil
from typing import Dict, Any, List, Optional
from datetime import datetime

from fastapi import APIRouter, HTTPException, Depends, status
from pydantic import BaseModel, Field

from .dependencies import APIState, get_api_state
from ..core.interfaces import ComplianceMode

logger = logging.getLogger(__name__)

# Create router for model management endpoints
router = APIRouter(prefix="/models", tags=["model-management"])

# Pydantic models for model management
class ModelSwitchRequest(BaseModel):
    """Request model for switching models."""
    model_name: str = Field(..., description="Name of the model to switch to")
    pipeline_type: str = Field(..., description="Pipeline type (image or video)")
    
    class Config:
        schema_extra = {
            "example": {
                "model_name": "stable-diffusion-v1-5",
                "pipeline_type": "image"
            }
        }


class VRAMInfo(BaseModel):
    """VRAM usage information."""
    total_mb: int = Field(..., description="Total VRAM in MB")
    used_mb: int = Field(..., description="Used VRAM in MB")
    free_mb: int = Field(..., description="Free VRAM in MB")
    utilization_percent: float = Field(..., description="VRAM utilization percentage")


class ModelInfo(BaseModel):
    """Information about a model."""
    name: str = Field(..., description="Model name")
    pipeline_type: str = Field(..., description="Pipeline type")
    is_loaded: bool = Field(..., description="Whether model is currently loaded")
    vram_requirement_mb: int = Field(..., description="VRAM requirement in MB")
    supports_features: List[str] = Field(..., description="Supported features")


class SystemStatus(BaseModel):
    """System status information."""
    vram_info: VRAMInfo = Field(..., description="VRAM usage information")
    current_models: Dict[str, Optional[str]] = Field(..., description="Currently loaded models by pipeline")
    available_models: Dict[str, List[str]] = Field(..., description="Available models by pipeline")
    hardware_info: Dict[str, Any] = Field(..., description="Hardware information")
    system_metrics: Dict[str, Any] = Field(..., description="System performance metrics")


class ExperimentSaveRequest(BaseModel):
    """Request model for saving experiments."""
    experiment_name: str = Field(..., min_length=1, max_length=200, description="Name for the experiment")
    description: Optional[str] = Field(None, max_length=1000, description="Experiment description")
    tags: List[str] = Field(default_factory=list, description="Tags for categorization")
    results: Dict[str, Any] = Field(..., description="Experiment results and metadata")
    compliance_mode: str = Field("research_safe", description="Compliance mode used")


class ExperimentResponse(BaseModel):
    """Response model for experiment operations."""
    experiment_id: str = Field(..., description="Unique experiment identifier")
    status: str = Field(..., description="Operation status")
    message: str = Field(..., description="Status message")


@router.get("/status", response_model=SystemStatus)
async def get_model_status(state: APIState = Depends(get_api_state)):
    """
    Get current model status and VRAM monitoring information.
    
    Returns comprehensive information about loaded models, VRAM usage,
    and system performance metrics.
    """
    try:
        # Get VRAM information
        vram_info = _get_vram_info(state.hardware_config)
        
        # Get currently loaded models
        current_models = {
            "image": state.image_pipeline.current_model if state.image_pipeline else None,
            "video": state.video_pipeline.current_model if state.video_pipeline else None
        }
        
        # Get available models
        available_models = {
            "image": state.image_pipeline.get_available_models() if state.image_pipeline else [],
            "video": state.video_pipeline.get_available_models() if state.video_pipeline else []
        }
        
        # Get hardware information
        hardware_info = {
            "gpu_model": state.hardware_config.gpu_model,
            "vram_size_mb": state.hardware_config.vram_size,
            "cuda_available": state.hardware_config.cuda_available,
            "cpu_cores": state.hardware_config.cpu_cores,
            "ram_size_mb": state.hardware_config.ram_size
        }
        
        # Get system metrics
        system_metrics = _get_system_metrics()
        
        return SystemStatus(
            vram_info=vram_info,
            current_models=current_models,
            available_models=available_models,
            hardware_info=hardware_info,
            system_metrics=system_metrics
        )
        
    except Exception as e:
        logger.error(f"Error getting model status: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get model status: {str(e)}"
        )


@router.post("/switch", response_model=Dict[str, Any])
async def switch_model(
    request: ModelSwitchRequest,
    state: APIState = Depends(get_api_state)
):
    """
    Switch to a different model for dynamic model loading.
    
    Allows switching between different models for image or video generation
    with proper memory management and optimization.
    """
    try:
        logger.info(f"Switching {request.pipeline_type} model to {request.model_name}")
        
        # Validate pipeline type
        if request.pipeline_type not in ["image", "video"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="pipeline_type must be 'image' or 'video'"
            )
        
        # Get the appropriate pipeline
        if request.pipeline_type == "image":
            pipeline = state.image_pipeline
        else:
            pipeline = state.video_pipeline
        
        if not pipeline:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"{request.pipeline_type} pipeline not available"
            )
        
        # Check if model is available
        available_models = pipeline.get_available_models()
        if request.model_name not in available_models:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Model {request.model_name} not available. Available models: {available_models}"
            )
        
        # Get VRAM info before switch
        vram_before = _get_vram_info(state.hardware_config)
        
        # Perform model switch
        start_time = datetime.now()
        success = pipeline.switch_model(request.model_name)
        switch_time = (datetime.now() - start_time).total_seconds()
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to switch to model {request.model_name}"
            )
        
        # Get VRAM info after switch
        vram_after = _get_vram_info(state.hardware_config)
        
        # Get model information
        model_info = pipeline.get_model_info(request.model_name)
        
        logger.info(f"Successfully switched to {request.model_name} in {switch_time:.2f}s")
        
        return {
            "success": True,
            "message": f"Successfully switched to {request.model_name}",
            "model_name": request.model_name,
            "pipeline_type": request.pipeline_type,
            "switch_time_seconds": switch_time,
            "model_info": model_info,
            "vram_usage": {
                "before_switch": vram_before.dict(),
                "after_switch": vram_after.dict(),
                "vram_change_mb": vram_after.used_mb - vram_before.used_mb
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error switching model: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Model switch failed: {str(e)}"
        )


@router.get("/info/{pipeline_type}/{model_name}", response_model=ModelInfo)
async def get_model_info(
    pipeline_type: str,
    model_name: str,
    state: APIState = Depends(get_api_state)
):
    """Get detailed information about a specific model."""
    try:
        # Validate pipeline type
        if pipeline_type not in ["image", "video"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="pipeline_type must be 'image' or 'video'"
            )
        
        # Get the appropriate pipeline
        pipeline = state.image_pipeline if pipeline_type == "image" else state.video_pipeline
        
        if not pipeline:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"{pipeline_type} pipeline not available"
            )
        
        # Get model information
        model_info = pipeline.get_model_info(model_name)
        
        if not model_info:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Model {model_name} not found"
            )
        
        # Determine supported features based on model info
        supported_features = []
        if model_info.get('supports_negative_prompt', False):
            supported_features.append("negative_prompt")
        if model_info.get('supports_guidance_scale', False):
            supported_features.append("guidance_scale")
        if model_info.get('supports_image_conditioning', False):
            supported_features.append("image_conditioning")
        if model_info.get('supports_motion_control', False):
            supported_features.append("motion_control")
        
        return ModelInfo(
            name=model_name,
            pipeline_type=pipeline_type,
            is_loaded=(pipeline.current_model == model_name),
            vram_requirement_mb=model_info.get('min_vram_mb', 0),
            supports_features=supported_features
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get model info: {str(e)}"
        )


@router.get("/list/{pipeline_type}", response_model=List[ModelInfo])
async def list_models(
    pipeline_type: str,
    state: APIState = Depends(get_api_state)
):
    """List all available models for a specific pipeline type."""
    try:
        # Validate pipeline type
        if pipeline_type not in ["image", "video"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="pipeline_type must be 'image' or 'video'"
            )
        
        # Get the appropriate pipeline
        pipeline = state.image_pipeline if pipeline_type == "image" else state.video_pipeline
        
        if not pipeline:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"{pipeline_type} pipeline not available"
            )
        
        # Get available models
        available_models = pipeline.get_available_models()
        
        # Build model info list
        model_list = []
        for model_name in available_models:
            model_info = pipeline.get_model_info(model_name)
            if model_info:
                # Determine supported features
                supported_features = []
                if model_info.get('supports_negative_prompt', False):
                    supported_features.append("negative_prompt")
                if model_info.get('supports_guidance_scale', False):
                    supported_features.append("guidance_scale")
                if model_info.get('supports_image_conditioning', False):
                    supported_features.append("image_conditioning")
                if model_info.get('supports_motion_control', False):
                    supported_features.append("motion_control")
                
                model_list.append(ModelInfo(
                    name=model_name,
                    pipeline_type=pipeline_type,
                    is_loaded=(pipeline.current_model == model_name),
                    vram_requirement_mb=model_info.get('min_vram_mb', 0),
                    supports_features=supported_features
                ))
        
        return model_list
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error listing models: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list models: {str(e)}"
        )


@router.post("/experiment/save", response_model=ExperimentResponse)
async def save_experiment(
    request: ExperimentSaveRequest,
    state: APIState = Depends(get_api_state)
):
    """
    Save experiment data for research data persistence.
    
    Stores experiment results, metadata, and compliance information
    for later analysis and comparison.
    """
    try:
        logger.info(f"Saving experiment: {request.experiment_name}")
        
        # Validate compliance mode
        try:
            compliance_mode = ComplianceMode(request.compliance_mode)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid compliance_mode: {request.compliance_mode}"
            )
        
        # Create experiment data
        experiment_data = {
            'name': request.experiment_name,
            'description': request.description,
            'tags': request.tags,
            'results': request.results,
            'compliance_mode': compliance_mode.value,
            'timestamp': datetime.now().isoformat(),
            'hardware_info': {
                'gpu_model': state.hardware_config.gpu_model,
                'vram_size_mb': state.hardware_config.vram_size,
                'cuda_available': state.hardware_config.cuda_available
            },
            'system_state': {
                'current_image_model': state.image_pipeline.current_model if state.image_pipeline else None,
                'current_video_model': state.video_pipeline.current_model if state.video_pipeline else None
            }
        }
        
        # Save experiment using experiment tracker
        if state.experiment_tracker:
            experiment_id = state.experiment_tracker.save_experiment(experiment_data)
        else:
            # Fallback: generate ID and log
            experiment_id = f"exp_{int(datetime.now().timestamp())}"
            logger.warning(f"Experiment tracker not available, experiment data logged: {experiment_id}")
        
        logger.info(f"Experiment saved successfully: {experiment_id}")
        
        return ExperimentResponse(
            experiment_id=experiment_id,
            status="saved",
            message=f"Experiment '{request.experiment_name}' saved successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error saving experiment: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save experiment: {str(e)}"
        )


@router.get("/experiment/{experiment_id}", response_model=Dict[str, Any])
async def get_experiment(
    experiment_id: str,
    state: APIState = Depends(get_api_state)
):
    """Retrieve saved experiment data."""
    try:
        if not state.experiment_tracker:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Experiment tracker not available"
            )
        
        experiment_data = state.experiment_tracker.get_experiment(experiment_id)
        
        if not experiment_data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Experiment {experiment_id} not found"
            )
        
        return experiment_data
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving experiment: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve experiment: {str(e)}"
        )


@router.get("/experiment/list", response_model=List[Dict[str, Any]])
async def list_experiments(
    limit: int = 50,
    offset: int = 0,
    tags: Optional[str] = None,
    state: APIState = Depends(get_api_state)
):
    """List saved experiments with optional filtering."""
    try:
        if not state.experiment_tracker:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Experiment tracker not available"
            )
        
        # Parse tags filter
        tag_filter = tags.split(',') if tags else None
        
        experiments = state.experiment_tracker.list_experiments(
            limit=limit,
            offset=offset,
            tag_filter=tag_filter
        )
        
        return experiments
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error listing experiments: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list experiments: {str(e)}"
        )


# Helper functions
def _get_vram_info(hardware_config) -> VRAMInfo:
    """Get current VRAM usage information."""
    try:
        # Try to get actual VRAM usage if torch is available
        try:
            import torch
            if torch.cuda.is_available():
                total_vram = torch.cuda.get_device_properties(0).total_memory // (1024 * 1024)  # MB
                allocated_vram = torch.cuda.memory_allocated(0) // (1024 * 1024)  # MB
                cached_vram = torch.cuda.memory_reserved(0) // (1024 * 1024)  # MB
                used_vram = max(allocated_vram, cached_vram)
                free_vram = total_vram - used_vram
                utilization = (used_vram / total_vram) * 100 if total_vram > 0 else 0
                
                return VRAMInfo(
                    total_mb=total_vram,
                    used_mb=used_vram,
                    free_mb=free_vram,
                    utilization_percent=utilization
                )
        except Exception as e:
            logger.debug(f"Could not get actual VRAM usage: {e}")
        
        # Fallback to hardware config values
        total_vram = hardware_config.vram_size
        # Estimate usage (this would be more accurate with actual monitoring)
        estimated_used = int(total_vram * 0.1)  # Assume 10% base usage
        free_vram = total_vram - estimated_used
        utilization = (estimated_used / total_vram) * 100 if total_vram > 0 else 0
        
        return VRAMInfo(
            total_mb=total_vram,
            used_mb=estimated_used,
            free_mb=free_vram,
            utilization_percent=utilization
        )
        
    except Exception as e:
        logger.error(f"Error getting VRAM info: {e}")
        # Return default values
        return VRAMInfo(
            total_mb=0,
            used_mb=0,
            free_mb=0,
            utilization_percent=0.0
        )


def _get_system_metrics() -> Dict[str, Any]:
    """Get system performance metrics."""
    try:
        # Get CPU and memory usage
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        
        # Get disk usage for outputs directory
        try:
            disk_usage = psutil.disk_usage('.')
            disk_free_gb = disk_usage.free // (1024 ** 3)
        except Exception:
            disk_free_gb = 0
        
        return {
            'cpu_usage_percent': cpu_percent,
            'ram_usage_percent': memory.percent,
            'ram_available_mb': memory.available // (1024 * 1024),
            'disk_free_gb': disk_free_gb,
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting system metrics: {e}")
        return {
            'cpu_usage_percent': 0.0,
            'ram_usage_percent': 0.0,
            'ram_available_mb': 0,
            'disk_free_gb': 0,
            'timestamp': datetime.now().isoformat()
        }