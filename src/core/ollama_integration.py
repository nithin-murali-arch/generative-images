"""
Ollama Integration for Local LLM Serving.

This module provides integration with Ollama for serving local LLMs including
Llama 3.1 8B and Phi-3-mini, with CPU offloading support for memory-constrained systems.
"""

import json
import time
import requests
import subprocess
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from .interfaces import HardwareConfig, SystemError
from .logging import get_logger

logger = get_logger(__name__)


class ModelSize(Enum):
    """Available model sizes for different hardware configurations."""
    PHI3_MINI = "phi3:mini"          # 3.8B parameters, ~2.3GB VRAM
    LLAMA31_8B = "llama3.1:8b"      # 8B parameters, ~4.7GB VRAM
    LLAMA31_70B = "llama3.1:70b"    # 70B parameters, ~40GB VRAM (CPU only)


@dataclass
class ModelInfo:
    """Information about an available model."""
    name: str
    size: str
    parameters: str
    vram_requirement: int  # MB
    supports_cpu_offload: bool
    download_size: int     # MB


@dataclass
class GenerationConfig:
    """Configuration for text generation."""
    temperature: float = 0.7
    max_tokens: int = 512
    top_p: float = 0.9
    top_k: int = 40
    repeat_penalty: float = 1.1
    stop_sequences: List[str] = None


class OllamaIntegration:
    """
    Integration with Ollama for local LLM serving.
    
    Provides model management, generation capabilities, and hardware optimization
    for running LLMs locally with CPU offloading support.
    """
    
    def __init__(self, hardware_config: HardwareConfig, base_url: str = "http://localhost:11434"):
        """Initialize Ollama integration."""
        self.hardware_config = hardware_config
        self.base_url = base_url
        self.current_model: Optional[str] = None
        self.available_models: Dict[str, ModelInfo] = {}
        
        # Model specifications
        self._init_model_specs()
        
        # Check Ollama availability
        self._check_ollama_availability()
        
        logger.info(f"Ollama integration initialized for {hardware_config.gpu_model}")
    
    def _init_model_specs(self) -> None:
        """Initialize model specifications."""
        self.model_specs = {
            ModelSize.PHI3_MINI: ModelInfo(
                name="phi3:mini",
                size="3.8B",
                parameters="3.8 billion",
                vram_requirement=2300,  # MB
                supports_cpu_offload=True,
                download_size=2200
            ),
            ModelSize.LLAMA31_8B: ModelInfo(
                name="llama3.1:8b",
                size="8B", 
                parameters="8 billion",
                vram_requirement=4700,  # MB
                supports_cpu_offload=True,
                download_size=4700
            ),
            ModelSize.LLAMA31_70B: ModelInfo(
                name="llama3.1:70b",
                size="70B",
                parameters="70 billion", 
                vram_requirement=40000,  # MB (CPU only for most systems)
                supports_cpu_offload=True,
                download_size=40000
            )
        }
    
    def _check_ollama_availability(self) -> bool:
        """Check if Ollama is running and accessible."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                logger.info("Ollama server is running and accessible")
                return True
            else:
                logger.warning(f"Ollama server responded with status {response.status_code}")
                return False
        except requests.exceptions.RequestException as e:
            logger.error(f"Cannot connect to Ollama server: {str(e)}")
            return False
    
    def start_ollama_server(self) -> bool:
        """Start Ollama server if not running."""
        try:
            # Check if already running
            if self._check_ollama_availability():
                return True
            
            logger.info("Starting Ollama server...")
            
            # Try to start Ollama
            process = subprocess.Popen(
                ["ollama", "serve"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, 'CREATE_NO_WINDOW') else 0
            )
            
            # Wait a moment for server to start
            time.sleep(3)
            
            # Check if it's now available
            if self._check_ollama_availability():
                logger.info("Ollama server started successfully")
                return True
            else:
                logger.error("Failed to start Ollama server")
                return False
                
        except FileNotFoundError:
            logger.error("Ollama not found. Please install Ollama first.")
            return False
        except Exception as e:
            logger.error(f"Error starting Ollama server: {str(e)}")
            return False
    
    def get_recommended_model(self) -> ModelSize:
        """Get recommended model based on hardware configuration."""
        vram_size = self.hardware_config.vram_size
        
        if vram_size >= 8000:  # 8GB+ VRAM
            return ModelSize.LLAMA31_8B
        elif vram_size >= 4000:  # 4GB+ VRAM  
            return ModelSize.PHI3_MINI
        else:  # Less than 4GB VRAM
            logger.warning("Very limited VRAM detected, will use CPU offloading")
            return ModelSize.PHI3_MINI
    
    def list_available_models(self) -> Dict[str, ModelInfo]:
        """List models available on the system."""
        try:
            response = requests.get(f"{self.base_url}/api/tags")
            if response.status_code == 200:
                data = response.json()
                available = {}
                
                for model in data.get('models', []):
                    model_name = model['name']
                    # Match with our known models
                    for model_size, model_info in self.model_specs.items():
                        if model_info.name in model_name:
                            available[model_name] = model_info
                            break
                
                self.available_models = available
                logger.info(f"Found {len(available)} available models")
                return available
            else:
                logger.error(f"Failed to list models: {response.status_code}")
                return {}
                
        except Exception as e:
            logger.error(f"Error listing models: {str(e)}")
            return {}
    
    def download_model(self, model_size: ModelSize) -> bool:
        """Download a model if not already available."""
        model_info = self.model_specs[model_size]
        model_name = model_info.name
        
        logger.info(f"Downloading model {model_name} ({model_info.download_size}MB)...")
        
        try:
            # Use Ollama pull command
            response = requests.post(
                f"{self.base_url}/api/pull",
                json={"name": model_name},
                stream=True,
                timeout=1800  # 30 minutes timeout for large models
            )
            
            if response.status_code == 200:
                # Stream the download progress
                for line in response.iter_lines():
                    if line:
                        try:
                            progress_data = json.loads(line)
                            if 'status' in progress_data:
                                logger.info(f"Download progress: {progress_data['status']}")
                        except json.JSONDecodeError:
                            continue
                
                logger.info(f"Successfully downloaded {model_name}")
                return True
            else:
                logger.error(f"Failed to download model: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"Error downloading model: {str(e)}")
            return False
    
    def load_model(self, model_size: ModelSize, cpu_offload: bool = None) -> bool:
        """Load a model for inference."""
        model_info = self.model_specs[model_size]
        model_name = model_info.name
        
        # Auto-determine CPU offloading if not specified
        if cpu_offload is None:
            cpu_offload = self.hardware_config.vram_size < model_info.vram_requirement
        
        logger.info(f"Loading model {model_name} (CPU offload: {cpu_offload})")
        
        try:
            # Check if model is available
            if model_name not in self.available_models:
                logger.info(f"Model {model_name} not found locally, downloading...")
                if not self.download_model(model_size):
                    return False
            
            # Load the model with a simple generation request
            load_payload = {
                "model": model_name,
                "prompt": "Hello",
                "stream": False,
                "options": {
                    "num_ctx": 2048,  # Context length
                    "num_gpu": 0 if cpu_offload else -1,  # -1 = auto, 0 = CPU only
                    "num_thread": self.hardware_config.cpu_cores if cpu_offload else 4
                }
            }
            
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=load_payload,
                timeout=120  # 2 minutes timeout for model loading
            )
            
            if response.status_code == 200:
                self.current_model = model_name
                logger.info(f"Successfully loaded {model_name}")
                return True
            else:
                logger.error(f"Failed to load model: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return False
    
    def generate_text(self, prompt: str, config: GenerationConfig = None) -> Dict[str, Any]:
        """Generate text using the currently loaded model."""
        if not self.current_model:
            raise SystemError("No model loaded. Please load a model first.")
        
        if config is None:
            config = GenerationConfig()
        
        logger.debug(f"Generating text with {self.current_model}")
        
        try:
            payload = {
                "model": self.current_model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": config.temperature,
                    "num_predict": config.max_tokens,
                    "top_p": config.top_p,
                    "top_k": config.top_k,
                    "repeat_penalty": config.repeat_penalty,
                    "stop": config.stop_sequences or []
                }
            }
            
            start_time = time.time()
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=300  # 5 minutes timeout
            )
            generation_time = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                
                return {
                    'success': True,
                    'text': data.get('response', ''),
                    'model': self.current_model,
                    'generation_time': generation_time,
                    'tokens_generated': len(data.get('response', '').split()),
                    'context_length': data.get('context', []),
                    'done': data.get('done', True)
                }
            else:
                logger.error(f"Generation failed: {response.status_code}")
                return {
                    'success': False,
                    'error': f"HTTP {response.status_code}",
                    'generation_time': generation_time
                }
                
        except Exception as e:
            logger.error(f"Error during text generation: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'generation_time': 0
            }
    
    def switch_model(self, new_model_size: ModelSize, cpu_offload: bool = None) -> bool:
        """Switch to a different model."""
        new_model_info = self.model_specs[new_model_size]
        new_model_name = new_model_info.name
        
        if self.current_model == new_model_name:
            logger.info(f"Model {new_model_name} is already loaded")
            return True
        
        logger.info(f"Switching from {self.current_model} to {new_model_name}")
        
        # Unload current model (Ollama handles this automatically)
        old_model = self.current_model
        self.current_model = None
        
        # Load new model
        success = self.load_model(new_model_size, cpu_offload)
        
        if success:
            logger.info(f"Successfully switched to {new_model_name}")
        else:
            logger.error(f"Failed to switch to {new_model_name}, reverting...")
            # Try to reload the old model
            if old_model:
                for model_size, model_info in self.model_specs.items():
                    if model_info.name == old_model:
                        self.load_model(model_size, cpu_offload)
                        break
        
        return success
    
    def get_model_status(self) -> Dict[str, Any]:
        """Get current model status and system information."""
        try:
            # Get running models
            response = requests.get(f"{self.base_url}/api/ps")
            running_models = []
            
            if response.status_code == 200:
                data = response.json()
                running_models = data.get('models', [])
            
            return {
                'current_model': self.current_model,
                'running_models': running_models,
                'available_models': list(self.available_models.keys()),
                'hardware_config': {
                    'vram_size': self.hardware_config.vram_size,
                    'gpu_model': self.hardware_config.gpu_model,
                    'cpu_cores': self.hardware_config.cpu_cores
                },
                'server_status': 'running' if self._check_ollama_availability() else 'stopped'
            }
            
        except Exception as e:
            logger.error(f"Error getting model status: {str(e)}")
            return {
                'current_model': self.current_model,
                'error': str(e),
                'server_status': 'error'
            }
    
    def optimize_for_hardware(self) -> Dict[str, Any]:
        """Optimize model selection and configuration for current hardware."""
        recommended_model = self.get_recommended_model()
        model_info = self.model_specs[recommended_model]
        
        # Determine if CPU offloading is needed
        needs_cpu_offload = self.hardware_config.vram_size < model_info.vram_requirement
        
        optimization_config = {
            'recommended_model': recommended_model.value,
            'model_info': {
                'name': model_info.name,
                'parameters': model_info.parameters,
                'vram_requirement': model_info.vram_requirement
            },
            'cpu_offload_recommended': needs_cpu_offload,
            'optimization_level': self.hardware_config.optimization_level,
            'generation_config': {
                'max_tokens': 256 if needs_cpu_offload else 512,
                'temperature': 0.7,
                'batch_size': 1  # Always use batch size 1 for memory efficiency
            }
        }
        
        logger.info(f"Hardware optimization: {recommended_model.value} with CPU offload: {needs_cpu_offload}")
        
        return optimization_config
    
    def cleanup(self) -> None:
        """Clean up resources and unload models."""
        logger.info("Cleaning up Ollama integration...")
        
        try:
            # Ollama automatically manages model unloading
            # Just reset our state
            self.current_model = None
            self.available_models.clear()
            
            logger.info("Ollama integration cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")
    
    def health_check(self) -> Dict[str, Any]:
        """Perform a health check of the Ollama integration."""
        health_status = {
            'server_running': False,
            'models_available': 0,
            'current_model_loaded': False,
            'can_generate': False,
            'errors': []
        }
        
        try:
            # Check server
            health_status['server_running'] = self._check_ollama_availability()
            
            if health_status['server_running']:
                # Check available models
                available = self.list_available_models()
                health_status['models_available'] = len(available)
                
                # Check current model
                health_status['current_model_loaded'] = self.current_model is not None
                
                # Test generation if model is loaded
                if self.current_model:
                    test_result = self.generate_text("Test", GenerationConfig(max_tokens=5))
                    health_status['can_generate'] = test_result.get('success', False)
                    if not health_status['can_generate']:
                        health_status['errors'].append(f"Generation test failed: {test_result.get('error', 'Unknown error')}")
            else:
                health_status['errors'].append("Ollama server not running")
                
        except Exception as e:
            health_status['errors'].append(f"Health check error: {str(e)}")
        
        return health_status