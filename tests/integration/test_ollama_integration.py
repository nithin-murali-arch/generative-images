"""
Integration tests for Ollama Integration.

These tests require Ollama to be installed and running locally.
They test the actual integration with Ollama server and model management.
"""

import pytest
import time
import requests
from unittest.mock import Mock, patch, MagicMock

from src.core.ollama_integration import OllamaIntegration, ModelSize, GenerationConfig
from src.core.interfaces import HardwareConfig


@pytest.fixture
def hardware_config():
    """Create test hardware configuration."""
    return HardwareConfig(
        vram_size=8192,  # 8GB VRAM
        gpu_model="RTX 3070",
        cpu_cores=8,
        ram_size=16384,
        cuda_available=True,
        optimization_level="balanced"
    )


@pytest.fixture
def limited_hardware_config():
    """Create limited hardware configuration."""
    return HardwareConfig(
        vram_size=4096,  # 4GB VRAM
        gpu_model="GTX 1650",
        cpu_cores=4,
        ram_size=8192,
        cuda_available=True,
        optimization_level="aggressive"
    )


@pytest.fixture
def ollama_integration(hardware_config):
    """Create Ollama integration instance."""
    return OllamaIntegration(hardware_config)


class TestOllamaIntegrationInitialization:
    """Test Ollama integration initialization."""
    
    def test_initialization_success(self, hardware_config):
        """Test successful initialization."""
        integration = OllamaIntegration(hardware_config)
        
        assert integration.hardware_config == hardware_config
        assert integration.base_url == "http://localhost:11434"
        assert integration.current_model is None
        assert len(integration.model_specs) == 3
    
    def test_custom_base_url(self, hardware_config):
        """Test initialization with custom base URL."""
        custom_url = "http://custom-host:8080"
        integration = OllamaIntegration(hardware_config, base_url=custom_url)
        
        assert integration.base_url == custom_url
    
    def test_model_specs_loaded(self, ollama_integration):
        """Test that model specifications are properly loaded."""
        specs = ollama_integration.model_specs
        
        assert ModelSize.PHI3_MINI in specs
        assert ModelSize.LLAMA31_8B in specs
        assert ModelSize.LLAMA31_70B in specs
        
        # Check PHI3 mini specs
        phi3_spec = specs[ModelSize.PHI3_MINI]
        assert phi3_spec.name == "phi3:mini"
        assert phi3_spec.vram_requirement == 2300
        assert phi3_spec.supports_cpu_offload is True


class TestModelRecommendation:
    """Test model recommendation based on hardware."""
    
    def test_recommend_llama_for_high_vram(self, hardware_config):
        """Test that Llama 3.1 8B is recommended for high VRAM systems."""
        integration = OllamaIntegration(hardware_config)
        recommended = integration.get_recommended_model()
        
        assert recommended == ModelSize.LLAMA31_8B
    
    def test_recommend_phi3_for_limited_vram(self, limited_hardware_config):
        """Test that Phi-3 mini is recommended for limited VRAM systems."""
        integration = OllamaIntegration(limited_hardware_config)
        recommended = integration.get_recommended_model()
        
        assert recommended == ModelSize.PHI3_MINI
    
    def test_recommend_phi3_for_very_limited_vram(self, hardware_config):
        """Test recommendation for very limited VRAM."""
        hardware_config.vram_size = 2048  # 2GB VRAM
        integration = OllamaIntegration(hardware_config)
        
        with patch('src.core.ollama_integration.logger') as mock_logger:
            recommended = integration.get_recommended_model()
            assert recommended == ModelSize.PHI3_MINI
            mock_logger.warning.assert_called()


class TestServerManagement:
    """Test Ollama server management."""
    
    @patch('requests.get')
    def test_check_ollama_availability_success(self, mock_get, ollama_integration):
        """Test successful Ollama availability check."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_get.return_value = mock_response
        
        result = ollama_integration._check_ollama_availability()
        assert result is True
        mock_get.assert_called_with("http://localhost:11434/api/tags", timeout=5)
    
    @patch('requests.get')
    def test_check_ollama_availability_failure(self, mock_get, ollama_integration):
        """Test Ollama availability check failure."""
        mock_get.side_effect = requests.exceptions.RequestException("Connection failed")
        
        result = ollama_integration._check_ollama_availability()
        assert result is False
    
    @patch('subprocess.Popen')
    @patch.object(OllamaIntegration, '_check_ollama_availability')
    def test_start_ollama_server_success(self, mock_check, mock_popen, ollama_integration):
        """Test successful Ollama server start."""
        # First call returns False (not running), second returns True (started)
        mock_check.side_effect = [False, True]
        mock_process = Mock()
        mock_popen.return_value = mock_process
        
        result = ollama_integration.start_ollama_server()
        assert result is True
        mock_popen.assert_called_once()
    
    @patch('subprocess.Popen')
    def test_start_ollama_server_not_found(self, mock_popen, ollama_integration):
        """Test Ollama server start when Ollama is not installed."""
        mock_popen.side_effect = FileNotFoundError("Ollama not found")
        
        result = ollama_integration.start_ollama_server()
        assert result is False


class TestModelManagement:
    """Test model management functionality."""
    
    @patch('requests.get')
    def test_list_available_models_success(self, mock_get, ollama_integration):
        """Test successful model listing."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'models': [
                {'name': 'phi3:mini'},
                {'name': 'llama3.1:8b'},
                {'name': 'other:model'}  # Should be ignored
            ]
        }
        mock_get.return_value = mock_response
        
        available = ollama_integration.list_available_models()
        
        assert len(available) == 2
        assert 'phi3:mini' in available
        assert 'llama3.1:8b' in available
        assert 'other:model' not in available
    
    @patch('requests.get')
    def test_list_available_models_failure(self, mock_get, ollama_integration):
        """Test model listing failure."""
        mock_response = Mock()
        mock_response.status_code = 500
        mock_get.return_value = mock_response
        
        available = ollama_integration.list_available_models()
        assert len(available) == 0
    
    @patch('requests.post')
    def test_download_model_success(self, mock_post, ollama_integration):
        """Test successful model download."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.iter_lines.return_value = [
            b'{"status": "downloading"}',
            b'{"status": "complete"}'
        ]
        mock_post.return_value = mock_response
        
        result = ollama_integration.download_model(ModelSize.PHI3_MINI)
        assert result is True
        
        mock_post.assert_called_once()
        call_args = mock_post.call_args
        assert call_args[1]['json']['name'] == 'phi3:mini'
    
    @patch('requests.post')
    def test_download_model_failure(self, mock_post, ollama_integration):
        """Test model download failure."""
        mock_response = Mock()
        mock_response.status_code = 500
        mock_post.return_value = mock_response
        
        result = ollama_integration.download_model(ModelSize.PHI3_MINI)
        assert result is False


class TestModelLoading:
    """Test model loading functionality."""
    
    @patch('requests.post')
    def test_load_model_success(self, mock_post, ollama_integration):
        """Test successful model loading."""
        # Mock successful model availability check
        ollama_integration.available_models = {'phi3:mini': ollama_integration.model_specs[ModelSize.PHI3_MINI]}
        
        mock_response = Mock()
        mock_response.status_code = 200
        mock_post.return_value = mock_response
        
        result = ollama_integration.load_model(ModelSize.PHI3_MINI)
        assert result is True
        assert ollama_integration.current_model == 'phi3:mini'
    
    @patch('requests.post')
    def test_load_model_with_cpu_offload(self, mock_post, ollama_integration):
        """Test model loading with CPU offloading."""
        ollama_integration.available_models = {'phi3:mini': ollama_integration.model_specs[ModelSize.PHI3_MINI]}
        
        mock_response = Mock()
        mock_response.status_code = 200
        mock_post.return_value = mock_response
        
        result = ollama_integration.load_model(ModelSize.PHI3_MINI, cpu_offload=True)
        assert result is True
        
        # Check that CPU offloading was configured
        call_args = mock_post.call_args
        options = call_args[1]['json']['options']
        assert options['num_gpu'] == 0  # CPU only
        assert options['num_thread'] == ollama_integration.hardware_config.cpu_cores
    
    @patch.object(OllamaIntegration, 'download_model')
    @patch('requests.post')
    def test_load_model_downloads_if_missing(self, mock_post, mock_download, ollama_integration):
        """Test that model is downloaded if not available locally."""
        # Model not in available_models
        ollama_integration.available_models = {}
        
        mock_download.return_value = True
        mock_response = Mock()
        mock_response.status_code = 200
        mock_post.return_value = mock_response
        
        result = ollama_integration.load_model(ModelSize.PHI3_MINI)
        
        mock_download.assert_called_once_with(ModelSize.PHI3_MINI)
        assert result is True


class TestTextGeneration:
    """Test text generation functionality."""
    
    def test_generate_text_no_model_loaded(self, ollama_integration):
        """Test text generation when no model is loaded."""
        with pytest.raises(Exception, match="No model loaded"):
            ollama_integration.generate_text("Test prompt")
    
    @patch('requests.post')
    def test_generate_text_success(self, mock_post, ollama_integration):
        """Test successful text generation."""
        ollama_integration.current_model = 'phi3:mini'
        
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'response': 'Generated text response',
            'done': True,
            'context': [1, 2, 3]
        }
        mock_post.return_value = mock_response
        
        config = GenerationConfig(temperature=0.8, max_tokens=100)
        result = ollama_integration.generate_text("Test prompt", config)
        
        assert result['success'] is True
        assert result['text'] == 'Generated text response'
        assert result['model'] == 'phi3:mini'
        assert 'generation_time' in result
        
        # Check request parameters
        call_args = mock_post.call_args
        options = call_args[1]['json']['options']
        assert options['temperature'] == 0.8
        assert options['num_predict'] == 100
    
    @patch('requests.post')
    def test_generate_text_failure(self, mock_post, ollama_integration):
        """Test text generation failure."""
        ollama_integration.current_model = 'phi3:mini'
        
        mock_response = Mock()
        mock_response.status_code = 500
        mock_post.return_value = mock_response
        
        result = ollama_integration.generate_text("Test prompt")
        
        assert result['success'] is False
        assert 'error' in result
        assert 'generation_time' in result


class TestModelSwitching:
    """Test model switching functionality."""
    
    @patch.object(OllamaIntegration, 'load_model')
    def test_switch_model_success(self, mock_load, ollama_integration):
        """Test successful model switching."""
        ollama_integration.current_model = 'phi3:mini'
        mock_load.return_value = True
        
        result = ollama_integration.switch_model(ModelSize.LLAMA31_8B)
        
        assert result is True
        mock_load.assert_called_once_with(ModelSize.LLAMA31_8B, None)
    
    @patch.object(OllamaIntegration, 'load_model')
    def test_switch_to_same_model(self, mock_load, ollama_integration):
        """Test switching to the same model (should be no-op)."""
        ollama_integration.current_model = 'phi3:mini'
        
        result = ollama_integration.switch_model(ModelSize.PHI3_MINI)
        
        assert result is True
        mock_load.assert_not_called()
    
    @patch.object(OllamaIntegration, 'load_model')
    def test_switch_model_failure_with_rollback(self, mock_load, ollama_integration):
        """Test model switching failure with rollback."""
        ollama_integration.current_model = 'phi3:mini'
        
        # First call (new model) fails, second call (rollback) succeeds
        mock_load.side_effect = [False, True]
        
        result = ollama_integration.switch_model(ModelSize.LLAMA31_8B)
        
        assert result is False
        assert mock_load.call_count == 2  # Original attempt + rollback


class TestStatusAndOptimization:
    """Test status reporting and optimization functionality."""
    
    @patch('requests.get')
    def test_get_model_status_success(self, mock_get, ollama_integration):
        """Test successful model status retrieval."""
        ollama_integration.current_model = 'phi3:mini'
        ollama_integration.available_models = {'phi3:mini': Mock()}
        
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'models': [{'name': 'phi3:mini', 'size': '2.3GB'}]
        }
        mock_get.return_value = mock_response
        
        status = ollama_integration.get_model_status()
        
        assert status['current_model'] == 'phi3:mini'
        assert len(status['running_models']) == 1
        assert len(status['available_models']) == 1
        assert 'hardware_config' in status
    
    def test_optimize_for_hardware_high_vram(self, ollama_integration):
        """Test hardware optimization for high VRAM system."""
        config = ollama_integration.optimize_for_hardware()
        
        assert config['recommended_model'] == ModelSize.LLAMA31_8B.value
        assert config['cpu_offload_recommended'] is False
        assert config['generation_config']['max_tokens'] == 512
    
    def test_optimize_for_hardware_limited_vram(self, limited_hardware_config):
        """Test hardware optimization for limited VRAM system."""
        # Reduce VRAM to force CPU offloading
        limited_hardware_config.vram_size = 2048  # 2GB VRAM, less than PHI3's 2.3GB requirement
        integration = OllamaIntegration(limited_hardware_config)
        config = integration.optimize_for_hardware()
        
        assert config['recommended_model'] == ModelSize.PHI3_MINI.value
        assert config['cpu_offload_recommended'] is True
        assert config['generation_config']['max_tokens'] == 256


class TestHealthCheck:
    """Test health check functionality."""
    
    @patch.object(OllamaIntegration, '_check_ollama_availability')
    @patch.object(OllamaIntegration, 'list_available_models')
    @patch.object(OllamaIntegration, 'generate_text')
    def test_health_check_all_good(self, mock_generate, mock_list, mock_check, ollama_integration):
        """Test health check when everything is working."""
        ollama_integration.current_model = 'phi3:mini'
        
        mock_check.return_value = True
        mock_list.return_value = {'phi3:mini': Mock()}
        mock_generate.return_value = {'success': True}
        
        health = ollama_integration.health_check()
        
        assert health['server_running'] is True
        assert health['models_available'] == 1
        assert health['current_model_loaded'] is True
        assert health['can_generate'] is True
        assert len(health['errors']) == 0
    
    @patch.object(OllamaIntegration, '_check_ollama_availability')
    def test_health_check_server_down(self, mock_check, ollama_integration):
        """Test health check when server is down."""
        mock_check.return_value = False
        
        health = ollama_integration.health_check()
        
        assert health['server_running'] is False
        assert health['models_available'] == 0
        assert health['current_model_loaded'] is False
        assert health['can_generate'] is False
        assert 'Ollama server not running' in health['errors']


class TestCleanup:
    """Test cleanup functionality."""
    
    def test_cleanup_success(self, ollama_integration):
        """Test successful cleanup."""
        ollama_integration.current_model = 'phi3:mini'
        ollama_integration.available_models = {'phi3:mini': Mock()}
        
        ollama_integration.cleanup()
        
        assert ollama_integration.current_model is None
        assert len(ollama_integration.available_models) == 0


if __name__ == "__main__":
    pytest.main([__file__])