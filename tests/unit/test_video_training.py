"""
Unit tests for video training components.

Tests the video LoRA adapter system, temporal consistency training,
and custom dataset functionality.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from src.core.interfaces import HardwareConfig
from src.pipelines.video_lora_adapter import (
    VideoLoRAAdapter, LoRAConfig, MotionLoRAManager, 
    MotionType, AdapterType, LinearLoRALayer
)
from src.pipelines.temporal_training import (
    TemporalTrainingPipeline, TemporalTrainingConfig, 
    TemporalConsistencyLoss, TrainingMode, LossType
)
from src.pipelines.video_dataset import (
    VideoDatasetFactory, DatasetConfig, DatasetType,
    MotionSpecificDataset, SyntheticVideoDataset
)


class TestVideoLoRAAdapter:
    """Test cases for VideoLoRAAdapter."""
    
    @pytest.fixture
    def lora_config(self):
        """Create test LoRA configuration."""
        return LoRAConfig(
            adapter_name="test_motion_adapter",
            adapter_type=AdapterType.MOTION_LORA,
            motion_type=MotionType.CAMERA_PAN,
            rank=16,
            alpha=32.0,
            dropout=0.1,
            target_modules=['attn', 'to_q', 'to_k', 'to_v']
        )
    
    @pytest.fixture
    def mock_model(self):
        """Create mock model for testing."""
        model = Mock()
        
        # Create mock modules
        mock_linear = Mock()
        mock_linear.in_features = 512
        mock_linear.out_features = 512
        mock_linear.weight = Mock()
        mock_linear.weight.data = Mock()
        mock_linear.weight.data.clone.return_value = Mock()
        
        # Set up named_modules
        model.named_modules.return_value = [
            ('transformer.attn.to_q', mock_linear),
            ('transformer.attn.to_k', mock_linear),
            ('transformer.attn.to_v', mock_linear),
        ]
        
        return model
    
    def test_lora_adapter_creation(self, lora_config):
        """Test LoRA adapter creation."""
        adapter = VideoLoRAAdapter(lora_config)
        
        assert adapter.config == lora_config
        assert not adapter.is_loaded
        assert adapter.target_model is None
        assert len(adapter.adapter_weights) == 0
    
    @patch('src.pipelines.video_lora_adapter.TORCH_AVAILABLE', True)
    @patch('src.pipelines.video_lora_adapter.nn')
    def test_create_adapter_layers(self, mock_nn, lora_config, mock_model):
        """Test creation of adapter layers."""
        # Mock nn.Linear as a proper type for isinstance
        mock_linear = type('MockLinear', (), {})
        mock_nn.Linear = mock_linear
        mock_nn.Conv2d = type('MockConv2d', (), {})
        mock_nn.Conv3d = type('MockConv3d', (), {})
        
        # Set up mock model with named_modules that returns matching modules
        mock_module = mock_linear()
        mock_model.named_modules.return_value = [
            ('transformer.attn.to_q', mock_module),
            ('transformer.attn.to_k', mock_module),
        ]
        
        adapter = VideoLoRAAdapter(lora_config)
        
        with patch.object(adapter, '_create_lora_layer', return_value=Mock()) as mock_create:
            adapter_layers = adapter.create_adapter_layers(mock_model)
            
            # Should create layers for matching modules
            assert len(adapter_layers) > 0
            assert mock_create.call_count > 0
    
    @patch('src.pipelines.video_lora_adapter.TORCH_AVAILABLE', True)
    def test_apply_to_model_success(self, lora_config, mock_model):
        """Test successful application of LoRA to model."""
        adapter = VideoLoRAAdapter(lora_config)
        
        with patch.object(adapter, 'create_adapter_layers', return_value={'test_layer': Mock()}), \
             patch.object(adapter, '_apply_lora_to_module'):
            
            result = adapter.apply_to_model(mock_model)
            
            assert result is True
            assert adapter.is_loaded
            assert adapter.target_model == mock_model
    
    def test_apply_to_model_no_layers(self, lora_config, mock_model):
        """Test application fails when no compatible layers found."""
        adapter = VideoLoRAAdapter(lora_config)
        
        with patch.object(adapter, 'create_adapter_layers', return_value={}):
            result = adapter.apply_to_model(mock_model)
            
            assert result is False
            assert not adapter.is_loaded
    
    def test_remove_from_model(self, lora_config, mock_model):
        """Test removal of LoRA from model."""
        adapter = VideoLoRAAdapter(lora_config)
        adapter.is_loaded = True
        adapter.target_model = mock_model
        adapter.original_weights = {'test_layer': Mock()}
        
        # Mock the model structure
        mock_module = Mock()
        mock_module.original_module = Mock()
        mock_model.named_modules.return_value = [('test_layer', mock_module)]
        
        result = adapter.remove_from_model()
        
        assert result is True
        assert not adapter.is_loaded
        assert adapter.target_model is None
        assert len(adapter.original_weights) == 0
    
    def test_remove_from_model_not_loaded(self, lora_config):
        """Test removal fails when adapter not loaded."""
        adapter = VideoLoRAAdapter(lora_config)
        
        result = adapter.remove_from_model()
        
        assert result is False
    
    @patch('src.pipelines.video_lora_adapter.torch')
    def test_save_adapter(self, mock_torch, lora_config):
        """Test saving LoRA adapter."""
        adapter = VideoLoRAAdapter(lora_config)
        
        # Mock adapter weights
        mock_layer = Mock()
        mock_layer.state_dict.return_value = {'weight': Mock()}
        adapter.adapter_weights = {'test_layer': mock_layer}
        
        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = Path(temp_dir) / "test_adapter.pt"
            
            result = adapter.save_adapter(save_path)
            
            assert result is True
            mock_torch.save.assert_called_once()
    
    def test_save_adapter_no_weights(self, lora_config):
        """Test saving fails when no weights available."""
        adapter = VideoLoRAAdapter(lora_config)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = Path(temp_dir) / "test_adapter.pt"
            
            result = adapter.save_adapter(save_path)
            
            assert result is False
    
    @patch('src.pipelines.video_lora_adapter.torch')
    def test_load_adapter(self, mock_torch, lora_config):
        """Test loading LoRA adapter."""
        adapter = VideoLoRAAdapter(lora_config)
        
        # Mock loaded state
        mock_state = {
            'config': {
                'adapter_name': 'loaded_adapter',
                'adapter_type': 'motion_lora',
                'motion_type': 'camera_pan',
                'rank': 16,
                'alpha': 32.0,
                'dropout': 0.1,
                'target_modules': ['attn']
            },
            'weights': {'test_layer': {'weight': Mock()}}
        }
        mock_torch.load.return_value = mock_state
        
        with tempfile.TemporaryDirectory() as temp_dir:
            load_path = Path(temp_dir) / "test_adapter.pt"
            load_path.touch()  # Create file
            
            result = adapter.load_adapter(load_path)
            
            assert result is True
            assert adapter.config.adapter_name == 'loaded_adapter'
            assert adapter.config.is_trained is True
    
    def test_load_adapter_file_not_found(self, lora_config):
        """Test loading fails when file not found."""
        adapter = VideoLoRAAdapter(lora_config)
        
        result = adapter.load_adapter(Path("nonexistent.pt"))
        
        assert result is False


class TestLinearLoRALayer:
    """Test cases for LinearLoRALayer."""
    
    @patch('src.pipelines.video_lora_adapter.TORCH_AVAILABLE', True)
    @patch('src.pipelines.video_lora_adapter.nn')
    @patch('src.pipelines.video_lora_adapter.math')
    def test_linear_lora_layer_creation(self, mock_math, mock_nn, ):
        """Test LinearLoRALayer creation."""
        # Mock nn.Linear and initialization functions
        mock_nn.Linear.return_value = Mock()
        mock_nn.Dropout.return_value = Mock()
        mock_nn.init.kaiming_uniform_ = Mock()
        mock_nn.init.zeros_ = Mock()
        mock_math.sqrt.return_value = 2.0
        
        layer = LinearLoRALayer(512, 512, 16, 32.0, 0.1)
        
        assert layer.rank == 16
        assert layer.alpha == 32.0
        assert layer.scaling == 32.0 / 16
    
    @patch('src.pipelines.video_lora_adapter.TORCH_AVAILABLE', True)
    def test_linear_lora_layer_forward(self):
        """Test LinearLoRALayer forward pass."""
        with patch('src.pipelines.video_lora_adapter.nn'), \
             patch('src.pipelines.video_lora_adapter.math'):
            
            layer = LinearLoRALayer(512, 512, 16, 32.0, 0.1)
            
            # Mock the sub-layers
            layer.lora_A = Mock()
            layer.lora_B = Mock()
            layer.dropout = Mock()
            
            # Mock forward pass
            mock_input = Mock()
            layer.lora_A.return_value = Mock()
            layer.dropout.return_value = Mock()
            layer.lora_B.return_value = Mock()
            
            result = layer.forward(mock_input)
            
            # Verify forward pass was called
            layer.lora_A.assert_called_once_with(mock_input)
            layer.dropout.assert_called_once()
            layer.lora_B.assert_called_once()


class TestMotionLoRAManager:
    """Test cases for MotionLoRAManager."""
    
    @pytest.fixture
    def hardware_config(self):
        """Create test hardware configuration."""
        return HardwareConfig(
            vram_size=8192,
            gpu_model="RTX 3070",
            cpu_cores=8,
            ram_size=16384,
            cuda_available=True,
            optimization_level="balanced"
        )
    
    def test_manager_creation(self, hardware_config):
        """Test MotionLoRAManager creation."""
        manager = MotionLoRAManager(hardware_config)
        
        assert manager.hardware_config == hardware_config
        assert len(manager.adapters) == 0
        assert len(manager.active_adapters) == 0
        assert len(manager.adapter_configs) > 0
    
    def test_create_motion_adapter(self, hardware_config):
        """Test creation of motion-specific adapter."""
        manager = MotionLoRAManager(hardware_config)
        
        adapter = manager.create_motion_adapter(
            "test_pan_adapter",
            MotionType.CAMERA_PAN
        )
        
        assert adapter is not None
        assert adapter.config.adapter_name == "test_pan_adapter"
        assert adapter.config.motion_type == MotionType.CAMERA_PAN
        assert "test_pan_adapter" in manager.adapters
    
    def test_create_adapter_low_vram_optimization(self):
        """Test adapter creation with low VRAM optimization."""
        low_vram_config = HardwareConfig(
            vram_size=4096,  # Low VRAM
            gpu_model="GTX 1650",
            cpu_cores=4,
            ram_size=8192,
            cuda_available=True,
            optimization_level="aggressive"
        )
        
        manager = MotionLoRAManager(low_vram_config)
        adapter = manager.create_motion_adapter("low_vram_adapter", MotionType.CAMERA_PAN)
        
        # Should have reduced rank for low VRAM
        assert adapter.config.rank <= 8
        assert adapter.config.dropout >= 0.2
    
    def test_load_adapter(self, hardware_config):
        """Test loading pre-trained adapter."""
        manager = MotionLoRAManager(hardware_config)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            adapter_path = Path(temp_dir) / "test_adapter.pt"
            
            # Mock successful loading
            with patch.object(VideoLoRAAdapter, 'load_adapter', return_value=True):
                result = manager.load_adapter("loaded_adapter", adapter_path)
                
                assert result is True
                assert "loaded_adapter" in manager.adapters
    
    def test_apply_adapter(self, hardware_config):
        """Test applying adapter to model."""
        manager = MotionLoRAManager(hardware_config)
        
        # Create adapter
        adapter = manager.create_motion_adapter("test_adapter", MotionType.CAMERA_PAN)
        
        # Mock model
        mock_model = Mock()
        
        with patch.object(adapter, 'apply_to_model', return_value=True):
            result = manager.apply_adapter("test_adapter", mock_model)
            
            assert result is True
            assert "test_adapter" in manager.active_adapters
    
    def test_apply_nonexistent_adapter(self, hardware_config):
        """Test applying nonexistent adapter fails."""
        manager = MotionLoRAManager(hardware_config)
        mock_model = Mock()
        
        result = manager.apply_adapter("nonexistent", mock_model)
        
        assert result is False
    
    def test_remove_adapter(self, hardware_config):
        """Test removing adapter from model."""
        manager = MotionLoRAManager(hardware_config)
        
        # Create and activate adapter
        adapter = manager.create_motion_adapter("test_adapter", MotionType.CAMERA_PAN)
        manager.active_adapters.append("test_adapter")
        
        with patch.object(adapter, 'remove_from_model', return_value=True):
            result = manager.remove_adapter("test_adapter")
            
            assert result is True
            assert "test_adapter" not in manager.active_adapters
    
    def test_get_adapter_info(self, hardware_config):
        """Test getting adapter information."""
        manager = MotionLoRAManager(hardware_config)
        adapter = manager.create_motion_adapter("info_adapter", MotionType.FLUID_MOTION)
        
        info = manager.get_adapter_info("info_adapter")
        
        assert info is not None
        assert info['adapter_name'] == "info_adapter"
        assert info['motion_type'] == MotionType.FLUID_MOTION.value
        assert 'rank' in info
        assert 'alpha' in info
    
    def test_get_adapter_info_nonexistent(self, hardware_config):
        """Test getting info for nonexistent adapter."""
        manager = MotionLoRAManager(hardware_config)
        
        info = manager.get_adapter_info("nonexistent")
        
        assert info is None
    
    def test_list_adapters(self, hardware_config):
        """Test listing all adapters."""
        manager = MotionLoRAManager(hardware_config)
        
        # Create multiple adapters
        manager.create_motion_adapter("adapter1", MotionType.CAMERA_PAN)
        manager.create_motion_adapter("adapter2", MotionType.CAMERA_ZOOM)
        
        adapter_list = manager.list_adapters()
        
        assert len(adapter_list) == 2
        assert "adapter1" in adapter_list
        assert "adapter2" in adapter_list
    
    def test_clear_all_adapters(self, hardware_config):
        """Test clearing all adapters."""
        manager = MotionLoRAManager(hardware_config)
        
        # Create and activate adapters
        adapter1 = manager.create_motion_adapter("adapter1", MotionType.CAMERA_PAN)
        adapter2 = manager.create_motion_adapter("adapter2", MotionType.CAMERA_ZOOM)
        manager.active_adapters = ["adapter1", "adapter2"]
        
        with patch.object(adapter1, 'remove_from_model', return_value=True), \
             patch.object(adapter2, 'remove_from_model', return_value=True):
            
            manager.clear_all_adapters()
            
            assert len(manager.adapters) == 0
            assert len(manager.active_adapters) == 0


class TestTemporalConsistencyLoss:
    """Test cases for TemporalConsistencyLoss."""
    
    @patch('src.pipelines.temporal_training.TORCH_AVAILABLE', True)
    def test_loss_creation(self):
        """Test TemporalConsistencyLoss creation."""
        with patch('src.pipelines.temporal_training.nn'):
            loss_fn = TemporalConsistencyLoss(
                temporal_weight=1.0,
                consistency_weight=0.5,
                perceptual_weight=0.3
            )
            
            assert loss_fn.temporal_weight == 1.0
            assert loss_fn.consistency_weight == 0.5
            assert loss_fn.perceptual_weight == 0.3
    
    @patch('src.pipelines.temporal_training.TORCH_AVAILABLE', True)
    @patch('src.pipelines.temporal_training.torch')
    @patch('src.pipelines.temporal_training.nn')
    def test_loss_forward(self, mock_nn, mock_torch):
        """Test TemporalConsistencyLoss forward pass."""
        # Mock tensor operations
        mock_tensor = Mock()
        mock_tensor.values.return_value = [mock_tensor]
        mock_torch.sum.return_value = mock_tensor
        
        # Mock MSE loss
        mock_mse = Mock()
        mock_mse.return_value = mock_tensor
        mock_nn.MSELoss.return_value = mock_mse
        
        loss_fn = TemporalConsistencyLoss()
        
        # Mock input tensors
        predicted = Mock()
        target = Mock()
        
        with patch.object(loss_fn, '_calculate_temporal_loss', return_value=mock_tensor), \
             patch.object(loss_fn, '_calculate_consistency_loss', return_value=mock_tensor), \
             patch.object(loss_fn, '_calculate_perceptual_loss', return_value=mock_tensor):
            
            losses = loss_fn.forward(predicted, target)
            
            assert 'total' in losses
            assert 'reconstruction' in losses
            assert 'temporal' in losses
            assert 'consistency' in losses


class TestTemporalTrainingPipeline:
    """Test cases for TemporalTrainingPipeline."""
    
    @pytest.fixture
    def training_config(self):
        """Create test training configuration."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            yield TemporalTrainingConfig(
                training_mode=TrainingMode.LORA_ONLY,
                loss_type=LossType.TEMPORAL_CONSISTENCY,
                learning_rate=1e-4,
                batch_size=2,
                num_epochs=1,
                gradient_accumulation_steps=1,
                warmup_steps=10,
                save_steps=50,
                eval_steps=25,
                max_grad_norm=1.0,
                temporal_weight=1.0,
                consistency_weight=0.5,
                perceptual_weight=0.3,
                use_mixed_precision=False,
                checkpoint_dir=temp_path / "checkpoints",
                log_dir=temp_path / "logs"
            )
    
    @pytest.fixture
    def hardware_config(self):
        """Create test hardware configuration."""
        return HardwareConfig(
            vram_size=8192,
            gpu_model="RTX 3070",
            cpu_cores=8,
            ram_size=16384,
            cuda_available=True,
            optimization_level="balanced"
        )
    
    def test_pipeline_creation(self, training_config, hardware_config):
        """Test TemporalTrainingPipeline creation."""
        pipeline = TemporalTrainingPipeline(training_config, hardware_config)
        
        assert pipeline.config == training_config
        assert pipeline.hardware_config == hardware_config
        assert pipeline.model is None
        assert pipeline.optimizer is None
    
    def test_setup_model_lora_only(self, training_config, hardware_config):
        """Test model setup for LoRA-only training."""
        pipeline = TemporalTrainingPipeline(training_config, hardware_config)
        
        # Mock model
        mock_model = Mock()
        mock_param = Mock()
        mock_param.requires_grad = True
        mock_model.parameters.return_value = [mock_param]
        mock_model.named_parameters.return_value = [('lora.weight', mock_param)]
        mock_model.to.return_value = mock_model
        
        with patch.object(pipeline, '_enable_lora_parameters'):
            result = pipeline.setup_model(mock_model)
            
            assert result is True
            assert pipeline.model == mock_model
    
    @patch('src.pipelines.temporal_training.DataLoader')
    @patch('src.pipelines.temporal_training.optim')
    def test_setup_training(self, mock_optim, mock_dataloader, training_config, hardware_config):
        """Test training setup."""
        pipeline = TemporalTrainingPipeline(training_config, hardware_config)
        
        # Mock model
        mock_model = Mock()
        mock_param = Mock()
        mock_param.requires_grad = True
        mock_model.parameters.return_value = [mock_param]
        pipeline.model = mock_model
        
        # Mock dataset
        mock_dataset = Mock()
        mock_dataset.__len__ = Mock(return_value=10)
        
        # Mock optimizer and scheduler
        mock_optimizer = Mock()
        mock_optim.AdamW.return_value = mock_optimizer
        
        result = pipeline.setup_training(mock_dataset)
        
        assert result is True
        assert pipeline.optimizer is not None
        assert pipeline.scheduler is not None
        assert pipeline.loss_function is not None


class TestVideoDataset:
    """Test cases for video dataset classes."""
    
    @pytest.fixture
    def dataset_config(self):
        """Create test dataset configuration."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield DatasetConfig(
                dataset_type=DatasetType.MOTION_SPECIFIC,
                data_root=Path(temp_dir),
                sequence_length=8,
                resolution=(256, 256),
                fps=8,
                overlap_ratio=0.5,
                min_motion_threshold=0.1,
                max_motion_threshold=1.0,
                augmentation_config={},
                compliance_mode="research_safe"
            )
    
    def test_synthetic_dataset_creation(self, dataset_config):
        """Test SyntheticVideoDataset creation."""
        motion_patterns = ['camera_pan', 'camera_zoom']
        dataset = SyntheticVideoDataset(dataset_config, motion_patterns, num_sequences=10)
        
        assert len(dataset) == 10
    
    @patch('src.pipelines.video_dataset.TORCH_AVAILABLE', True)
    @patch('src.pipelines.video_dataset.NUMPY_AVAILABLE', True)
    @patch('src.pipelines.video_dataset.torch')
    @patch('src.pipelines.video_dataset.np')
    def test_synthetic_dataset_getitem(self, mock_np, mock_torch, dataset_config):
        """Test SyntheticVideoDataset __getitem__."""
        # Mock numpy operations
        mock_np.random.seed = Mock()
        mock_np.zeros.return_value = Mock()
        mock_np.sin.return_value = 0.5
        mock_np.pi = 3.14159
        mock_np.stack.return_value = Mock()
        mock_np.clip.return_value = Mock()
        
        # Mock torch operations
        mock_tensor = Mock()
        mock_tensor.permute.return_value = mock_tensor
        mock_torch.from_numpy.return_value = mock_tensor
        
        motion_patterns = ['camera_pan']
        dataset = SyntheticVideoDataset(dataset_config, motion_patterns, num_sequences=5)
        
        item = dataset[0]
        
        assert 'frames' in item
        assert 'motion_pattern' in item
        assert 'annotations' in item
        assert item['motion_pattern'] == 'camera_pan'
        assert item['is_synthetic'] is True
    
    def test_dataset_factory_create_synthetic(self, dataset_config):
        """Test VideoDatasetFactory for synthetic dataset."""
        dataset = VideoDatasetFactory.create_dataset(
            DatasetType.SYNTHETIC,
            dataset_config,
            motion_patterns=['camera_pan', 'camera_zoom'],
            num_sequences=20
        )
        
        assert isinstance(dataset, SyntheticVideoDataset)
        assert len(dataset) == 20
    
    def test_dataset_factory_create_config(self):
        """Test VideoDatasetFactory config creation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = VideoDatasetFactory.create_config(
                DatasetType.MOTION_SPECIFIC,
                Path(temp_dir),
                sequence_length=16,
                resolution=(512, 512),
                fps=12
            )
            
            assert config.dataset_type == DatasetType.MOTION_SPECIFIC
            assert config.sequence_length == 16
            assert config.resolution == (512, 512)
            assert config.fps == 12
    
    def test_dataset_factory_unknown_type(self, dataset_config):
        """Test VideoDatasetFactory with unknown dataset type."""
        with pytest.raises(ValueError, match="Unknown dataset type"):
            VideoDatasetFactory.create_dataset("unknown_type", dataset_config)