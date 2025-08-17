"""
Unit tests for LoRA training pipeline.

Tests LoRA training functionality with copyright-aware dataset loading,
hardware optimization, and compliance mode validation.
"""

import pytest
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

from src.core.interfaces import HardwareConfig, ComplianceMode
from src.pipelines.lora_training import (
    LoRAConfig, LoRATrainer, LoRADataset, TrainingProgress, 
    TrainingResult, LoRATarget
)
from src.data.dataset_organizer import OrganizedDataset, DatasetStats, DatasetConfig
from src.data.license_classifier import LicenseType


class TestLoRAConfig:
    """Test LoRA configuration."""
    
    def test_default_config(self):
        """Test default LoRA configuration."""
        config = LoRAConfig()
        
        assert config.rank == 4
        assert config.alpha == 32.0
        assert config.dropout == 0.1
        assert config.learning_rate == 1e-4
        assert config.batch_size == 1
        assert config.num_epochs == 10
        assert config.resolution == 512
        assert config.compliance_mode == ComplianceMode.RESEARCH_SAFE
    
    def test_custom_config(self):
        """Test custom LoRA configuration."""
        config = LoRAConfig(
            rank=16,
            alpha=64.0,
            learning_rate=5e-4,
            batch_size=2,
            resolution=768,
            compliance_mode=ComplianceMode.OPEN_SOURCE_ONLY
        )
        
        assert config.rank == 16
        assert config.alpha == 64.0
        assert config.learning_rate == 5e-4
        assert config.batch_size == 2
        assert config.resolution == 768
        assert config.compliance_mode == ComplianceMode.OPEN_SOURCE_ONLY


class TestLoRADataset:
    """Test LoRA dataset functionality."""
    
    @pytest.fixture
    def mock_organized_dataset(self):
        """Create mock organized dataset."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create mock train split
            train_split_path = temp_path / "train_split.json"
            train_data = [
                {
                    "url": "http://example.com/image1.jpg",
                    "local_path": str(temp_path / "image1.jpg"),
                    "title": "Test image 1",
                    "attribution": "Test attribution",
                    "license_type": "creative_commons"
                },
                {
                    "url": "http://example.com/image2.jpg", 
                    "local_path": str(temp_path / "image2.jpg"),
                    "title": "Test image 2",
                    "attribution": "Test attribution 2",
                    "license_type": "public_domain"
                }
            ]
            
            with open(train_split_path, 'w') as f:
                json.dump(train_data, f)
            
            # Create mock images
            for i in range(1, 3):
                image_path = temp_path / f"image{i}.jpg"
                image_path.touch()
            
            stats = DatasetStats(
                total_items=2,
                license_distribution={"creative_commons": 1, "public_domain": 1},
                compliance_safe_count=2,
                compliance_safe_percentage=100.0
            )
            
            dataset = OrganizedDataset(
                name="test_dataset",
                path=temp_path,
                config=DatasetConfig(),
                stats=stats,
                train_split_path=train_split_path,
                val_split_path=None
            )
            
            yield dataset
    
    @pytest.fixture
    def mock_tokenizer(self):
        """Create mock tokenizer."""
        tokenizer = Mock()
        tokenizer.model_max_length = 77
        tokenizer.return_value = Mock(
            input_ids=Mock(squeeze=Mock(return_value="mock_input_ids")),
            attention_mask=Mock(squeeze=Mock(return_value="mock_attention_mask"))
        )
        return tokenizer
    
    def test_dataset_initialization(self, mock_organized_dataset, mock_tokenizer):
        """Test LoRA dataset initialization."""
        config = LoRAConfig()
        
        dataset = LoRADataset(mock_organized_dataset, config, mock_tokenizer, "train")
        
        assert len(dataset) == 2
        assert dataset.split == "train"
        assert dataset.config == config
    
    @patch('src.pipelines.lora_training.PIL_AVAILABLE', False)
    @patch('src.pipelines.lora_training.TORCH_AVAILABLE', False)
    def test_dataset_getitem_no_pil(self, mock_organized_dataset, mock_tokenizer):
        """Test dataset __getitem__ without PIL."""
        config = LoRAConfig()
        dataset = LoRADataset(mock_organized_dataset, config, mock_tokenizer, "train")
        
        with patch('src.pipelines.lora_training.torch') as mock_torch:
            mock_torch.randn.return_value = "mock_tensor"
            
            item = dataset[0]
            
            assert item['pixel_values'] == "mock_tensor"
            assert item['prompt'] == "Test image 1"
            assert item['attribution'] == "Test attribution"
            assert item['license_type'] == "creative_commons"


class TestLoRATrainer:
    """Test LoRA trainer functionality."""
    
    @pytest.fixture
    def hardware_config(self):
        """Create test hardware configuration."""
        return HardwareConfig(
            vram_size=8192,
            gpu_model="RTX 3070",
            cpu_cores=8,
            ram_size=32768,
            cuda_available=True,
            optimization_level="balanced"
        )
    
    @pytest.fixture
    def mock_dataset_organizer(self):
        """Mock dataset organizer."""
        with patch('src.pipelines.lora_training.DatasetOrganizer') as mock_organizer_class:
            mock_organizer = Mock()
            mock_organizer_class.return_value = mock_organizer
            
            # Mock dataset
            stats = DatasetStats(
                total_items=100,
                compliance_safe_count=100,
                compliance_safe_percentage=100.0
            )
            
            mock_dataset = OrganizedDataset(
                name="test_dataset",
                path=Path("/mock/path"),
                config=DatasetConfig(),
                stats=stats
            )
            
            mock_organizer.load_dataset.return_value = mock_dataset
            mock_organizer.get_compliance_summary.return_value = {
                "compliance_safe_percentage": 100.0
            }
            
            yield mock_organizer
    
    def test_trainer_initialization(self, hardware_config):
        """Test LoRA trainer initialization."""
        trainer = LoRATrainer(hardware_config)
        
        assert trainer.hardware_config == hardware_config
        assert trainer.memory_manager is not None
        assert trainer.profile_manager is not None
        assert not trainer.is_training
        assert trainer.current_progress is None
    
    @patch('src.pipelines.lora_training.TORCH_AVAILABLE', False)
    @patch('src.pipelines.lora_training.DIFFUSERS_AVAILABLE', False)
    def test_train_no_dependencies(self, hardware_config, mock_dataset_organizer):
        """Test training without required dependencies."""
        trainer = LoRATrainer(hardware_config)
        config = LoRAConfig()
        
        result = trainer.train("test_dataset", config)
        
        assert not result.success
        assert "dependencies not available" in result.error_message
    
    @patch('src.pipelines.lora_training.TORCH_AVAILABLE', True)
    @patch('src.pipelines.lora_training.DIFFUSERS_AVAILABLE', True)
    @patch('src.pipelines.lora_training.TRANSFORMERS_AVAILABLE', True)
    def test_train_success_mock(self, hardware_config, mock_dataset_organizer):
        """Test successful training with mocked components."""
        trainer = LoRATrainer(hardware_config)
        config = LoRAConfig(num_epochs=1)  # Short training for test
        
        # Mock training components
        with patch.object(trainer, '_initialize_training_components', return_value=True), \
             patch.object(trainer, '_setup_lora_layers', return_value=True), \
             patch.object(trainer, '_create_data_loaders') as mock_loaders, \
             patch.object(trainer, '_setup_optimizer_and_scheduler') as mock_opt, \
             patch.object(trainer, '_training_loop', return_value=0.1), \
             patch.object(trainer, '_save_lora_weights') as mock_save:
            
            # Setup mocks
            mock_train_loader = Mock()
            mock_train_loader.__len__ = Mock(return_value=10)
            mock_loaders.return_value = (mock_train_loader, None)
            mock_opt.return_value = (Mock(), Mock())
            mock_save.return_value = Path("/mock/output.pt")
            
            result = trainer.train("test_dataset", config)
            
            assert result.success
            assert result.output_path == Path("/mock/output.pt")
            assert result.final_loss == 0.1
    
    def test_train_already_running(self, hardware_config):
        """Test training when already in progress."""
        trainer = LoRATrainer(hardware_config)
        trainer.is_training = True
        
        config = LoRAConfig()
        result = trainer.train("test_dataset", config)
        
        assert not result.success
        assert "already in progress" in result.error_message
    
    def test_validate_dataset_compliance_valid(self, hardware_config, mock_dataset_organizer):
        """Test dataset compliance validation - valid case."""
        trainer = LoRATrainer(hardware_config)
        
        result = trainer.validate_dataset_compliance("test_dataset", ComplianceMode.RESEARCH_SAFE)
        
        assert result["valid"]
        assert len(result["warnings"]) == 0
    
    def test_validate_dataset_compliance_invalid(self, hardware_config):
        """Test dataset compliance validation - invalid case."""
        trainer = LoRATrainer(hardware_config)
        
        with patch('src.pipelines.lora_training.DatasetOrganizer') as mock_organizer_class:
            mock_organizer = Mock()
            mock_organizer_class.return_value = mock_organizer
            mock_organizer.load_dataset.return_value = None
            
            result = trainer.validate_dataset_compliance("nonexistent_dataset", ComplianceMode.OPEN_SOURCE_ONLY)
            
            assert not result["valid"]
            assert "not found" in result["error"]
    
    def test_list_available_datasets(self, hardware_config):
        """Test listing available datasets."""
        trainer = LoRATrainer(hardware_config)
        
        with patch('src.pipelines.lora_training.DatasetOrganizer') as mock_organizer_class:
            mock_organizer = Mock()
            mock_organizer_class.return_value = mock_organizer
            mock_organizer.list_datasets.return_value = ["dataset1", "dataset2"]
            
            datasets = trainer.list_available_datasets()
            
            assert datasets == ["dataset1", "dataset2"]
    
    def test_get_progress_no_session(self, hardware_config):
        """Test getting progress when no training session."""
        trainer = LoRATrainer(hardware_config)
        
        progress = trainer.get_progress()
        
        assert progress is None
    
    def test_stop_training_not_running(self, hardware_config):
        """Test stopping training when not running."""
        trainer = LoRATrainer(hardware_config)
        
        result = trainer.stop_training()
        
        assert not result
    
    def test_stop_training_running(self, hardware_config):
        """Test stopping training when running."""
        trainer = LoRATrainer(hardware_config)
        trainer.is_training = True
        
        result = trainer.stop_training()
        
        assert result
        assert not trainer.is_training


class TestTrainingProgress:
    """Test training progress tracking."""
    
    def test_progress_initialization(self):
        """Test training progress initialization."""
        progress = TrainingProgress()
        
        assert progress.epoch == 0
        assert progress.step == 0
        assert progress.total_steps == 0
        assert progress.loss == 0.0
        assert progress.learning_rate == 0.0
        assert progress.elapsed_time == 0.0
        assert progress.validation_loss is None
    
    def test_progress_with_values(self):
        """Test training progress with values."""
        progress = TrainingProgress(
            epoch=5,
            step=100,
            total_steps=1000,
            loss=0.5,
            learning_rate=1e-4,
            elapsed_time=3600.0,
            validation_loss=0.3
        )
        
        assert progress.epoch == 5
        assert progress.step == 100
        assert progress.total_steps == 1000
        assert progress.loss == 0.5
        assert progress.learning_rate == 1e-4
        assert progress.elapsed_time == 3600.0
        assert progress.validation_loss == 0.3


class TestTrainingResult:
    """Test training result."""
    
    def test_successful_result(self):
        """Test successful training result."""
        result = TrainingResult(
            success=True,
            output_path=Path("/path/to/model.pt"),
            training_time=3600.0,
            final_loss=0.1,
            total_steps=1000,
            model_size_mb=50.0
        )
        
        assert result.success
        assert result.output_path == Path("/path/to/model.pt")
        assert result.training_time == 3600.0
        assert result.final_loss == 0.1
        assert result.total_steps == 1000
        assert result.model_size_mb == 50.0
        assert result.error_message is None
    
    def test_failed_result(self):
        """Test failed training result."""
        result = TrainingResult(
            success=False,
            error_message="Training failed due to memory error"
        )
        
        assert not result.success
        assert result.error_message == "Training failed due to memory error"
        assert result.output_path is None


class TestIntegration:
    """Integration tests for LoRA training pipeline."""
    
    @pytest.fixture
    def hardware_config(self):
        """Create test hardware configuration."""
        return HardwareConfig(
            vram_size=8192,
            gpu_model="RTX 3070",
            cpu_cores=8,
            ram_size=32768,
            cuda_available=True,
            optimization_level="balanced"
        )
    
    def test_full_training_pipeline_mock(self, hardware_config):
        """Test full training pipeline with mocked components."""
        trainer = LoRATrainer(hardware_config)
        config = LoRAConfig(num_epochs=1, batch_size=1)
        
        # Mock all external dependencies
        with patch('src.pipelines.lora_training.TORCH_AVAILABLE', True), \
             patch('src.pipelines.lora_training.DIFFUSERS_AVAILABLE', True), \
             patch('src.pipelines.lora_training.TRANSFORMERS_AVAILABLE', True), \
             patch('src.pipelines.lora_training.DatasetOrganizer') as mock_organizer_class:
            
            # Setup dataset organizer mock
            mock_organizer = Mock()
            mock_organizer_class.return_value = mock_organizer
            
            stats = DatasetStats(
                total_items=10,
                compliance_safe_count=10,
                compliance_safe_percentage=100.0
            )
            
            mock_dataset = OrganizedDataset(
                name="test_dataset",
                path=Path("/mock/path"),
                config=DatasetConfig(),
                stats=stats
            )
            
            mock_organizer.load_dataset.return_value = mock_dataset
            mock_organizer.get_compliance_summary.return_value = {
                "compliance_safe_percentage": 100.0
            }
            
            # Mock training components
            with patch.object(trainer, '_initialize_training_components', return_value=True), \
                 patch.object(trainer, '_setup_lora_layers', return_value=True), \
                 patch.object(trainer, '_create_data_loaders') as mock_loaders, \
                 patch.object(trainer, '_setup_optimizer_and_scheduler') as mock_opt, \
                 patch.object(trainer, '_training_loop', return_value=0.05), \
                 patch.object(trainer, '_save_lora_weights') as mock_save, \
                 patch('tempfile.mkdtemp', return_value="/tmp/test"):
                
                # Setup data loader mock
                mock_train_loader = Mock()
                mock_train_loader.__len__ = Mock(return_value=10)
                mock_loaders.return_value = (mock_train_loader, None)
                
                # Setup optimizer mock
                mock_opt.return_value = (Mock(), Mock())
                
                # Setup save mock
                output_path = Path("/mock/lora_model.pt")
                mock_save.return_value = output_path
                
                # Run training
                result = trainer.train("test_dataset", config, "stable-diffusion-v1-5")
                
                # Verify result
                assert result.success
                assert result.output_path == output_path
                assert result.final_loss == 0.05
                assert result.training_time > 0
                assert result.compliance_report is not None
    
    def test_compliance_mode_filtering(self, hardware_config):
        """Test compliance mode filtering during training."""
        trainer = LoRATrainer(hardware_config)
        
        # Test open source only mode
        result = trainer.validate_dataset_compliance("test_dataset", ComplianceMode.OPEN_SOURCE_ONLY)
        
        # Should be called with dataset organizer
        assert isinstance(result, dict)
        assert "valid" in result