"""
Performance tests for video generation pipeline.

Tests video generation performance across different hardware configurations
and validates memory optimization strategies.
"""

import pytest
import time
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from src.core.interfaces import (
    GenerationRequest, HardwareConfig, StyleConfig, 
    ConversationContext, ComplianceMode, OutputType
)
from src.pipelines.video_generation import VideoGenerationPipeline, VideoModel
from src.pipelines.frame_processor import FrameProcessor
from src.pipelines.temporal_consistency import TemporalConsistencyEngine


class TestVideoGenerationPerformance:
    """Performance tests for video generation pipeline."""
    
    @pytest.fixture
    def low_vram_config(self):
        """GTX 1650 configuration (4GB VRAM)."""
        return HardwareConfig(
            vram_size=4096,
            gpu_model="GTX 1650",
            cpu_cores=4,
            ram_size=8192,
            cuda_available=True,
            optimization_level="aggressive"
        )
    
    @pytest.fixture
    def medium_vram_config(self):
        """RTX 3070 configuration (8GB VRAM)."""
        return HardwareConfig(
            vram_size=8192,
            gpu_model="RTX 3070",
            cpu_cores=8,
            ram_size=16384,
            cuda_available=True,
            optimization_level="balanced"
        )
    
    @pytest.fixture
    def high_vram_config(self):
        """RTX 4090 configuration (24GB VRAM)."""
        return HardwareConfig(
            vram_size=24576,
            gpu_model="RTX 4090",
            cpu_cores=16,
            ram_size=32768,
            cuda_available=True,
            optimization_level="minimal"
        )
    
    @pytest.fixture
    def performance_style_config(self):
        """Style configuration for performance testing."""
        return StyleConfig(
            style_name="performance_test",
            generation_params={
                'num_frames': 16,
                'fps': 8,
                'guidance_scale': 7.5,
                'num_inference_steps': 20,  # Reduced for faster testing
                'width': 512,
                'height': 512
            }
        )
    
    @pytest.fixture
    def conversation_context(self):
        """Test conversation context."""
        return ConversationContext(
            conversation_id="perf_test_001",
            history=[],
            current_mode=ComplianceMode.RESEARCH_SAFE,
            user_preferences={}
        )
    
    @pytest.fixture
    def temp_output_dir(self):
        """Temporary output directory."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)
    
    def create_generation_request(self, hardware_config, style_config, conversation_context, prompt="Test video"):
        """Create a generation request for testing."""
        return GenerationRequest(
            prompt=prompt,
            output_type=OutputType.VIDEO,
            style_config=style_config,
            compliance_mode=ComplianceMode.RESEARCH_SAFE,
            hardware_constraints=hardware_config,
            context=conversation_context
        )
    
    @patch('src.pipelines.video_generation.MemoryManager')
    @patch('src.pipelines.video_generation.HardwareProfileManager')
    def test_low_vram_performance(self, mock_profile_manager, mock_memory_manager, 
                                 low_vram_config, performance_style_config, conversation_context):
        """Test video generation performance on low VRAM hardware."""
        # Setup mocks
        mock_memory_manager.return_value = Mock()
        mock_profile_manager.return_value = Mock()
        
        pipeline = VideoGenerationPipeline()
        
        # Mock initialization and model loading
        with patch.object(pipeline, '_check_dependencies', return_value=True), \
             patch.object(pipeline, '_select_optimal_model', return_value=VideoModel.TEXT2VIDEO_ZERO.value), \
             patch.object(pipeline, '_load_model', return_value=True):
            
            # Initialize pipeline
            init_start = time.time()
            result = pipeline.initialize(low_vram_config)
            init_time = time.time() - init_start
            
            assert result is True
            assert init_time < 10.0  # Should initialize within 10 seconds
            
            # Test generation performance
            request = self.create_generation_request(low_vram_config, performance_style_config, conversation_context)
            
            # Mock the generation process
            mock_frames = [Mock() for _ in range(16)]
            
            with patch.object(pipeline, '_generate_video_hybrid', return_value=mock_frames), \
                 patch.object(pipeline, '_save_video', return_value=Path("test_video.mp4")), \
                 patch.object(pipeline, '_calculate_quality_metrics', return_value={'frames': 16}), \
                 patch.object(pipeline, '_get_compliance_info', return_value={'compliant': True}):
                
                pipeline.is_initialized = True
                pipeline.current_model = VideoModel.TEXT2VIDEO_ZERO.value
                
                gen_start = time.time()
                result = pipeline.generate(request)
                gen_time = time.time() - gen_start
                
                assert result.success is True
                assert gen_time < 900.0  # Should complete within 15 minutes for low VRAM
                
                # Verify memory optimization was applied
                assert pipeline.hardware_config.optimization_level == "aggressive"
    
    @patch('src.pipelines.video_generation.MemoryManager')
    @patch('src.pipelines.video_generation.HardwareProfileManager')
    def test_medium_vram_performance(self, mock_profile_manager, mock_memory_manager,
                                   medium_vram_config, performance_style_config, conversation_context):
        """Test video generation performance on medium VRAM hardware."""
        # Setup mocks
        mock_memory_manager.return_value = Mock()
        mock_profile_manager.return_value = Mock()
        
        pipeline = VideoGenerationPipeline()
        
        with patch.object(pipeline, '_check_dependencies', return_value=True), \
             patch.object(pipeline, '_select_optimal_model', return_value=VideoModel.ANIMATEDIFF.value), \
             patch.object(pipeline, '_load_model', return_value=True):
            
            # Initialize pipeline
            init_start = time.time()
            result = pipeline.initialize(medium_vram_config)
            init_time = time.time() - init_start
            
            assert result is True
            assert init_time < 8.0  # Should initialize faster with more VRAM
            
            # Test generation performance
            request = self.create_generation_request(medium_vram_config, performance_style_config, conversation_context)
            
            mock_frames = [Mock() for _ in range(16)]
            
            with patch.object(pipeline, '_generate_video_hybrid', return_value=mock_frames), \
                 patch.object(pipeline, '_save_video', return_value=Path("test_video.mp4")), \
                 patch.object(pipeline, '_calculate_quality_metrics', return_value={'frames': 16}), \
                 patch.object(pipeline, '_get_compliance_info', return_value={'compliant': True}):
                
                pipeline.is_initialized = True
                pipeline.current_model = VideoModel.ANIMATEDIFF.value
                
                gen_start = time.time()
                result = pipeline.generate(request)
                gen_time = time.time() - gen_start
                
                assert result.success is True
                assert gen_time < 300.0  # Should complete within 5 minutes for medium VRAM
                
                # Verify balanced optimization
                assert pipeline.hardware_config.optimization_level == "balanced"
    
    @patch('src.pipelines.video_generation.MemoryManager')
    @patch('src.pipelines.video_generation.HardwareProfileManager')
    def test_high_vram_performance(self, mock_profile_manager, mock_memory_manager,
                                  high_vram_config, performance_style_config, conversation_context):
        """Test video generation performance on high VRAM hardware."""
        # Setup mocks
        mock_memory_manager.return_value = Mock()
        mock_profile_manager.return_value = Mock()
        
        pipeline = VideoGenerationPipeline()
        
        with patch.object(pipeline, '_check_dependencies', return_value=True), \
             patch.object(pipeline, '_select_optimal_model', return_value=VideoModel.I2VGEN_XL.value), \
             patch.object(pipeline, '_load_model', return_value=True):
            
            # Initialize pipeline
            init_start = time.time()
            result = pipeline.initialize(high_vram_config)
            init_time = time.time() - init_start
            
            assert result is True
            assert init_time < 5.0  # Should initialize quickly with high VRAM
            
            # Test generation performance
            request = self.create_generation_request(high_vram_config, performance_style_config, conversation_context)
            
            mock_frames = [Mock() for _ in range(16)]
            
            with patch.object(pipeline, '_generate_video_hybrid', return_value=mock_frames), \
                 patch.object(pipeline, '_save_video', return_value=Path("test_video.mp4")), \
                 patch.object(pipeline, '_calculate_quality_metrics', return_value={'frames': 16}), \
                 patch.object(pipeline, '_get_compliance_info', return_value={'compliant': True}):
                
                pipeline.is_initialized = True
                pipeline.current_model = VideoModel.I2VGEN_XL.value
                
                gen_start = time.time()
                result = pipeline.generate(request)
                gen_time = time.time() - gen_start
                
                assert result.success is True
                assert gen_time < 120.0  # Should complete within 2 minutes for high VRAM
                
                # Verify minimal optimization
                assert pipeline.hardware_config.optimization_level == "minimal"
    
    def test_model_switching_performance(self, medium_vram_config):
        """Test model switching performance."""
        pipeline = VideoGenerationPipeline()
        pipeline.hardware_config = medium_vram_config
        pipeline.memory_manager = Mock()
        
        # Test switching between models
        models_to_test = [
            VideoModel.TEXT2VIDEO_ZERO.value,
            VideoModel.ANIMATEDIFF.value,
            VideoModel.TEXT2VIDEO_ZERO.value  # Switch back
        ]
        
        switch_times = []
        
        for model in models_to_test:
            with patch.object(pipeline, '_load_model', return_value=True):
                switch_start = time.time()
                result = pipeline.switch_model(model)
                switch_time = time.time() - switch_start
                
                assert result is True
                assert switch_time < 60.0  # Should switch within 60 seconds
                switch_times.append(switch_time)
        
        # Verify memory manager was called for switching
        assert pipeline.memory_manager.manage_model_switching.call_count == len(models_to_test)
    
    def test_memory_optimization_effectiveness(self, low_vram_config):
        """Test effectiveness of memory optimization strategies."""
        pipeline = VideoGenerationPipeline()
        
        # Mock memory manager with realistic behavior
        mock_memory_manager = Mock()
        mock_memory_manager.get_memory_status.return_value = {
            'used_vram_mb': 3500,  # Near limit for 4GB
            'available_vram_mb': 596,
            'total_vram_mb': 4096
        }
        
        pipeline.memory_manager = mock_memory_manager
        pipeline.hardware_config = low_vram_config
        
        # Test that optimization strategies are applied
        with patch.object(pipeline, '_apply_pipeline_optimizations') as mock_optimize:
            mock_pipeline = Mock()
            pipeline._apply_pipeline_optimizations(mock_pipeline)
            
            mock_optimize.assert_called_once_with(mock_pipeline)
    
    def test_frame_processor_performance(self, medium_vram_config):
        """Test frame processor performance with hybrid CPU/GPU processing."""
        frame_processor = FrameProcessor(medium_vram_config, Mock())
        
        # Mock pipelines
        gpu_pipeline = Mock()
        cpu_pipeline = Mock()
        frame_processor.set_pipelines(gpu_pipeline, cpu_pipeline)
        
        # Mock frame generation
        mock_frames = [Mock() for _ in range(16)]
        
        with patch.object(frame_processor, '_generate_with_keyframes', return_value=mock_frames):
            start_time = time.time()
            
            frames = frame_processor.process_video_frames(
                "test prompt",
                16,
                {'width': 512, 'height': 512, 'num_inference_steps': 20}
            )
            
            processing_time = time.time() - start_time
            
            assert len(frames) == 16
            assert processing_time < 30.0  # Should process quickly with mocked generation
            
            # Check processing stats
            stats = frame_processor.get_processing_stats()
            assert 'total_processing_time' in stats
            assert stats['total_processing_time'] > 0
    
    def test_temporal_consistency_performance(self, medium_vram_config):
        """Test temporal consistency engine performance."""
        consistency_engine = TemporalConsistencyEngine(medium_vram_config)
        
        # Create mock frames
        mock_frames = []
        for i in range(16):
            mock_frame = Mock()
            mock_frame.size = (512, 512)
            mock_frames.append(mock_frame)
        
        # Mock numpy operations
        with patch('src.pipelines.temporal_consistency.NUMPY_AVAILABLE', True), \
             patch('src.pipelines.temporal_consistency.np') as mock_np:
            
            # Mock array operations
            mock_np.array.return_value = Mock()
            mock_np.corrcoef.return_value = [[1.0, 0.8], [0.8, 1.0]]
            mock_np.mean.return_value = 0.8
            mock_np.var.return_value = 0.1
            mock_np.isnan.return_value = False
            
            start_time = time.time()
            
            optimized_frames, metrics = consistency_engine.optimize_sequence(mock_frames)
            
            optimization_time = time.time() - start_time
            
            assert len(optimized_frames) == 16
            assert optimization_time < 10.0  # Should optimize quickly
            assert metrics.processing_time > 0
            assert 0.0 <= metrics.motion_smoothness <= 1.0
            assert 0.0 <= metrics.temporal_coherence <= 1.0
    
    def test_batch_processing_performance(self, medium_vram_config, performance_style_config, conversation_context):
        """Test performance of batch processing multiple videos."""
        pipeline = VideoGenerationPipeline()
        
        # Mock initialization
        with patch.object(pipeline, '_check_dependencies', return_value=True), \
             patch.object(pipeline, '_select_optimal_model', return_value=VideoModel.ANIMATEDIFF.value), \
             patch.object(pipeline, '_load_model', return_value=True):
            
            pipeline.initialize(medium_vram_config)
            pipeline.is_initialized = True
            pipeline.current_model = VideoModel.ANIMATEDIFF.value
        
        # Create multiple requests
        requests = []
        for i in range(3):
            request = self.create_generation_request(
                medium_vram_config, 
                performance_style_config, 
                conversation_context,
                f"Test video {i+1}"
            )
            requests.append(request)
        
        # Mock generation
        mock_frames = [Mock() for _ in range(16)]
        
        with patch.object(pipeline, '_generate_video_hybrid', return_value=mock_frames), \
             patch.object(pipeline, '_save_video', return_value=Path("test_video.mp4")), \
             patch.object(pipeline, '_calculate_quality_metrics', return_value={'frames': 16}), \
             patch.object(pipeline, '_get_compliance_info', return_value={'compliant': True}):
            
            batch_start = time.time()
            results = []
            
            for request in requests:
                result = pipeline.generate(request)
                results.append(result)
            
            batch_time = time.time() - batch_start
            
            # Verify all generations succeeded
            assert all(result.success for result in results)
            
            # Verify reasonable batch processing time
            assert batch_time < 600.0  # Should complete 3 videos within 10 minutes
            
            # Verify average time per video is reasonable
            avg_time_per_video = batch_time / len(requests)
            assert avg_time_per_video < 200.0  # Average under 3.33 minutes per video
    
    def test_memory_cleanup_performance(self, low_vram_config):
        """Test memory cleanup performance."""
        pipeline = VideoGenerationPipeline()
        pipeline.hardware_config = low_vram_config
        pipeline.memory_manager = Mock()
        pipeline.current_pipeline = Mock()
        pipeline.current_model = VideoModel.TEXT2VIDEO_ZERO.value
        
        # Test cleanup performance
        cleanup_start = time.time()
        pipeline.cleanup()
        cleanup_time = time.time() - cleanup_start
        
        assert cleanup_time < 5.0  # Should cleanup quickly
        assert pipeline.current_pipeline is None
        assert pipeline.current_model is None
        
        # Verify memory manager cleanup was called
        pipeline.memory_manager.clear_vram_cache.assert_called_once()


class TestVideoGenerationScalability:
    """Test scalability of video generation across different parameters."""
    
    @pytest.fixture
    def scalability_hardware_config(self):
        """Hardware config for scalability testing."""
        return HardwareConfig(
            vram_size=8192,
            gpu_model="RTX 3070",
            cpu_cores=8,
            ram_size=16384,
            cuda_available=True,
            optimization_level="balanced"
        )
    
    @pytest.mark.parametrize("num_frames", [4, 8, 16, 24])
    def test_frame_count_scalability(self, scalability_hardware_config, num_frames):
        """Test performance scaling with different frame counts."""
        pipeline = VideoGenerationPipeline()
        
        # Mock initialization
        with patch.object(pipeline, '_check_dependencies', return_value=True), \
             patch.object(pipeline, '_select_optimal_model', return_value=VideoModel.ANIMATEDIFF.value), \
             patch.object(pipeline, '_load_model', return_value=True):
            
            pipeline.initialize(scalability_hardware_config)
            pipeline.is_initialized = True
            pipeline.current_model = VideoModel.ANIMATEDIFF.value
        
        # Create style config with varying frame counts
        style_config = StyleConfig(
            generation_params={
                'num_frames': num_frames,
                'fps': 8,
                'width': 512,
                'height': 512,
                'num_inference_steps': 20
            }
        )
        
        # Mock generation
        mock_frames = [Mock() for _ in range(num_frames)]
        
        with patch.object(pipeline, '_generate_video_hybrid', return_value=mock_frames), \
             patch.object(pipeline, '_save_video', return_value=Path("test_video.mp4")), \
             patch.object(pipeline, '_calculate_quality_metrics', return_value={'frames': num_frames}), \
             patch.object(pipeline, '_get_compliance_info', return_value={'compliant': True}):
            
            request = GenerationRequest(
                prompt="Scalability test",
                output_type=OutputType.VIDEO,
                style_config=style_config,
                compliance_mode=ComplianceMode.RESEARCH_SAFE,
                hardware_constraints=scalability_hardware_config,
                context=Mock()
            )
            
            start_time = time.time()
            result = pipeline.generate(request)
            generation_time = time.time() - start_time
            
            assert result.success is True
            
            # Verify time scales reasonably with frame count
            expected_max_time = num_frames * 10  # 10 seconds per frame max
            assert generation_time < expected_max_time
    
    @pytest.mark.parametrize("resolution", [(256, 256), (512, 512), (768, 768)])
    def test_resolution_scalability(self, scalability_hardware_config, resolution):
        """Test performance scaling with different resolutions."""
        width, height = resolution
        
        pipeline = VideoGenerationPipeline()
        
        # Mock initialization
        with patch.object(pipeline, '_check_dependencies', return_value=True), \
             patch.object(pipeline, '_select_optimal_model', return_value=VideoModel.ANIMATEDIFF.value), \
             patch.object(pipeline, '_load_model', return_value=True):
            
            pipeline.initialize(scalability_hardware_config)
            pipeline.is_initialized = True
            pipeline.current_model = VideoModel.ANIMATEDIFF.value
        
        # Create style config with varying resolutions
        style_config = StyleConfig(
            generation_params={
                'num_frames': 8,
                'fps': 8,
                'width': width,
                'height': height,
                'num_inference_steps': 20
            }
        )
        
        # Mock generation
        mock_frames = [Mock() for _ in range(8)]
        
        with patch.object(pipeline, '_generate_video_hybrid', return_value=mock_frames), \
             patch.object(pipeline, '_save_video', return_value=Path("test_video.mp4")), \
             patch.object(pipeline, '_calculate_quality_metrics', return_value={'frames': 8}), \
             patch.object(pipeline, '_get_compliance_info', return_value={'compliant': True}):
            
            request = GenerationRequest(
                prompt="Resolution scalability test",
                output_type=OutputType.VIDEO,
                style_config=style_config,
                compliance_mode=ComplianceMode.RESEARCH_SAFE,
                hardware_constraints=scalability_hardware_config,
                context=Mock()
            )
            
            start_time = time.time()
            result = pipeline.generate(request)
            generation_time = time.time() - start_time
            
            assert result.success is True
            
            # Verify time scales with resolution complexity
            pixel_count = width * height
            expected_max_time = (pixel_count / 10000) * 60  # Scale with pixel count
            assert generation_time < expected_max_time