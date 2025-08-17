"""
Unit tests for the research interface components.

Tests the basic functionality of the Gradio-based research interface
including component creation, event handling, and mock interactions.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import sys
from pathlib import Path

# Add src to path for imports
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

try:
    from src.ui.research_interface import ResearchInterface, UIState, create_research_interface
    from src.core.interfaces import ComplianceMode, OutputType, HardwareConfig
except ImportError:
    # Alternative import approach
    import os
    os.chdir(src_path.parent)
    from src.ui.research_interface import ResearchInterface, UIState, create_research_interface
    from src.core.interfaces import ComplianceMode, OutputType, HardwareConfig


class TestUIState(unittest.TestCase):
    """Test the UIState dataclass."""
    
    def test_ui_state_initialization(self):
        """Test UIState initialization with defaults."""
        state = UIState()
        
        self.assertEqual(state.current_compliance_mode, ComplianceMode.RESEARCH_SAFE)
        self.assertIsNone(state.current_model)
        self.assertIsNone(state.experiment_id)
        self.assertEqual(state.generation_history, [])
    
    def test_ui_state_custom_initialization(self):
        """Test UIState initialization with custom values."""
        state = UIState(
            current_compliance_mode=ComplianceMode.OPEN_SOURCE_ONLY,
            current_model="test-model",
            experiment_id="exp_001"
        )
        
        self.assertEqual(state.current_compliance_mode, ComplianceMode.OPEN_SOURCE_ONLY)
        self.assertEqual(state.current_model, "test-model")
        self.assertEqual(state.experiment_id, "exp_001")
        self.assertEqual(state.generation_history, [])


class TestResearchInterface(unittest.TestCase):
    """Test the ResearchInterface class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_system_controller = Mock()
        self.mock_experiment_tracker = Mock()
        self.mock_compliance_engine = Mock()
        
        self.interface = ResearchInterface(
            system_controller=self.mock_system_controller,
            experiment_tracker=self.mock_experiment_tracker,
            compliance_engine=self.mock_compliance_engine
        )
    
    def test_interface_initialization(self):
        """Test interface initialization."""
        self.assertEqual(self.interface.system_controller, self.mock_system_controller)
        self.assertEqual(self.interface.experiment_tracker, self.mock_experiment_tracker)
        self.assertEqual(self.interface.compliance_engine, self.mock_compliance_engine)
        self.assertIsInstance(self.interface.ui_state, UIState)
        self.assertIsNone(self.interface.gradio_app)
        self.assertFalse(self.interface.is_initialized)
    
    @patch('src.ui.research_interface.GRADIO_AVAILABLE', False)
    def test_initialize_without_gradio(self):
        """Test initialization when Gradio is not available."""
        result = self.interface.initialize()
        
        self.assertFalse(result)
        self.assertFalse(self.interface.is_initialized)
    
    @patch('src.ui.research_interface.GRADIO_AVAILABLE', True)
    @patch('src.ui.research_interface.gr')
    def test_initialize_with_gradio(self, mock_gr):
        """Test initialization when Gradio is available."""
        # Mock Gradio components
        mock_blocks = Mock()
        mock_gr.Blocks.return_value.__enter__.return_value = mock_blocks
        mock_gr.themes.Soft.return_value = Mock()
        
        # Mock all Gradio components
        mock_gr.Markdown.return_value = Mock()
        mock_gr.Row.return_value.__enter__.return_value = Mock()
        mock_gr.Column.return_value.__enter__.return_value = Mock()
        mock_gr.Tabs.return_value.__enter__.return_value = Mock()
        mock_gr.TabItem.return_value.__enter__.return_value = Mock()
        mock_gr.Dropdown.return_value = Mock()
        mock_gr.Textbox.return_value = Mock()
        mock_gr.Slider.return_value = Mock()
        mock_gr.Number.return_value = Mock()
        mock_gr.Button.return_value = Mock()
        mock_gr.Image.return_value = Mock()
        mock_gr.Video.return_value = Mock()
        mock_gr.JSON.return_value = Mock()
        mock_gr.Progress.return_value = Mock()
        mock_gr.Dataframe.return_value = Mock()
        mock_gr.Plot.return_value = Mock()
        mock_gr.File.return_value = Mock()
        
        result = self.interface.initialize()
        
        self.assertTrue(result)
        self.assertTrue(self.interface.is_initialized)
        self.assertIsNotNone(self.interface.gradio_app)
    
    def test_launch_without_initialization(self):
        """Test launching interface without initialization."""
        with patch('src.ui.research_interface.logger') as mock_logger:
            self.interface.launch()
            mock_logger.error.assert_called_with("Interface not initialized - call initialize() first")
    
    @patch('src.ui.research_interface.GRADIO_AVAILABLE', True)
    @patch('src.ui.research_interface.gr')
    def test_launch_with_initialization(self, mock_gr):
        """Test launching initialized interface."""
        # Initialize first
        mock_blocks = Mock()
        mock_gr.Blocks.return_value.__enter__.return_value = mock_blocks
        mock_gr.themes.Soft.return_value = Mock()
        
        # Mock all required components for initialization
        self._mock_gradio_components(mock_gr)
        
        self.interface.initialize()
        
        # Test launch
        mock_app = Mock()
        self.interface.gradio_app = mock_app
        
        self.interface.launch(share=True, server_port=8080)
        
        mock_app.launch.assert_called_once_with(
            share=True,
            server_name="127.0.0.1",
            server_port=8080,
            show_error=True,
            quiet=False
        )
    
    def test_compliance_mode_change(self):
        """Test compliance mode change handler."""
        result_model, result_stats = self.interface._on_compliance_mode_change("open_only")
        
        self.assertEqual(self.interface.ui_state.current_compliance_mode, ComplianceMode.OPEN_SOURCE_ONLY)
        self.assertIn("open_only", result_model)
        self.assertIsInstance(result_stats, dict)
        self.assertEqual(result_stats["fair_use_research"], 0)
        self.assertEqual(result_stats["copyrighted"], 0)
    
    def test_compliance_mode_change_full_dataset(self):
        """Test compliance mode change to full dataset."""
        result_model, result_stats = self.interface._on_compliance_mode_change("full_dataset")
        
        self.assertEqual(self.interface.ui_state.current_compliance_mode, ComplianceMode.FULL_DATASET)
        self.assertGreater(result_stats["fair_use_research"], 0)
        self.assertGreater(result_stats["copyrighted"], 0)
    
    @patch('src.ui.research_interface.PIL_AVAILABLE', True)
    def test_generate_image_success(self):
        """Test successful image generation."""
        # Mock PIL Image creation
        with patch('src.ui.research_interface.PIL.Image') as mock_pil_image:
            mock_image = Mock()
            mock_pil_image.new.return_value = mock_image
        
            result_image, result_info, result_status = self.interface._generate_image(
                prompt="test prompt",
                negative_prompt="test negative",
                width=512,
                height=512,
                steps=20,
                guidance_scale=7.5,
                seed=42,
                model="stable-diffusion-v1-5",
                compliance_mode="research_safe"
            )
            
            self.assertEqual(result_image, mock_image)
            self.assertIsInstance(result_info, dict)
            self.assertEqual(result_info["model"], "stable-diffusion-v1-5")
            self.assertEqual(result_info["prompt"], "test prompt")
            self.assertIn("successfully", result_status)
    
    @patch('src.ui.research_interface.PIL_AVAILABLE', False)
    def test_generate_image_without_pil(self):
        """Test image generation without PIL available."""
        result_image, result_info, result_status = self.interface._generate_image(
            prompt="test prompt",
            negative_prompt="",
            width=512,
            height=512,
            steps=20,
            guidance_scale=7.5,
            seed=None,
            model="stable-diffusion-v1-5",
            compliance_mode="research_safe"
        )
        
        self.assertIsNone(result_image)
        self.assertIsInstance(result_info, dict)
        self.assertIn("successfully", result_status)
    
    def test_generate_video_success(self):
        """Test successful video generation."""
        result_video, result_info, result_status = self.interface._generate_video(
            prompt="test video prompt",
            conditioning_image=None,
            width=512,
            height=512,
            num_frames=14,
            fps=7,
            steps=25,
            guidance_scale=7.5,
            motion_bucket_id=127,
            seed=42,
            model="stable-video-diffusion",
            compliance_mode="research_safe"
        )
        
        self.assertIsNone(result_video)  # Mock returns None
        self.assertIsInstance(result_info, dict)
        self.assertEqual(result_info["model"], "stable-video-diffusion")
        self.assertEqual(result_info["prompt"], "test video prompt")
        self.assertIn("successfully", result_status)
    
    def test_save_experiment(self):
        """Test experiment saving."""
        result = self.interface._save_experiment("Test experiment notes")
        
        self.assertIn("successfully", result)
    
    def test_refresh_experiment_history(self):
        """Test experiment history refresh."""
        result = self.interface._refresh_experiment_history()
        
        self.assertIsInstance(result, list)
        self.assertGreater(len(result), 0)
        # Check that each row has the expected number of columns
        for row in result:
            self.assertEqual(len(row), 6)  # ID, Timestamp, Type, Model, Prompt, Status
    
    def test_refresh_system_status(self):
        """Test system status refresh."""
        hardware_status, model_status, logs = self.interface._refresh_system_status()
        
        self.assertIsInstance(hardware_status, dict)
        self.assertIn("gpu_model", hardware_status)
        self.assertIn("vram_total", hardware_status)
        
        self.assertIsInstance(model_status, dict)
        self.assertIn("image_pipeline", model_status)
        
        self.assertIsInstance(logs, str)
        self.assertIn("System initialized", logs)
    
    def test_clear_vram_cache(self):
        """Test VRAM cache clearing."""
        result = self.interface._clear_vram_cache()
        
        self.assertIsInstance(result, dict)
        self.assertIn("vram_used", result)
        self.assertLess(result["vram_used"], 1000)  # Should be reduced after clearing
    
    def test_get_custom_css(self):
        """Test custom CSS generation."""
        css = self.interface._get_custom_css()
        
        self.assertIsInstance(css, str)
        self.assertIn("gradio-container", css)
        self.assertIn("max-width", css)
    
    def _mock_gradio_components(self, mock_gr):
        """Helper method to mock all Gradio components."""
        mock_gr.Markdown.return_value = Mock()
        mock_gr.Row.return_value.__enter__.return_value = Mock()
        mock_gr.Column.return_value.__enter__.return_value = Mock()
        mock_gr.Tabs.return_value.__enter__.return_value = Mock()
        mock_gr.TabItem.return_value.__enter__.return_value = Mock()
        mock_gr.Dropdown.return_value = Mock()
        mock_gr.Textbox.return_value = Mock()
        mock_gr.Slider.return_value = Mock()
        mock_gr.Number.return_value = Mock()
        mock_gr.Button.return_value = Mock()
        mock_gr.Image.return_value = Mock()
        mock_gr.Video.return_value = Mock()
        mock_gr.JSON.return_value = Mock()
        mock_gr.Progress.return_value = Mock()
        mock_gr.Dataframe.return_value = Mock()
        mock_gr.Plot.return_value = Mock()
        mock_gr.File.return_value = Mock()


class TestUtilityFunctions(unittest.TestCase):
    """Test utility functions for interface creation."""
    
    @patch('src.ui.research_interface.ResearchInterface')
    def test_create_research_interface_success(self, mock_interface_class):
        """Test successful interface creation."""
        mock_interface = Mock()
        mock_interface.initialize.return_value = True
        mock_interface_class.return_value = mock_interface
        
        result = create_research_interface()
        
        self.assertEqual(result, mock_interface)
        mock_interface.initialize.assert_called_once()
    
    @patch('src.ui.research_interface.ResearchInterface')
    def test_create_research_interface_failure(self, mock_interface_class):
        """Test interface creation failure."""
        mock_interface = Mock()
        mock_interface.initialize.return_value = False
        mock_interface_class.return_value = mock_interface
        
        result = create_research_interface()
        
        self.assertIsNone(result)
        mock_interface.initialize.assert_called_once()


if __name__ == '__main__':
    unittest.main()