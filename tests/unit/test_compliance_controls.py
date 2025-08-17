"""
Unit tests for the compliance controls module.

Tests the copyright compliance functionality including mode selection,
model filtering, and attribution management.
"""

import unittest
from unittest.mock import Mock, patch
import sys
from pathlib import Path

# Add src to path for imports
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from src.ui.compliance_controls import (
    ComplianceController, ModelLicenseInfo, DatasetStats,
    create_compliance_controller
)
from src.core.interfaces import ComplianceMode, LicenseType


class TestDatasetStats(unittest.TestCase):
    """Test the DatasetStats dataclass."""
    
    def test_dataset_stats_initialization(self):
        """Test DatasetStats initialization with defaults."""
        stats = DatasetStats()
        
        self.assertEqual(stats.public_domain, 0)
        self.assertEqual(stats.creative_commons, 0)
        self.assertEqual(stats.fair_use_research, 0)
        self.assertEqual(stats.copyrighted, 0)
        self.assertEqual(stats.unknown, 0)
    
    def test_dataset_stats_custom_initialization(self):
        """Test DatasetStats initialization with custom values."""
        stats = DatasetStats(
            public_domain=100,
            creative_commons=200,
            fair_use_research=50,
            copyrighted=75,
            unknown=25
        )
        
        self.assertEqual(stats.public_domain, 100)
        self.assertEqual(stats.creative_commons, 200)
        self.assertEqual(stats.fair_use_research, 50)
        self.assertEqual(stats.copyrighted, 75)
        self.assertEqual(stats.unknown, 25)
    
    def test_dataset_stats_properties(self):
        """Test DatasetStats computed properties."""
        stats = DatasetStats(
            public_domain=100,
            creative_commons=200,
            fair_use_research=50,
            copyrighted=75,
            unknown=25
        )
        
        self.assertEqual(stats.total, 450)
        self.assertEqual(stats.open_source_count, 300)  # PD + CC
        self.assertEqual(stats.research_safe_count, 350)  # PD + CC + Fair Use


class TestModelLicenseInfo(unittest.TestCase):
    """Test the ModelLicenseInfo dataclass."""
    
    def test_model_license_info_creation(self):
        """Test ModelLicenseInfo creation."""
        info = ModelLicenseInfo(
            model_name="test-model",
            license_type="Test License",
            training_data_sources=["Source1", "Source2"],
            compliance_level=ComplianceMode.RESEARCH_SAFE,
            attribution_required=True,
            commercial_use_allowed=False,
            research_use_allowed=True
        )
        
        self.assertEqual(info.model_name, "test-model")
        self.assertEqual(info.license_type, "Test License")
        self.assertEqual(info.training_data_sources, ["Source1", "Source2"])
        self.assertEqual(info.compliance_level, ComplianceMode.RESEARCH_SAFE)
        self.assertTrue(info.attribution_required)
        self.assertFalse(info.commercial_use_allowed)
        self.assertTrue(info.research_use_allowed)


class TestComplianceController(unittest.TestCase):
    """Test the ComplianceController class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_compliance_engine = Mock()
        self.mock_data_manager = Mock()
        
        self.controller = ComplianceController(
            compliance_engine=self.mock_compliance_engine,
            data_manager=self.mock_data_manager
        )
    
    def test_controller_initialization(self):
        """Test controller initialization."""
        self.assertEqual(self.controller.compliance_engine, self.mock_compliance_engine)
        self.assertEqual(self.controller.data_manager, self.mock_data_manager)
        self.assertEqual(self.controller.current_mode, ComplianceMode.RESEARCH_SAFE)
        self.assertIsInstance(self.controller.model_licenses, list)
        self.assertGreater(len(self.controller.model_licenses), 0)
        self.assertIsInstance(self.controller.dataset_stats, DatasetStats)
    
    def test_get_compliance_mode_info(self):
        """Test getting compliance mode information."""
        info = self.controller.get_compliance_mode_info()
        
        self.assertIsInstance(info, dict)
        self.assertIn(ComplianceMode.OPEN_SOURCE_ONLY.value, info)
        self.assertIn(ComplianceMode.RESEARCH_SAFE.value, info)
        self.assertIn(ComplianceMode.FULL_DATASET.value, info)
        
        # Check structure of mode info
        for mode_info in info.values():
            self.assertIn("title", mode_info)
            self.assertIn("description", mode_info)
            self.assertIn("data_types", mode_info)
            self.assertIn("restrictions", mode_info)
            self.assertIn("use_cases", mode_info)
    
    def test_set_compliance_mode_valid(self):
        """Test setting valid compliance mode."""
        success, message = self.controller.set_compliance_mode("open_only")
        
        self.assertTrue(success)
        self.assertIn("open_only", message)
        self.assertEqual(self.controller.current_mode, ComplianceMode.OPEN_SOURCE_ONLY)
    
    def test_set_compliance_mode_invalid(self):
        """Test setting invalid compliance mode."""
        success, message = self.controller.set_compliance_mode("invalid_mode")
        
        self.assertFalse(success)
        self.assertIn("Invalid compliance mode", message)
        # Mode should remain unchanged
        self.assertEqual(self.controller.current_mode, ComplianceMode.RESEARCH_SAFE)
    
    def test_get_available_models_open_source_only(self):
        """Test getting available models in open source only mode."""
        self.controller.set_compliance_mode("open_only")
        available_models = self.controller.get_available_models()
        
        self.assertIsInstance(available_models, list)
        # Should only include models with OPEN_SOURCE_ONLY compliance level
        for model_name in available_models:
            model_info = self.controller.get_model_license_info(model_name)
            self.assertEqual(model_info.compliance_level, ComplianceMode.OPEN_SOURCE_ONLY)
    
    def test_get_available_models_research_safe(self):
        """Test getting available models in research safe mode."""
        self.controller.set_compliance_mode("research_safe")
        available_models = self.controller.get_available_models()
        
        self.assertIsInstance(available_models, list)
        # Should include models with OPEN_SOURCE_ONLY or RESEARCH_SAFE compliance level
        for model_name in available_models:
            model_info = self.controller.get_model_license_info(model_name)
            self.assertIn(model_info.compliance_level, [ComplianceMode.OPEN_SOURCE_ONLY, ComplianceMode.RESEARCH_SAFE])
    
    def test_get_available_models_full_dataset(self):
        """Test getting available models in full dataset mode."""
        self.controller.set_compliance_mode("full_dataset")
        available_models = self.controller.get_available_models()
        
        self.assertIsInstance(available_models, list)
        # Should include all models
        all_model_names = [info.model_name for info in self.controller.model_licenses]
        self.assertEqual(set(available_models), set(all_model_names))
    
    def test_get_model_license_info_existing(self):
        """Test getting license info for existing model."""
        # Use the first model from the initialized list
        first_model = self.controller.model_licenses[0]
        model_info = self.controller.get_model_license_info(first_model.model_name)
        
        self.assertIsNotNone(model_info)
        self.assertEqual(model_info.model_name, first_model.model_name)
        self.assertIsInstance(model_info, ModelLicenseInfo)
    
    def test_get_model_license_info_nonexistent(self):
        """Test getting license info for non-existent model."""
        model_info = self.controller.get_model_license_info("nonexistent-model")
        
        self.assertIsNone(model_info)
    
    def test_get_dataset_stats(self):
        """Test getting dataset statistics."""
        stats = self.controller.get_dataset_stats()
        
        self.assertIsInstance(stats, dict)
        required_keys = [
            "public_domain", "creative_commons", "fair_use_research",
            "copyrighted", "unknown", "total", "open_source_count", "research_safe_count"
        ]
        for key in required_keys:
            self.assertIn(key, stats)
            self.assertIsInstance(stats[key], int)
    
    def test_get_attribution_text_open_source(self):
        """Test getting attribution text for open source mode."""
        self.controller.set_compliance_mode("open_only")
        attribution = self.controller.get_attribution_text()
        
        self.assertIsInstance(attribution, str)
        self.assertIn("Creative Commons", attribution)
        self.assertIn("Public Domain", attribution)
    
    def test_get_attribution_text_research_safe(self):
        """Test getting attribution text for research safe mode."""
        self.controller.set_compliance_mode("research_safe")
        attribution = self.controller.get_attribution_text()
        
        self.assertIsInstance(attribution, str)
        self.assertIn("Fair Use Research", attribution)
        self.assertIn("academic", attribution.lower())
    
    def test_get_attribution_text_full_dataset(self):
        """Test getting attribution text for full dataset mode."""
        self.controller.set_compliance_mode("full_dataset")
        attribution = self.controller.get_attribution_text()
        
        self.assertIsInstance(attribution, str)
        self.assertIn("Copyrighted", attribution)
        self.assertIn("fair use", attribution.lower())
    
    def test_validate_generation_request_valid(self):
        """Test validating a valid generation request."""
        # Use a model that should be available in research safe mode
        available_models = self.controller.get_available_models()
        if available_models:
            model_name = available_models[0]
            is_valid, message = self.controller.validate_generation_request(model_name, "a simple landscape")
            
            self.assertTrue(is_valid)
            self.assertIn("compliant", message)
    
    def test_validate_generation_request_unknown_model(self):
        """Test validating request with unknown model."""
        is_valid, message = self.controller.validate_generation_request("unknown-model", "test prompt")
        
        self.assertFalse(is_valid)
        self.assertIn("Unknown model", message)
    
    def test_validate_generation_request_non_compliant_model(self):
        """Test validating request with non-compliant model."""
        # Set to open source only mode
        self.controller.set_compliance_mode("open_only")
        
        # Try to use a research-safe model
        research_safe_models = [
            info.model_name for info in self.controller.model_licenses
            if info.compliance_level == ComplianceMode.RESEARCH_SAFE
        ]
        
        if research_safe_models:
            model_name = research_safe_models[0]
            is_valid, message = self.controller.validate_generation_request(model_name, "test prompt")
            
            self.assertFalse(is_valid)
            self.assertIn("not available", message)
    
    def test_validate_generation_request_problematic_prompt(self):
        """Test validating request with problematic prompt."""
        available_models = self.controller.get_available_models()
        if available_models:
            model_name = available_models[0]
            is_valid, message = self.controller.validate_generation_request(
                model_name, "create a copyrighted character from Disney"
            )
            
            self.assertFalse(is_valid)
            self.assertIn("compliance issues", message)
    
    @patch('src.ui.compliance_controls.GRADIO_AVAILABLE', True)
    @patch('src.ui.compliance_controls.gr')
    def test_create_compliance_components_with_gradio(self, mock_gr):
        """Test creating compliance components when Gradio is available."""
        # Mock Gradio components
        mock_gr.Dropdown.return_value = Mock()
        mock_gr.Dataframe.return_value = Mock()
        mock_gr.JSON.return_value = Mock()
        mock_gr.Textbox.return_value = Mock()
        mock_gr.Button.return_value = Mock()
        
        components = self.controller.create_compliance_components()
        
        self.assertIsInstance(components, dict)
        expected_components = [
            'compliance_mode', 'model_licenses', 'dataset_stats',
            'attribution_info', 'compliance_check_btn', 'compliance_results'
        ]
        for component_name in expected_components:
            self.assertIn(component_name, components)
    
    @patch('src.ui.compliance_controls.GRADIO_AVAILABLE', False)
    def test_create_compliance_components_without_gradio(self):
        """Test creating compliance components when Gradio is not available."""
        components = self.controller.create_compliance_components()
        
        self.assertEqual(components, {})
    
    @patch('src.ui.compliance_controls.GRADIO_AVAILABLE', True)
    @patch('src.ui.compliance_controls.gr')
    def test_create_compliance_info_panel(self, mock_gr):
        """Test creating compliance info panel."""
        mock_gr.Markdown.return_value = Mock()
        
        panel = self.controller.create_compliance_info_panel()
        
        self.assertIsNotNone(panel)
        mock_gr.Markdown.assert_called_once()


class TestUtilityFunctions(unittest.TestCase):
    """Test utility functions for compliance controls."""
    
    def test_create_compliance_controller(self):
        """Test creating compliance controller."""
        mock_compliance_engine = Mock()
        mock_data_manager = Mock()
        
        controller = create_compliance_controller(
            compliance_engine=mock_compliance_engine,
            data_manager=mock_data_manager
        )
        
        self.assertIsInstance(controller, ComplianceController)
        self.assertEqual(controller.compliance_engine, mock_compliance_engine)
        self.assertEqual(controller.data_manager, mock_data_manager)
    
    def test_create_compliance_controller_no_args(self):
        """Test creating compliance controller without arguments."""
        controller = create_compliance_controller()
        
        self.assertIsInstance(controller, ComplianceController)
        self.assertIsNone(controller.compliance_engine)
        self.assertIsNone(controller.data_manager)


if __name__ == '__main__':
    unittest.main()