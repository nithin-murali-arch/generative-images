"""
Copyright compliance controls for the research interface.

This module provides UI components and logic for managing copyright compliance
modes, model selection based on training data licenses, and attribution display.
"""

import logging
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

try:
    from ..core.interfaces import ComplianceMode, LicenseType
except ImportError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from core.interfaces import ComplianceMode, LicenseType

logger = logging.getLogger(__name__)

# Try to import Gradio
try:
    import gradio as gr
    GRADIO_AVAILABLE = True
except ImportError:
    # Create a mock gr module for type hints and basic functionality
    class MockGradio:
        class Dropdown:
            pass
        class Textbox:
            pass
        class Dataframe:
            pass
        class JSON:
            pass
        class Button:
            pass
        class Markdown:
            pass
        class Row:
            def __enter__(self):
                return self
            def __exit__(self, *args):
                pass
        class Column:
            def __enter__(self):
                return self
            def __exit__(self, *args):
                pass
    
    gr = MockGradio()
    GRADIO_AVAILABLE = False
    logger.warning("Gradio not available - compliance controls will be limited")


@dataclass
class ModelLicenseInfo:
    """Information about a model's training data license."""
    model_name: str
    license_type: str
    training_data_sources: List[str]
    compliance_level: ComplianceMode
    attribution_required: bool
    commercial_use_allowed: bool
    research_use_allowed: bool


@dataclass
class DatasetStats:
    """Statistics about dataset license breakdown."""
    public_domain: int = 0
    creative_commons: int = 0
    fair_use_research: int = 0
    copyrighted: int = 0
    unknown: int = 0
    
    @property
    def total(self) -> int:
        """Total number of items in dataset."""
        return (self.public_domain + self.creative_commons + 
                self.fair_use_research + self.copyrighted + self.unknown)
    
    @property
    def open_source_count(self) -> int:
        """Count of open source items (PD + CC)."""
        return self.public_domain + self.creative_commons
    
    @property
    def research_safe_count(self) -> int:
        """Count of research safe items (PD + CC + Fair Use)."""
        return self.open_source_count + self.fair_use_research


class ComplianceController:
    """
    Controller for managing copyright compliance in the research interface.
    
    Handles compliance mode selection, model filtering based on training data
    licenses, and attribution information display.
    """
    
    def __init__(self, compliance_engine=None, data_manager=None):
        """
        Initialize the compliance controller.
        
        Args:
            compliance_engine: Copyright compliance engine
            data_manager: Data management system
        """
        self.compliance_engine = compliance_engine
        self.data_manager = data_manager
        self.current_mode = ComplianceMode.RESEARCH_SAFE
        self.model_licenses = self._initialize_model_licenses()
        self.dataset_stats = DatasetStats()
        
        logger.info("ComplianceController initialized")
    
    def get_compliance_mode_info(self) -> Dict[str, str]:
        """Get information about available compliance modes."""
        return {
            ComplianceMode.OPEN_SOURCE_ONLY.value: {
                "title": "Open Source Only",
                "description": "Uses only Public Domain and Creative Commons content",
                "data_types": ["Public Domain", "Creative Commons"],
                "restrictions": "No copyrighted material",
                "use_cases": "Commercial and research use"
            },
            ComplianceMode.RESEARCH_SAFE.value: {
                "title": "Research Safe", 
                "description": "Adds Fair Use research content (academic use only)",
                "data_types": ["Public Domain", "Creative Commons", "Fair Use Research"],
                "restrictions": "Academic/research use only",
                "use_cases": "Academic research and education"
            },
            ComplianceMode.FULL_DATASET.value: {
                "title": "Full Dataset",
                "description": "All content including copyrighted material (comparison research only)",
                "data_types": ["All content types"],
                "restrictions": "Research comparison only - no distribution",
                "use_cases": "Academic comparison studies only"
            }
        }
    
    def set_compliance_mode(self, mode: str) -> Tuple[bool, str]:
        """
        Set the current compliance mode.
        
        Args:
            mode: Compliance mode string
            
        Returns:
            Tuple of (success, message)
        """
        try:
            new_mode = ComplianceMode(mode)
            self.current_mode = new_mode
            
            # Update dataset stats based on new mode
            self._update_dataset_stats()
            
            logger.info(f"Compliance mode set to: {mode}")
            return True, f"Compliance mode set to: {new_mode.value}"
            
        except ValueError as e:
            logger.error(f"Invalid compliance mode: {mode}")
            return False, f"Invalid compliance mode: {mode}"
    
    def get_available_models(self) -> List[str]:
        """Get models available for current compliance mode."""
        available_models = []
        
        for model_info in self.model_licenses:
            if self._is_model_compliant(model_info):
                available_models.append(model_info.model_name)
        
        return available_models
    
    def get_model_license_info(self, model_name: str) -> Optional[ModelLicenseInfo]:
        """Get license information for a specific model."""
        for model_info in self.model_licenses:
            if model_info.model_name == model_name:
                return model_info
        return None
    
    def get_dataset_stats(self) -> Dict[str, Any]:
        """Get current dataset statistics."""
        return {
            "public_domain": self.dataset_stats.public_domain,
            "creative_commons": self.dataset_stats.creative_commons,
            "fair_use_research": self.dataset_stats.fair_use_research,
            "copyrighted": self.dataset_stats.copyrighted,
            "unknown": self.dataset_stats.unknown,
            "total": self.dataset_stats.total,
            "open_source_count": self.dataset_stats.open_source_count,
            "research_safe_count": self.dataset_stats.research_safe_count
        }
    
    def get_attribution_text(self) -> str:
        """Get attribution text for current compliance mode."""
        if self.current_mode == ComplianceMode.OPEN_SOURCE_ONLY:
            return self._get_open_source_attribution()
        elif self.current_mode == ComplianceMode.RESEARCH_SAFE:
            return self._get_research_safe_attribution()
        else:  # FULL_DATASET
            return self._get_full_dataset_attribution()
    
    def validate_generation_request(self, model_name: str, prompt: str) -> Tuple[bool, str]:
        """
        Validate a generation request against compliance rules.
        
        Args:
            model_name: Name of the model to use
            prompt: Generation prompt
            
        Returns:
            Tuple of (is_valid, message)
        """
        # Check if model is compliant with current mode
        model_info = self.get_model_license_info(model_name)
        if not model_info:
            return False, f"Unknown model: {model_name}"
        
        if not self._is_model_compliant(model_info):
            return False, f"Model {model_name} not available in {self.current_mode.value} mode"
        
        # Check prompt for potential compliance issues
        compliance_issues = self._check_prompt_compliance(prompt)
        if compliance_issues:
            return False, f"Prompt compliance issues: {', '.join(compliance_issues)}"
        
        return True, "Request is compliant"
    
    def create_compliance_components(self) -> Dict[str, Any]:
        """Create Gradio components for compliance controls."""
        if not GRADIO_AVAILABLE:
            logger.warning("Gradio not available - returning empty components")
            return {}
        
        components = {}
        
        # Compliance mode selector
        components['compliance_mode'] = gr.Dropdown(
            choices=[mode.value for mode in ComplianceMode],
            value=self.current_mode.value,
            label="Copyright Compliance Mode",
            info="Controls which training data is used for generation"
        )
        
        # Model license information display
        components['model_licenses'] = gr.Dataframe(
            headers=["Model", "License", "Training Data", "Compliance Level"],
            value=self._get_model_license_table(),
            label="Model License Information",
            interactive=False
        )
        
        # Dataset statistics
        components['dataset_stats'] = gr.JSON(
            label="Dataset License Breakdown",
            value=self.get_dataset_stats()
        )
        
        # Attribution information
        components['attribution_info'] = gr.Textbox(
            label="Attribution Requirements",
            value=self.get_attribution_text(),
            lines=8,
            interactive=False
        )
        
        # Compliance check button
        components['compliance_check_btn'] = gr.Button(
            "Run Compliance Check",
            variant="secondary"
        )
        
        # Compliance results
        components['compliance_results'] = gr.JSON(
            label="Compliance Check Results",
            visible=False
        )
        
        return components
    
    def create_compliance_info_panel(self) -> Any:
        """Create an information panel explaining compliance modes."""
        if not GRADIO_AVAILABLE:
            return None
        
        info_text = """
        ## Copyright Compliance Modes
        
        ### Open Source Only
        - **Data**: Public Domain + Creative Commons content only
        - **Use**: Commercial and research applications
        - **Restrictions**: None for generated content
        - **Attribution**: Creative Commons sources require attribution
        
        ### Research Safe  
        - **Data**: Open Source + Fair Use research content
        - **Use**: Academic research and education only
        - **Restrictions**: No commercial use of generated content
        - **Attribution**: All sources require proper academic citation
        
        ### Full Dataset
        - **Data**: All available content including copyrighted material
        - **Use**: Research comparison studies only
        - **Restrictions**: No distribution of generated content
        - **Attribution**: Full source documentation required
        
        **Important**: Always verify compliance requirements for your specific use case.
        """
        
        return gr.Markdown(info_text)
    
    def _initialize_model_licenses(self) -> List[ModelLicenseInfo]:
        """Initialize model license information."""
        return [
            ModelLicenseInfo(
                model_name="stable-diffusion-v1-5",
                license_type="CreativeML Open RAIL-M",
                training_data_sources=["LAION-5B (filtered)", "OpenImages", "CC12M"],
                compliance_level=ComplianceMode.RESEARCH_SAFE,
                attribution_required=True,
                commercial_use_allowed=False,
                research_use_allowed=True
            ),
            ModelLicenseInfo(
                model_name="sdxl-turbo",
                license_type="CreativeML Open RAIL++-M",
                training_data_sources=["LAION-5B (filtered)", "Internal dataset"],
                compliance_level=ComplianceMode.RESEARCH_SAFE,
                attribution_required=True,
                commercial_use_allowed=False,
                research_use_allowed=True
            ),
            ModelLicenseInfo(
                model_name="flux.1-schnell",
                license_type="Apache 2.0",
                training_data_sources=["Curated open datasets"],
                compliance_level=ComplianceMode.OPEN_SOURCE_ONLY,
                attribution_required=False,
                commercial_use_allowed=True,
                research_use_allowed=True
            ),
            ModelLicenseInfo(
                model_name="stable-video-diffusion",
                license_type="Stability AI Community License",
                training_data_sources=["WebVid-10M", "Internal video dataset"],
                compliance_level=ComplianceMode.RESEARCH_SAFE,
                attribution_required=True,
                commercial_use_allowed=False,
                research_use_allowed=True
            ),
            ModelLicenseInfo(
                model_name="animatediff",
                license_type="CreativeML Open RAIL-M",
                training_data_sources=["WebVid-10M", "LAION-5B"],
                compliance_level=ComplianceMode.RESEARCH_SAFE,
                attribution_required=True,
                commercial_use_allowed=False,
                research_use_allowed=True
            )
        ]
    
    def _update_dataset_stats(self):
        """Update dataset statistics based on current compliance mode."""
        # Base statistics (these would come from actual data manager)
        base_stats = DatasetStats(
            public_domain=1000,
            creative_commons=2500,
            fair_use_research=800,
            copyrighted=1200,
            unknown=150
        )
        
        if self.current_mode == ComplianceMode.OPEN_SOURCE_ONLY:
            self.dataset_stats = DatasetStats(
                public_domain=base_stats.public_domain,
                creative_commons=base_stats.creative_commons,
                fair_use_research=0,
                copyrighted=0,
                unknown=0
            )
        elif self.current_mode == ComplianceMode.RESEARCH_SAFE:
            self.dataset_stats = DatasetStats(
                public_domain=base_stats.public_domain,
                creative_commons=base_stats.creative_commons,
                fair_use_research=base_stats.fair_use_research,
                copyrighted=0,
                unknown=base_stats.unknown
            )
        else:  # FULL_DATASET
            self.dataset_stats = base_stats
    
    def _is_model_compliant(self, model_info: ModelLicenseInfo) -> bool:
        """Check if a model is compliant with current mode."""
        if self.current_mode == ComplianceMode.OPEN_SOURCE_ONLY:
            return model_info.compliance_level == ComplianceMode.OPEN_SOURCE_ONLY
        elif self.current_mode == ComplianceMode.RESEARCH_SAFE:
            return model_info.compliance_level in [ComplianceMode.OPEN_SOURCE_ONLY, ComplianceMode.RESEARCH_SAFE]
        else:  # FULL_DATASET
            return True  # All models available in full dataset mode
    
    def _get_model_license_table(self) -> List[List[str]]:
        """Get model license information as table data."""
        table_data = []
        for model_info in self.model_licenses:
            compliance_status = "✓ Available" if self._is_model_compliant(model_info) else "✗ Restricted"
            table_data.append([
                model_info.model_name,
                model_info.license_type,
                ", ".join(model_info.training_data_sources[:2]),  # Limit to first 2 sources
                compliance_status
            ])
        return table_data
    
    def _get_open_source_attribution(self) -> str:
        """Get attribution text for open source mode."""
        return """
Attribution Requirements for Open Source Mode:

Creative Commons Content:
- Attribution required for CC-BY licensed content
- Include creator name, source, and license type
- Example: "Image by [Creator] from [Source] (CC BY 4.0)"

Public Domain Content:
- No attribution required but recommended
- Example: "Image from [Source] (Public Domain)"

Generated Content:
- May be used commercially with proper attribution
- Include model name and license in derivative works
- Example: "Generated using FLUX.1-schnell (Apache 2.0)"
        """.strip()
    
    def _get_research_safe_attribution(self) -> str:
        """Get attribution text for research safe mode."""
        return """
Attribution Requirements for Research Safe Mode:

All Open Source Requirements Plus:

Fair Use Research Content:
- Full academic citation required
- Include DOI or permanent link when available
- Example: "Smith, J. (2023). Dataset Name. Journal, 1(1), 1-10. DOI:10.1000/xyz"

Generated Content:
- Academic use only - no commercial distribution
- Include full methodology and model information
- Example: "Generated using Stable Diffusion v1.5 for academic research purposes"

Research Publications:
- Acknowledge all data sources in methodology section
- Include compliance mode in reproducibility statement
- Follow institutional ethics guidelines
        """.strip()
    
    def _get_full_dataset_attribution(self) -> str:
        """Get attribution text for full dataset mode."""
        return """
Attribution Requirements for Full Dataset Mode:

All Previous Requirements Plus:

Copyrighted Content:
- Full source documentation required
- Fair use justification must be documented
- No distribution of generated content permitted
- Example: "Copyrighted image from [Source] used under fair use for research comparison"

Generated Content:
- Research comparison only - no publication of outputs
- Full dataset composition must be documented
- Include copyright analysis in research methodology

Legal Compliance:
- Verify fair use applies to your jurisdiction
- Obtain institutional legal review if required
- Document all copyright holders and usage justification
- Maintain detailed audit trail of all sources
        """.strip()
    
    def _check_prompt_compliance(self, prompt: str) -> List[str]:
        """Check prompt for potential compliance issues."""
        issues = []
        
        # Check for potentially problematic content
        problematic_terms = [
            "copyrighted character", "trademarked", "brand logo", 
            "celebrity", "famous person", "movie character"
        ]
        
        prompt_lower = prompt.lower()
        for term in problematic_terms:
            if term in prompt_lower:
                issues.append(f"Potential copyright issue: '{term}' detected")
        
        # Check for commercial terms in non-commercial modes
        if self.current_mode != ComplianceMode.OPEN_SOURCE_ONLY:
            commercial_terms = ["for sale", "commercial use", "marketing", "advertisement"]
            for term in commercial_terms:
                if term in prompt_lower:
                    issues.append(f"Commercial use detected in non-commercial mode: '{term}'")
        
        return issues


def create_compliance_controller(compliance_engine=None, data_manager=None) -> ComplianceController:
    """
    Create and initialize a compliance controller.
    
    Args:
        compliance_engine: Copyright compliance engine
        data_manager: Data management system
        
    Returns:
        ComplianceController: Initialized controller
    """
    controller = ComplianceController(compliance_engine, data_manager)
    logger.info("Compliance controller created successfully")
    return controller