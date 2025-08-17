"""
Unit tests for LLM Controller Foundation.

Tests the core functionality of request parsing, intent classification,
conversation context management, and workflow coordination.
"""

import pytest
import uuid
from datetime import datetime
from unittest.mock import Mock, patch

from src.core.llm_controller import LLMController, IntentClassification, WorkflowStep
from src.core.interfaces import (
    HardwareConfig, ConversationContext, ComplianceMode, OutputType,
    GenerationRequest, StyleConfig, SystemError
)


@pytest.fixture
def hardware_config():
    """Create test hardware configuration."""
    return HardwareConfig(
        vram_size=8192,  # 8GB VRAM
        gpu_model="RTX 3070",
        cpu_cores=8,
        ram_size=16384,  # 16GB RAM
        cuda_available=True,
        optimization_level="balanced"
    )


@pytest.fixture
def conversation_context():
    """Create test conversation context."""
    return ConversationContext(
        conversation_id="test_conv_123",
        history=[],
        current_mode=ComplianceMode.RESEARCH_SAFE,
        user_preferences={}
    )


@pytest.fixture
def llm_controller(hardware_config):
    """Create LLM controller instance for testing."""
    return LLMController(hardware_config)


class TestLLMControllerInitialization:
    """Test LLM controller initialization."""
    
    def test_initialization_success(self, hardware_config):
        """Test successful controller initialization."""
        controller = LLMController(hardware_config)
        
        assert controller.hardware_config == hardware_config
        assert isinstance(controller.conversations, dict)
        assert isinstance(controller.active_workflows, dict)
        assert len(controller.image_patterns) > 0
        assert len(controller.video_patterns) > 0
    
    def test_classification_patterns_loaded(self, llm_controller):
        """Test that classification patterns are properly loaded."""
        assert hasattr(llm_controller, 'image_patterns')
        assert hasattr(llm_controller, 'video_patterns')
        assert hasattr(llm_controller, 'multimodal_patterns')
        assert hasattr(llm_controller, 'style_patterns')


class TestIntentClassification:
    """Test intent classification functionality."""
    
    def test_image_intent_classification(self, llm_controller):
        """Test classification of image generation requests."""
        prompts = [
            "Generate an image of a sunset",
            "Create a picture of a cat",
            "Draw a portrait in realistic style",
            "Make an illustration of a forest"
        ]
        
        for prompt in prompts:
            intent = llm_controller._classify_intent(prompt)
            assert intent.output_type == OutputType.IMAGE
            assert intent.confidence > 0.0
            assert "image" in intent.reasoning.lower() or "patterns" in intent.reasoning.lower()
    
    def test_video_intent_classification(self, llm_controller):
        """Test classification of video generation requests."""
        prompts = [
            "Generate a video of waves crashing",
            "Create an animation of a dancing robot",
            "Make a 10-second clip of rain falling",
            "Animate this character walking"
        ]
        
        for prompt in prompts:
            intent = llm_controller._classify_intent(prompt)
            assert intent.output_type == OutputType.VIDEO
            assert intent.confidence > 0.0
            assert "video" in intent.reasoning.lower()
    
    def test_multimodal_intent_classification(self, llm_controller):
        """Test classification of multimodal requests."""
        prompts = [
            "First create an image, then animate it into a video",
            "Generate both a picture and a video of this scene",
            "Make an image and then create a sequence showing motion"
        ]
        
        for prompt in prompts:
            intent = llm_controller._classify_intent(prompt)
            assert intent.output_type == OutputType.MULTIMODAL
            assert intent.confidence > 0.0
    
    def test_ambiguous_prompt_defaults_to_image(self, llm_controller):
        """Test that ambiguous prompts default to image generation."""
        ambiguous_prompts = [
            "Create something beautiful",
            "Make art",
            "Generate content"
        ]
        
        for prompt in ambiguous_prompts:
            intent = llm_controller._classify_intent(prompt)
            assert intent.output_type == OutputType.IMAGE
            assert intent.confidence == 0.5
            assert "defaulting" in intent.reasoning.lower()
    
    def test_parameter_extraction(self, llm_controller):
        """Test extraction of parameters from prompts."""
        prompt = "Create a high quality 1024x1024 image in anime style"
        params = llm_controller._extract_parameters(prompt)
        
        assert 'quality' in params
        assert 'resolution' in params
        assert 'style' in params or params.get('quality') == 'high'


class TestRequestParsing:
    """Test request parsing functionality."""
    
    def test_parse_simple_image_request(self, llm_controller, conversation_context):
        """Test parsing a simple image generation request."""
        prompt = "Generate an image of a mountain landscape"
        
        request = llm_controller.parse_request(prompt, conversation_context)
        
        assert isinstance(request, GenerationRequest)
        assert request.prompt == prompt
        assert request.output_type == OutputType.IMAGE
        assert request.compliance_mode == ComplianceMode.RESEARCH_SAFE
        assert request.context == conversation_context
        assert isinstance(request.style_config, StyleConfig)
    
    def test_parse_video_request_with_parameters(self, llm_controller, conversation_context):
        """Test parsing video request with extracted parameters."""
        prompt = "Create a 10-second video of ocean waves in high quality"
        
        request = llm_controller.parse_request(prompt, conversation_context)
        
        assert request.output_type == OutputType.VIDEO
        assert 'duration' in request.additional_params or 'duration' in request.style_config.generation_params
    
    def test_conversation_history_updated(self, llm_controller, conversation_context):
        """Test that conversation history is updated after parsing."""
        initial_history_length = len(conversation_context.history)
        prompt = "Generate an image of a cat"
        
        llm_controller.parse_request(prompt, conversation_context)
        
        assert len(conversation_context.history) == initial_history_length + 1
        assert conversation_context.history[-1]['prompt'] == prompt
    
    def test_parse_request_error_handling(self, llm_controller, conversation_context):
        """Test error handling in request parsing."""
        # Mock an error in intent classification
        with patch.object(llm_controller, '_classify_intent', side_effect=Exception("Test error")):
            with pytest.raises(SystemError, match="Failed to parse request"):
                llm_controller.parse_request("test prompt", conversation_context)


class TestRequestRouting:
    """Test request routing functionality."""
    
    def test_route_image_request(self, llm_controller, hardware_config, conversation_context):
        """Test routing of image generation requests."""
        request = GenerationRequest(
            prompt="test",
            output_type=OutputType.IMAGE,
            style_config=StyleConfig(),
            compliance_mode=ComplianceMode.RESEARCH_SAFE,
            hardware_constraints=hardware_config,
            context=conversation_context
        )
        
        pipeline = llm_controller.route_request(request)
        assert pipeline == "image"
    
    def test_route_video_request(self, llm_controller, hardware_config, conversation_context):
        """Test routing of video generation requests."""
        request = GenerationRequest(
            prompt="test",
            output_type=OutputType.VIDEO,
            style_config=StyleConfig(),
            compliance_mode=ComplianceMode.RESEARCH_SAFE,
            hardware_constraints=hardware_config,
            context=conversation_context
        )
        
        pipeline = llm_controller.route_request(request)
        assert pipeline == "video"
    
    def test_route_multimodal_request(self, llm_controller, hardware_config, conversation_context):
        """Test routing of multimodal requests."""
        request = GenerationRequest(
            prompt="test",
            output_type=OutputType.MULTIMODAL,
            style_config=StyleConfig(),
            compliance_mode=ComplianceMode.RESEARCH_SAFE,
            hardware_constraints=hardware_config,
            context=conversation_context
        )
        
        pipeline = llm_controller.route_request(request)
        assert pipeline == "multimodal"
    
    def test_route_with_limited_vram_warning(self, llm_controller, conversation_context):
        """Test routing with limited VRAM shows appropriate warnings."""
        limited_hardware = HardwareConfig(
            vram_size=4096,  # 4GB VRAM
            gpu_model="GTX 1650",
            cpu_cores=4,
            ram_size=8192,
            cuda_available=True,
            optimization_level="aggressive"
        )
        
        request = GenerationRequest(
            prompt="test",
            output_type=OutputType.VIDEO,
            style_config=StyleConfig(),
            compliance_mode=ComplianceMode.RESEARCH_SAFE,
            hardware_constraints=limited_hardware,
            context=conversation_context
        )
        
        with patch('src.core.llm_controller.logger') as mock_logger:
            pipeline = llm_controller.route_request(request)
            assert pipeline == "video"
            mock_logger.warning.assert_called()


class TestConversationManagement:
    """Test conversation context management."""
    
    def test_create_new_conversation(self, llm_controller):
        """Test creation of new conversation context."""
        conv_id = "new_conversation_123"
        
        context = llm_controller.manage_context(conv_id)
        
        assert context.conversation_id == conv_id
        assert len(context.history) == 0
        assert context.current_mode == ComplianceMode.RESEARCH_SAFE
        assert conv_id in llm_controller.conversations
    
    def test_retrieve_existing_conversation(self, llm_controller):
        """Test retrieval of existing conversation context."""
        conv_id = "existing_conversation_456"
        
        # Create conversation first
        original_context = llm_controller.manage_context(conv_id)
        original_context.history.append({"test": "data"})
        
        # Retrieve same conversation
        retrieved_context = llm_controller.manage_context(conv_id)
        
        assert retrieved_context == original_context
        assert len(retrieved_context.history) == 1
    
    def test_conversation_summary(self, llm_controller):
        """Test conversation summary generation."""
        conv_id = "summary_test_789"
        context = llm_controller.manage_context(conv_id)
        
        # Add some history
        context.history = [
            {'intent': {'output_type': 'image'}},
            {'intent': {'output_type': 'video'}},
            {'intent': {'output_type': 'image'}}
        ]
        
        summary = llm_controller.get_conversation_summary(conv_id)
        
        assert summary['conversation_id'] == conv_id
        assert summary['total_interactions'] == 3
        assert summary['current_mode'] == ComplianceMode.RESEARCH_SAFE.value
        assert len(summary['recent_intents']) == 3
    
    def test_update_compliance_mode(self, llm_controller):
        """Test updating compliance mode for conversation."""
        conv_id = "compliance_test_101"
        context = llm_controller.manage_context(conv_id)
        
        # Update compliance mode
        new_mode = ComplianceMode.OPEN_SOURCE_ONLY
        llm_controller.update_compliance_mode(conv_id, new_mode)
        
        assert context.current_mode == new_mode
    
    def test_cleanup_conversation(self, llm_controller):
        """Test conversation cleanup."""
        conv_id = "cleanup_test_202"
        llm_controller.manage_context(conv_id)
        
        assert conv_id in llm_controller.conversations
        
        llm_controller.cleanup_conversation(conv_id)
        
        assert conv_id not in llm_controller.conversations


class TestWorkflowCoordination:
    """Test workflow coordination functionality."""
    
    def test_coordinate_simple_workflow(self, llm_controller):
        """Test coordination of a simple workflow."""
        steps = [
            {
                'type': 'generate',
                'pipeline': 'image',
                'prompt': 'Create an image of a cat',
                'params': {'style': 'realistic'}
            },
            {
                'type': 'generate',
                'pipeline': 'video',
                'prompt': 'Animate the cat walking',
                'params': {'duration': 5}
            }
        ]
        
        result = llm_controller.coordinate_workflow(steps)
        
        assert 'workflow_id' in result
        assert result['steps'] == 2
        assert result['status'] == 'planned'
        assert 'execution_plan' in result
        assert len(result['execution_plan']) == 2
    
    def test_workflow_step_creation(self, llm_controller):
        """Test creation of workflow steps."""
        steps = [
            {
                'type': 'generate',
                'pipeline': 'image',
                'prompt': 'Test prompt',
                'params': {},
                'dependencies': []
            }
        ]
        
        result = llm_controller.coordinate_workflow(steps)
        workflow_id = result['workflow_id']
        
        assert workflow_id in llm_controller.active_workflows
        workflow_steps = llm_controller.active_workflows[workflow_id]
        assert len(workflow_steps) == 1
        assert isinstance(workflow_steps[0], WorkflowStep)
    
    def test_execution_plan_generation(self, llm_controller):
        """Test execution plan generation."""
        steps = [
            {
                'type': 'generate',
                'pipeline': 'image',
                'prompt': 'Test image',
                'params': {}
            }
        ]
        
        result = llm_controller.coordinate_workflow(steps)
        plan = result['execution_plan']
        
        assert len(plan) == 1
        assert 'step_id' in plan[0]
        assert 'pipeline' in plan[0]
        assert 'estimated_time' in plan[0]
        assert 'memory_requirements' in plan[0]
    
    def test_workflow_error_handling(self, llm_controller):
        """Test error handling in workflow coordination."""
        # Test with invalid step data
        invalid_steps = [{'invalid': 'data'}]
        
        # Should not raise exception but handle gracefully
        result = llm_controller.coordinate_workflow(invalid_steps)
        assert 'workflow_id' in result


class TestMemoryAndPerformance:
    """Test memory management and performance considerations."""
    
    def test_conversation_history_trimming(self, llm_controller):
        """Test that conversation history is trimmed to prevent memory issues."""
        conv_id = "history_trim_test"
        context = llm_controller.manage_context(conv_id)
        
        # Add more than max history entries
        for i in range(60):  # More than the 50 limit
            llm_controller._update_conversation_history(
                context, f"prompt {i}", 
                IntentClassification(OutputType.IMAGE, 0.8, "test", {})
            )
        
        assert len(context.history) == 50  # Should be trimmed to max
    
    def test_time_estimation_hardware_adjustment(self, llm_controller):
        """Test that time estimation adjusts for hardware capabilities."""
        # Test with high-end hardware
        high_end_config = HardwareConfig(
            vram_size=24576,  # 24GB
            gpu_model="RTX 4090",
            cpu_cores=16,
            ram_size=32768,
            cuda_available=True,
            optimization_level="minimal"
        )
        
        controller_high_end = LLMController(high_end_config)
        
        step = WorkflowStep("test", "generate", "image", "test", {}, [])
        
        time_high_end = controller_high_end._estimate_step_time(step)
        time_regular = llm_controller._estimate_step_time(step)
        
        assert time_high_end < time_regular  # High-end should be faster
    
    def test_memory_requirements_estimation(self, llm_controller):
        """Test memory requirements estimation for different pipelines."""
        image_step = WorkflowStep("test", "generate", "image", "test", {}, [])
        video_step = WorkflowStep("test", "generate", "video", "test", {}, [])
        
        image_reqs = llm_controller._estimate_memory_requirements(image_step)
        video_reqs = llm_controller._estimate_memory_requirements(video_step)
        
        assert video_reqs['vram'] > image_reqs['vram']  # Video needs more VRAM
        assert video_reqs['ram'] > image_reqs['ram']    # Video needs more RAM


if __name__ == "__main__":
    pytest.main([__file__])