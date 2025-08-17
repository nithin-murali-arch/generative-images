"""
Unit tests for Workflow Coordination System.

Tests workflow creation, execution, state management, prompt optimization,
and error recovery functionality.
"""

import pytest
import time
from unittest.mock import Mock, patch
from datetime import datetime

from src.core.workflow_coordinator import (
    WorkflowCoordinator, PromptOptimizer, WorkflowStep, WorkflowContext,
    WorkflowDefinition, WorkflowStatus, StepStatus
)
from src.core.interfaces import (
    HardwareConfig, ComplianceMode, GenerationRequest, GenerationResult,
    OutputType, StyleConfig
)


@pytest.fixture
def hardware_config():
    """Create test hardware configuration."""
    return HardwareConfig(
        vram_size=8192,
        gpu_model="RTX 3070",
        cpu_cores=8,
        ram_size=16384,
        cuda_available=True,
        optimization_level="balanced"
    )


@pytest.fixture
def workflow_coordinator(hardware_config):
    """Create workflow coordinator instance."""
    return WorkflowCoordinator(hardware_config)


@pytest.fixture
def prompt_optimizer():
    """Create prompt optimizer instance."""
    return PromptOptimizer()


@pytest.fixture
def sample_workflow_config():
    """Create sample workflow configuration."""
    return [
        {
            'step_id': 'step_1',
            'type': 'generate',
            'pipeline': 'image',
            'prompt': 'Create an image of a sunset',
            'params': {'style': 'realistic'},
            'dependencies': []
        },
        {
            'step_id': 'step_2',
            'type': 'generate',
            'pipeline': 'video',
            'prompt': 'Animate the sunset scene',
            'params': {'duration': 5},
            'dependencies': ['step_1']
        }
    ]


class TestPromptOptimizer:
    """Test prompt optimization functionality."""
    
    def test_optimize_image_prompt_basic(self, prompt_optimizer):
        """Test basic image prompt optimization."""
        context = WorkflowContext(
            workflow_id="test",
            user_id="user1",
            compliance_mode=ComplianceMode.RESEARCH_SAFE,
            hardware_config=HardwareConfig(8192, "RTX 3070", 8, 16384, True, "balanced"),
            global_params={},
            intermediate_results={},
            shared_state={}
        )
        
        original = "A beautiful landscape"
        optimized = prompt_optimizer.optimize_prompt(original, "image", context, {})
        
        assert len(optimized) > len(original)
        assert "quality" in optimized.lower()
        assert "detailed" in optimized.lower()
    
    def test_optimize_image_prompt_with_previous_results(self, prompt_optimizer):
        """Test image prompt optimization with previous results."""
        context = WorkflowContext(
            workflow_id="test",
            user_id="user1", 
            compliance_mode=ComplianceMode.RESEARCH_SAFE,
            hardware_config=HardwareConfig(8192, "RTX 3070", 8, 16384, True, "balanced"),
            global_params={},
            intermediate_results={},
            shared_state={}
        )
        
        previous_results = {
            'step_1': {
                'output_type': 'image',
                'style_detected': 'anime'
            }
        }
        
        original = "A character portrait"
        optimized = prompt_optimizer.optimize_prompt(original, "image", context, previous_results)
        
        assert "anime style" in optimized.lower()
    
    def test_optimize_video_prompt(self, prompt_optimizer):
        """Test video prompt optimization."""
        context = WorkflowContext(
            workflow_id="test",
            user_id="user1",
            compliance_mode=ComplianceMode.RESEARCH_SAFE,
            hardware_config=HardwareConfig(4096, "GTX 1650", 4, 8192, True, "aggressive"),
            global_params={'duration': 3},
            intermediate_results={},
            shared_state={}
        )
        
        original = "A dancing robot"
        optimized = prompt_optimizer.optimize_prompt(original, "video", context, {})
        
        assert "motion" in optimized.lower() or "movement" in optimized.lower()
        assert "3 second" in optimized
        assert "limited resources" in optimized.lower()  # Due to low VRAM
    
    def test_optimize_text_prompt_with_context(self, prompt_optimizer):
        """Test text prompt optimization with context."""
        context = WorkflowContext(
            workflow_id="test",
            user_id="user1",
            compliance_mode=ComplianceMode.RESEARCH_SAFE,
            hardware_config=HardwareConfig(8192, "RTX 3070", 8, 16384, True, "balanced"),
            global_params={},
            intermediate_results={},
            shared_state={}
        )
        
        previous_results = {
            'step_1': {
                'description': 'A sunset over mountains'
            }
        }
        
        original = "Write a story about this scene"
        optimized = prompt_optimizer.optimize_prompt(original, "text", context, previous_results)
        
        assert "Based on:" in optimized
        assert "sunset over mountains" in optimized
        assert "research and educational purposes" in optimized
    
    def test_optimize_compliance_modes(self, prompt_optimizer):
        """Test prompt optimization for different compliance modes."""
        context_open_source = WorkflowContext(
            workflow_id="test",
            user_id="user1",
            compliance_mode=ComplianceMode.OPEN_SOURCE_ONLY,
            hardware_config=HardwareConfig(8192, "RTX 3070", 8, 16384, True, "balanced"),
            global_params={},
            intermediate_results={},
            shared_state={}
        )
        
        original = "Create an artwork"
        optimized = prompt_optimizer.optimize_prompt(original, "image", context_open_source, {})
        
        assert "open source" in optimized.lower()
        assert "creative commons" in optimized.lower()


class TestWorkflowCoordinator:
    """Test workflow coordinator functionality."""
    
    def test_initialization(self, hardware_config):
        """Test workflow coordinator initialization."""
        coordinator = WorkflowCoordinator(hardware_config)
        
        assert coordinator.hardware_config == hardware_config
        assert len(coordinator.active_workflows) == 0
        assert len(coordinator.completed_workflows) == 0
        assert isinstance(coordinator.prompt_optimizer, PromptOptimizer)
    
    def test_register_pipeline_executor(self, workflow_coordinator):
        """Test pipeline executor registration."""
        mock_executor = Mock()
        
        workflow_coordinator.register_pipeline_executor("image", mock_executor)
        
        assert "image" in workflow_coordinator.pipeline_executors
        assert workflow_coordinator.pipeline_executors["image"] == mock_executor
    
    def test_create_workflow_success(self, workflow_coordinator, sample_workflow_config):
        """Test successful workflow creation."""
        workflow_id = workflow_coordinator.create_workflow(
            name="Test Workflow",
            description="A test workflow",
            steps_config=sample_workflow_config,
            user_id="user1",
            compliance_mode=ComplianceMode.RESEARCH_SAFE
        )
        
        assert workflow_id in workflow_coordinator.active_workflows
        workflow = workflow_coordinator.active_workflows[workflow_id]
        
        assert workflow.name == "Test Workflow"
        assert len(workflow.steps) == 2
        assert workflow.status == WorkflowStatus.PENDING
        assert workflow.context.user_id == "user1"
    
    def test_create_workflow_with_circular_dependencies(self, workflow_coordinator):
        """Test workflow creation with circular dependencies."""
        circular_config = [
            {
                'step_id': 'step_1',
                'type': 'generate',
                'pipeline': 'image',
                'prompt': 'Test',
                'dependencies': ['step_2']
            },
            {
                'step_id': 'step_2', 
                'type': 'generate',
                'pipeline': 'video',
                'prompt': 'Test',
                'dependencies': ['step_1']
            }
        ]
        
        with pytest.raises(Exception, match="Invalid workflow configuration"):
            workflow_coordinator.create_workflow(
                name="Circular Test",
                description="Test circular dependencies",
                steps_config=circular_config,
                user_id="user1",
                compliance_mode=ComplianceMode.RESEARCH_SAFE
            )
    
    def test_validate_workflow_missing_dependency(self, workflow_coordinator):
        """Test workflow validation with missing dependency."""
        invalid_config = [
            {
                'step_id': 'step_1',
                'type': 'generate',
                'pipeline': 'image',
                'prompt': 'Test',
                'dependencies': ['nonexistent_step']
            }
        ]
        
        with pytest.raises(Exception, match="Invalid workflow configuration"):
            workflow_coordinator.create_workflow(
                name="Invalid Test",
                description="Test missing dependency",
                steps_config=invalid_config,
                user_id="user1",
                compliance_mode=ComplianceMode.RESEARCH_SAFE
            )
    
    def test_execution_order_calculation(self, workflow_coordinator):
        """Test execution order calculation with dependencies."""
        complex_config = [
            {
                'step_id': 'step_1',
                'type': 'generate',
                'pipeline': 'image',
                'prompt': 'Base image',
                'dependencies': []
            },
            {
                'step_id': 'step_2',
                'type': 'generate',
                'pipeline': 'image',
                'prompt': 'Another image',
                'dependencies': []
            },
            {
                'step_id': 'step_3',
                'type': 'generate',
                'pipeline': 'video',
                'prompt': 'Combine images',
                'dependencies': ['step_1', 'step_2']
            }
        ]
        
        workflow_id = workflow_coordinator.create_workflow(
            name="Complex Test",
            description="Test execution order",
            steps_config=complex_config,
            user_id="user1",
            compliance_mode=ComplianceMode.RESEARCH_SAFE
        )
        
        workflow = workflow_coordinator.active_workflows[workflow_id]
        execution_order = workflow_coordinator._get_execution_order(workflow.steps)
        
        # step_3 should come after step_1 and step_2
        step_3_index = next(i for i, step in enumerate(execution_order) if step.step_id == 'step_3')
        step_1_index = next(i for i, step in enumerate(execution_order) if step.step_id == 'step_1')
        step_2_index = next(i for i, step in enumerate(execution_order) if step.step_id == 'step_2')
        
        assert step_3_index > step_1_index
        assert step_3_index > step_2_index


class TestWorkflowExecution:
    """Test workflow execution functionality."""
    
    def test_execute_workflow_success(self, workflow_coordinator, sample_workflow_config):
        """Test successful workflow execution."""
        # Register mock executors
        mock_image_executor = Mock()
        mock_video_executor = Mock()
        
        mock_image_result = Mock()
        mock_image_result.success = True
        mock_video_result = Mock()
        mock_video_result.success = True
        
        mock_image_executor.return_value = mock_image_result
        mock_video_executor.return_value = mock_video_result
        
        workflow_coordinator.register_pipeline_executor("image", mock_image_executor)
        workflow_coordinator.register_pipeline_executor("video", mock_video_executor)
        
        # Create and execute workflow
        workflow_id = workflow_coordinator.create_workflow(
            name="Test Execution",
            description="Test workflow execution",
            steps_config=sample_workflow_config,
            user_id="user1",
            compliance_mode=ComplianceMode.RESEARCH_SAFE
        )
        
        result = workflow_coordinator.execute_workflow(workflow_id)
        
        assert result['success'] is True
        assert result['completed_steps'] == 2
        assert result['total_steps'] == 2
        assert workflow_id in workflow_coordinator.completed_workflows
        assert workflow_id not in workflow_coordinator.active_workflows
        
        # Check that executors were called
        mock_image_executor.assert_called_once()
        mock_video_executor.assert_called_once()
    
    def test_execute_workflow_step_failure(self, workflow_coordinator, sample_workflow_config):
        """Test workflow execution with step failure."""
        # Register mock executors - image fails, video should not be called
        mock_image_executor = Mock()
        mock_video_executor = Mock()
        
        mock_image_result = Mock()
        mock_image_result.success = False
        
        mock_image_executor.return_value = mock_image_result
        
        workflow_coordinator.register_pipeline_executor("image", mock_image_executor)
        workflow_coordinator.register_pipeline_executor("video", mock_video_executor)
        
        # Create and execute workflow
        workflow_id = workflow_coordinator.create_workflow(
            name="Test Failure",
            description="Test workflow failure",
            steps_config=sample_workflow_config,
            user_id="user1",
            compliance_mode=ComplianceMode.RESEARCH_SAFE
        )
        
        result = workflow_coordinator.execute_workflow(workflow_id)
        
        assert result['success'] is False
        assert 'error' in result
        
        # Image executor should be called, video should not (due to dependency)
        mock_image_executor.assert_called()
        mock_video_executor.assert_not_called()
    
    def test_execute_workflow_with_retries(self, workflow_coordinator):
        """Test workflow execution with step retries."""
        retry_config = [
            {
                'step_id': 'retry_step',
                'type': 'generate',
                'pipeline': 'image',
                'prompt': 'Test retry',
                'params': {'critical': True},
                'dependencies': [],
                'max_retries': 2
            }
        ]
        
        # Mock executor that fails twice then succeeds
        mock_executor = Mock()
        mock_fail_result = Mock()
        mock_fail_result.success = False
        mock_success_result = Mock()
        mock_success_result.success = True
        
        mock_executor.side_effect = [mock_fail_result, mock_fail_result, mock_success_result]
        
        workflow_coordinator.register_pipeline_executor("image", mock_executor)
        
        workflow_id = workflow_coordinator.create_workflow(
            name="Retry Test",
            description="Test retry mechanism",
            steps_config=retry_config,
            user_id="user1",
            compliance_mode=ComplianceMode.RESEARCH_SAFE
        )
        
        result = workflow_coordinator.execute_workflow(workflow_id)
        
        assert result['success'] is True
        assert mock_executor.call_count == 3  # Initial + 2 retries
    
    def test_execute_workflow_missing_executor(self, workflow_coordinator):
        """Test workflow execution with missing pipeline executor."""
        config = [
            {
                'step_id': 'missing_executor_step',
                'type': 'generate',
                'pipeline': 'nonexistent_pipeline',
                'prompt': 'Test',
                'dependencies': []
            }
        ]
        
        workflow_id = workflow_coordinator.create_workflow(
            name="Missing Executor Test",
            description="Test missing executor",
            steps_config=config,
            user_id="user1",
            compliance_mode=ComplianceMode.RESEARCH_SAFE
        )
        
        result = workflow_coordinator.execute_workflow(workflow_id)
        
        assert result['success'] is False
        assert 'No executor available' in result['error']


class TestWorkflowManagement:
    """Test workflow management functionality."""
    
    def test_get_workflow_status(self, workflow_coordinator, sample_workflow_config):
        """Test workflow status retrieval."""
        workflow_id = workflow_coordinator.create_workflow(
            name="Status Test",
            description="Test status retrieval",
            steps_config=sample_workflow_config,
            user_id="user1",
            compliance_mode=ComplianceMode.RESEARCH_SAFE
        )
        
        status = workflow_coordinator.get_workflow_status(workflow_id)
        
        assert status['workflow_id'] == workflow_id
        assert status['name'] == "Status Test"
        assert status['status'] == WorkflowStatus.PENDING.value
        assert status['progress'] == 0.0
        assert len(status['steps']) == 2
    
    def test_get_workflow_status_not_found(self, workflow_coordinator):
        """Test workflow status for non-existent workflow."""
        status = workflow_coordinator.get_workflow_status("nonexistent")
        
        assert 'error' in status
        assert status['error'] == 'Workflow not found'
    
    def test_pause_resume_workflow(self, workflow_coordinator, sample_workflow_config):
        """Test workflow pause and resume functionality."""
        workflow_id = workflow_coordinator.create_workflow(
            name="Pause Test",
            description="Test pause/resume",
            steps_config=sample_workflow_config,
            user_id="user1",
            compliance_mode=ComplianceMode.RESEARCH_SAFE
        )
        
        # Set workflow to running state
        workflow = workflow_coordinator.active_workflows[workflow_id]
        workflow.status = WorkflowStatus.RUNNING
        
        # Test pause
        assert workflow_coordinator.pause_workflow(workflow_id) is True
        assert workflow.status == WorkflowStatus.PAUSED
        
        # Test resume
        assert workflow_coordinator.resume_workflow(workflow_id) is True
        assert workflow.status == WorkflowStatus.RUNNING
    
    def test_cancel_workflow(self, workflow_coordinator, sample_workflow_config):
        """Test workflow cancellation."""
        workflow_id = workflow_coordinator.create_workflow(
            name="Cancel Test",
            description="Test cancellation",
            steps_config=sample_workflow_config,
            user_id="user1",
            compliance_mode=ComplianceMode.RESEARCH_SAFE
        )
        
        assert workflow_coordinator.cancel_workflow(workflow_id) is True
        
        # Should be moved to completed workflows
        assert workflow_id not in workflow_coordinator.active_workflows
        assert workflow_id in workflow_coordinator.completed_workflows
        
        workflow = workflow_coordinator.completed_workflows[workflow_id]
        assert workflow.status == WorkflowStatus.CANCELLED
    
    def test_cleanup_completed_workflows(self, workflow_coordinator, sample_workflow_config):
        """Test cleanup of old completed workflows."""
        # Create and complete a workflow
        workflow_id = workflow_coordinator.create_workflow(
            name="Cleanup Test",
            description="Test cleanup",
            steps_config=sample_workflow_config,
            user_id="user1",
            compliance_mode=ComplianceMode.RESEARCH_SAFE
        )
        
        # Move to completed
        workflow = workflow_coordinator.active_workflows[workflow_id]
        workflow.status = WorkflowStatus.COMPLETED
        workflow_coordinator.completed_workflows[workflow_id] = workflow
        del workflow_coordinator.active_workflows[workflow_id]
        
        # Modify creation time to be old
        workflow.created_at = datetime.fromtimestamp(time.time() - 48 * 3600)  # 48 hours ago
        
        # Cleanup with 24 hour threshold
        cleaned_count = workflow_coordinator.cleanup_completed_workflows(max_age_hours=24)
        
        assert cleaned_count == 1
        assert workflow_id not in workflow_coordinator.completed_workflows
    
    def test_get_workflow_results(self, workflow_coordinator, sample_workflow_config):
        """Test retrieval of workflow results."""
        workflow_id = workflow_coordinator.create_workflow(
            name="Results Test",
            description="Test results retrieval",
            steps_config=sample_workflow_config,
            user_id="user1",
            compliance_mode=ComplianceMode.RESEARCH_SAFE
        )
        
        # Move to completed with mock results
        workflow = workflow_coordinator.active_workflows[workflow_id]
        workflow.status = WorkflowStatus.COMPLETED
        workflow.context.intermediate_results = {
            'step_1': {'output_type': 'image', 'result': 'mock_result_1'},
            'step_2': {'output_type': 'video', 'result': 'mock_result_2'}
        }
        workflow.total_execution_time = 120.5
        
        workflow_coordinator.completed_workflows[workflow_id] = workflow
        del workflow_coordinator.active_workflows[workflow_id]
        
        results = workflow_coordinator.get_workflow_results(workflow_id)
        
        assert results['workflow_id'] == workflow_id
        assert results['name'] == "Results Test"
        assert len(results['results']) == 2
        assert results['execution_time'] == 120.5


class TestStepExecution:
    """Test individual step execution functionality."""
    
    def test_step_ready_check(self, workflow_coordinator, sample_workflow_config):
        """Test step readiness checking."""
        workflow_id = workflow_coordinator.create_workflow(
            name="Ready Test",
            description="Test step readiness",
            steps_config=sample_workflow_config,
            user_id="user1",
            compliance_mode=ComplianceMode.RESEARCH_SAFE
        )
        
        workflow = workflow_coordinator.active_workflows[workflow_id]
        step_1 = workflow.steps[0]  # No dependencies
        step_2 = workflow.steps[1]  # Depends on step_1
        
        # Step 1 should be ready, step 2 should not
        assert workflow_coordinator._is_step_ready(step_1, workflow) is True
        assert workflow_coordinator._is_step_ready(step_2, workflow) is False
        
        # Complete step 1
        step_1.status = StepStatus.COMPLETED
        
        # Now step 2 should be ready
        assert workflow_coordinator._is_step_ready(step_2, workflow) is True
    
    def test_prompt_optimization_in_execution(self, workflow_coordinator):
        """Test that prompt optimization is applied during step execution."""
        config = [
            {
                'step_id': 'optimize_test',
                'type': 'generate',
                'pipeline': 'image',
                'prompt': 'A simple image',
                'params': {},
                'dependencies': []
            }
        ]
        
        # Mock executor to capture the optimized prompt
        captured_request = None
        
        def mock_executor(request):
            nonlocal captured_request
            captured_request = request
            result = Mock()
            result.success = True
            return result
        
        workflow_coordinator.register_pipeline_executor("image", mock_executor)
        
        workflow_id = workflow_coordinator.create_workflow(
            name="Optimization Test",
            description="Test prompt optimization",
            steps_config=config,
            user_id="user1",
            compliance_mode=ComplianceMode.RESEARCH_SAFE
        )
        
        workflow_coordinator.execute_workflow(workflow_id)
        
        # Check that the prompt was optimized
        assert captured_request is not None
        assert len(captured_request.prompt) > len("A simple image")
        assert "quality" in captured_request.prompt.lower()


if __name__ == "__main__":
    pytest.main([__file__])