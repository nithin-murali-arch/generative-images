"""
Workflow Coordination System for Multi-step Generation.

This module implements workflow orchestration for complex multi-step generation
scenarios, including prompt optimization, state management, and error recovery.
"""

import uuid
import json
import time
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, asdict
from enum import Enum
from datetime import datetime

from .interfaces import (
    GenerationRequest, GenerationResult, OutputType, ComplianceMode,
    HardwareConfig, SystemError
)
from .logging import get_logger

logger = get_logger(__name__)


class WorkflowStatus(Enum):
    """Status of workflow execution."""
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class StepStatus(Enum):
    """Status of individual workflow steps."""
    WAITING = "waiting"
    READY = "ready"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class WorkflowStep:
    """Individual step in a workflow."""
    step_id: str
    step_type: str  # "generate", "optimize", "refine", "combine"
    pipeline: str   # "image", "video", "text", "multimodal"
    prompt: str
    params: Dict[str, Any]
    dependencies: List[str]  # Step IDs this step depends on
    status: StepStatus = StepStatus.WAITING
    result: Optional[GenerationResult] = None
    error_message: Optional[str] = None
    execution_time: float = 0.0
    retry_count: int = 0
    max_retries: int = 3


@dataclass
class WorkflowContext:
    """Context and state for workflow execution."""
    workflow_id: str
    user_id: str
    compliance_mode: ComplianceMode
    hardware_config: HardwareConfig
    global_params: Dict[str, Any]
    intermediate_results: Dict[str, Any]  # Results from completed steps
    shared_state: Dict[str, Any]  # Shared state between steps


@dataclass
class WorkflowDefinition:
    """Complete workflow definition."""
    workflow_id: str
    name: str
    description: str
    steps: List[WorkflowStep]
    context: WorkflowContext
    created_at: datetime
    status: WorkflowStatus = WorkflowStatus.PENDING
    progress: float = 0.0  # 0.0 to 1.0
    total_execution_time: float = 0.0


class PromptOptimizer:
    """Optimizes prompts for downstream models based on workflow context."""
    
    def __init__(self):
        """Initialize prompt optimizer."""
        self.optimization_strategies = {
            'image': self._optimize_image_prompt,
            'video': self._optimize_video_prompt,
            'text': self._optimize_text_prompt,
            'multimodal': self._optimize_multimodal_prompt
        }
    
    def optimize_prompt(self, original_prompt: str, pipeline: str, 
                       context: WorkflowContext, previous_results: Dict[str, Any]) -> str:
        """Optimize prompt for specific pipeline and context."""
        logger.debug(f"Optimizing prompt for {pipeline} pipeline")
        
        strategy = self.optimization_strategies.get(pipeline, self._optimize_generic_prompt)
        optimized = strategy(original_prompt, context, previous_results)
        
        logger.debug(f"Prompt optimized: {len(original_prompt)} -> {len(optimized)} chars")
        return optimized
    
    def _optimize_image_prompt(self, prompt: str, context: WorkflowContext, 
                              previous_results: Dict[str, Any]) -> str:
        """Optimize prompt for image generation."""
        optimized = prompt
        
        # Add style consistency if previous images exist
        if previous_results:
            for result_id, result in previous_results.items():
                if result.get('output_type') == 'image':
                    # Extract style information from previous results
                    if 'style_detected' in result:
                        style = result['style_detected']
                        if style not in optimized.lower():
                            optimized += f", in {style} style"
                    break
        
        # Add quality and technical parameters based on hardware
        if context.hardware_config.vram_size > 8000:
            optimized += ", high quality, detailed, 4k resolution"
        else:
            optimized += ", good quality, detailed"
        
        # Add compliance-aware terms
        if context.compliance_mode == ComplianceMode.OPEN_SOURCE_ONLY:
            optimized += ", open source style, creative commons inspired"
        
        return optimized
    
    def _optimize_video_prompt(self, prompt: str, context: WorkflowContext,
                              previous_results: Dict[str, Any]) -> str:
        """Optimize prompt for video generation."""
        optimized = prompt
        
        # Add temporal consistency cues
        if "motion" not in optimized.lower() and "movement" not in optimized.lower():
            optimized += ", smooth motion, temporal consistency"
        
        # Add duration context
        duration = context.global_params.get('duration', 4)
        optimized += f", {duration} second duration"
        
        # Reference previous image if available
        if previous_results:
            for result_id, result in previous_results.items():
                if result.get('output_type') == 'image':
                    optimized = f"Animate this scene: {optimized}"
                    break
        
        # Hardware-specific optimizations
        if context.hardware_config.vram_size < 6000:
            optimized += ", simple motion, optimized for limited resources"
        
        return optimized
    
    def _optimize_text_prompt(self, prompt: str, context: WorkflowContext,
                             previous_results: Dict[str, Any]) -> str:
        """Optimize prompt for text generation."""
        optimized = prompt
        
        # Add context from previous results
        if previous_results:
            context_info = []
            for result_id, result in previous_results.items():
                if result.get('description'):
                    context_info.append(result['description'])
            
            if context_info:
                context_str = " ".join(context_info[:2])  # Limit context length
                optimized = f"Based on: {context_str}. {optimized}"
        
        # Add compliance context
        if context.compliance_mode == ComplianceMode.RESEARCH_SAFE:
            optimized += " (For research and educational purposes)"
        
        return optimized
    
    def _optimize_multimodal_prompt(self, prompt: str, context: WorkflowContext,
                                   previous_results: Dict[str, Any]) -> str:
        """Optimize prompt for multimodal generation."""
        # Use the most appropriate single-modal optimization
        if "image" in prompt.lower():
            return self._optimize_image_prompt(prompt, context, previous_results)
        elif "video" in prompt.lower():
            return self._optimize_video_prompt(prompt, context, previous_results)
        else:
            return self._optimize_generic_prompt(prompt, context, previous_results)
    
    def _optimize_generic_prompt(self, prompt: str, context: WorkflowContext,
                                previous_results: Dict[str, Any]) -> str:
        """Generic prompt optimization."""
        return prompt  # No specific optimization


class WorkflowCoordinator:
    """
    Coordinates multi-step generation workflows with state management and error recovery.
    
    Handles workflow execution, step dependencies, prompt optimization, and provides
    comprehensive error recovery and retry mechanisms.
    """
    
    def __init__(self, hardware_config: HardwareConfig):
        """Initialize workflow coordinator."""
        self.hardware_config = hardware_config
        self.active_workflows: Dict[str, WorkflowDefinition] = {}
        self.completed_workflows: Dict[str, WorkflowDefinition] = {}
        self.prompt_optimizer = PromptOptimizer()
        
        # Pipeline execution callbacks (to be set by main controller)
        self.pipeline_executors: Dict[str, Callable] = {}
        
        logger.info("Workflow coordinator initialized")
    
    def register_pipeline_executor(self, pipeline: str, executor: Callable) -> None:
        """Register a pipeline executor function."""
        self.pipeline_executors[pipeline] = executor
        logger.info(f"Registered executor for {pipeline} pipeline")
    
    def create_workflow(self, name: str, description: str, steps_config: List[Dict[str, Any]],
                       user_id: str, compliance_mode: ComplianceMode,
                       global_params: Dict[str, Any] = None) -> str:
        """Create a new workflow from configuration."""
        workflow_id = str(uuid.uuid4())
        
        # Create workflow context
        context = WorkflowContext(
            workflow_id=workflow_id,
            user_id=user_id,
            compliance_mode=compliance_mode,
            hardware_config=self.hardware_config,
            global_params=global_params or {},
            intermediate_results={},
            shared_state={}
        )
        
        # Create workflow steps
        steps = []
        for i, step_config in enumerate(steps_config):
            step = WorkflowStep(
                step_id=step_config.get('step_id', f"{workflow_id}_step_{i}"),
                step_type=step_config.get('type', 'generate'),
                pipeline=step_config.get('pipeline', 'image'),
                prompt=step_config.get('prompt', ''),
                params=step_config.get('params', {}),
                dependencies=step_config.get('dependencies', []),
                max_retries=step_config.get('max_retries', 3)
            )
            steps.append(step)
        
        # Create workflow definition
        workflow = WorkflowDefinition(
            workflow_id=workflow_id,
            name=name,
            description=description,
            steps=steps,
            context=context,
            created_at=datetime.now()
        )
        
        # Validate workflow
        if not self._validate_workflow(workflow):
            raise SystemError(f"Invalid workflow configuration: {workflow_id}")
        
        self.active_workflows[workflow_id] = workflow
        logger.info(f"Created workflow {workflow_id} with {len(steps)} steps")
        
        return workflow_id
    
    def _validate_workflow(self, workflow: WorkflowDefinition) -> bool:
        """Validate workflow configuration."""
        try:
            # Check for circular dependencies
            if self._has_circular_dependencies(workflow.steps):
                logger.error("Workflow has circular dependencies")
                return False
            
            # Check that all dependencies exist
            step_ids = {step.step_id for step in workflow.steps}
            for step in workflow.steps:
                for dep in step.dependencies:
                    if dep not in step_ids:
                        logger.error(f"Step {step.step_id} depends on non-existent step {dep}")
                        return False
            
            # Check pipeline availability
            for step in workflow.steps:
                if step.pipeline not in self.pipeline_executors:
                    logger.warning(f"No executor registered for pipeline {step.pipeline}")
            
            return True
            
        except Exception as e:
            logger.error(f"Workflow validation error: {str(e)}")
            return False
    
    def _has_circular_dependencies(self, steps: List[WorkflowStep]) -> bool:
        """Check for circular dependencies using DFS."""
        # Build adjacency list
        graph = {step.step_id: step.dependencies for step in steps}
        
        # DFS to detect cycles
        visited = set()
        rec_stack = set()
        
        def dfs(node):
            if node in rec_stack:
                return True  # Cycle found
            if node in visited:
                return False
            
            visited.add(node)
            rec_stack.add(node)
            
            for neighbor in graph.get(node, []):
                if dfs(neighbor):
                    return True
            
            rec_stack.remove(node)
            return False
        
        for step_id in graph:
            if step_id not in visited:
                if dfs(step_id):
                    return True
        
        return False
    
    def execute_workflow(self, workflow_id: str) -> Dict[str, Any]:
        """Execute a workflow asynchronously."""
        if workflow_id not in self.active_workflows:
            raise SystemError(f"Workflow {workflow_id} not found")
        
        workflow = self.active_workflows[workflow_id]
        workflow.status = WorkflowStatus.RUNNING
        
        logger.info(f"Starting execution of workflow {workflow_id}")
        
        try:
            # Execute workflow steps
            execution_result = self._execute_workflow_steps(workflow)
            
            # Update workflow status
            if execution_result['success']:
                workflow.status = WorkflowStatus.COMPLETED
                workflow.progress = 1.0
                
                # Move to completed workflows
                self.completed_workflows[workflow_id] = workflow
                del self.active_workflows[workflow_id]
            else:
                workflow.status = WorkflowStatus.FAILED
            
            return execution_result
            
        except Exception as e:
            logger.error(f"Workflow execution error: {str(e)}")
            workflow.status = WorkflowStatus.FAILED
            return {
                'success': False,
                'workflow_id': workflow_id,
                'error': str(e),
                'completed_steps': 0,
                'total_steps': len(workflow.steps)
            }
    
    def _execute_workflow_steps(self, workflow: WorkflowDefinition) -> Dict[str, Any]:
        """Execute workflow steps in dependency order."""
        start_time = time.time()
        completed_steps = 0
        total_steps = len(workflow.steps)
        
        # Create execution order based on dependencies
        execution_order = self._get_execution_order(workflow.steps)
        
        for step in execution_order:
            try:
                # Check if step is ready to execute
                if not self._is_step_ready(step, workflow):
                    step.status = StepStatus.SKIPPED
                    logger.warning(f"Skipping step {step.step_id} - dependencies not met")
                    continue
                
                # Execute step with retry logic
                step_success = False
                while step.retry_count <= step.max_retries and not step_success:
                    step.status = StepStatus.RUNNING
                    step_result = self._execute_step(step, workflow)
                
                    if step_result['success']:
                        step.status = StepStatus.COMPLETED
                        step.result = step_result.get('result')
                        
                        # Store intermediate result
                        workflow.context.intermediate_results[step.step_id] = {
                            'output_type': step.pipeline,
                            'result': step_result.get('result'),
                            'execution_time': step.execution_time,
                            'step_type': step.step_type
                        }
                        
                        completed_steps += 1
                        workflow.progress = completed_steps / total_steps
                        step_success = True
                        
                        logger.info(f"Step {step.step_id} completed successfully")
                    else:
                        step.status = StepStatus.FAILED
                        step.error_message = step_result.get('error', 'Unknown error')
                        
                        if step.retry_count < step.max_retries:
                            step.retry_count += 1
                            logger.warning(f"Step {step.step_id} failed, retrying ({step.retry_count}/{step.max_retries})")
                        else:
                            logger.error(f"Step {step.step_id} failed after {step.max_retries} retries")
                            # Decide whether to continue or fail entire workflow
                            if step.params.get('critical', True):
                                return {
                                    'success': False,
                                    'workflow_id': workflow.workflow_id,
                                    'error': f"Critical step {step.step_id} failed: {step.error_message}",
                                    'completed_steps': completed_steps,
                                    'total_steps': total_steps
                                }
                            break  # Exit retry loop for non-critical steps
                
            except Exception as e:
                logger.error(f"Error executing step {step.step_id}: {str(e)}")
                step.status = StepStatus.FAILED
                step.error_message = str(e)
                
                if step.params.get('critical', True):
                    return {
                        'success': False,
                        'workflow_id': workflow.workflow_id,
                        'error': f"Step execution error: {str(e)}",
                        'completed_steps': completed_steps,
                        'total_steps': total_steps
                    }
        
        workflow.total_execution_time = time.time() - start_time
        
        return {
            'success': True,
            'workflow_id': workflow.workflow_id,
            'completed_steps': completed_steps,
            'total_steps': total_steps,
            'execution_time': workflow.total_execution_time,
            'results': workflow.context.intermediate_results
        }
    
    def _get_execution_order(self, steps: List[WorkflowStep]) -> List[WorkflowStep]:
        """Get execution order based on dependencies (topological sort)."""
        # Build dependency graph
        in_degree = {step.step_id: 0 for step in steps}
        graph = {step.step_id: [] for step in steps}
        step_map = {step.step_id: step for step in steps}
        
        for step in steps:
            for dep in step.dependencies:
                graph[dep].append(step.step_id)
                in_degree[step.step_id] += 1
        
        # Topological sort using Kahn's algorithm
        queue = [step_id for step_id, degree in in_degree.items() if degree == 0]
        execution_order = []
        
        while queue:
            current = queue.pop(0)
            execution_order.append(step_map[current])
            
            for neighbor in graph[current]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        
        return execution_order
    
    def _is_step_ready(self, step: WorkflowStep, workflow: WorkflowDefinition) -> bool:
        """Check if step is ready to execute (all dependencies completed)."""
        for dep_id in step.dependencies:
            dep_step = next((s for s in workflow.steps if s.step_id == dep_id), None)
            if not dep_step or dep_step.status != StepStatus.COMPLETED:
                return False
        return True
    
    def _execute_step(self, step: WorkflowStep, workflow: WorkflowDefinition) -> Dict[str, Any]:
        """Execute a single workflow step."""
        logger.info(f"Executing step {step.step_id} ({step.step_type} -> {step.pipeline})")
        
        start_time = time.time()
        
        try:
            # Optimize prompt based on context and previous results
            optimized_prompt = self.prompt_optimizer.optimize_prompt(
                step.prompt, step.pipeline, workflow.context,
                workflow.context.intermediate_results
            )
            
            # Get pipeline executor
            executor = self.pipeline_executors.get(step.pipeline)
            if not executor:
                return {
                    'success': False,
                    'error': f"No executor available for pipeline {step.pipeline}"
                }
            
            # Create generation request
            request = GenerationRequest(
                prompt=optimized_prompt,
                output_type=OutputType(step.pipeline) if step.pipeline in ['image', 'video', 'text'] else OutputType.IMAGE,
                style_config=step.params.get('style_config'),
                compliance_mode=workflow.context.compliance_mode,
                hardware_constraints=workflow.context.hardware_config,
                context=None,  # Will be set by executor
                additional_params=step.params
            )
            
            # Execute pipeline
            result = executor(request)
            
            step.execution_time = time.time() - start_time
            
            return {
                'success': result.success if hasattr(result, 'success') else True,
                'result': result,
                'execution_time': step.execution_time
            }
            
        except Exception as e:
            step.execution_time = time.time() - start_time
            logger.error(f"Step execution failed: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'execution_time': step.execution_time
            }
    
    def get_workflow_status(self, workflow_id: str) -> Dict[str, Any]:
        """Get current status of a workflow."""
        workflow = self.active_workflows.get(workflow_id) or self.completed_workflows.get(workflow_id)
        
        if not workflow:
            return {'error': 'Workflow not found'}
        
        step_statuses = []
        for step in workflow.steps:
            step_statuses.append({
                'step_id': step.step_id,
                'step_type': step.step_type,
                'pipeline': step.pipeline,
                'status': step.status.value,
                'execution_time': step.execution_time,
                'retry_count': step.retry_count,
                'error_message': step.error_message
            })
        
        return {
            'workflow_id': workflow_id,
            'name': workflow.name,
            'status': workflow.status.value,
            'progress': workflow.progress,
            'total_execution_time': workflow.total_execution_time,
            'steps': step_statuses,
            'intermediate_results': len(workflow.context.intermediate_results),
            'created_at': workflow.created_at.isoformat()
        }
    
    def pause_workflow(self, workflow_id: str) -> bool:
        """Pause a running workflow."""
        if workflow_id in self.active_workflows:
            workflow = self.active_workflows[workflow_id]
            if workflow.status == WorkflowStatus.RUNNING:
                workflow.status = WorkflowStatus.PAUSED
                logger.info(f"Paused workflow {workflow_id}")
                return True
        return False
    
    def resume_workflow(self, workflow_id: str) -> bool:
        """Resume a paused workflow."""
        if workflow_id in self.active_workflows:
            workflow = self.active_workflows[workflow_id]
            if workflow.status == WorkflowStatus.PAUSED:
                workflow.status = WorkflowStatus.RUNNING
                logger.info(f"Resumed workflow {workflow_id}")
                return True
        return False
    
    def cancel_workflow(self, workflow_id: str) -> bool:
        """Cancel a workflow."""
        if workflow_id in self.active_workflows:
            workflow = self.active_workflows[workflow_id]
            workflow.status = WorkflowStatus.CANCELLED
            
            # Move to completed workflows
            self.completed_workflows[workflow_id] = workflow
            del self.active_workflows[workflow_id]
            
            logger.info(f"Cancelled workflow {workflow_id}")
            return True
        return False
    
    def cleanup_completed_workflows(self, max_age_hours: int = 24) -> int:
        """Clean up old completed workflows."""
        cutoff_time = datetime.now().timestamp() - (max_age_hours * 3600)
        
        to_remove = []
        for workflow_id, workflow in self.completed_workflows.items():
            if workflow.created_at.timestamp() < cutoff_time:
                to_remove.append(workflow_id)
        
        for workflow_id in to_remove:
            del self.completed_workflows[workflow_id]
        
        logger.info(f"Cleaned up {len(to_remove)} old workflows")
        return len(to_remove)
    
    def get_workflow_results(self, workflow_id: str) -> Dict[str, Any]:
        """Get final results from a completed workflow."""
        workflow = self.completed_workflows.get(workflow_id)
        
        if not workflow:
            return {'error': 'Completed workflow not found'}
        
        if workflow.status != WorkflowStatus.COMPLETED:
            return {'error': f'Workflow not completed (status: {workflow.status.value})'}
        
        return {
            'workflow_id': workflow_id,
            'name': workflow.name,
            'results': workflow.context.intermediate_results,
            'execution_time': workflow.total_execution_time,
            'completed_at': workflow.created_at.isoformat()
        }