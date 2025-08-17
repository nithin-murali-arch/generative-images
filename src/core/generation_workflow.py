"""
Complete Generation Workflow Implementation

This module implements the complete end-to-end generation workflow that connects
text prompts through LLM processing to image and video generation, with proper
error handling and user feedback.
"""

import logging
import time
import uuid
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
from dataclasses import dataclass, asdict
from enum import Enum

from .interfaces import (
    GenerationRequest, GenerationResult, OutputType, ComplianceMode,
    HardwareConfig, ConversationContext, StyleConfig, SystemError
)

logger = logging.getLogger(__name__)


@dataclass
class WorkflowStep:
    """Individual step in the generation workflow."""
    step_id: str
    step_type: str  # "parse", "route", "generate", "post_process"
    status: str     # "pending", "running", "completed", "failed"
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    result: Optional[Any] = None
    error: Optional[str] = None


@dataclass
class WorkflowProgress:
    """Progress information for the workflow."""
    workflow_id: str
    total_steps: int
    completed_steps: int
    current_step: Optional[WorkflowStep]
    estimated_remaining_time: float
    status: str  # "running", "completed", "failed"


class GenerationWorkflow:
    """
    Complete generation workflow that orchestrates the entire process from
    text prompt to final output with progress tracking and error handling.
    """
    
    def __init__(self, system_integration):
        """
        Initialize the generation workflow.
        
        Args:
            system_integration: SystemIntegration instance with all components
        """
        self.system_integration = system_integration
        self.active_workflows: Dict[str, Dict[str, Any]] = {}
        
        logger.info("GenerationWorkflow initialized")
    
    def execute_complete_workflow(self, 
                                prompt: str, 
                                conversation_id: str = None,
                                compliance_mode: ComplianceMode = ComplianceMode.RESEARCH_SAFE,
                                additional_params: Optional[Dict[str, Any]] = None,
                                progress_callback: Optional[callable] = None) -> GenerationResult:
        """
        Execute the complete generation workflow from prompt to output.
        
        This implements the full flow:
        1. Text prompt → LLM processing (intent classification, prompt optimization)
        2. LLM processing → Pipeline routing (image/video/multimodal)
        3. Pipeline execution → Content generation
        4. Post-processing → Final output with metadata
        
        Args:
            prompt: User's text prompt
            conversation_id: Optional conversation ID for context
            compliance_mode: Copyright compliance mode
            additional_params: Additional generation parameters
            progress_callback: Optional callback for progress updates
            
        Returns:
            GenerationResult: Complete generation result with metadata
        """
        workflow_id = str(uuid.uuid4())
        start_time = time.time()
        
        if not conversation_id:
            conversation_id = f"conv_{int(time.time())}"
        
        try:
            logger.info(f"Starting complete generation workflow {workflow_id}")
            logger.info(f"Prompt: '{prompt[:100]}...'")
            
            # Initialize workflow tracking
            workflow_steps = self._create_workflow_steps()
            self.active_workflows[workflow_id] = {
                'steps': workflow_steps,
                'start_time': start_time,
                'prompt': prompt,
                'conversation_id': conversation_id,
                'compliance_mode': compliance_mode
            }
            
            # Step 1: Parse and classify request
            step_result = self._execute_step_with_progress(
                workflow_id, "parse_request", 
                lambda: self._parse_request_step(prompt, conversation_id, compliance_mode, additional_params),
                progress_callback
            )
            
            if not step_result['success']:
                return self._create_error_result(workflow_id, step_result['error'], start_time)
            
            request = step_result['result']
            
            # Step 2: Route to appropriate pipeline
            step_result = self._execute_step_with_progress(
                workflow_id, "route_request",
                lambda: self._route_request_step(request),
                progress_callback
            )
            
            if not step_result['success']:
                return self._create_error_result(workflow_id, step_result['error'], start_time)
            
            pipeline_name = step_result['result']
            
            # Step 3: Execute generation
            step_result = self._execute_step_with_progress(
                workflow_id, "generate_content",
                lambda: self._generate_content_step(request, pipeline_name),
                progress_callback
            )
            
            if not step_result['success']:
                return self._create_error_result(workflow_id, step_result['error'], start_time)
            
            generation_result = step_result['result']
            
            # Step 4: Post-process and finalize
            step_result = self._execute_step_with_progress(
                workflow_id, "post_process",
                lambda: self._post_process_step(generation_result, request, workflow_id),
                progress_callback
            )
            
            if not step_result['success']:
                return self._create_error_result(workflow_id, step_result['error'], start_time)
            
            final_result = step_result['result']
            
            # Mark workflow as completed
            self._complete_workflow(workflow_id, final_result)
            
            total_time = time.time() - start_time
            logger.info(f"Workflow {workflow_id} completed successfully in {total_time:.2f}s")
            
            return final_result
            
        except Exception as e:
            logger.error(f"Workflow {workflow_id} failed: {e}")
            return self._create_error_result(workflow_id, str(e), start_time)
        finally:
            # Clean up workflow tracking
            if workflow_id in self.active_workflows:
                self.active_workflows[workflow_id]['end_time'] = time.time()
    
    def _create_workflow_steps(self) -> List[WorkflowStep]:
        """Create the standard workflow steps."""
        return [
            WorkflowStep("parse_request", "parse", "pending"),
            WorkflowStep("route_request", "route", "pending"),
            WorkflowStep("generate_content", "generate", "pending"),
            WorkflowStep("post_process", "post_process", "pending")
        ]
    
    def _execute_step_with_progress(self, 
                                  workflow_id: str, 
                                  step_id: str, 
                                  step_function: callable,
                                  progress_callback: Optional[callable] = None) -> Dict[str, Any]:
        """Execute a workflow step with progress tracking."""
        workflow = self.active_workflows[workflow_id]
        steps = workflow['steps']
        
        # Find the step
        step = next((s for s in steps if s.step_id == step_id), None)
        if not step:
            return {'success': False, 'error': f"Step {step_id} not found"}
        
        try:
            # Update step status
            step.status = "running"
            step.start_time = time.time()
            
            # Send progress update
            if progress_callback:
                progress = self._calculate_progress(workflow_id)
                progress_callback(progress)
            
            # Execute the step
            result = step_function()
            
            # Update step completion
            step.status = "completed"
            step.end_time = time.time()
            step.result = result
            
            logger.debug(f"Step {step_id} completed in {step.end_time - step.start_time:.2f}s")
            
            return {'success': True, 'result': result}
            
        except Exception as e:
            step.status = "failed"
            step.end_time = time.time()
            step.error = str(e)
            
            logger.error(f"Step {step_id} failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def _parse_request_step(self, 
                          prompt: str, 
                          conversation_id: str, 
                          compliance_mode: ComplianceMode,
                          additional_params: Optional[Dict[str, Any]]) -> GenerationRequest:
        """Execute the request parsing step."""
        logger.info("Executing request parsing step")
        
        # Get or create conversation context
        context = self.system_integration.llm_controller.manage_context(conversation_id)
        context.current_mode = compliance_mode
        
        # Parse the request using LLM controller
        request = self.system_integration.llm_controller.parse_request(prompt, context)
        
        # Apply additional parameters
        if additional_params:
            if not request.additional_params:
                request.additional_params = {}
            request.additional_params.update(additional_params)
        
        logger.info(f"Request parsed: {request.output_type.value} generation")
        return request
    
    def _route_request_step(self, request: GenerationRequest) -> str:
        """Execute the request routing step."""
        logger.info("Executing request routing step")
        
        # Route request to appropriate pipeline
        pipeline_name = self.system_integration.llm_controller.route_request(request)
        
        logger.info(f"Request routed to {pipeline_name} pipeline")
        return pipeline_name
    
    def _generate_content_step(self, request: GenerationRequest, pipeline_name: str) -> GenerationResult:
        """Execute the content generation step."""
        logger.info(f"Executing content generation step using {pipeline_name} pipeline")
        
        # Execute generation using system integration
        result = self.system_integration._execute_generation(request, pipeline_name)
        
        if result.success:
            logger.info(f"Content generated successfully: {result.output_path}")
        else:
            logger.error(f"Content generation failed: {result.error_message}")
        
        return result
    
    def _post_process_step(self, 
                         generation_result: GenerationResult, 
                         request: GenerationRequest,
                         workflow_id: str) -> GenerationResult:
        """Execute the post-processing step."""
        logger.info("Executing post-processing step")
        
        # Add workflow metadata to the result
        if not generation_result.metadata:
            generation_result.metadata = {}
        
        generation_result.metadata.update({
            'workflow_id': workflow_id,
            'prompt': request.prompt,
            'output_type': request.output_type.value,
            'compliance_mode': request.compliance_mode.value,
            'hardware_config': asdict(request.hardware_constraints) if request.hardware_constraints else None,
            'processing_timestamp': time.time()
        })
        
        # Track experiment if successful
        if generation_result.success and self.system_integration.experiment_tracker:
            try:
                self.system_integration._track_experiment(workflow_id, request, generation_result)
            except Exception as e:
                logger.warning(f"Failed to track experiment: {e}")
        
        logger.info("Post-processing completed")
        return generation_result
    
    def _calculate_progress(self, workflow_id: str) -> WorkflowProgress:
        """Calculate current workflow progress."""
        workflow = self.active_workflows[workflow_id]
        steps = workflow['steps']
        
        completed_steps = len([s for s in steps if s.status == "completed"])
        current_step = next((s for s in steps if s.status == "running"), None)
        
        # Estimate remaining time based on completed steps
        if completed_steps > 0:
            elapsed_time = time.time() - workflow['start_time']
            avg_step_time = elapsed_time / completed_steps
            remaining_steps = len(steps) - completed_steps
            estimated_remaining_time = avg_step_time * remaining_steps
        else:
            estimated_remaining_time = 60.0  # Default estimate
        
        # Determine overall status
        if all(s.status == "completed" for s in steps):
            status = "completed"
        elif any(s.status == "failed" for s in steps):
            status = "failed"
        else:
            status = "running"
        
        return WorkflowProgress(
            workflow_id=workflow_id,
            total_steps=len(steps),
            completed_steps=completed_steps,
            current_step=current_step,
            estimated_remaining_time=estimated_remaining_time,
            status=status
        )
    
    def _complete_workflow(self, workflow_id: str, result: GenerationResult) -> None:
        """Mark workflow as completed."""
        if workflow_id in self.active_workflows:
            workflow = self.active_workflows[workflow_id]
            workflow['status'] = 'completed' if result.success else 'failed'
            workflow['result'] = result
            workflow['end_time'] = time.time()
    
    def _create_error_result(self, workflow_id: str, error_message: str, start_time: float) -> GenerationResult:
        """Create an error result for failed workflow."""
        generation_time = time.time() - start_time
        
        # Mark workflow as failed
        if workflow_id in self.active_workflows:
            self.active_workflows[workflow_id]['status'] = 'failed'
            self.active_workflows[workflow_id]['error'] = error_message
            self.active_workflows[workflow_id]['end_time'] = time.time()
        
        return GenerationResult(
            success=False,
            output_path=None,
            generation_time=generation_time,
            model_used="unknown",
            error_message=error_message,
            metadata={'workflow_id': workflow_id}
        )
    
    def get_workflow_status(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific workflow."""
        if workflow_id not in self.active_workflows:
            return None
        
        workflow = self.active_workflows[workflow_id]
        progress = self._calculate_progress(workflow_id)
        
        return {
            'workflow_id': workflow_id,
            'prompt': workflow['prompt'],
            'conversation_id': workflow['conversation_id'],
            'compliance_mode': workflow['compliance_mode'].value if hasattr(workflow['compliance_mode'], 'value') else str(workflow['compliance_mode']),
            'start_time': workflow['start_time'],
            'end_time': workflow.get('end_time'),
            'status': workflow.get('status', 'running'),
            'progress': asdict(progress),
            'steps': [asdict(step) for step in workflow['steps']],
            'error': workflow.get('error')
        }
    
    def get_active_workflows(self) -> List[Dict[str, Any]]:
        """Get all active workflows."""
        active = []
        for workflow_id in self.active_workflows:
            status = self.get_workflow_status(workflow_id)
            if status and status['status'] == 'running':
                active.append(status)
        return active
    
    def cleanup_completed_workflows(self, max_age_hours: int = 24) -> None:
        """Clean up old completed workflows."""
        current_time = time.time()
        max_age_seconds = max_age_hours * 3600
        
        workflows_to_remove = []
        for workflow_id, workflow in self.active_workflows.items():
            end_time = workflow.get('end_time')
            if end_time and (current_time - end_time) > max_age_seconds:
                workflows_to_remove.append(workflow_id)
        
        for workflow_id in workflows_to_remove:
            del self.active_workflows[workflow_id]
        
        if workflows_to_remove:
            logger.info(f"Cleaned up {len(workflows_to_remove)} old workflows")


class WorkflowManager:
    """
    Manager class for handling multiple generation workflows with queue management
    and resource allocation.
    """
    
    def __init__(self, system_integration, max_concurrent_workflows: int = 3):
        """
        Initialize the workflow manager.
        
        Args:
            system_integration: SystemIntegration instance
            max_concurrent_workflows: Maximum number of concurrent workflows
        """
        self.system_integration = system_integration
        self.max_concurrent_workflows = max_concurrent_workflows
        self.workflow_engine = GenerationWorkflow(system_integration)
        self.workflow_queue: List[Dict[str, Any]] = []
        self.processing_workflows: Dict[str, Dict[str, Any]] = {}
        
        logger.info(f"WorkflowManager initialized with max {max_concurrent_workflows} concurrent workflows")
    
    def submit_generation_request(self, 
                                prompt: str,
                                conversation_id: str = None,
                                compliance_mode: ComplianceMode = ComplianceMode.RESEARCH_SAFE,
                                additional_params: Optional[Dict[str, Any]] = None,
                                priority: int = 0,
                                progress_callback: Optional[callable] = None) -> str:
        """
        Submit a generation request to the workflow queue.
        
        Args:
            prompt: User's text prompt
            conversation_id: Optional conversation ID
            compliance_mode: Copyright compliance mode
            additional_params: Additional generation parameters
            priority: Request priority (higher = more priority)
            progress_callback: Optional progress callback
            
        Returns:
            str: Workflow ID for tracking
        """
        workflow_id = str(uuid.uuid4())
        
        request_data = {
            'workflow_id': workflow_id,
            'prompt': prompt,
            'conversation_id': conversation_id,
            'compliance_mode': compliance_mode,
            'additional_params': additional_params,
            'priority': priority,
            'progress_callback': progress_callback,
            'submitted_time': time.time(),
            'status': 'queued'
        }
        
        # Add to queue (sorted by priority)
        self.workflow_queue.append(request_data)
        self.workflow_queue.sort(key=lambda x: x['priority'], reverse=True)
        
        logger.info(f"Generation request {workflow_id} submitted to queue (priority: {priority})")
        
        # Try to process immediately if capacity available
        self._process_queue()
        
        return workflow_id
    
    def _process_queue(self) -> None:
        """Process queued workflows if capacity is available."""
        while (len(self.processing_workflows) < self.max_concurrent_workflows and 
               self.workflow_queue):
            
            # Get next request from queue
            request_data = self.workflow_queue.pop(0)
            workflow_id = request_data['workflow_id']
            
            # Move to processing
            self.processing_workflows[workflow_id] = request_data
            request_data['status'] = 'processing'
            request_data['start_time'] = time.time()
            
            # Start workflow execution (this would typically be async in a real implementation)
            try:
                result = self.workflow_engine.execute_complete_workflow(
                    prompt=request_data['prompt'],
                    conversation_id=request_data['conversation_id'],
                    compliance_mode=request_data['compliance_mode'],
                    additional_params=request_data['additional_params'],
                    progress_callback=request_data['progress_callback']
                )
                
                # Mark as completed
                request_data['status'] = 'completed'
                request_data['result'] = result
                request_data['end_time'] = time.time()
                
                logger.info(f"Workflow {workflow_id} completed")
                
            except Exception as e:
                # Mark as failed
                request_data['status'] = 'failed'
                request_data['error'] = str(e)
                request_data['end_time'] = time.time()
                
                logger.error(f"Workflow {workflow_id} failed: {e}")
            
            # Remove from processing (in a real async implementation, this would be done in a callback)
            if workflow_id in self.processing_workflows:
                del self.processing_workflows[workflow_id]
    
    def get_workflow_status(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a workflow by ID."""
        # Check processing workflows
        if workflow_id in self.processing_workflows:
            return self.processing_workflows[workflow_id]
        
        # Check queued workflows
        for request in self.workflow_queue:
            if request['workflow_id'] == workflow_id:
                return request
        
        # Check workflow engine
        return self.workflow_engine.get_workflow_status(workflow_id)
    
    def cancel_workflow(self, workflow_id: str) -> bool:
        """Cancel a queued or processing workflow."""
        # Remove from queue if present
        for i, request in enumerate(self.workflow_queue):
            if request['workflow_id'] == workflow_id:
                del self.workflow_queue[i]
                logger.info(f"Workflow {workflow_id} cancelled (was queued)")
                return True
        
        # For processing workflows, we would need to implement cancellation
        # This is a simplified implementation
        if workflow_id in self.processing_workflows:
            logger.warning(f"Cannot cancel processing workflow {workflow_id} - not implemented")
            return False
        
        return False
    
    def get_queue_status(self) -> Dict[str, Any]:
        """Get current queue status."""
        return {
            'queued_workflows': len(self.workflow_queue),
            'processing_workflows': len(self.processing_workflows),
            'max_concurrent': self.max_concurrent_workflows,
            'queue_details': [
                {
                    'workflow_id': req['workflow_id'],
                    'prompt': req['prompt'][:50] + '...' if len(req['prompt']) > 50 else req['prompt'],
                    'priority': req['priority'],
                    'submitted_time': req['submitted_time']
                }
                for req in self.workflow_queue
            ]
        }