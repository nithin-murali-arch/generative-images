"""
LLM Controller Foundation - Central intelligence for request routing and workflow coordination.

This module implements the core LLM controller that serves as the central hub for
parsing user requests, classifying intent, and coordinating between specialized
generation pipelines.
"""

import re
import json
import uuid
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime

from .interfaces import (
    ILLMController, GenerationRequest, ConversationContext, OutputType,
    ComplianceMode, StyleConfig, HardwareConfig, SystemError
)
from .logging import get_logger

logger = get_logger(__name__)


@dataclass
class IntentClassification:
    """Result of intent classification."""
    output_type: OutputType
    confidence: float
    reasoning: str
    extracted_params: Dict[str, Any]


@dataclass
class WorkflowStep:
    """Individual step in a multi-step workflow."""
    step_id: str
    step_type: str  # "generate", "optimize", "refine"
    pipeline: str   # "image", "video", "text"
    prompt: str
    params: Dict[str, Any]
    dependencies: List[str]  # IDs of steps this depends on


class LLMController(ILLMController):
    """
    Central LLM controller for request parsing, routing, and workflow coordination.
    
    This controller serves as the intelligent hub that:
    - Parses natural language requests into structured generation requests
    - Classifies intent to determine appropriate output type
    - Manages conversation context and memory
    - Coordinates multi-step generation workflows
    """
    
    def __init__(self, hardware_config: HardwareConfig):
        """Initialize the LLM controller with hardware configuration."""
        self.hardware_config = hardware_config
        self.conversations: Dict[str, ConversationContext] = {}
        self.active_workflows: Dict[str, List[WorkflowStep]] = {}
        
        # Intent classification patterns
        self._init_classification_patterns()
        
        logger.info(f"LLM Controller initialized for {hardware_config.gpu_model}")
    
    def _init_classification_patterns(self) -> None:
        """Initialize patterns for intent classification."""
        self.image_patterns = [
            r'\b(image|picture|photo|drawing|artwork|illustration|painting)\b',
            r'\b(generate|create|make|draw|paint|design)\s+.*\b(image|picture|photo)\b',
            r'\bshow me\b.*\b(image|picture|visual)\b',
            r'\b(style|artistic|visual|aesthetic)\b',
            r'\b(portrait|landscape|abstract|realistic)\b'
        ]
        
        self.video_patterns = [
            r'\b(video|animation|movie|clip|sequence|motion)\b',
            r'\b(animate|moving|temporal|time|duration)\b',
            r'\b(generate|create|make)\s+.*\b(video|animation)\b',
            r'\b(frames|fps|seconds|minutes)\b',
            r'\b(motion|movement|action|dynamic)\b'
        ]
        
        self.multimodal_patterns = [
            r'\b(both|and|also|then|next|after)\b.*\b(image|video)\b',
            r'\b(first.*then|start.*with.*then)\b',
            r'\b(sequence|series|workflow|pipeline)\b',
            r'\b(multiple|several|various)\s+.*\b(types|formats|outputs)\b'
        ]
        
        # Style and parameter extraction patterns
        self.style_patterns = {
            'style': r'\b(?:in\s+)?(\w+)\s+style\b|\bstyle:\s*([^,\n]+)',
            'lora': r'\b(lora|adapter|fine-tuned?):\s*([^,\n]+)',
            'resolution': r'\b(\d+x\d+|\d+p)\b',
            'duration': r'\b(\d+)\s*(second|sec|minute|min)s?\b',
            'quality': r'\b(high|low|medium|best|fast)\s*(?:quality|resolution|mode)?\b'
        }
    
    def parse_request(self, prompt: str, context: ConversationContext) -> GenerationRequest:
        """
        Parse user input and create a structured generation request.
        
        Args:
            prompt: Natural language prompt from user
            context: Current conversation context
            
        Returns:
            GenerationRequest: Structured request for generation pipeline
        """
        logger.info(f"Parsing request: {prompt[:100]}...")
        
        try:
            # Classify intent and extract parameters
            intent = self._classify_intent(prompt)
            
            # Extract style and generation parameters
            style_config = self._extract_style_config(prompt)
            
            # Create generation request
            request = GenerationRequest(
                prompt=prompt,
                output_type=intent.output_type,
                style_config=style_config,
                compliance_mode=context.current_mode,
                hardware_constraints=self.hardware_config,
                context=context,
                additional_params=intent.extracted_params
            )
            
            # Update conversation history
            self._update_conversation_history(context, prompt, intent)
            
            logger.info(f"Request parsed successfully: {intent.output_type.value} generation")
            return request
            
        except Exception as e:
            logger.error(f"Error parsing request: {str(e)}")
            raise SystemError(f"Failed to parse request: {str(e)}")
    
    def _classify_intent(self, prompt: str) -> IntentClassification:
        """
        Classify user intent to determine output type.
        
        Args:
            prompt: User's natural language prompt
            
        Returns:
            IntentClassification: Classification result with confidence
        """
        prompt_lower = prompt.lower()
        
        # Count pattern matches for each type
        image_score = sum(1 for pattern in self.image_patterns 
                         if re.search(pattern, prompt_lower, re.IGNORECASE))
        video_score = sum(1 for pattern in self.video_patterns 
                         if re.search(pattern, prompt_lower, re.IGNORECASE))
        multimodal_score = sum(1 for pattern in self.multimodal_patterns 
                              if re.search(pattern, prompt_lower, re.IGNORECASE))
        
        # Extract additional parameters
        extracted_params = self._extract_parameters(prompt)
        
        # Determine output type based on scores
        if multimodal_score > 0 or (image_score > 0 and video_score > 0):
            output_type = OutputType.MULTIMODAL
            confidence = min(0.9, (multimodal_score + image_score + video_score) / 10)
            reasoning = "Detected multimodal request with both image and video elements"
        elif video_score > image_score:
            output_type = OutputType.VIDEO
            confidence = min(0.9, video_score / 5)
            reasoning = f"Video patterns detected (score: {video_score})"
        elif image_score > 0:
            output_type = OutputType.IMAGE
            confidence = min(0.9, image_score / 5)
            reasoning = f"Image patterns detected (score: {image_score})"
        else:
            # Default to image for ambiguous cases
            output_type = OutputType.IMAGE
            confidence = 0.5
            reasoning = "No clear patterns detected, defaulting to image generation"
        
        return IntentClassification(
            output_type=output_type,
            confidence=confidence,
            reasoning=reasoning,
            extracted_params=extracted_params
        )
    
    def _extract_parameters(self, prompt: str) -> Dict[str, Any]:
        """Extract generation parameters from prompt."""
        params = {}
        
        for param_name, pattern in self.style_patterns.items():
            match = re.search(pattern, prompt, re.IGNORECASE)
            if match:
                # For style pattern, try both groups
                if param_name == 'style':
                    params[param_name] = match.group(1) or match.group(2) if match.lastindex >= 2 else match.group(1)
                else:
                    params[param_name] = match.group(2) if match.lastindex >= 2 else match.group(1)
        
        return params
    
    def _extract_style_config(self, prompt: str) -> StyleConfig:
        """Extract style configuration from prompt."""
        params = self._extract_parameters(prompt)
        
        return StyleConfig(
            style_name=params.get('style'),
            lora_path=None,  # Will be resolved later by pipeline
            controlnet_config=None,
            generation_params={
                'quality': params.get('quality', 'medium'),
                'resolution': params.get('resolution', '512x512'),
                'duration': params.get('duration', '4')
            }
        )
    
    def route_request(self, request: GenerationRequest) -> str:
        """
        Determine which pipeline should handle the request.
        
        Args:
            request: Structured generation request
            
        Returns:
            str: Pipeline identifier ("image", "video", "multimodal")
        """
        logger.info(f"Routing request for {request.output_type.value} generation")
        
        # Route based on output type and hardware constraints
        if request.output_type == OutputType.IMAGE:
            return "image"
        elif request.output_type == OutputType.VIDEO:
            # Check if hardware can handle video generation
            if request.hardware_constraints.vram_size < 6000:  # Less than 6GB VRAM
                logger.warning("Limited VRAM detected, may need CPU offloading for video")
            return "video"
        elif request.output_type == OutputType.MULTIMODAL:
            return "multimodal"
        else:
            # Default fallback
            logger.warning(f"Unknown output type {request.output_type}, defaulting to image")
            return "image"
    
    def coordinate_workflow(self, steps: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Coordinate multi-step generation workflows.
        
        Args:
            steps: List of workflow step definitions
            
        Returns:
            Dict containing workflow execution results
        """
        workflow_id = str(uuid.uuid4())
        logger.info(f"Starting workflow coordination: {workflow_id}")
        
        try:
            # Convert steps to WorkflowStep objects
            workflow_steps = []
            for i, step_data in enumerate(steps):
                step = WorkflowStep(
                    step_id=f"{workflow_id}_step_{i}",
                    step_type=step_data.get('type', 'generate'),
                    pipeline=step_data.get('pipeline', 'image'),
                    prompt=step_data.get('prompt', ''),
                    params=step_data.get('params', {}),
                    dependencies=step_data.get('dependencies', [])
                )
                workflow_steps.append(step)
            
            # Store workflow for tracking
            self.active_workflows[workflow_id] = workflow_steps
            
            # Create execution plan
            execution_plan = self._create_execution_plan(workflow_steps)
            
            return {
                'workflow_id': workflow_id,
                'steps': len(workflow_steps),
                'execution_plan': execution_plan,
                'status': 'planned'
            }
            
        except Exception as e:
            logger.error(f"Error coordinating workflow: {str(e)}")
            raise SystemError(f"Workflow coordination failed: {str(e)}")
    
    def _create_execution_plan(self, steps: List[WorkflowStep]) -> List[Dict[str, Any]]:
        """Create optimized execution plan for workflow steps."""
        # Simple dependency-based ordering for now
        # In a full implementation, this would use topological sorting
        
        plan = []
        for step in steps:
            plan.append({
                'step_id': step.step_id,
                'pipeline': step.pipeline,
                'estimated_time': self._estimate_step_time(step),
                'memory_requirements': self._estimate_memory_requirements(step)
            })
        
        return plan
    
    def _estimate_step_time(self, step: WorkflowStep) -> float:
        """Estimate execution time for a workflow step."""
        # Basic time estimation based on pipeline and hardware
        base_times = {
            'image': 30.0,  # seconds
            'video': 300.0,  # 5 minutes
            'text': 5.0
        }
        
        base_time = base_times.get(step.pipeline, 30.0)
        
        # Adjust for hardware
        if self.hardware_config.vram_size < 6000:
            base_time *= 2.0  # Slower on limited VRAM
        elif self.hardware_config.vram_size > 16000:
            base_time *= 0.5  # Faster on high-end hardware
        
        return base_time
    
    def _estimate_memory_requirements(self, step: WorkflowStep) -> Dict[str, int]:
        """Estimate memory requirements for a workflow step."""
        # Basic memory estimation
        memory_reqs = {
            'image': {'vram': 4000, 'ram': 2000},  # MB
            'video': {'vram': 8000, 'ram': 4000},
            'text': {'vram': 2000, 'ram': 1000}
        }
        
        return memory_reqs.get(step.pipeline, {'vram': 4000, 'ram': 2000})
    
    def manage_context(self, conversation_id: str) -> ConversationContext:
        """
        Manage conversation context and memory.
        
        Args:
            conversation_id: Unique identifier for conversation
            
        Returns:
            ConversationContext: Current or new conversation context
        """
        if conversation_id not in self.conversations:
            # Create new conversation context
            context = ConversationContext(
                conversation_id=conversation_id,
                history=[],
                current_mode=ComplianceMode.RESEARCH_SAFE,  # Default mode
                user_preferences={}
            )
            self.conversations[conversation_id] = context
            logger.info(f"Created new conversation context: {conversation_id}")
        else:
            context = self.conversations[conversation_id]
            logger.debug(f"Retrieved existing conversation context: {conversation_id}")
        
        return context
    
    def _update_conversation_history(self, context: ConversationContext, 
                                   prompt: str, intent: IntentClassification) -> None:
        """Update conversation history with new interaction."""
        history_entry = {
            'timestamp': datetime.now().isoformat(),
            'prompt': prompt,
            'intent': asdict(intent),
            'context_length': len(context.history)
        }
        
        context.history.append(history_entry)
        
        # Limit history size to prevent memory issues
        max_history = 50
        if len(context.history) > max_history:
            context.history = context.history[-max_history:]
            logger.debug(f"Trimmed conversation history to {max_history} entries")
    
    def get_conversation_summary(self, conversation_id: str) -> Dict[str, Any]:
        """Get summary of conversation context."""
        if conversation_id not in self.conversations:
            return {'error': 'Conversation not found'}
        
        context = self.conversations[conversation_id]
        
        return {
            'conversation_id': conversation_id,
            'total_interactions': len(context.history),
            'current_mode': context.current_mode.value,
            'recent_intents': [entry['intent']['output_type'] 
                             for entry in context.history[-5:]],
            'user_preferences': context.user_preferences
        }
    
    def update_compliance_mode(self, conversation_id: str, mode: ComplianceMode) -> None:
        """Update compliance mode for a conversation."""
        if conversation_id in self.conversations:
            self.conversations[conversation_id].current_mode = mode
            logger.info(f"Updated compliance mode to {mode.value} for {conversation_id}")
        else:
            logger.warning(f"Attempted to update mode for unknown conversation: {conversation_id}")
    
    def cleanup_conversation(self, conversation_id: str) -> None:
        """Clean up conversation context to free memory."""
        if conversation_id in self.conversations:
            del self.conversations[conversation_id]
            logger.info(f"Cleaned up conversation context: {conversation_id}")
        
        # Also clean up any associated workflows
        workflows_to_remove = [wid for wid in self.active_workflows 
                              if wid.startswith(conversation_id)]
        for workflow_id in workflows_to_remove:
            del self.active_workflows[workflow_id]
            logger.info(f"Cleaned up workflow: {workflow_id}")