"""
Logging configuration for the Academic Multimodal LLM Experiment System.

This module provides centralized logging with support for different log levels,
file output, and structured logging for research tracking.
"""

import logging
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime


class ResearchLogger:
    """Enhanced logger for research activities with structured output."""
    
    def __init__(self, name: str, log_file: Optional[Path] = None, level: str = "INFO"):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, level.upper()))
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Create formatters
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        )
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
        
        # File handler if specified
        if log_file:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(file_formatter)
            self.logger.addHandler(file_handler)
    
    def log_experiment_start(self, experiment_id: str, prompt: str, model: str) -> None:
        """Log the start of an experiment."""
        self.logger.info(f"EXPERIMENT_START - ID: {experiment_id}, Model: {model}, Prompt: {prompt[:100]}...")
    
    def log_experiment_end(self, experiment_id: str, success: bool, duration: float) -> None:
        """Log the end of an experiment."""
        status = "SUCCESS" if success else "FAILED"
        self.logger.info(f"EXPERIMENT_END - ID: {experiment_id}, Status: {status}, Duration: {duration:.2f}s")
    
    def log_model_load(self, model_name: str, vram_usage: Optional[float] = None) -> None:
        """Log model loading."""
        vram_info = f", VRAM: {vram_usage:.1f}MB" if vram_usage else ""
        self.logger.info(f"MODEL_LOAD - {model_name}{vram_info}")
    
    def log_compliance_check(self, content_type: str, license_type: str, approved: bool) -> None:
        """Log compliance checking."""
        status = "APPROVED" if approved else "REJECTED"
        self.logger.info(f"COMPLIANCE_CHECK - Type: {content_type}, License: {license_type}, Status: {status}")
    
    def log_memory_usage(self, vram_used: float, vram_total: float, optimization_applied: str = None) -> None:
        """Log memory usage and optimizations."""
        usage_percent = (vram_used / vram_total) * 100
        opt_info = f", Optimization: {optimization_applied}" if optimization_applied else ""
        self.logger.info(f"MEMORY_USAGE - {vram_used:.1f}MB/{vram_total:.1f}MB ({usage_percent:.1f}%){opt_info}")
    
    def log_error_with_context(self, error: Exception, context: dict) -> None:
        """Log error with additional context."""
        context_str = ", ".join([f"{k}: {v}" for k, v in context.items()])
        self.logger.error(f"ERROR - {type(error).__name__}: {str(error)} | Context: {context_str}")


def setup_logging(log_file: Optional[Path] = None, level: str = "INFO") -> ResearchLogger:
    """Set up logging for the entire system."""
    return ResearchLogger("AcademicMultimodalLLM", log_file, level)


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance for a specific module."""
    return logging.getLogger(f"AcademicMultimodalLLM.{name}")


# Performance logging decorator
def log_performance(logger: logging.Logger):
    """Decorator to log function performance."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = datetime.now()
            try:
                result = func(*args, **kwargs)
                duration = (datetime.now() - start_time).total_seconds()
                logger.info(f"PERFORMANCE - {func.__name__} completed in {duration:.2f}s")
                return result
            except Exception as e:
                duration = (datetime.now() - start_time).total_seconds()
                logger.error(f"PERFORMANCE - {func.__name__} failed after {duration:.2f}s: {str(e)}")
                raise
        return wrapper
    return decorator