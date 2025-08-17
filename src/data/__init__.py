"""
Data Management Module for Academic Multimodal LLM System

This module provides copyright-aware data collection, classification,
and organization capabilities for research compliance.
"""

from .license_classifier import LicenseClassifier, LicenseType, LicenseInfo
from .research_crawler import ResearchCrawler, CrawlConfig, ContentItem, CrawlResult, RateLimiter
from .dataset_organizer import DatasetOrganizer, DatasetConfig, ComplianceMode, OrganizedDataset, DatasetStats

__all__ = [
    # License Classification
    "LicenseClassifier",
    "LicenseType", 
    "LicenseInfo",
    
    # Research Crawling
    "ResearchCrawler",
    "CrawlConfig",
    "ContentItem",
    "CrawlResult",
    "RateLimiter",
    
    # Dataset Organization
    "DatasetOrganizer",
    "DatasetConfig",
    "ComplianceMode",
    "OrganizedDataset",
    "DatasetStats"
]