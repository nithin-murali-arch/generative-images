"""
Research Crawler with Copyright Awareness

This module provides ethical web crawling capabilities with built-in copyright
classification, attribution tracking, and respectful scraping practices for
academic research compliance.
"""

import asyncio
import aiohttp
import time
import logging
from typing import Dict, List, Optional, Set, Tuple, AsyncGenerator
from urllib.parse import urljoin, urlparse, parse_qs
from urllib.robotparser import RobotFileParser
from dataclasses import dataclass, field
from pathlib import Path
import hashlib
import json
from datetime import datetime, timedelta

from .license_classifier import LicenseClassifier, LicenseInfo, LicenseType


@dataclass
class CrawlConfig:
    """Configuration for research crawler"""
    max_requests_per_second: float = 1.0  # Respectful rate limiting
    max_concurrent_requests: int = 5
    request_timeout: int = 30
    user_agent: str = "Academic-Research-Crawler/1.0 (Educational Use)"
    respect_robots_txt: bool = True
    max_retries: int = 3
    retry_delay: float = 2.0
    max_content_size: int = 10 * 1024 * 1024  # 10MB limit
    allowed_content_types: Set[str] = field(default_factory=lambda: {
        "text/html", "text/plain", "application/json",
        "image/jpeg", "image/png", "image/gif", "image/webp",
        "video/mp4", "video/webm", "video/avi"
    })


@dataclass
class ContentItem:
    """Container for crawled content with copyright information"""
    url: str
    local_path: Optional[Path] = None
    content_type: str = ""
    content_size: int = 0
    title: str = ""
    description: str = ""
    license_info: Optional[LicenseInfo] = None
    metadata: Dict = field(default_factory=dict)
    attribution: str = ""
    crawl_timestamp: datetime = field(default_factory=datetime.now)
    source_domain: str = ""
    research_safe: bool = False
    content_hash: str = ""


@dataclass
class CrawlResult:
    """Result of a crawling operation"""
    success_count: int = 0
    error_count: int = 0
    skipped_count: int = 0
    total_size: int = 0
    items: List[ContentItem] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    duration: float = 0.0


class RateLimiter:
    """Rate limiter for respectful crawling"""
    
    def __init__(self, max_requests_per_second: float):
        self.max_requests_per_second = max_requests_per_second
        self.min_interval = 1.0 / max_requests_per_second if max_requests_per_second > 0 else 0
        self.last_request_time = 0.0
        self._lock = asyncio.Lock()
    
    async def acquire(self):
        """Acquire permission to make a request"""
        async with self._lock:
            current_time = time.time()
            time_since_last = current_time - self.last_request_time
            
            if time_since_last < self.min_interval:
                sleep_time = self.min_interval - time_since_last
                await asyncio.sleep(sleep_time)
            
            self.last_request_time = time.time()


class ResearchCrawler:
    """
    Ethical web crawler for academic research with copyright awareness.
    
    Features:
    - Respectful rate limiting and robots.txt compliance
    - Automatic copyright classification and attribution tracking
    - Content type filtering and size limits
    - Retry logic with exponential backoff
    - Comprehensive error handling and logging
    """
    
    def __init__(self, config: Optional[CrawlConfig] = None, 
                 storage_path: Optional[Path] = None):
        self.config = config or CrawlConfig()
        self.storage_path = storage_path or Path("./data/crawled")
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.license_classifier = LicenseClassifier()
        self.rate_limiter = RateLimiter(self.config.max_requests_per_second)
        self.robots_cache: Dict[str, RobotFileParser] = {}
        self.session: Optional[aiohttp.ClientSession] = None
        
        # Set up logging
        self.logger = logging.getLogger(__name__)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
    
    async def __aenter__(self):
        """Async context manager entry"""
        connector = aiohttp.TCPConnector(limit=self.config.max_concurrent_requests)
        timeout = aiohttp.ClientTimeout(total=self.config.request_timeout)
        
        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            headers={"User-Agent": self.config.user_agent}
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    async def crawl_urls(self, urls: List[str], 
                        compliance_mode: str = "research_safe") -> CrawlResult:
        """
        Crawl a list of URLs with copyright awareness.
        
        Args:
            urls: List of URLs to crawl
            compliance_mode: "open_source", "research_safe", or "full_dataset"
            
        Returns:
            CrawlResult with crawled items and statistics
        """
        start_time = time.time()
        result = CrawlResult()
        
        self.logger.info(f"Starting crawl of {len(urls)} URLs in {compliance_mode} mode")
        
        # Create semaphore for concurrent requests
        semaphore = asyncio.Semaphore(self.config.max_concurrent_requests)
        
        # Create tasks for all URLs
        tasks = [
            self._crawl_single_url(url, semaphore, compliance_mode)
            for url in urls
        ]
        
        # Execute tasks and collect results
        completed_tasks = await asyncio.gather(*tasks, return_exceptions=True)
        
        for i, task_result in enumerate(completed_tasks):
            if isinstance(task_result, Exception):
                result.error_count += 1
                result.errors.append(f"URL {urls[i]}: {str(task_result)}")
                self.logger.error(f"Error crawling {urls[i]}: {task_result}")
            elif task_result is None:
                result.skipped_count += 1
            else:
                result.success_count += 1
                result.total_size += task_result.content_size
                result.items.append(task_result)
        
        result.duration = time.time() - start_time
        
        self.logger.info(
            f"Crawl completed: {result.success_count} success, "
            f"{result.error_count} errors, {result.skipped_count} skipped "
            f"in {result.duration:.2f}s"
        )
        
        return result
    
    async def _crawl_single_url(self, url: str, semaphore: asyncio.Semaphore,
                               compliance_mode: str) -> Optional[ContentItem]:
        """Crawl a single URL with rate limiting and error handling"""
        async with semaphore:
            await self.rate_limiter.acquire()
            
            try:
                # Check robots.txt if enabled
                if self.config.respect_robots_txt:
                    if not await self._can_fetch(url):
                        self.logger.info(f"Robots.txt disallows crawling: {url}")
                        return None
                
                # Attempt to fetch content with retries
                for attempt in range(self.config.max_retries):
                    try:
                        content_item = await self._fetch_content(url)
                        if content_item:
                            # Classify copyright and check compliance
                            await self._classify_and_validate(content_item, compliance_mode)
                            
                            # Save content if research safe or in full dataset mode
                            if (content_item.research_safe or 
                                compliance_mode == "full_dataset"):
                                await self._save_content(content_item)
                                return content_item
                            else:
                                self.logger.info(
                                    f"Content not research safe, skipping: {url}"
                                )
                                return None
                        
                    except aiohttp.ClientError as e:
                        if attempt == self.config.max_retries - 1:
                            raise
                        
                        wait_time = self.config.retry_delay * (2 ** attempt)
                        self.logger.warning(
                            f"Attempt {attempt + 1} failed for {url}, "
                            f"retrying in {wait_time}s: {e}"
                        )
                        await asyncio.sleep(wait_time)
                
            except Exception as e:
                self.logger.error(f"Failed to crawl {url}: {e}")
                raise
        
        return None
    
    async def _fetch_content(self, url: str) -> Optional[ContentItem]:
        """Fetch content from URL with validation"""
        if not self.session:
            raise RuntimeError("Session not initialized. Use async context manager.")
        
        try:
            async with self.session.get(url) as response:
                # Check response status
                if response.status != 200:
                    self.logger.warning(f"HTTP {response.status} for {url}")
                    return None
                
                # Check content type
                content_type = response.headers.get("content-type", "").split(";")[0]
                if content_type not in self.config.allowed_content_types:
                    self.logger.info(f"Unsupported content type {content_type} for {url}")
                    return None
                
                # Check content size
                content_length = response.headers.get("content-length")
                if content_length and int(content_length) > self.config.max_content_size:
                    self.logger.info(f"Content too large ({content_length} bytes) for {url}")
                    return None
                
                # Read content
                content = await response.read()
                if len(content) > self.config.max_content_size:
                    self.logger.info(f"Content too large ({len(content)} bytes) for {url}")
                    return None
                
                # Create content item
                content_item = ContentItem(
                    url=url,
                    content_type=content_type,
                    content_size=len(content),
                    source_domain=urlparse(url).netloc,
                    content_hash=hashlib.sha256(content).hexdigest()
                )
                
                # Extract metadata from response headers and content
                await self._extract_metadata(content_item, response, content)
                
                return content_item
                
        except asyncio.TimeoutError:
            self.logger.warning(f"Timeout fetching {url}")
            return None
        except Exception as e:
            self.logger.error(f"Error fetching {url}: {e}")
            return None
    
    async def _extract_metadata(self, content_item: ContentItem, 
                               response: aiohttp.ClientResponse, content: bytes):
        """Extract metadata from response and content"""
        # Extract from headers
        content_item.metadata.update({
            "last_modified": response.headers.get("last-modified"),
            "etag": response.headers.get("etag"),
            "server": response.headers.get("server"),
            "content_encoding": response.headers.get("content-encoding")
        })
        
        # For HTML content, extract additional metadata
        if content_item.content_type == "text/html":
            try:
                content_text = content.decode("utf-8", errors="ignore")
                await self._extract_html_metadata(content_item, content_text)
            except Exception as e:
                self.logger.warning(f"Error extracting HTML metadata: {e}")
        
        # For JSON content, parse and extract metadata
        elif content_item.content_type == "application/json":
            try:
                content_text = content.decode("utf-8", errors="ignore")
                json_data = json.loads(content_text)
                content_item.metadata.update(json_data)
            except Exception as e:
                self.logger.warning(f"Error parsing JSON metadata: {e}")
    
    async def _extract_html_metadata(self, content_item: ContentItem, html_content: str):
        """Extract metadata from HTML content"""
        import re
        
        # Extract title
        title_match = re.search(r"<title[^>]*>([^<]+)</title>", html_content, re.IGNORECASE)
        if title_match:
            content_item.title = title_match.group(1).strip()
        
        # Extract meta description
        desc_match = re.search(
            r'<meta[^>]*name=["\']description["\'][^>]*content=["\']([^"\']+)["\']',
            html_content, re.IGNORECASE
        )
        if desc_match:
            content_item.description = desc_match.group(1).strip()
        
        # Extract license information from meta tags
        license_patterns = [
            r'<meta[^>]*name=["\']license["\'][^>]*content=["\']([^"\']+)["\']',
            r'<meta[^>]*name=["\']copyright["\'][^>]*content=["\']([^"\']+)["\']',
            r'<meta[^>]*name=["\']rights["\'][^>]*content=["\']([^"\']+)["\']'
        ]
        
        for pattern in license_patterns:
            match = re.search(pattern, html_content, re.IGNORECASE)
            if match:
                field_name = re.search(r'name=["\']([^"\']+)["\']', match.group(0)).group(1)
                content_item.metadata[field_name] = match.group(1).strip()
        
        # Extract Creative Commons license links
        cc_link_match = re.search(
            r'<link[^>]*rel=["\']license["\'][^>]*href=["\']([^"\']*creativecommons[^"\']*)["\']',
            html_content, re.IGNORECASE
        )
        if cc_link_match:
            content_item.metadata["license_url"] = cc_link_match.group(1)
        
        # Extract attribution information
        attribution_patterns = [
            r'<meta[^>]*name=["\']author["\'][^>]*content=["\']([^"\']+)["\']',
            r'<meta[^>]*name=["\']creator["\'][^>]*content=["\']([^"\']+)["\']'
        ]
        
        for pattern in attribution_patterns:
            match = re.search(pattern, html_content, re.IGNORECASE)
            if match:
                content_item.attribution = match.group(1).strip()
                break
    
    async def _classify_and_validate(self, content_item: ContentItem, compliance_mode: str):
        """Classify copyright and validate against compliance mode"""
        # Classify license using the license classifier
        content_item.license_info = self.license_classifier.classify_url(
            content_item.url, content_item.metadata
        )
        
        # Set research safety based on classification and compliance mode
        if compliance_mode == "open_source":
            content_item.research_safe = content_item.license_info.license_type in [
                LicenseType.PUBLIC_DOMAIN, LicenseType.CREATIVE_COMMONS
            ]
        elif compliance_mode == "research_safe":
            content_item.research_safe = content_item.license_info.research_safe
        else:  # full_dataset
            content_item.research_safe = True  # Include everything but flag appropriately
        
        # Generate attribution string
        content_item.attribution = self._generate_attribution(content_item)
        
        self.logger.debug(
            f"Classified {content_item.url}: {content_item.license_info.license_type.value} "
            f"(research_safe: {content_item.research_safe})"
        )
    
    def _generate_attribution(self, content_item: ContentItem) -> str:
        """Generate proper attribution string for content"""
        parts = []
        
        # Add title if available
        if content_item.title:
            parts.append(f'"{content_item.title}"')
        
        # Add author/creator if available
        if content_item.attribution:
            parts.append(f"by {content_item.attribution}")
        
        # Add source URL
        parts.append(f"Source: {content_item.url}")
        
        # Add license information
        if content_item.license_info:
            license_type = content_item.license_info.license_type.value.replace("_", " ").title()
            parts.append(f"License: {license_type}")
            
            if content_item.license_info.license_type == LicenseType.CREATIVE_COMMONS:
                if "license_url" in content_item.metadata:
                    parts.append(f"License URL: {content_item.metadata['license_url']}")
        
        # Add access date
        parts.append(f"Accessed: {content_item.crawl_timestamp.strftime('%Y-%m-%d')}")
        
        return " | ".join(parts)
    
    async def _save_content(self, content_item: ContentItem):
        """Save content item to local storage"""
        # Create directory structure based on domain
        domain_path = self.storage_path / content_item.source_domain
        domain_path.mkdir(parents=True, exist_ok=True)
        
        # Generate filename based on content hash
        file_extension = self._get_file_extension(content_item.content_type)
        filename = f"{content_item.content_hash[:16]}{file_extension}"
        content_item.local_path = domain_path / filename
        
        # Save metadata as JSON
        metadata_path = domain_path / f"{content_item.content_hash[:16]}.json"
        metadata = {
            "url": content_item.url,
            "content_type": content_item.content_type,
            "content_size": content_item.content_size,
            "title": content_item.title,
            "description": content_item.description,
            "attribution": content_item.attribution,
            "crawl_timestamp": content_item.crawl_timestamp.isoformat(),
            "source_domain": content_item.source_domain,
            "research_safe": content_item.research_safe,
            "license_info": {
                "license_type": content_item.license_info.license_type.value,
                "confidence": content_item.license_info.confidence,
                "attribution_required": content_item.license_info.attribution_required,
                "commercial_use_allowed": content_item.license_info.commercial_use_allowed,
                "research_safe": content_item.license_info.research_safe,
                "details": content_item.license_info.details
            } if content_item.license_info else None,
            "metadata": content_item.metadata
        }
        
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        self.logger.debug(f"Saved content metadata: {metadata_path}")
    
    def _get_file_extension(self, content_type: str) -> str:
        """Get appropriate file extension for content type"""
        extensions = {
            "text/html": ".html",
            "text/plain": ".txt",
            "application/json": ".json",
            "image/jpeg": ".jpg",
            "image/png": ".png",
            "image/gif": ".gif",
            "image/webp": ".webp",
            "video/mp4": ".mp4",
            "video/webm": ".webm",
            "video/avi": ".avi"
        }
        return extensions.get(content_type, ".bin")
    
    async def _can_fetch(self, url: str) -> bool:
        """Check if URL can be fetched according to robots.txt"""
        parsed_url = urlparse(url)
        base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
        
        if base_url not in self.robots_cache:
            robots_url = urljoin(base_url, "/robots.txt")
            
            try:
                rp = RobotFileParser()
                rp.set_url(robots_url)
                
                # Fetch robots.txt with timeout
                if self.session:
                    async with self.session.get(robots_url) as response:
                        if response.status == 200:
                            robots_content = await response.text()
                            # Parse robots.txt content
                            for line in robots_content.split('\n'):
                                rp.read_line(line.strip())
                
                self.robots_cache[base_url] = rp
                
            except Exception as e:
                self.logger.warning(f"Could not fetch robots.txt for {base_url}: {e}")
                # If we can't fetch robots.txt, assume we can crawl
                rp = RobotFileParser()
                self.robots_cache[base_url] = rp
        
        rp = self.robots_cache[base_url]
        return rp.can_fetch(self.config.user_agent, url)
    
    def get_crawl_statistics(self, result: CrawlResult) -> Dict:
        """Generate detailed statistics from crawl result"""
        if not result.items:
            return {"total_items": 0}
        
        license_counts = {}
        domain_counts = {}
        content_type_counts = {}
        research_safe_count = 0
        
        for item in result.items:
            # Count license types
            if item.license_info:
                license_type = item.license_info.license_type.value
                license_counts[license_type] = license_counts.get(license_type, 0) + 1
            
            # Count domains
            domain_counts[item.source_domain] = domain_counts.get(item.source_domain, 0) + 1
            
            # Count content types
            content_type_counts[item.content_type] = content_type_counts.get(item.content_type, 0) + 1
            
            # Count research safe items
            if item.research_safe:
                research_safe_count += 1
        
        return {
            "total_items": len(result.items),
            "success_count": result.success_count,
            "error_count": result.error_count,
            "skipped_count": result.skipped_count,
            "total_size_mb": result.total_size / (1024 * 1024),
            "duration_seconds": result.duration,
            "research_safe_count": research_safe_count,
            "research_safe_percentage": (research_safe_count / len(result.items)) * 100,
            "license_distribution": license_counts,
            "domain_distribution": domain_counts,
            "content_type_distribution": content_type_counts,
            "average_confidence": sum(
                item.license_info.confidence for item in result.items 
                if item.license_info
            ) / len([item for item in result.items if item.license_info]) if result.items else 0
        }