"""
Integration tests for Research Crawler

Tests the research crawler functionality with mock HTTP responses,
copyright awareness, rate limiting, and ethical crawling practices.
"""

import pytest
import asyncio
import aiohttp
from unittest.mock import AsyncMock, MagicMock, patch
from pathlib import Path
import tempfile
import json
from datetime import datetime

from src.data.research_crawler import (
    ResearchCrawler, CrawlConfig, ContentItem, CrawlResult, RateLimiter
)
from src.data.license_classifier import LicenseType


class MockResponse:
    """Mock HTTP response for testing"""
    
    def __init__(self, status=200, content_type="text/html", content=b"", headers=None):
        self.status = status
        self.headers = headers or {"content-type": content_type}
        self._content = content
    
    async def read(self):
        return self._content
    
    async def text(self):
        return self._content.decode("utf-8", errors="ignore")
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass


class TestRateLimiter:
    """Test rate limiting functionality"""
    
    @pytest.mark.asyncio
    async def test_rate_limiter_timing(self):
        """Test that rate limiter enforces timing constraints"""
        limiter = RateLimiter(max_requests_per_second=2.0)  # 0.5 second intervals
        
        start_time = asyncio.get_event_loop().time()
        
        # Make 3 requests
        await limiter.acquire()
        first_time = asyncio.get_event_loop().time()
        
        await limiter.acquire()
        second_time = asyncio.get_event_loop().time()
        
        await limiter.acquire()
        third_time = asyncio.get_event_loop().time()
        
        # Check timing intervals
        assert (second_time - first_time) >= 0.4  # Allow some tolerance
        assert (third_time - second_time) >= 0.4
    
    @pytest.mark.asyncio
    async def test_rate_limiter_concurrent(self):
        """Test rate limiter with concurrent requests"""
        limiter = RateLimiter(max_requests_per_second=5.0)
        
        start_time = asyncio.get_event_loop().time()
        
        # Make 5 concurrent requests
        tasks = [limiter.acquire() for _ in range(5)]
        await asyncio.gather(*tasks)
        
        end_time = asyncio.get_event_loop().time()
        
        # Should take at least 0.8 seconds (4 intervals of 0.2s each)
        assert (end_time - start_time) >= 0.6


class TestResearchCrawler:
    """Test research crawler functionality"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.config = CrawlConfig(
            max_requests_per_second=10.0,  # Fast for testing
            max_concurrent_requests=2,
            request_timeout=5,
            respect_robots_txt=False  # Disable for testing
        )
        self.crawler = ResearchCrawler(self.config, self.temp_dir)
    
    def teardown_method(self):
        """Clean up test fixtures"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @pytest.mark.asyncio
    async def test_crawler_context_manager(self):
        """Test crawler async context manager"""
        async with self.crawler as crawler:
            assert crawler.session is not None
            assert isinstance(crawler.session, aiohttp.ClientSession)
        
        # Session should be closed after context exit
        assert crawler.session.closed
    
    @pytest.mark.asyncio
    async def test_fetch_content_success(self):
        """Test successful content fetching"""
        html_content = b"""
        <html>
        <head>
            <title>Test Page</title>
            <meta name="description" content="Test description">
            <meta name="license" content="CC BY 4.0">
            <meta name="author" content="Test Author">
        </head>
        <body>Test content</body>
        </html>
        """
        
        mock_response = MockResponse(
            status=200,
            content_type="text/html",
            content=html_content,
            headers={
                "content-type": "text/html; charset=utf-8",
                "content-length": str(len(html_content)),
                "last-modified": "Wed, 21 Oct 2024 07:28:00 GMT"
            }
        )
        
        with patch.object(self.crawler, 'session') as mock_session:
            mock_session.get.return_value = mock_response
            
            result = await self.crawler._fetch_content("https://example.com/test")
            
            assert result is not None
            assert result.url == "https://example.com/test"
            assert result.content_type == "text/html"
            assert result.content_size == len(html_content)
            assert result.title == "Test Page"
            assert result.description == "Test description"
            assert result.attribution == "Test Author"
            assert "license" in result.metadata
            assert result.metadata["license"] == "CC BY 4.0"
    
    @pytest.mark.asyncio
    async def test_fetch_content_unsupported_type(self):
        """Test handling of unsupported content types"""
        mock_response = MockResponse(
            status=200,
            content_type="application/pdf",
            content=b"PDF content"
        )
        
        with patch.object(self.crawler, 'session') as mock_session:
            mock_session.get.return_value = mock_response
            
            result = await self.crawler._fetch_content("https://example.com/test.pdf")
            
            assert result is None
    
    @pytest.mark.asyncio
    async def test_fetch_content_too_large(self):
        """Test handling of content that's too large"""
        large_content = b"x" * (self.config.max_content_size + 1)
        
        mock_response = MockResponse(
            status=200,
            content_type="text/html",
            content=large_content,
            headers={
                "content-type": "text/html",
                "content-length": str(len(large_content))
            }
        )
        
        with patch.object(self.crawler, 'session') as mock_session:
            mock_session.get.return_value = mock_response
            
            result = await self.crawler._fetch_content("https://example.com/large")
            
            assert result is None
    
    @pytest.mark.asyncio
    async def test_fetch_content_http_error(self):
        """Test handling of HTTP errors"""
        mock_response = MockResponse(status=404)
        
        with patch.object(self.crawler, 'session') as mock_session:
            mock_session.get.return_value = mock_response
            
            result = await self.crawler._fetch_content("https://example.com/notfound")
            
            assert result is None
    
    @pytest.mark.asyncio
    async def test_classify_and_validate_open_source_mode(self):
        """Test classification and validation in open source mode"""
        content_item = ContentItem(
            url="https://commons.wikimedia.org/wiki/File:Test.jpg",
            content_type="image/jpeg",
            source_domain="commons.wikimedia.org"
        )
        
        await self.crawler._classify_and_validate(content_item, "open_source")
        
        assert content_item.license_info is not None
        assert content_item.license_info.license_type == LicenseType.PUBLIC_DOMAIN
        assert content_item.research_safe is True
        assert content_item.attribution != ""
    
    @pytest.mark.asyncio
    async def test_classify_and_validate_research_safe_mode(self):
        """Test classification and validation in research safe mode"""
        content_item = ContentItem(
            url="https://deviantart.com/artist/art/Test-123456",
            content_type="image/jpeg",
            source_domain="deviantart.com"
        )
        
        await self.crawler._classify_and_validate(content_item, "research_safe")
        
        assert content_item.license_info is not None
        assert content_item.license_info.license_type == LicenseType.FAIR_USE_RESEARCH
        assert content_item.research_safe is True
    
    @pytest.mark.asyncio
    async def test_classify_and_validate_full_dataset_mode(self):
        """Test classification and validation in full dataset mode"""
        content_item = ContentItem(
            url="https://shutterstock.com/image/test-123456",
            content_type="image/jpeg",
            source_domain="shutterstock.com"
        )
        
        await self.crawler._classify_and_validate(content_item, "full_dataset")
        
        assert content_item.license_info is not None
        assert content_item.license_info.license_type == LicenseType.COPYRIGHTED
        assert content_item.research_safe is True  # Full dataset includes everything
    
    @pytest.mark.asyncio
    async def test_save_content(self):
        """Test content saving functionality"""
        content_item = ContentItem(
            url="https://example.com/test.jpg",
            content_type="image/jpeg",
            content_size=1024,
            title="Test Image",
            description="Test description",
            source_domain="example.com",
            content_hash="abcdef1234567890",
            research_safe=True
        )
        
        # Mock license info
        from src.data.license_classifier import LicenseInfo
        content_item.license_info = LicenseInfo(
            license_type=LicenseType.CREATIVE_COMMONS,
            confidence=0.9,
            source_domain="example.com",
            attribution_required=True,
            commercial_use_allowed=True,
            research_safe=True,
            details="Test license"
        )
        
        await self.crawler._save_content(content_item)
        
        # Check that metadata file was created
        expected_metadata_path = self.temp_dir / "example.com" / "abcdef1234567890.json"
        assert expected_metadata_path.exists()
        
        # Check metadata content
        with open(expected_metadata_path, "r") as f:
            metadata = json.load(f)
        
        assert metadata["url"] == content_item.url
        assert metadata["title"] == content_item.title
        assert metadata["research_safe"] is True
        assert metadata["license_info"]["license_type"] == "creative_commons"
    
    def test_generate_attribution(self):
        """Test attribution string generation"""
        content_item = ContentItem(
            url="https://example.com/test.jpg",
            title="Test Image",
            attribution="Test Author",
            crawl_timestamp=datetime(2024, 1, 1, 12, 0, 0)
        )
        
        from src.data.license_classifier import LicenseInfo
        content_item.license_info = LicenseInfo(
            license_type=LicenseType.CREATIVE_COMMONS,
            confidence=0.9,
            source_domain="example.com",
            attribution_required=True,
            commercial_use_allowed=True,
            research_safe=True,
            details="Test"
        )
        
        content_item.metadata = {"license_url": "https://creativecommons.org/licenses/by/4.0/"}
        
        attribution = self.crawler._generate_attribution(content_item)
        
        assert "Test Image" in attribution
        assert "Test Author" in attribution
        assert "https://example.com/test.jpg" in attribution
        assert "Creative Commons" in attribution
        assert "2024-01-01" in attribution
    
    def test_get_file_extension(self):
        """Test file extension mapping"""
        assert self.crawler._get_file_extension("text/html") == ".html"
        assert self.crawler._get_file_extension("image/jpeg") == ".jpg"
        assert self.crawler._get_file_extension("image/png") == ".png"
        assert self.crawler._get_file_extension("video/mp4") == ".mp4"
        assert self.crawler._get_file_extension("unknown/type") == ".bin"
    
    @pytest.mark.asyncio
    async def test_crawl_urls_integration(self):
        """Test full crawl integration with multiple URLs"""
        urls = [
            "https://commons.wikimedia.org/wiki/File:Test1.jpg",
            "https://unsplash.com/photos/test2",
            "https://shutterstock.com/image/test3-123456"
        ]
        
        # Mock responses for each URL
        responses = [
            MockResponse(
                status=200,
                content_type="text/html",
                content=b"<html><head><title>Test 1</title></head><body>Content 1</body></html>"
            ),
            MockResponse(
                status=200,
                content_type="text/html", 
                content=b"<html><head><title>Test 2</title></head><body>Content 2</body></html>"
            ),
            MockResponse(
                status=200,
                content_type="text/html",
                content=b"<html><head><title>Test 3</title></head><body>Content 3</body></html>"
            )
        ]
        
        async with self.crawler as crawler:
            with patch.object(crawler.session, 'get') as mock_get:
                mock_get.side_effect = responses
                
                result = await crawler.crawl_urls(urls, "research_safe")
                
                assert isinstance(result, CrawlResult)
                assert result.success_count >= 2  # At least wikimedia and unsplash should be research safe
                assert len(result.items) >= 2
                assert result.duration > 0
    
    def test_get_crawl_statistics(self):
        """Test crawl statistics generation"""
        # Create mock crawl result
        from src.data.license_classifier import LicenseInfo
        
        items = [
            ContentItem(
                url="https://commons.wikimedia.org/test1",
                source_domain="commons.wikimedia.org",
                content_type="image/jpeg",
                content_size=1024,
                research_safe=True,
                license_info=LicenseInfo(
                    license_type=LicenseType.PUBLIC_DOMAIN,
                    confidence=0.95,
                    source_domain="commons.wikimedia.org",
                    attribution_required=True,
                    commercial_use_allowed=True,
                    research_safe=True,
                    details="Test"
                )
            ),
            ContentItem(
                url="https://shutterstock.com/test2",
                source_domain="shutterstock.com",
                content_type="image/png",
                content_size=2048,
                research_safe=False,
                license_info=LicenseInfo(
                    license_type=LicenseType.COPYRIGHTED,
                    confidence=0.90,
                    source_domain="shutterstock.com",
                    attribution_required=True,
                    commercial_use_allowed=False,
                    research_safe=False,
                    details="Test"
                )
            )
        ]
        
        result = CrawlResult(
            success_count=2,
            error_count=0,
            skipped_count=0,
            total_size=3072,
            items=items,
            duration=5.0
        )
        
        stats = self.crawler.get_crawl_statistics(result)
        
        assert stats["total_items"] == 2
        assert stats["success_count"] == 2
        assert stats["research_safe_count"] == 1
        assert stats["research_safe_percentage"] == 50.0
        assert "public_domain" in stats["license_distribution"]
        assert "copyrighted" in stats["license_distribution"]
        assert "commons.wikimedia.org" in stats["domain_distribution"]
        assert "shutterstock.com" in stats["domain_distribution"]
        assert stats["average_confidence"] == 0.925


class TestCrawlConfig:
    """Test crawl configuration"""
    
    def test_default_config(self):
        """Test default configuration values"""
        config = CrawlConfig()
        
        assert config.max_requests_per_second == 1.0
        assert config.max_concurrent_requests == 5
        assert config.request_timeout == 30
        assert config.respect_robots_txt is True
        assert config.max_retries == 3
        assert "text/html" in config.allowed_content_types
        assert "image/jpeg" in config.allowed_content_types
    
    def test_custom_config(self):
        """Test custom configuration"""
        config = CrawlConfig(
            max_requests_per_second=2.0,
            max_concurrent_requests=10,
            respect_robots_txt=False
        )
        
        assert config.max_requests_per_second == 2.0
        assert config.max_concurrent_requests == 10
        assert config.respect_robots_txt is False


class TestContentItem:
    """Test ContentItem data structure"""
    
    def test_content_item_creation(self):
        """Test ContentItem creation and defaults"""
        item = ContentItem(url="https://example.com/test")
        
        assert item.url == "https://example.com/test"
        assert item.local_path is None
        assert item.content_type == ""
        assert item.content_size == 0
        assert item.research_safe is False
        assert isinstance(item.metadata, dict)
        assert isinstance(item.crawl_timestamp, datetime)
    
    def test_content_item_with_data(self):
        """Test ContentItem with full data"""
        timestamp = datetime.now()
        item = ContentItem(
            url="https://example.com/test",
            content_type="image/jpeg",
            content_size=1024,
            title="Test Image",
            description="Test description",
            crawl_timestamp=timestamp,
            research_safe=True
        )
        
        assert item.content_type == "image/jpeg"
        assert item.content_size == 1024
        assert item.title == "Test Image"
        assert item.description == "Test description"
        assert item.crawl_timestamp == timestamp
        assert item.research_safe is True


if __name__ == "__main__":
    pytest.main([__file__])