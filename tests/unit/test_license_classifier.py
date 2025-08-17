"""
Unit tests for License Classification Engine

Tests license classification accuracy, domain-based rules, metadata analysis,
and content indicator detection for research compliance.
"""

import pytest
from src.data.license_classifier import LicenseClassifier, LicenseType, LicenseInfo


class TestLicenseClassifier:
    """Test suite for LicenseClassifier"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.classifier = LicenseClassifier()
    
    def test_license_type_enum(self):
        """Test LicenseType enumeration values"""
        assert LicenseType.PUBLIC_DOMAIN.value == "public_domain"
        assert LicenseType.CREATIVE_COMMONS.value == "creative_commons"
        assert LicenseType.FAIR_USE_RESEARCH.value == "fair_use_research"
        assert LicenseType.COPYRIGHTED.value == "copyrighted"
        assert LicenseType.UNKNOWN.value == "unknown"
    
    def test_public_domain_classification(self):
        """Test classification of public domain sources"""
        test_cases = [
            "https://commons.wikimedia.org/wiki/File:Example.jpg",
            "https://archive.org/details/example",
            "https://pixabay.com/photos/example-123456/"
        ]
        
        for url in test_cases:
            result = self.classifier.classify_url(url)
            assert result.license_type == LicenseType.PUBLIC_DOMAIN
            assert result.confidence > 0.8
            assert result.research_safe is True
            assert result.source_domain in url
    
    def test_creative_commons_classification(self):
        """Test classification of Creative Commons sources"""
        test_cases = [
            "https://unsplash.com/photos/example",
            "https://pexels.com/photo/example-123456/",
            "https://flickr.com/photos/user/123456/"
        ]
        
        for url in test_cases:
            result = self.classifier.classify_url(url)
            assert result.license_type == LicenseType.CREATIVE_COMMONS
            assert result.confidence > 0.6
            assert result.research_safe is True
            assert result.attribution_required is True
    
    def test_fair_use_research_classification(self):
        """Test classification of fair use research sources"""
        test_cases = [
            "https://deviantart.com/user/art/example-123456",
            "https://artstation.com/artwork/example",
            "https://youtube.com/watch?v=example123",
            "https://youtu.be/example123"
        ]
        
        for url in test_cases:
            result = self.classifier.classify_url(url)
            assert result.license_type == LicenseType.FAIR_USE_RESEARCH
            assert result.confidence > 0.7
            assert result.research_safe is True
            assert result.commercial_use_allowed is False
    
    def test_copyrighted_classification(self):
        """Test classification of copyrighted sources"""
        test_cases = [
            "https://shutterstock.com/image/example-123456",
            "https://gettyimages.com/detail/example",
            "https://instagram.com/p/example123/",
            "https://facebook.com/photo/example",
            "https://twitter.com/user/status/123456",
            "https://x.com/user/status/123456"
        ]
        
        for url in test_cases:
            result = self.classifier.classify_url(url)
            assert result.license_type == LicenseType.COPYRIGHTED
            assert result.confidence > 0.7
            assert result.research_safe is False
            assert result.commercial_use_allowed is False
    
    def test_unknown_domain_classification(self):
        """Test classification of unknown domains"""
        test_cases = [
            "https://unknown-site.com/image.jpg",
            "https://random-blog.net/photo.png",
            "https://example.org/content.gif"
        ]
        
        for url in test_cases:
            result = self.classifier.classify_url(url)
            assert result.license_type == LicenseType.UNKNOWN
            assert result.confidence <= 0.6
            assert result.research_safe is False  # Conservative approach
            assert result.attribution_required is True
    
    def test_subdomain_matching(self):
        """Test subdomain pattern matching"""
        test_cases = [
            {
                "url": "https://en.wikipedia.org/wiki/Example",
                "expected_type": LicenseType.PUBLIC_DOMAIN,
                "min_confidence": 0.8
            },
            {
                "url": "https://subdomain.unsplash.com/photo", 
                "expected_type": LicenseType.CREATIVE_COMMONS,
                "min_confidence": 0.7
            }
        ]
        
        for case in test_cases:
            result = self.classifier.classify_url(case["url"])
            assert result.license_type == case["expected_type"]
            assert result.confidence >= case["min_confidence"]
            assert "match" in result.details.lower()
    
    def test_metadata_classification(self):
        """Test classification based on metadata"""
        test_cases = [
            # CC0 metadata
            {
                "url": "https://example.com/image.jpg",
                "metadata": {"license": "CC0", "rights": "Public Domain"},
                "expected_type": LicenseType.PUBLIC_DOMAIN,
                "expected_attribution": False
            },
            # CC BY metadata
            {
                "url": "https://example.com/image.jpg", 
                "metadata": {"license": "CC BY 4.0", "usage_rights": "Attribution required"},
                "expected_type": LicenseType.CREATIVE_COMMONS,
                "expected_attribution": True
            },
            # Creative Commons general
            {
                "url": "https://example.com/image.jpg",
                "metadata": {"rights": "Creative Commons License"},
                "expected_type": LicenseType.CREATIVE_COMMONS,
                "expected_attribution": True
            }
        ]
        
        for case in test_cases:
            result = self.classifier.classify_url(case["url"], case["metadata"])
            assert result.license_type == case["expected_type"]
            assert result.attribution_required == case["expected_attribution"]
            assert result.confidence > 0.8
    
    def test_content_analysis(self):
        """Test content-based license detection"""
        test_cases = [
            {
                "content": "This image is licensed under CC BY 4.0",
                "url": "https://example.com/image.jpg",
                "expected_adjustment": "positive"
            },
            {
                "content": "© 2024 All Rights Reserved. Proprietary content.",
                "url": "https://example.com/image.jpg", 
                "expected_adjustment": "negative"
            },
            {
                "content": "Please provide attribution when using this image",
                "url": "https://example.com/image.jpg",
                "expected_adjustment": "slight_positive"
            }
        ]
        
        for case in test_cases:
            result = self.classifier.classify_content(
                case["content"], case["url"]
            )
            
            # Content analysis should influence the result
            assert "indicators" in result.details.lower() or "content" in result.details.lower()
            
            if case["expected_adjustment"] == "negative":
                # Strong copyright indicators should make it less research-safe
                assert result.research_safe is False or result.license_type == LicenseType.COPYRIGHTED
    
    def test_research_safety_methods(self):
        """Test research safety utility methods"""
        # Test research-safe content
        safe_result = LicenseInfo(
            license_type=LicenseType.CREATIVE_COMMONS,
            confidence=0.9,
            source_domain="unsplash.com",
            attribution_required=True,
            commercial_use_allowed=True,
            research_safe=True,
            details="Test"
        )
        
        assert self.classifier.is_research_safe(safe_result) is True
        assert self.classifier.requires_attribution(safe_result) is True
        
        # Test copyrighted content
        unsafe_result = LicenseInfo(
            license_type=LicenseType.COPYRIGHTED,
            confidence=0.9,
            source_domain="shutterstock.com",
            attribution_required=True,
            commercial_use_allowed=False,
            research_safe=False,
            details="Test"
        )
        
        assert self.classifier.is_research_safe(unsafe_result) is False
        assert self.classifier.requires_attribution(unsafe_result) is True
    
    def test_confidence_levels(self):
        """Test confidence level accuracy"""
        # High confidence cases
        high_confidence_urls = [
            "https://commons.wikimedia.org/wiki/File:Example.jpg",
            "https://shutterstock.com/image/example-123456"
        ]
        
        for url in high_confidence_urls:
            result = self.classifier.classify_url(url)
            assert result.confidence > 0.8
        
        # Lower confidence cases
        lower_confidence_urls = [
            "https://flickr.com/photos/user/123456/",  # Mixed licenses
            "https://unknown-domain.com/image.jpg"
        ]
        
        for url in lower_confidence_urls:
            result = self.classifier.classify_url(url)
            assert result.confidence <= 0.8
    
    def test_attribution_requirements(self):
        """Test attribution requirement detection"""
        # Attribution required
        attribution_required_urls = [
            "https://unsplash.com/photos/example",
            "https://commons.wikimedia.org/wiki/File:Example.jpg",
            "https://shutterstock.com/image/example-123456"
        ]
        
        for url in attribution_required_urls:
            result = self.classifier.classify_url(url)
            assert result.attribution_required is True
        
        # Attribution not required (CC0/Public Domain)
        no_attribution_urls = [
            "https://pixabay.com/photos/example-123456/"
        ]
        
        for url in no_attribution_urls:
            result = self.classifier.classify_url(url)
            assert result.attribution_required is False
    
    def test_commercial_use_permissions(self):
        """Test commercial use permission detection"""
        # Commercial use allowed
        commercial_allowed_urls = [
            "https://unsplash.com/photos/example",
            "https://pixabay.com/photos/example-123456/",
            "https://commons.wikimedia.org/wiki/File:Example.jpg"
        ]
        
        for url in commercial_allowed_urls:
            result = self.classifier.classify_url(url)
            assert result.commercial_use_allowed is True
        
        # Commercial use not allowed
        commercial_restricted_urls = [
            "https://deviantart.com/user/art/example-123456",
            "https://shutterstock.com/image/example-123456",
            "https://flickr.com/photos/user/123456/"
        ]
        
        for url in commercial_restricted_urls:
            result = self.classifier.classify_url(url)
            assert result.commercial_use_allowed is False
    
    def test_edge_cases(self):
        """Test edge cases and error handling"""
        # Invalid URLs
        invalid_urls = [
            "",
            "not-a-url",
            "ftp://example.com/file.jpg",
            "https://",
            None
        ]
        
        for url in invalid_urls:
            if url is None:
                continue
            result = self.classifier.classify_url(url)
            assert isinstance(result, LicenseInfo)
            assert result.license_type in [LicenseType.UNKNOWN, LicenseType.COPYRIGHTED]
        
        # Empty metadata
        result = self.classifier.classify_url("https://example.com/image.jpg", {})
        assert isinstance(result, LicenseInfo)
        
        # Empty content
        result = self.classifier.classify_content("", "https://example.com/image.jpg")
        assert isinstance(result, LicenseInfo)
    
    def test_license_info_dataclass(self):
        """Test LicenseInfo dataclass functionality"""
        info = LicenseInfo(
            license_type=LicenseType.CREATIVE_COMMONS,
            confidence=0.85,
            source_domain="unsplash.com",
            attribution_required=True,
            commercial_use_allowed=True,
            research_safe=True,
            details="Test license info"
        )
        
        assert info.license_type == LicenseType.CREATIVE_COMMONS
        assert info.confidence == 0.85
        assert info.source_domain == "unsplash.com"
        assert info.attribution_required is True
        assert info.commercial_use_allowed is True
        assert info.research_safe is True
        assert info.details == "Test license info"


class TestLicenseClassifierIntegration:
    """Integration tests for license classifier with real-world scenarios"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.classifier = LicenseClassifier()
    
    def test_research_workflow_compliance(self):
        """Test typical research workflow compliance scenarios"""
        # Scenario 1: Researcher wants only open source content
        open_source_urls = [
            "https://commons.wikimedia.org/wiki/File:Example.jpg",
            "https://pixabay.com/photos/example-123456/",
            "https://unsplash.com/photos/example"
        ]
        
        for url in open_source_urls:
            result = self.classifier.classify_url(url)
            assert result.research_safe is True
            assert result.license_type in [LicenseType.PUBLIC_DOMAIN, LicenseType.CREATIVE_COMMONS]
        
        # Scenario 2: Researcher needs to avoid copyrighted content
        copyrighted_urls = [
            "https://shutterstock.com/image/example-123456",
            "https://gettyimages.com/detail/example",
            "https://instagram.com/p/example123/"
        ]
        
        for url in copyrighted_urls:
            result = self.classifier.classify_url(url)
            assert result.research_safe is False
            assert result.license_type == LicenseType.COPYRIGHTED
    
    def test_mixed_source_dataset(self):
        """Test classification of mixed source datasets"""
        mixed_urls = [
            "https://commons.wikimedia.org/wiki/File:Example1.jpg",  # Public Domain
            "https://unsplash.com/photos/example2",                  # Creative Commons
            "https://deviantart.com/user/art/example3-123456",       # Fair Use Research
            "https://shutterstock.com/image/example4-123456",        # Copyrighted
            "https://unknown-site.com/example5.jpg"                 # Unknown
        ]
        
        results = [self.classifier.classify_url(url) for url in mixed_urls]
        
        # Should have variety of license types
        license_types = [r.license_type for r in results]
        assert LicenseType.PUBLIC_DOMAIN in license_types
        assert LicenseType.CREATIVE_COMMONS in license_types
        assert LicenseType.FAIR_USE_RESEARCH in license_types
        assert LicenseType.COPYRIGHTED in license_types
        
        # Research safe count should be appropriate
        research_safe_count = sum(1 for r in results if r.research_safe)
        assert research_safe_count >= 3  # First 3 should be research safe
    
    def test_compliance_mode_filtering(self):
        """Test filtering based on different compliance modes"""
        test_urls = [
            "https://commons.wikimedia.org/wiki/File:Example1.jpg",
            "https://unsplash.com/photos/example2", 
            "https://deviantart.com/user/art/example3-123456",
            "https://shutterstock.com/image/example4-123456"
        ]
        
        results = [self.classifier.classify_url(url) for url in test_urls]
        
        # Open Source Only mode (PUBLIC_DOMAIN + CREATIVE_COMMONS)
        open_source_safe = [r for r in results if r.license_type in 
                           [LicenseType.PUBLIC_DOMAIN, LicenseType.CREATIVE_COMMONS]]
        assert len(open_source_safe) == 2
        
        # Research Safe mode (includes FAIR_USE_RESEARCH)
        research_safe = [r for r in results if r.research_safe]
        assert len(research_safe) == 3
        
        # Full Dataset mode (includes everything, but flags copyrighted)
        copyrighted = [r for r in results if r.license_type == LicenseType.COPYRIGHTED]
        assert len(copyrighted) == 1
        assert not copyrighted[0].research_safe


if __name__ == "__main__":
    pytest.main([__file__])