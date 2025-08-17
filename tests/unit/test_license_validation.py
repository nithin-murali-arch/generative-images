"""
Validation tests for License Classification Engine

Tests license classification accuracy with different content sources,
edge cases, and real-world scenarios for research compliance validation.
"""

import pytest
from src.data.license_classifier import LicenseClassifier, LicenseType, LicenseInfo


class TestLicenseValidation:
    """Validation tests for license classifier with diverse content sources"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.classifier = LicenseClassifier()
    
    def test_academic_source_validation(self):
        """Test classification of academic and educational sources"""
        academic_sources = [
            {
                "url": "https://commons.wikimedia.org/wiki/File:Academic_diagram.svg",
                "expected_type": LicenseType.PUBLIC_DOMAIN,
                "expected_research_safe": True,
                "description": "Wikimedia Commons academic content"
            },
            {
                "url": "https://archive.org/details/academic_paper_2024",
                "expected_type": LicenseType.PUBLIC_DOMAIN,
                "expected_research_safe": True,
                "description": "Internet Archive academic paper"
            },
            {
                "url": "https://en.wikipedia.org/wiki/Machine_Learning",
                "expected_type": LicenseType.PUBLIC_DOMAIN,
                "expected_research_safe": True,
                "description": "Wikipedia educational content"
            }
        ]
        
        for source in academic_sources:
            result = self.classifier.classify_url(source["url"])
            assert result.license_type == source["expected_type"], \
                f"Failed for {source['description']}: expected {source['expected_type']}, got {result.license_type}"
            assert result.research_safe == source["expected_research_safe"], \
                f"Research safety failed for {source['description']}"
            assert result.confidence > 0.8, \
                f"Low confidence for {source['description']}: {result.confidence}"
    
    def test_creative_platform_validation(self):
        """Test classification of creative platforms with mixed licenses"""
        creative_platforms = [
            {
                "url": "https://unsplash.com/photos/nature-landscape-abc123",
                "expected_type": LicenseType.CREATIVE_COMMONS,
                "expected_research_safe": True,
                "attribution_required": True
            },
            {
                "url": "https://pexels.com/photo/abstract-art-456789/",
                "expected_type": LicenseType.CREATIVE_COMMONS,
                "expected_research_safe": True,
                "attribution_required": True
            },
            {
                "url": "https://pixabay.com/illustrations/digital-art-123456/",
                "expected_type": LicenseType.PUBLIC_DOMAIN,
                "expected_research_safe": True,
                "attribution_required": False
            },
            {
                "url": "https://flickr.com/photos/photographer/987654321/",
                "expected_type": LicenseType.CREATIVE_COMMONS,
                "expected_research_safe": True,
                "attribution_required": True
            }
        ]
        
        for platform in creative_platforms:
            result = self.classifier.classify_url(platform["url"])
            assert result.license_type == platform["expected_type"]
            assert result.research_safe == platform["expected_research_safe"]
            assert result.attribution_required == platform["attribution_required"]
    
    def test_social_media_validation(self):
        """Test classification of social media platforms"""
        social_media_sources = [
            {
                "url": "https://instagram.com/p/ABC123xyz/",
                "expected_type": LicenseType.COPYRIGHTED,
                "expected_research_safe": False,
                "description": "Instagram post"
            },
            {
                "url": "https://twitter.com/user/status/1234567890",
                "expected_type": LicenseType.COPYRIGHTED,
                "expected_research_safe": False,
                "description": "Twitter post"
            },
            {
                "url": "https://x.com/artist/status/9876543210",
                "expected_type": LicenseType.COPYRIGHTED,
                "expected_research_safe": False,
                "description": "X (Twitter) post"
            },
            {
                "url": "https://facebook.com/photo.php?fbid=123456789",
                "expected_type": LicenseType.COPYRIGHTED,
                "expected_research_safe": False,
                "description": "Facebook photo"
            }
        ]
        
        for source in social_media_sources:
            result = self.classifier.classify_url(source["url"])
            assert result.license_type == source["expected_type"], \
                f"Failed for {source['description']}"
            assert result.research_safe == source["expected_research_safe"], \
                f"Research safety failed for {source['description']}"
            assert result.attribution_required is True, \
                f"Attribution should be required for {source['description']}"
    
    def test_commercial_stock_validation(self):
        """Test classification of commercial stock photo sites"""
        commercial_sources = [
            {
                "url": "https://shutterstock.com/image-photo/business-meeting-123456789",
                "expected_type": LicenseType.COPYRIGHTED,
                "expected_research_safe": False,
                "commercial_allowed": False
            },
            {
                "url": "https://gettyimages.com/detail/photo/corporate-team-987654321",
                "expected_type": LicenseType.COPYRIGHTED,
                "expected_research_safe": False,
                "commercial_allowed": False
            },
            {
                "url": "https://adobe.stock.com/images/technology-concept/456789123",
                "expected_type": LicenseType.COPYRIGHTED,
                "expected_research_safe": False,
                "commercial_allowed": False
            }
        ]
        
        for source in commercial_sources:
            result = self.classifier.classify_url(source["url"])
            assert result.license_type == source["expected_type"]
            assert result.research_safe == source["expected_research_safe"]
            assert result.commercial_use_allowed == source["commercial_allowed"]
            assert result.confidence > 0.8
    
    def test_art_community_validation(self):
        """Test classification of art community platforms"""
        art_platforms = [
            {
                "url": "https://deviantart.com/artist/art/Digital-Painting-123456789",
                "expected_type": LicenseType.FAIR_USE_RESEARCH,
                "expected_research_safe": True,
                "commercial_allowed": False
            },
            {
                "url": "https://artstation.com/artwork/concept-art-xyz123",
                "expected_type": LicenseType.FAIR_USE_RESEARCH,
                "expected_research_safe": True,
                "commercial_allowed": False
            },
            {
                "url": "https://behance.net/gallery/12345678/Creative-Project",
                "expected_type": LicenseType.FAIR_USE_RESEARCH,
                "expected_research_safe": True,
                "commercial_allowed": False
            }
        ]
        
        for platform in art_platforms:
            result = self.classifier.classify_url(platform["url"])
            assert result.license_type == platform["expected_type"]
            assert result.research_safe == platform["expected_research_safe"]
            assert result.commercial_use_allowed == platform["commercial_allowed"]
    
    def test_video_platform_validation(self):
        """Test classification of video platforms"""
        video_sources = [
            {
                "url": "https://youtube.com/watch?v=dQw4w9WgXcQ",
                "expected_type": LicenseType.FAIR_USE_RESEARCH,
                "expected_research_safe": True,
                "description": "YouTube video"
            },
            {
                "url": "https://youtu.be/dQw4w9WgXcQ",
                "expected_type": LicenseType.FAIR_USE_RESEARCH,
                "expected_research_safe": True,
                "description": "YouTube short URL"
            },
            {
                "url": "https://vimeo.com/123456789",
                "expected_type": LicenseType.FAIR_USE_RESEARCH,
                "expected_research_safe": True,
                "description": "Vimeo video"
            }
        ]
        
        for source in video_sources:
            result = self.classifier.classify_url(source["url"])
            assert result.license_type == source["expected_type"], \
                f"Failed for {source['description']}"
            assert result.research_safe == source["expected_research_safe"], \
                f"Research safety failed for {source['description']}"
    
    def test_metadata_override_validation(self):
        """Test metadata override scenarios"""
        test_cases = [
            {
                "url": "https://unknown-site.com/image.jpg",
                "metadata": {"license": "CC0", "rights": "Public Domain"},
                "expected_type": LicenseType.PUBLIC_DOMAIN,
                "expected_confidence_boost": True
            },
            {
                "url": "https://example.com/photo.png",
                "metadata": {"license": "CC BY-SA 4.0", "attribution": "Required"},
                "expected_type": LicenseType.CREATIVE_COMMONS,
                "expected_confidence_boost": True
            },
            {
                "url": "https://random-blog.net/artwork.gif",
                "metadata": {"copyright": "© 2024 Artist Name", "rights": "All Rights Reserved"},
                "expected_type": LicenseType.COPYRIGHTED,
                "expected_research_safe": False
            }
        ]
        
        for case in test_cases:
            result_with_metadata = self.classifier.classify_url(case["url"], case["metadata"])
            result_without_metadata = self.classifier.classify_url(case["url"])
            
            assert result_with_metadata.license_type == case["expected_type"]
            
            if case.get("expected_confidence_boost"):
                assert result_with_metadata.confidence > result_without_metadata.confidence
    
    def test_content_analysis_validation(self):
        """Test content-based license detection"""
        content_scenarios = [
            {
                "content": "This work is licensed under a Creative Commons Attribution 4.0 International License.",
                "url": "https://example.com/image.jpg",
                "expected_type": LicenseType.CREATIVE_COMMONS,
                "expected_research_safe": True
            },
            {
                "content": "Image released under CC0 - No Rights Reserved. Free for any use.",
                "url": "https://example.com/photo.png",
                "expected_type": LicenseType.PUBLIC_DOMAIN,
                "expected_attribution": False
            },
            {
                "content": "© 2024 Professional Photography Studio. All rights reserved. Unauthorized use prohibited.",
                "url": "https://example.com/professional.jpg",
                "expected_research_safe": False,
                "expected_copyright_detected": True
            },
            {
                "content": "Please provide attribution when using this image in your research.",
                "url": "https://example.com/research-image.jpg",
                "expected_attribution": True,
                "expected_positive_indicator": True
            }
        ]
        
        for scenario in content_scenarios:
            result = self.classifier.classify_content(
                scenario["content"], scenario["url"]
            )
            
            if "expected_type" in scenario:
                assert result.license_type == scenario["expected_type"]
            
            if "expected_research_safe" in scenario:
                assert result.research_safe == scenario["expected_research_safe"]
            
            if "expected_attribution" in scenario:
                assert result.attribution_required == scenario["expected_attribution"]
            
            if scenario.get("expected_copyright_detected"):
                assert "copyright" in result.details.lower() or result.license_type == LicenseType.COPYRIGHTED
    
    def test_edge_case_validation(self):
        """Test edge cases and unusual scenarios"""
        edge_cases = [
            {
                "url": "https://subdomain.commons.wikimedia.org/wiki/File:Test.jpg",
                "expected_research_safe": True,
                "description": "Wikimedia subdomain"
            },
            {
                "url": "https://images.unsplash.com/photo-123456789",
                "expected_type": LicenseType.CREATIVE_COMMONS,
                "description": "Unsplash CDN URL"
            },
            {
                "url": "https://cdn.pixabay.com/photo/2024/01/01/image.jpg",
                "expected_type": LicenseType.PUBLIC_DOMAIN,
                "description": "Pixabay CDN URL"
            },
            {
                "url": "https://completely-unknown-domain-12345.com/image.jpg",
                "expected_type": LicenseType.UNKNOWN,
                "expected_research_safe": False,
                "description": "Completely unknown domain"
            }
        ]
        
        for case in edge_cases:
            result = self.classifier.classify_url(case["url"])
            
            if "expected_type" in case:
                assert result.license_type == case["expected_type"], \
                    f"Failed for {case['description']}"
            
            if "expected_research_safe" in case:
                assert result.research_safe == case["expected_research_safe"], \
                    f"Research safety failed for {case['description']}"
    
    def test_confidence_calibration(self):
        """Test confidence score calibration across different scenarios"""
        confidence_tests = [
            {
                "url": "https://commons.wikimedia.org/wiki/File:Test.jpg",
                "min_confidence": 0.9,
                "description": "High confidence - well-known public domain"
            },
            {
                "url": "https://shutterstock.com/image/test-123456",
                "min_confidence": 0.9,
                "description": "High confidence - well-known copyrighted"
            },
            {
                "url": "https://flickr.com/photos/user/123456/",
                "max_confidence": 0.8,
                "description": "Medium confidence - mixed license platform"
            },
            {
                "url": "https://unknown-domain.com/image.jpg",
                "max_confidence": 0.6,
                "description": "Low confidence - unknown domain"
            }
        ]
        
        for test in confidence_tests:
            result = self.classifier.classify_url(test["url"])
            
            if "min_confidence" in test:
                assert result.confidence >= test["min_confidence"], \
                    f"Confidence too low for {test['description']}: {result.confidence}"
            
            if "max_confidence" in test:
                assert result.confidence <= test["max_confidence"], \
                    f"Confidence too high for {test['description']}: {result.confidence}"
    
    def test_research_compliance_scenarios(self):
        """Test specific research compliance scenarios"""
        compliance_scenarios = [
            {
                "name": "Open Source Only Dataset",
                "urls": [
                    "https://commons.wikimedia.org/wiki/File:Example1.jpg",
                    "https://pixabay.com/photos/example2-123456/",
                    "https://unsplash.com/photos/example3"
                ],
                "expected_all_research_safe": True,
                "expected_types": [LicenseType.PUBLIC_DOMAIN, LicenseType.CREATIVE_COMMONS]
            },
            {
                "name": "Mixed Research Dataset",
                "urls": [
                    "https://commons.wikimedia.org/wiki/File:Example1.jpg",
                    "https://deviantart.com/user/art/example2-123456",
                    "https://shutterstock.com/image/example3-789012"
                ],
                "expected_research_safe_count": 2,
                "expected_copyrighted_count": 1
            },
            {
                "name": "Commercial Dataset (Problematic)",
                "urls": [
                    "https://shutterstock.com/image/example1-123456",
                    "https://gettyimages.com/detail/example2",
                    "https://instagram.com/p/example3/"
                ],
                "expected_all_research_safe": False,
                "expected_all_copyrighted": True
            }
        ]
        
        for scenario in compliance_scenarios:
            results = [self.classifier.classify_url(url) for url in scenario["urls"]]
            
            if scenario.get("expected_all_research_safe"):
                assert all(r.research_safe for r in results), \
                    f"Not all URLs research safe in {scenario['name']}"
            
            if "expected_research_safe_count" in scenario:
                research_safe_count = sum(1 for r in results if r.research_safe)
                assert research_safe_count == scenario["expected_research_safe_count"], \
                    f"Wrong research safe count in {scenario['name']}"
            
            if "expected_copyrighted_count" in scenario:
                copyrighted_count = sum(1 for r in results if r.license_type == LicenseType.COPYRIGHTED)
                assert copyrighted_count == scenario["expected_copyrighted_count"], \
                    f"Wrong copyrighted count in {scenario['name']}"
            
            if scenario.get("expected_all_copyrighted"):
                assert all(r.license_type == LicenseType.COPYRIGHTED for r in results), \
                    f"Not all URLs copyrighted in {scenario['name']}"
            
            if "expected_types" in scenario:
                license_types = [r.license_type for r in results]
                for expected_type in scenario["expected_types"]:
                    assert expected_type in license_types, \
                        f"Expected type {expected_type} not found in {scenario['name']}"


if __name__ == "__main__":
    pytest.main([__file__])