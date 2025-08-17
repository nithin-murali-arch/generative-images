"""
License Classification Engine for Academic Multimodal LLM System

This module provides copyright-aware classification of content based on domain patterns,
metadata analysis, and license detection for research compliance.
"""

from enum import Enum
from typing import Dict, List, Optional, Tuple, Set
from urllib.parse import urlparse
import re
from dataclasses import dataclass


class LicenseType(Enum):
    """License classification types for research compliance"""
    PUBLIC_DOMAIN = "public_domain"
    CREATIVE_COMMONS = "creative_commons"
    FAIR_USE_RESEARCH = "fair_use_research"
    COPYRIGHTED = "copyrighted"
    UNKNOWN = "unknown"


@dataclass
class LicenseInfo:
    """Container for license classification results"""
    license_type: LicenseType
    confidence: float  # 0.0 to 1.0
    source_domain: str
    attribution_required: bool
    commercial_use_allowed: bool
    research_safe: bool
    details: str


class LicenseClassifier:
    """
    Domain-based license classifier for research compliance.
    
    Classifies content based on source domain patterns, metadata analysis,
    and known license indicators to ensure academic research compliance.
    """
    
    def __init__(self):
        self._domain_rules = self._initialize_domain_rules()
        self._cc_patterns = self._initialize_cc_patterns()
        self._copyright_indicators = self._initialize_copyright_indicators()
    
    def classify_url(self, url: str, metadata: Optional[Dict] = None) -> LicenseInfo:
        """
        Classify license type based on URL and optional metadata.
        
        Args:
            url: Source URL of the content
            metadata: Optional metadata dictionary with license information
            
        Returns:
            LicenseInfo with classification results
        """
        domain = self._extract_domain(url)
        
        # Check metadata first if available
        if metadata:
            metadata_result = self._classify_from_metadata(metadata, domain)
            if metadata_result.confidence > 0.8:
                return metadata_result
        
        # Domain-based classification
        domain_result = self._classify_from_domain(domain, url)
        
        # If we have both metadata and domain results, combine them
        if metadata and domain_result.confidence < 0.9:
            metadata_result = self._classify_from_metadata(metadata, domain)
            if metadata_result.confidence > domain_result.confidence:
                return metadata_result
        
        return domain_result
    
    def classify_content(self, content_text: str, url: str, 
                        metadata: Optional[Dict] = None) -> LicenseInfo:
        """
        Classify license based on content analysis, URL, and metadata.
        
        Args:
            content_text: Text content to analyze for license indicators
            url: Source URL
            metadata: Optional metadata dictionary
            
        Returns:
            LicenseInfo with classification results
        """
        # Start with URL-based classification
        base_result = self.classify_url(url, metadata)
        
        # Analyze content for license indicators
        content_indicators = self._analyze_content_indicators(content_text)
        
        # Adjust classification based on content analysis
        if content_indicators:
            return self._merge_classifications(base_result, content_indicators)
        
        return base_result
    
    def is_research_safe(self, license_info: LicenseInfo) -> bool:
        """Check if content is safe for academic research use"""
        return license_info.research_safe
    
    def requires_attribution(self, license_info: LicenseInfo) -> bool:
        """Check if content requires attribution"""
        return license_info.attribution_required
    
    def _extract_domain(self, url: str) -> str:
        """Extract domain from URL"""
        try:
            parsed = urlparse(url)
            return parsed.netloc.lower()
        except Exception:
            return ""
    
    def _initialize_domain_rules(self) -> Dict[str, Tuple[LicenseType, float, Dict]]:
        """Initialize domain-based classification rules"""
        return {
            # Public Domain sources
            "commons.wikimedia.org": (
                LicenseType.PUBLIC_DOMAIN, 0.95,
                {"attribution_required": True, "commercial_use_allowed": True, "research_safe": True}
            ),
            "archive.org": (
                LicenseType.PUBLIC_DOMAIN, 0.90,
                {"attribution_required": True, "commercial_use_allowed": True, "research_safe": True}
            ),
            "pixabay.com": (
                LicenseType.PUBLIC_DOMAIN, 0.85,
                {"attribution_required": False, "commercial_use_allowed": True, "research_safe": True}
            ),
            
            # Creative Commons sources
            "unsplash.com": (
                LicenseType.CREATIVE_COMMONS, 0.90,
                {"attribution_required": True, "commercial_use_allowed": True, "research_safe": True}
            ),
            "pexels.com": (
                LicenseType.CREATIVE_COMMONS, 0.90,
                {"attribution_required": True, "commercial_use_allowed": True, "research_safe": True}
            ),
            "flickr.com": (
                LicenseType.CREATIVE_COMMONS, 0.70,  # Mixed licenses on Flickr
                {"attribution_required": True, "commercial_use_allowed": False, "research_safe": True}
            ),
            
            # Fair Use Research sources
            "deviantart.com": (
                LicenseType.FAIR_USE_RESEARCH, 0.80,
                {"attribution_required": True, "commercial_use_allowed": False, "research_safe": True}
            ),
            "artstation.com": (
                LicenseType.FAIR_USE_RESEARCH, 0.80,
                {"attribution_required": True, "commercial_use_allowed": False, "research_safe": True}
            ),
            "behance.net": (
                LicenseType.FAIR_USE_RESEARCH, 0.80,
                {"attribution_required": True, "commercial_use_allowed": False, "research_safe": True}
            ),
            "youtube.com": (
                LicenseType.FAIR_USE_RESEARCH, 0.75,
                {"attribution_required": True, "commercial_use_allowed": False, "research_safe": True}
            ),
            "youtu.be": (
                LicenseType.FAIR_USE_RESEARCH, 0.75,
                {"attribution_required": True, "commercial_use_allowed": False, "research_safe": True}
            ),
            "vimeo.com": (
                LicenseType.FAIR_USE_RESEARCH, 0.75,
                {"attribution_required": True, "commercial_use_allowed": False, "research_safe": True}
            ),
            
            # Copyrighted sources
            "shutterstock.com": (
                LicenseType.COPYRIGHTED, 0.95,
                {"attribution_required": True, "commercial_use_allowed": False, "research_safe": False}
            ),
            "gettyimages.com": (
                LicenseType.COPYRIGHTED, 0.95,
                {"attribution_required": True, "commercial_use_allowed": False, "research_safe": False}
            ),
            "adobe.stock.com": (
                LicenseType.COPYRIGHTED, 0.95,
                {"attribution_required": True, "commercial_use_allowed": False, "research_safe": False}
            ),
            "instagram.com": (
                LicenseType.COPYRIGHTED, 0.85,
                {"attribution_required": True, "commercial_use_allowed": False, "research_safe": False}
            ),
            "facebook.com": (
                LicenseType.COPYRIGHTED, 0.85,
                {"attribution_required": True, "commercial_use_allowed": False, "research_safe": False}
            ),
            "twitter.com": (
                LicenseType.COPYRIGHTED, 0.80,
                {"attribution_required": True, "commercial_use_allowed": False, "research_safe": False}
            ),
            "x.com": (
                LicenseType.COPYRIGHTED, 0.80,
                {"attribution_required": True, "commercial_use_allowed": False, "research_safe": False}
            ),
        }
    
    def _initialize_cc_patterns(self) -> List[Tuple[re.Pattern, LicenseType, Dict]]:
        """Initialize Creative Commons license patterns"""
        patterns = []
        
        # CC license patterns
        cc_patterns = [
            (r"cc0|public domain", LicenseType.PUBLIC_DOMAIN, 
             {"attribution_required": False, "commercial_use_allowed": True}),
            (r"cc by\b", LicenseType.CREATIVE_COMMONS,
             {"attribution_required": True, "commercial_use_allowed": True}),
            (r"cc by-sa", LicenseType.CREATIVE_COMMONS,
             {"attribution_required": True, "commercial_use_allowed": True}),
            (r"cc by-nc", LicenseType.CREATIVE_COMMONS,
             {"attribution_required": True, "commercial_use_allowed": False}),
            (r"creative commons", LicenseType.CREATIVE_COMMONS,
             {"attribution_required": True, "commercial_use_allowed": False}),
        ]
        
        for pattern_str, license_type, attrs in cc_patterns:
            pattern = re.compile(pattern_str, re.IGNORECASE)
            attrs.update({"research_safe": True})
            patterns.append((pattern, license_type, attrs))
        
        return patterns
    
    def _initialize_copyright_indicators(self) -> List[Tuple[re.Pattern, float]]:
        """Initialize copyright indicator patterns"""
        indicators = [
            (re.compile(r"©|\(c\)|copyright", re.IGNORECASE), -0.3),
            (re.compile(r"all rights reserved", re.IGNORECASE), -0.4),
            (re.compile(r"proprietary|trademark", re.IGNORECASE), -0.3),
            (re.compile(r"license|licensed", re.IGNORECASE), 0.2),
            (re.compile(r"attribution|credit", re.IGNORECASE), 0.1),
        ]
        return indicators
    
    def _classify_from_domain(self, domain: str, url: str) -> LicenseInfo:
        """Classify based on domain rules"""
        # Direct domain match
        if domain in self._domain_rules:
            license_type, confidence, attrs = self._domain_rules[domain]
            return LicenseInfo(
                license_type=license_type,
                confidence=confidence,
                source_domain=domain,
                attribution_required=attrs["attribution_required"],
                commercial_use_allowed=attrs["commercial_use_allowed"],
                research_safe=attrs["research_safe"],
                details=f"Domain-based classification: {domain}"
            )
        
        # Subdomain matching and related domain patterns
        for known_domain, (license_type, confidence, attrs) in self._domain_rules.items():
            if domain.endswith(known_domain):
                return LicenseInfo(
                    license_type=license_type,
                    confidence=confidence * 0.8,  # Lower confidence for subdomain
                    source_domain=domain,
                    attribution_required=attrs["attribution_required"],
                    commercial_use_allowed=attrs["commercial_use_allowed"],
                    research_safe=attrs["research_safe"],
                    details=f"Subdomain match: {domain} -> {known_domain}"
                )
        
        # Special case for Wikipedia domains (all should be treated as Wikimedia)
        if "wikipedia.org" in domain:
            return LicenseInfo(
                license_type=LicenseType.PUBLIC_DOMAIN,
                confidence=0.85,
                source_domain=domain,
                attribution_required=True,
                commercial_use_allowed=True,
                research_safe=True,
                details=f"Wikipedia domain match: {domain}"
            )
        
        # Unknown domain - default to copyrighted for safety
        return LicenseInfo(
            license_type=LicenseType.UNKNOWN,
            confidence=0.5,
            source_domain=domain,
            attribution_required=True,
            commercial_use_allowed=False,
            research_safe=False,
            details=f"Unknown domain: {domain}"
        )
    
    def _classify_from_metadata(self, metadata: Dict, domain: str) -> LicenseInfo:
        """Classify based on metadata analysis"""
        license_text = ""
        
        # Collect license-related metadata
        license_fields = ["license", "rights", "copyright", "usage_rights", "license_url"]
        for field in license_fields:
            if field in metadata and metadata[field]:
                license_text += f" {metadata[field]}"
        
        if not license_text.strip():
            return LicenseInfo(
                license_type=LicenseType.UNKNOWN,
                confidence=0.3,
                source_domain=domain,
                attribution_required=True,
                commercial_use_allowed=False,
                research_safe=False,
                details="No license metadata found"
            )
        
        # Check for copyright indicators first
        copyright_indicators = ["©", "(c)", "copyright", "all rights reserved", "proprietary"]
        license_lower = license_text.lower()
        
        if any(indicator in license_lower for indicator in copyright_indicators):
            return LicenseInfo(
                license_type=LicenseType.COPYRIGHTED,
                confidence=0.90,
                source_domain=domain,
                attribution_required=True,
                commercial_use_allowed=False,
                research_safe=False,
                details=f"Copyright metadata detected: {license_text[:100]}"
            )
        
        # Check against CC patterns
        for pattern, license_type, attrs in self._cc_patterns:
            if pattern.search(license_text):
                return LicenseInfo(
                    license_type=license_type,
                    confidence=0.85,
                    source_domain=domain,
                    attribution_required=attrs["attribution_required"],
                    commercial_use_allowed=attrs["commercial_use_allowed"],
                    research_safe=attrs["research_safe"],
                    details=f"Metadata license match: {pattern.pattern}"
                )
        
        # Default for metadata without clear license
        return LicenseInfo(
            license_type=LicenseType.UNKNOWN,
            confidence=0.4,
            source_domain=domain,
            attribution_required=True,
            commercial_use_allowed=False,
            research_safe=False,
            details=f"Metadata found but no clear license: {license_text[:100]}"
        )
    
    def _analyze_content_indicators(self, content: str) -> Optional[Dict]:
        """Analyze content for copyright indicators"""
        if not content:
            return None
        
        confidence_adjustment = 0.0
        indicators_found = []
        content_text = content.lower()
        
        # Add the actual content text for pattern matching
        indicators_found.append(content)
        
        for pattern, adjustment in self._copyright_indicators:
            if pattern.search(content):
                confidence_adjustment += adjustment
                indicators_found.append(pattern.pattern)
        
        if indicators_found:
            return {
                "confidence_adjustment": confidence_adjustment,
                "indicators": indicators_found
            }
        
        return None
    
    def _merge_classifications(self, base_result: LicenseInfo, 
                             content_indicators: Dict) -> LicenseInfo:
        """Merge base classification with content analysis"""
        adjusted_confidence = max(0.1, min(1.0, 
            base_result.confidence + content_indicators["confidence_adjustment"]))
        
        # Check if content contains strong license indicators
        content_text = " ".join(content_indicators.get("indicators", []))
        
        # Check for Creative Commons in content
        for pattern, license_type, attrs in self._cc_patterns:
            if pattern.search(content_text):
                return LicenseInfo(
                    license_type=license_type,
                    confidence=0.85,
                    source_domain=base_result.source_domain,
                    attribution_required=attrs["attribution_required"],
                    commercial_use_allowed=attrs["commercial_use_allowed"],
                    research_safe=attrs["research_safe"],
                    details=f"Content license match: {pattern.pattern}; {base_result.details}"
                )
        
        # If content indicates strong copyright, be more conservative
        if content_indicators["confidence_adjustment"] < -0.2:
            research_safe = False
            license_type = LicenseType.COPYRIGHTED
        else:
            research_safe = base_result.research_safe
            license_type = base_result.license_type
        
        return LicenseInfo(
            license_type=license_type,
            confidence=adjusted_confidence,
            source_domain=base_result.source_domain,
            attribution_required=base_result.attribution_required,
            commercial_use_allowed=base_result.commercial_use_allowed,
            research_safe=research_safe,
            details=f"{base_result.details}; Content indicators: {content_indicators['indicators']}"
        )