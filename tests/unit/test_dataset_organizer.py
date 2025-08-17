"""
Unit tests for Dataset Organization System

Tests dataset organization, license-based filtering, metadata tracking,
and compliance mode enforcement for research datasets.
"""

import pytest
import tempfile
import shutil
import json
import sqlite3
from pathlib import Path
from datetime import datetime
from typing import List
from unittest.mock import patch, MagicMock

from src.data.dataset_organizer import (
    DatasetOrganizer, DatasetConfig, ComplianceMode, DatasetStats, OrganizedDataset
)
from src.data.research_crawler import ContentItem, CrawlResult
from src.data.license_classifier import LicenseType, LicenseInfo


class TestDatasetConfig:
    """Test dataset configuration"""
    
    def test_default_config(self):
        """Test default configuration values"""
        config = DatasetConfig()
        
        assert config.base_path == Path("./datasets")
        assert config.compliance_mode == ComplianceMode.RESEARCH_SAFE
        assert config.preserve_attribution is True
        assert config.create_license_reports is True
        assert config.max_items_per_category is None
        assert config.include_metadata is True
        assert config.create_train_val_split is True
        assert config.validation_split == 0.2
        assert config.random_seed == 42
    
    def test_custom_config(self):
        """Test custom configuration"""
        config = DatasetConfig(
            base_path=Path("/custom/path"),
            compliance_mode=ComplianceMode.OPEN_SOURCE_ONLY,
            preserve_attribution=False,
            max_items_per_category=100,
            validation_split=0.3
        )
        
        assert config.base_path == Path("/custom/path")
        assert config.compliance_mode == ComplianceMode.OPEN_SOURCE_ONLY
        assert config.preserve_attribution is False
        assert config.max_items_per_category == 100
        assert config.validation_split == 0.3


class TestComplianceMode:
    """Test compliance mode enumeration"""
    
    def test_compliance_mode_values(self):
        """Test compliance mode enum values"""
        assert ComplianceMode.OPEN_SOURCE_ONLY.value == "open_source_only"
        assert ComplianceMode.RESEARCH_SAFE.value == "research_safe"
        assert ComplianceMode.FULL_DATASET.value == "full_dataset"


class TestDatasetOrganizer:
    """Test dataset organizer functionality"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.config = DatasetConfig(base_path=self.temp_dir)
        self.organizer = DatasetOrganizer(self.config)
        
        # Create sample content items
        self.sample_items = self._create_sample_items()
    
    def teardown_method(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def _create_sample_items(self) -> List[ContentItem]:
        """Create sample content items for testing"""
        items = []
        
        # Public domain item
        item1 = ContentItem(
            url="https://commons.wikimedia.org/wiki/File:Test1.jpg",
            content_type="image/jpeg",
            content_size=1024,
            title="Test Image 1",
            source_domain="commons.wikimedia.org",
            content_hash="hash1234567890abcdef",
            research_safe=True,
            attribution="Wikimedia Commons"
        )
        item1.license_info = LicenseInfo(
            license_type=LicenseType.PUBLIC_DOMAIN,
            confidence=0.95,
            source_domain="commons.wikimedia.org",
            attribution_required=True,
            commercial_use_allowed=True,
            research_safe=True,
            details="Public domain"
        )
        
        # Create temporary file for item1
        item1_file = self.temp_dir / "temp1.jpg"
        item1_file.write_bytes(b"fake image data 1")
        item1.local_path = item1_file
        
        items.append(item1)
        
        # Creative Commons item
        item2 = ContentItem(
            url="https://unsplash.com/photos/test2",
            content_type="image/png",
            content_size=2048,
            title="Test Image 2",
            source_domain="unsplash.com",
            content_hash="hash2345678901bcdefg",
            research_safe=True,
            attribution="Unsplash Photographer"
        )
        item2.license_info = LicenseInfo(
            license_type=LicenseType.CREATIVE_COMMONS,
            confidence=0.90,
            source_domain="unsplash.com",
            attribution_required=True,
            commercial_use_allowed=True,
            research_safe=True,
            details="CC BY 4.0"
        )
        
        # Create temporary file for item2
        item2_file = self.temp_dir / "temp2.png"
        item2_file.write_bytes(b"fake image data 2")
        item2.local_path = item2_file
        
        items.append(item2)
        
        # Fair use research item
        item3 = ContentItem(
            url="https://deviantart.com/artist/art/test3-123456",
            content_type="image/gif",
            content_size=512,
            title="Test Art 3",
            source_domain="deviantart.com",
            content_hash="hash3456789012cdefgh",
            research_safe=True,
            attribution="DeviantArt Artist"
        )
        item3.license_info = LicenseInfo(
            license_type=LicenseType.FAIR_USE_RESEARCH,
            confidence=0.80,
            source_domain="deviantart.com",
            attribution_required=True,
            commercial_use_allowed=False,
            research_safe=True,
            details="Fair use research"
        )
        
        # Create temporary file for item3
        item3_file = self.temp_dir / "temp3.gif"
        item3_file.write_bytes(b"fake image data 3")
        item3.local_path = item3_file
        
        items.append(item3)
        
        # Copyrighted item
        item4 = ContentItem(
            url="https://shutterstock.com/image/test4-789012",
            content_type="image/jpeg",
            content_size=4096,
            title="Test Stock Photo",
            source_domain="shutterstock.com",
            content_hash="hash4567890123defghi",
            research_safe=False,
            attribution="Shutterstock"
        )
        item4.license_info = LicenseInfo(
            license_type=LicenseType.COPYRIGHTED,
            confidence=0.95,
            source_domain="shutterstock.com",
            attribution_required=True,
            commercial_use_allowed=False,
            research_safe=False,
            details="Copyrighted"
        )
        
        # Create temporary file for item4
        item4_file = self.temp_dir / "temp4.jpg"
        item4_file.write_bytes(b"fake image data 4")
        item4.local_path = item4_file
        
        items.append(item4)
        
        return items
    
    def test_organizer_initialization(self):
        """Test organizer initialization"""
        assert self.organizer.config == self.config
        assert self.temp_dir.exists()
    
    def test_filter_by_compliance_open_source_only(self):
        """Test filtering for open source only compliance mode"""
        self.config.compliance_mode = ComplianceMode.OPEN_SOURCE_ONLY
        
        filtered = self.organizer._filter_by_compliance(self.sample_items)
        
        # Should include only public domain and creative commons
        assert len(filtered) == 2
        license_types = [item.license_info.license_type for item in filtered]
        assert LicenseType.PUBLIC_DOMAIN in license_types
        assert LicenseType.CREATIVE_COMMONS in license_types
        assert LicenseType.FAIR_USE_RESEARCH not in license_types
        assert LicenseType.COPYRIGHTED not in license_types
    
    def test_filter_by_compliance_research_safe(self):
        """Test filtering for research safe compliance mode"""
        self.config.compliance_mode = ComplianceMode.RESEARCH_SAFE
        
        filtered = self.organizer._filter_by_compliance(self.sample_items)
        
        # Should include research safe items (first 3)
        assert len(filtered) == 3
        research_safe_count = sum(1 for item in filtered if item.research_safe)
        assert research_safe_count == 3
    
    def test_filter_by_compliance_full_dataset(self):
        """Test filtering for full dataset compliance mode"""
        self.config.compliance_mode = ComplianceMode.FULL_DATASET
        
        filtered = self.organizer._filter_by_compliance(self.sample_items)
        
        # Should include all items
        assert len(filtered) == 4
    
    def test_organize_by_license(self):
        """Test organization by license type"""
        dataset_path = self.temp_dir / "test_dataset"
        dataset_path.mkdir()
        
        organized = self.organizer._organize_by_license(self.sample_items, dataset_path)
        
        # Check organization structure
        assert LicenseType.PUBLIC_DOMAIN in organized
        assert LicenseType.CREATIVE_COMMONS in organized
        assert LicenseType.FAIR_USE_RESEARCH in organized
        assert LicenseType.COPYRIGHTED in organized
        
        # Check directory creation
        assert (dataset_path / "public_domain").exists()
        assert (dataset_path / "creative_commons").exists()
        assert (dataset_path / "fair_use_research").exists()
        assert (dataset_path / "copyrighted").exists()
        
        # Check file copying
        public_domain_files = list((dataset_path / "public_domain").glob("*"))
        assert len(public_domain_files) >= 1  # At least the copied file
    
    def test_organize_by_license_with_attribution(self):
        """Test organization with attribution preservation"""
        self.config.preserve_attribution = True
        dataset_path = self.temp_dir / "test_dataset"
        dataset_path.mkdir()
        
        organized = self.organizer._organize_by_license(self.sample_items, dataset_path)
        
        # Check attribution files
        public_domain_dir = dataset_path / "public_domain"
        attribution_files = list(public_domain_dir.glob("*_attribution.txt"))
        assert len(attribution_files) >= 1
        
        # Check attribution content
        with open(attribution_files[0], "r") as f:
            content = f.read()
            assert "Wikimedia Commons" in content
    
    def test_organize_by_license_max_items_limit(self):
        """Test organization with max items per category limit"""
        self.config.max_items_per_category = 1
        dataset_path = self.temp_dir / "test_dataset"
        dataset_path.mkdir()
        
        # Create multiple items of same license type
        extra_items = []
        for i in range(3):
            item = ContentItem(
                url=f"https://commons.wikimedia.org/wiki/File:Extra{i}.jpg",
                content_type="image/jpeg",
                content_size=1024,
                source_domain="commons.wikimedia.org",
                content_hash=f"extrahash{i}",
                research_safe=True
            )
            item.license_info = LicenseInfo(
                license_type=LicenseType.PUBLIC_DOMAIN,
                confidence=0.95,
                source_domain="commons.wikimedia.org",
                attribution_required=True,
                commercial_use_allowed=True,
                research_safe=True,
                details="Public domain"
            )
            extra_items.append(item)
        
        all_items = [self.sample_items[0]] + extra_items  # Only public domain items
        organized = self.organizer._organize_by_license(all_items, dataset_path)
        
        # Should only have 1 item due to limit
        assert len(organized[LicenseType.PUBLIC_DOMAIN]) == 1
    
    def test_create_metadata_database(self):
        """Test metadata database creation"""
        dataset_path = self.temp_dir / "test_dataset"
        dataset_path.mkdir()
        
        organized = {
            LicenseType.PUBLIC_DOMAIN: [self.sample_items[0]],
            LicenseType.CREATIVE_COMMONS: [self.sample_items[1]]
        }
        
        db_path = self.organizer._create_metadata_database(organized, dataset_path)
        
        assert db_path.exists()
        assert db_path.name == "metadata.db"
        
        # Check database content
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM content_items")
        count = cursor.fetchone()[0]
        assert count == 2
        
        cursor.execute("SELECT url, license_type FROM content_items")
        rows = cursor.fetchall()
        urls = [row[0] for row in rows]
        assert "https://commons.wikimedia.org/wiki/File:Test1.jpg" in urls
        assert "https://unsplash.com/photos/test2" in urls
        
        conn.close()
    
    def test_create_train_val_splits(self):
        """Test train/validation split creation"""
        dataset_path = self.temp_dir / "test_dataset"
        dataset_path.mkdir()
        
        organized = {
            LicenseType.PUBLIC_DOMAIN: self.sample_items[:2],
            LicenseType.CREATIVE_COMMONS: self.sample_items[2:4]
        }
        
        train_path, val_path = self.organizer._create_train_val_splits(organized, dataset_path)
        
        assert train_path.exists()
        assert val_path.exists()
        
        # Check split content
        with open(train_path, "r") as f:
            train_data = json.load(f)
        
        with open(val_path, "r") as f:
            val_data = json.load(f)
        
        total_items = len(train_data) + len(val_data)
        assert total_items == 4
        
        # Check validation split ratio (approximately)
        # With small datasets, validation might be 0
        val_ratio = len(val_data) / total_items if total_items > 0 else 0
        assert 0.0 <= val_ratio <= 0.3  # Allow 0 for very small datasets
    
    def test_generate_statistics(self):
        """Test statistics generation"""
        organized = {
            LicenseType.PUBLIC_DOMAIN: [self.sample_items[0]],
            LicenseType.CREATIVE_COMMONS: [self.sample_items[1]],
            LicenseType.FAIR_USE_RESEARCH: [self.sample_items[2]],
            LicenseType.COPYRIGHTED: [self.sample_items[3]]
        }
        
        stats = self.organizer._generate_statistics(organized)
        
        assert stats.total_items == 4
        assert stats.license_distribution["public_domain"] == 1
        assert stats.license_distribution["creative_commons"] == 1
        assert stats.license_distribution["fair_use_research"] == 1
        assert stats.license_distribution["copyrighted"] == 1
        
        assert "commons.wikimedia.org" in stats.domain_distribution
        assert "unsplash.com" in stats.domain_distribution
        assert "deviantart.com" in stats.domain_distribution
        assert "shutterstock.com" in stats.domain_distribution
        
        assert stats.compliance_safe_count == 3  # First 3 items are research safe
        assert stats.compliance_safe_percentage == 75.0
        
        assert stats.attribution_required_count == 4  # All items require attribution
        assert stats.commercial_use_allowed_count == 2  # First 2 items allow commercial use
    
    def test_create_license_report(self):
        """Test license report creation"""
        dataset_path = self.temp_dir / "test_dataset"
        dataset_path.mkdir()
        
        organized = {
            LicenseType.PUBLIC_DOMAIN: [self.sample_items[0]],
            LicenseType.CREATIVE_COMMONS: [self.sample_items[1]]
        }
        
        stats = self.organizer._generate_statistics(organized)
        report_path = self.organizer._create_license_report(organized, stats, dataset_path)
        
        assert report_path.exists()
        assert report_path.name == "license_report.md"
        
        # Check report content
        with open(report_path, "r") as f:
            content = f.read()
        
        assert "# License Report" in content
        assert "**Total Items:** 2" in content
        assert "**Research Safe:** 2" in content
        assert "Public Domain" in content
        assert "Creative Commons" in content
        assert self.config.compliance_mode.value in content
    
    def test_organize_content_items(self):
        """Test organizing content items into dataset"""
        dataset = self.organizer.organize_content_items(self.sample_items[:2], "test_dataset")
        
        assert isinstance(dataset, OrganizedDataset)
        assert dataset.name == "test_dataset"
        assert dataset.path.exists()
        assert dataset.stats.total_items == 2
        
        # Check directory structure
        assert (dataset.path / "public_domain").exists()
        assert (dataset.path / "creative_commons").exists()
        
        # Check files
        assert dataset.license_report_path.exists()
        assert dataset.train_split_path.exists()
        assert dataset.val_split_path.exists()
        assert dataset.metadata_db_path.exists()
    
    def test_organize_crawl_result(self):
        """Test organizing crawl result into dataset"""
        crawl_result = CrawlResult(
            success_count=len(self.sample_items),
            items=self.sample_items,
            total_size=sum(item.content_size for item in self.sample_items)
        )
        
        dataset = self.organizer.organize_crawl_result(crawl_result, "crawl_dataset")
        
        assert isinstance(dataset, OrganizedDataset)
        assert dataset.name == "crawl_dataset"
        # In research_safe mode, copyrighted items are filtered out
        assert dataset.stats.total_items == 3  # Only research safe items
    
    def test_save_and_load_dataset(self):
        """Test saving and loading dataset configuration"""
        dataset = self.organizer.organize_content_items(self.sample_items[:2], "save_load_test")
        
        # Load the dataset
        loaded_dataset = self.organizer.load_dataset("save_load_test")
        
        assert loaded_dataset is not None
        assert loaded_dataset.name == dataset.name
        assert loaded_dataset.stats.total_items == dataset.stats.total_items
        assert loaded_dataset.path == dataset.path
    
    def test_load_nonexistent_dataset(self):
        """Test loading non-existent dataset"""
        result = self.organizer.load_dataset("nonexistent_dataset")
        assert result is None
    
    def test_list_datasets(self):
        """Test listing available datasets"""
        # Create a few datasets
        self.organizer.organize_content_items(self.sample_items[:1], "dataset1")
        self.organizer.organize_content_items(self.sample_items[1:2], "dataset2")
        
        datasets = self.organizer.list_datasets()
        
        assert "dataset1" in datasets
        assert "dataset2" in datasets
        assert len(datasets) >= 2
    
    def test_get_compliance_summary(self):
        """Test compliance summary generation"""
        dataset = self.organizer.organize_content_items(self.sample_items[:3], "compliance_test")
        
        summary = self.organizer.get_compliance_summary(dataset)
        
        assert summary["dataset_name"] == "compliance_test"
        assert summary["compliance_mode"] == self.config.compliance_mode.value
        assert summary["total_items"] == 3
        assert "license_distribution" in summary
        assert "created_at" in summary
    
    def test_merge_datasets(self):
        """Test merging multiple datasets"""
        # Create two datasets
        dataset1 = self.organizer.organize_content_items(self.sample_items[:2], "merge_test1")
        dataset2 = self.organizer.organize_content_items(self.sample_items[2:4], "merge_test2")
        
        # Merge datasets
        merged = self.organizer.merge_datasets(["merge_test1", "merge_test2"], "merged_dataset")
        
        assert merged is not None
        assert merged.name == "merged_dataset"
        # In research_safe mode, only research safe items are included
        assert merged.stats.total_items == 3  # Only research safe items from both datasets
    
    def test_merge_nonexistent_datasets(self):
        """Test merging with non-existent datasets"""
        result = self.organizer.merge_datasets(["nonexistent1", "nonexistent2"], "merged")
        assert result is None


class TestDatasetStats:
    """Test dataset statistics data structure"""
    
    def test_dataset_stats_creation(self):
        """Test DatasetStats creation and defaults"""
        stats = DatasetStats()
        
        assert stats.total_items == 0
        assert isinstance(stats.license_distribution, dict)
        assert isinstance(stats.domain_distribution, dict)
        assert isinstance(stats.content_type_distribution, dict)
        assert stats.compliance_safe_count == 0
        assert stats.compliance_safe_percentage == 0.0
        assert stats.total_size_mb == 0.0
        assert isinstance(stats.created_at, datetime)
    
    def test_dataset_stats_with_data(self):
        """Test DatasetStats with data"""
        stats = DatasetStats(
            total_items=100,
            compliance_safe_count=80,
            compliance_safe_percentage=80.0,
            total_size_mb=50.5
        )
        
        assert stats.total_items == 100
        assert stats.compliance_safe_count == 80
        assert stats.compliance_safe_percentage == 80.0
        assert stats.total_size_mb == 50.5


class TestOrganizedDataset:
    """Test organized dataset data structure"""
    
    def test_organized_dataset_creation(self):
        """Test OrganizedDataset creation"""
        config = DatasetConfig()
        stats = DatasetStats(total_items=10)
        path = Path("/test/path")
        
        dataset = OrganizedDataset(
            name="test_dataset",
            path=path,
            config=config,
            stats=stats
        )
        
        assert dataset.name == "test_dataset"
        assert dataset.path == path
        assert dataset.config == config
        assert dataset.stats == stats
        assert dataset.license_report_path is None
        assert dataset.train_split_path is None
        assert dataset.val_split_path is None
        assert dataset.metadata_db_path is None


if __name__ == "__main__":
    pytest.main([__file__])