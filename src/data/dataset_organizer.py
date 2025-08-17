"""
Dataset Organization System for Academic Multimodal LLM System

This module provides license-based content organization, compliance mode filtering,
metadata tracking, and attribution preservation for research datasets.
"""

import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import sqlite3
import hashlib
import logging

from .license_classifier import LicenseType, LicenseInfo
from .research_crawler import ContentItem, CrawlResult


class ComplianceMode(Enum):
    """Dataset compliance modes for research use"""
    OPEN_SOURCE_ONLY = "open_source_only"      # Public Domain + Creative Commons
    RESEARCH_SAFE = "research_safe"            # Open Source + Fair Use Research
    FULL_DATASET = "full_dataset"              # All content (with warnings)


@dataclass
class DatasetConfig:
    """Configuration for dataset organization"""
    base_path: Path = field(default_factory=lambda: Path("./datasets"))
    compliance_mode: ComplianceMode = ComplianceMode.RESEARCH_SAFE
    preserve_attribution: bool = True
    create_license_reports: bool = True
    max_items_per_category: Optional[int] = None
    include_metadata: bool = True
    create_train_val_split: bool = True
    validation_split: float = 0.2
    random_seed: int = 42


@dataclass
class DatasetStats:
    """Statistics for organized dataset"""
    total_items: int = 0
    license_distribution: Dict[str, int] = field(default_factory=dict)
    domain_distribution: Dict[str, int] = field(default_factory=dict)
    content_type_distribution: Dict[str, int] = field(default_factory=dict)
    compliance_safe_count: int = 0
    compliance_safe_percentage: float = 0.0
    total_size_mb: float = 0.0
    attribution_required_count: int = 0
    commercial_use_allowed_count: int = 0
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class OrganizedDataset:
    """Container for organized dataset information"""
    name: str
    path: Path
    config: DatasetConfig
    stats: DatasetStats
    license_report_path: Optional[Path] = None
    train_split_path: Optional[Path] = None
    val_split_path: Optional[Path] = None
    metadata_db_path: Optional[Path] = None


class DatasetOrganizer:
    """
    Dataset organization system with license-based content separation.
    
    Features:
    - License-based directory organization
    - Compliance mode filtering
    - Metadata tracking and preservation
    - Attribution requirement tracking
    - Train/validation split creation
    - Comprehensive reporting and statistics
    """
    
    def __init__(self, config: Optional[DatasetConfig] = None):
        self.config = config or DatasetConfig()
        self.config.base_path.mkdir(parents=True, exist_ok=True)
        
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
    
    def organize_crawl_result(self, crawl_result: CrawlResult, 
                             dataset_name: str) -> OrganizedDataset:
        """
        Organize crawled content into a structured dataset.
        
        Args:
            crawl_result: Result from research crawler
            dataset_name: Name for the organized dataset
            
        Returns:
            OrganizedDataset with organization information
        """
        self.logger.info(f"Organizing dataset '{dataset_name}' with {len(crawl_result.items)} items")
        
        # Create dataset directory
        dataset_path = self.config.base_path / dataset_name
        dataset_path.mkdir(parents=True, exist_ok=True)
        
        # Filter items based on compliance mode
        filtered_items = self._filter_by_compliance(crawl_result.items)
        
        # Organize items by license type
        organized_items = self._organize_by_license(filtered_items, dataset_path)
        
        # Create metadata database
        metadata_db_path = None
        if self.config.include_metadata:
            metadata_db_path = self._create_metadata_database(organized_items, dataset_path)
        
        # Create train/validation splits
        train_split_path = None
        val_split_path = None
        if self.config.create_train_val_split:
            train_split_path, val_split_path = self._create_train_val_splits(
                organized_items, dataset_path
            )
        
        # Generate statistics
        stats = self._generate_statistics(organized_items)
        
        # Create license report
        license_report_path = None
        if self.config.create_license_reports:
            license_report_path = self._create_license_report(organized_items, stats, dataset_path)
        
        # Create organized dataset object
        organized_dataset = OrganizedDataset(
            name=dataset_name,
            path=dataset_path,
            config=self.config,
            stats=stats,
            license_report_path=license_report_path,
            train_split_path=train_split_path,
            val_split_path=val_split_path,
            metadata_db_path=metadata_db_path
        )
        
        # Save dataset configuration
        self._save_dataset_config(organized_dataset)
        
        self.logger.info(
            f"Dataset '{dataset_name}' organized: {stats.total_items} items, "
            f"{stats.compliance_safe_percentage:.1f}% compliance safe"
        )
        
        return organized_dataset
    
    def organize_content_items(self, items: List[ContentItem], 
                              dataset_name: str) -> OrganizedDataset:
        """
        Organize a list of content items into a structured dataset.
        
        Args:
            items: List of ContentItem objects
            dataset_name: Name for the organized dataset
            
        Returns:
            OrganizedDataset with organization information
        """
        # Create a mock crawl result
        crawl_result = CrawlResult(
            success_count=len(items),
            items=items,
            total_size=sum(item.content_size for item in items)
        )
        
        return self.organize_crawl_result(crawl_result, dataset_name)
    
    def load_dataset(self, dataset_name: str) -> Optional[OrganizedDataset]:
        """
        Load an existing organized dataset.
        
        Args:
            dataset_name: Name of the dataset to load
            
        Returns:
            OrganizedDataset if found, None otherwise
        """
        dataset_path = self.config.base_path / dataset_name
        config_path = dataset_path / "dataset_config.json"
        
        if not config_path.exists():
            return None
        
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config_data = json.load(f)
            
            # Reconstruct dataset object
            stats = DatasetStats(**config_data["stats"])
            
            return OrganizedDataset(
                name=dataset_name,
                path=dataset_path,
                config=self.config,
                stats=stats,
                license_report_path=Path(config_data["license_report_path"]) if config_data.get("license_report_path") else None,
                train_split_path=Path(config_data["train_split_path"]) if config_data.get("train_split_path") else None,
                val_split_path=Path(config_data["val_split_path"]) if config_data.get("val_split_path") else None,
                metadata_db_path=Path(config_data["metadata_db_path"]) if config_data.get("metadata_db_path") else None
            )
            
        except Exception as e:
            self.logger.error(f"Error loading dataset '{dataset_name}': {e}")
            return None
    
    def list_datasets(self) -> List[str]:
        """List all available organized datasets"""
        datasets = []
        
        if not self.config.base_path.exists():
            return datasets
        
        for item in self.config.base_path.iterdir():
            if item.is_dir() and (item / "dataset_config.json").exists():
                datasets.append(item.name)
        
        return sorted(datasets)
    
    def get_compliance_summary(self, dataset: OrganizedDataset) -> Dict:
        """Get compliance summary for a dataset"""
        return {
            "dataset_name": dataset.name,
            "compliance_mode": self.config.compliance_mode.value,
            "total_items": dataset.stats.total_items,
            "compliance_safe_count": dataset.stats.compliance_safe_count,
            "compliance_safe_percentage": dataset.stats.compliance_safe_percentage,
            "attribution_required_count": dataset.stats.attribution_required_count,
            "commercial_use_allowed_count": dataset.stats.commercial_use_allowed_count,
            "license_distribution": dataset.stats.license_distribution,
            "created_at": dataset.stats.created_at.isoformat()
        }
    
    def merge_datasets(self, dataset_names: List[str], 
                      merged_name: str) -> Optional[OrganizedDataset]:
        """
        Merge multiple datasets into a single organized dataset.
        
        Args:
            dataset_names: List of dataset names to merge
            merged_name: Name for the merged dataset
            
        Returns:
            OrganizedDataset for merged dataset
        """
        self.logger.info(f"Merging datasets {dataset_names} into '{merged_name}'")
        
        all_items = []
        
        # Collect items from all datasets
        for dataset_name in dataset_names:
            dataset = self.load_dataset(dataset_name)
            if not dataset:
                self.logger.warning(f"Dataset '{dataset_name}' not found, skipping")
                continue
            
            # Load items from metadata database
            if dataset.metadata_db_path and dataset.metadata_db_path.exists():
                items = self._load_items_from_database(dataset.metadata_db_path)
                all_items.extend(items)
        
        if not all_items:
            self.logger.error("No items found in datasets to merge")
            return None
        
        # Organize merged items
        return self.organize_content_items(all_items, merged_name)
    
    def _filter_by_compliance(self, items: List[ContentItem]) -> List[ContentItem]:
        """Filter items based on compliance mode"""
        if self.config.compliance_mode == ComplianceMode.FULL_DATASET:
            return items
        
        filtered_items = []
        
        for item in items:
            if not item.license_info:
                continue
            
            license_type = item.license_info.license_type
            
            if self.config.compliance_mode == ComplianceMode.OPEN_SOURCE_ONLY:
                if license_type in [LicenseType.PUBLIC_DOMAIN, LicenseType.CREATIVE_COMMONS]:
                    filtered_items.append(item)
            elif self.config.compliance_mode == ComplianceMode.RESEARCH_SAFE:
                if item.license_info.research_safe:
                    filtered_items.append(item)
        
        self.logger.info(
            f"Filtered {len(items)} items to {len(filtered_items)} "
            f"for compliance mode: {self.config.compliance_mode.value}"
        )
        
        return filtered_items
    
    def _organize_by_license(self, items: List[ContentItem], 
                           dataset_path: Path) -> Dict[LicenseType, List[ContentItem]]:
        """Organize items by license type into directories"""
        organized = {}
        
        for item in items:
            if not item.license_info:
                continue
            
            license_type = item.license_info.license_type
            
            if license_type not in organized:
                organized[license_type] = []
            
            # Apply max items per category limit
            if (self.config.max_items_per_category and 
                len(organized[license_type]) >= self.config.max_items_per_category):
                continue
            
            organized[license_type].append(item)
        
        # Create directories and copy files
        for license_type, license_items in organized.items():
            license_dir = dataset_path / license_type.value
            license_dir.mkdir(parents=True, exist_ok=True)
            
            for item in license_items:
                if item.local_path and item.local_path.exists():
                    # Generate new filename with hash to avoid conflicts
                    file_extension = item.local_path.suffix
                    new_filename = f"{item.content_hash[:16]}{file_extension}"
                    new_path = license_dir / new_filename
                    
                    # Copy file to organized location
                    shutil.copy2(item.local_path, new_path)
                    item.local_path = new_path
                    
                    # Create attribution file if required
                    if self.config.preserve_attribution and item.attribution:
                        attribution_path = license_dir / f"{item.content_hash[:16]}_attribution.txt"
                        with open(attribution_path, "w", encoding="utf-8") as f:
                            f.write(item.attribution)
        
        return organized
    
    def _create_metadata_database(self, organized_items: Dict[LicenseType, List[ContentItem]], 
                                 dataset_path: Path) -> Path:
        """Create SQLite database with item metadata"""
        db_path = dataset_path / "metadata.db"
        
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Create tables
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS content_items (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                url TEXT NOT NULL,
                local_path TEXT,
                content_type TEXT,
                content_size INTEGER,
                title TEXT,
                description TEXT,
                attribution TEXT,
                source_domain TEXT,
                content_hash TEXT UNIQUE,
                license_type TEXT,
                license_confidence REAL,
                attribution_required BOOLEAN,
                commercial_use_allowed BOOLEAN,
                research_safe BOOLEAN,
                license_details TEXT,
                crawl_timestamp TEXT,
                metadata_json TEXT
            )
        """)
        
        # Insert items
        for license_type, items in organized_items.items():
            for item in items:
                cursor.execute("""
                    INSERT OR REPLACE INTO content_items (
                        url, local_path, content_type, content_size, title, description,
                        attribution, source_domain, content_hash, license_type,
                        license_confidence, attribution_required, commercial_use_allowed,
                        research_safe, license_details, crawl_timestamp, metadata_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    item.url,
                    str(item.local_path) if item.local_path else None,
                    item.content_type,
                    item.content_size,
                    item.title,
                    item.description,
                    item.attribution,
                    item.source_domain,
                    item.content_hash,
                    item.license_info.license_type.value if item.license_info else None,
                    item.license_info.confidence if item.license_info else None,
                    item.license_info.attribution_required if item.license_info else None,
                    item.license_info.commercial_use_allowed if item.license_info else None,
                    item.license_info.research_safe if item.license_info else None,
                    item.license_info.details if item.license_info else None,
                    item.crawl_timestamp.isoformat(),
                    json.dumps(item.metadata)
                ))
        
        conn.commit()
        conn.close()
        
        self.logger.info(f"Created metadata database: {db_path}")
        return db_path
    
    def _create_train_val_splits(self, organized_items: Dict[LicenseType, List[ContentItem]], 
                                dataset_path: Path) -> Tuple[Path, Path]:
        """Create train/validation splits"""
        import random
        
        random.seed(self.config.random_seed)
        
        train_split_path = dataset_path / "train_split.json"
        val_split_path = dataset_path / "val_split.json"
        
        train_items = []
        val_items = []
        
        # Split items from each license category
        for license_type, items in organized_items.items():
            # Shuffle items
            shuffled_items = items.copy()
            random.shuffle(shuffled_items)
            
            # Calculate split point
            val_count = int(len(shuffled_items) * self.config.validation_split)
            
            # Split items
            val_items.extend(shuffled_items[:val_count])
            train_items.extend(shuffled_items[val_count:])
        
        # Save splits
        train_data = [
            {
                "url": item.url,
                "local_path": str(item.local_path) if item.local_path else None,
                "title": item.title,
                "license_type": item.license_info.license_type.value if item.license_info else None,
                "attribution": item.attribution
            }
            for item in train_items
        ]
        
        val_data = [
            {
                "url": item.url,
                "local_path": str(item.local_path) if item.local_path else None,
                "title": item.title,
                "license_type": item.license_info.license_type.value if item.license_info else None,
                "attribution": item.attribution
            }
            for item in val_items
        ]
        
        with open(train_split_path, "w", encoding="utf-8") as f:
            json.dump(train_data, f, indent=2, ensure_ascii=False)
        
        with open(val_split_path, "w", encoding="utf-8") as f:
            json.dump(val_data, f, indent=2, ensure_ascii=False)
        
        self.logger.info(
            f"Created splits: {len(train_items)} train, {len(val_items)} validation"
        )
        
        return train_split_path, val_split_path
    
    def _generate_statistics(self, organized_items: Dict[LicenseType, List[ContentItem]]) -> DatasetStats:
        """Generate comprehensive dataset statistics"""
        stats = DatasetStats()
        
        all_items = []
        for items in organized_items.values():
            all_items.extend(items)
        
        stats.total_items = len(all_items)
        
        if not all_items:
            return stats
        
        # License distribution
        for license_type, items in organized_items.items():
            stats.license_distribution[license_type.value] = len(items)
        
        # Domain and content type distribution
        for item in all_items:
            # Domain distribution
            domain = item.source_domain
            stats.domain_distribution[domain] = stats.domain_distribution.get(domain, 0) + 1
            
            # Content type distribution
            content_type = item.content_type
            stats.content_type_distribution[content_type] = stats.content_type_distribution.get(content_type, 0) + 1
            
            # Size calculation
            stats.total_size_mb += item.content_size / (1024 * 1024)
            
            # Compliance and attribution counts
            if item.license_info:
                if item.license_info.research_safe:
                    stats.compliance_safe_count += 1
                if item.license_info.attribution_required:
                    stats.attribution_required_count += 1
                if item.license_info.commercial_use_allowed:
                    stats.commercial_use_allowed_count += 1
        
        # Calculate percentages
        stats.compliance_safe_percentage = (stats.compliance_safe_count / stats.total_items) * 100
        
        return stats
    
    def _create_license_report(self, organized_items: Dict[LicenseType, List[ContentItem]], 
                              stats: DatasetStats, dataset_path: Path) -> Path:
        """Create comprehensive license report"""
        report_path = dataset_path / "license_report.md"
        
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(f"# License Report\n\n")
            f.write(f"**Dataset Created:** {stats.created_at.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**Compliance Mode:** {self.config.compliance_mode.value}\n\n")
            
            f.write(f"## Summary\n\n")
            f.write(f"- **Total Items:** {stats.total_items}\n")
            f.write(f"- **Research Safe:** {stats.compliance_safe_count} ({stats.compliance_safe_percentage:.1f}%)\n")
            f.write(f"- **Attribution Required:** {stats.attribution_required_count}\n")
            f.write(f"- **Commercial Use Allowed:** {stats.commercial_use_allowed_count}\n")
            f.write(f"- **Total Size:** {stats.total_size_mb:.2f} MB\n\n")
            
            f.write(f"## License Distribution\n\n")
            for license_type, count in stats.license_distribution.items():
                percentage = (count / stats.total_items) * 100
                f.write(f"- **{license_type.replace('_', ' ').title()}:** {count} ({percentage:.1f}%)\n")
            
            f.write(f"\n## Domain Distribution\n\n")
            sorted_domains = sorted(stats.domain_distribution.items(), key=lambda x: x[1], reverse=True)
            for domain, count in sorted_domains[:10]:  # Top 10 domains
                percentage = (count / stats.total_items) * 100
                f.write(f"- **{domain}:** {count} ({percentage:.1f}%)\n")
            
            f.write(f"\n## Content Type Distribution\n\n")
            for content_type, count in stats.content_type_distribution.items():
                percentage = (count / stats.total_items) * 100
                f.write(f"- **{content_type}:** {count} ({percentage:.1f}%)\n")
            
            f.write(f"\n## Compliance Information\n\n")
            f.write(f"This dataset was organized using compliance mode: **{self.config.compliance_mode.value}**\n\n")
            
            if self.config.compliance_mode == ComplianceMode.OPEN_SOURCE_ONLY:
                f.write("- Only Public Domain and Creative Commons licensed content included\n")
                f.write("- Safe for commercial use and redistribution\n")
            elif self.config.compliance_mode == ComplianceMode.RESEARCH_SAFE:
                f.write("- Includes Public Domain, Creative Commons, and Fair Use Research content\n")
                f.write("- Safe for academic research and educational use\n")
            else:
                f.write("- Includes all content types including copyrighted material\n")
                f.write("- **WARNING:** Contains copyrighted content - use only for research comparison\n")
            
            f.write(f"\n## Attribution Requirements\n\n")
            f.write(f"Items requiring attribution: {stats.attribution_required_count}/{stats.total_items}\n\n")
            f.write("Please ensure proper attribution is provided when using this dataset. ")
            f.write("Attribution information is preserved in individual item metadata and attribution files.\n")
        
        self.logger.info(f"Created license report: {report_path}")
        return report_path
    
    def _save_dataset_config(self, dataset: OrganizedDataset):
        """Save dataset configuration and metadata"""
        config_path = dataset.path / "dataset_config.json"
        
        config_data = {
            "name": dataset.name,
            "compliance_mode": self.config.compliance_mode.value,
            "created_at": dataset.stats.created_at.isoformat(),
            "stats": {
                "total_items": dataset.stats.total_items,
                "license_distribution": dataset.stats.license_distribution,
                "domain_distribution": dataset.stats.domain_distribution,
                "content_type_distribution": dataset.stats.content_type_distribution,
                "compliance_safe_count": dataset.stats.compliance_safe_count,
                "compliance_safe_percentage": dataset.stats.compliance_safe_percentage,
                "total_size_mb": dataset.stats.total_size_mb,
                "attribution_required_count": dataset.stats.attribution_required_count,
                "commercial_use_allowed_count": dataset.stats.commercial_use_allowed_count,
                "created_at": dataset.stats.created_at.isoformat()
            },
            "license_report_path": str(dataset.license_report_path) if dataset.license_report_path else None,
            "train_split_path": str(dataset.train_split_path) if dataset.train_split_path else None,
            "val_split_path": str(dataset.val_split_path) if dataset.val_split_path else None,
            "metadata_db_path": str(dataset.metadata_db_path) if dataset.metadata_db_path else None
        }
        
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config_data, f, indent=2, ensure_ascii=False)
    
    def _load_items_from_database(self, db_path: Path) -> List[ContentItem]:
        """Load content items from metadata database"""
        items = []
        
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM content_items")
        rows = cursor.fetchall()
        
        for row in rows:
            # Reconstruct ContentItem from database row
            item = ContentItem(
                url=row[1],
                local_path=Path(row[2]) if row[2] else None,
                content_type=row[3],
                content_size=row[4],
                title=row[5],
                description=row[6],
                attribution=row[7],
                source_domain=row[8],
                content_hash=row[9],
                crawl_timestamp=datetime.fromisoformat(row[16]),
                metadata=json.loads(row[17]) if row[17] else {}
            )
            
            # Reconstruct license info
            if row[10]:  # license_type
                from .license_classifier import LicenseInfo
                item.license_info = LicenseInfo(
                    license_type=LicenseType(row[10]),
                    confidence=row[11],
                    source_domain=row[8],
                    attribution_required=bool(row[12]),
                    commercial_use_allowed=bool(row[13]),
                    research_safe=bool(row[14]),
                    details=row[15]
                )
                item.research_safe = item.license_info.research_safe
            
            items.append(item)
        
        conn.close()
        return items