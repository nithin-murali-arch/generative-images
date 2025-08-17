"""Setup script for the Academic Multimodal LLM Experiment System."""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

# Read requirements
requirements_path = Path(__file__).parent / "requirements.txt"
requirements = []
if requirements_path.exists():
    with open(requirements_path, 'r') as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name="academic-multimodal-llm-system",
    version="0.1.0",
    author="Academic Research Team",
    author_email="research@example.com",
    description="A research-focused platform for ethical experimentation with multimodal AI generation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/example/academic-multimodal-llm-system",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Multimedia :: Graphics",
        "Topic :: Multimedia :: Video",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-asyncio>=0.21.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
        ],
        "gpu": [
            "xformers>=0.0.20",
            "bitsandbytes>=0.41.0",
        ],
        "tracking": [
            "wandb>=0.15.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "academic-multimodal=src.main:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)