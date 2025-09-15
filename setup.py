"""
Vision Transformer PyTorch Package Setup
Modern setup.py configuration for distribution and installation
"""

from setuptools import setup, find_packages
import os
import re

# Read version from __init__.py
def get_version():
    with open(os.path.join(os.path.dirname(__file__), '__init__.py'), 'r') as f:
        version_file = f.read()
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")

# Read long description from README
def get_long_description():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements
def get_requirements():
    with open("requirements.txt", "r") as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="vision-transformer-pytorch",
    version=get_version(),
    author="Your Name",
    author_email="your.email@example.com",
    description="Advanced Vision Transformer implementation in PyTorch with state-of-the-art optimizations",
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/vision-transformer-pytorch",
    project_urls={
        "Bug Tracker": "https://github.com/yourusername/vision-transformer-pytorch/issues",
        "Documentation": "https://github.com/yourusername/vision-transformer-pytorch/docs",
        "Source Code": "https://github.com/yourusername/vision-transformer-pytorch",
    },
    packages=find_packages(exclude=["tests*", "docs*", "examples*"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=get_requirements(),
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
            "isort>=5.10.0",
            "pre-commit>=2.20.0",
        ],
        "docs": [
            "sphinx>=5.0.0",
            "sphinx-rtd-theme>=1.0.0",
            "myst-parser>=0.18.0",
        ],
        "benchmark": [
            "memory-profiler>=0.60.0",
            "psutil>=5.9.0",
            "matplotlib>=3.5.0",
            "seaborn>=0.11.0",
            "pandas>=1.5.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "vit-train=train:main",
            "vit-inference=inference:main",
            "vit-benchmark=benchmarks.performance_comparison:run_comprehensive_benchmark",
        ],
    },
    keywords="vision transformer, pytorch, deep learning, computer vision, attention, neural networks",
    zip_safe=False,
    include_package_data=True,
    package_data={
        "": ["*.md", "*.txt", "*.yaml", "*.yml"],
    },
)
