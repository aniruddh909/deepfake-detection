"""
Setup script for the Deepfake Detection System.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text() if readme_path.exists() else ""

# Read requirements
requirements_path = Path(__file__).parent / "requirements.txt"
requirements = []
if requirements_path.exists():
    requirements = requirements_path.read_text().strip().split('\n')
    requirements = [req.strip() for req in requirements if req.strip() and not req.startswith('#')]

setup(
    name="deepfake-detection",
    version="1.0.0",
    description="Real-Time Deepfake Detection System using MediaPipe and Conv1D CNN",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Deepfake Detection Team",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "deepfake-detect=main:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Multimedia :: Video",
        "Topic :: Security",
    ],
    keywords="deepfake detection computer-vision machine-learning tensorflow mediapipe",
    project_urls={
        "Documentation": "https://github.com/username/deepfake-detection",
        "Source": "https://github.com/username/deepfake-detection",
        "Tracker": "https://github.com/username/deepfake-detection/issues",
    },
)
