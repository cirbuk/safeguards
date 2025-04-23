"""DEPRECATED: This file is maintained for backward compatibility only.

The project now uses pyproject.toml for build configuration.
Please refer to pyproject.toml for the current build configuration.
"""

# This file is kept for backward compatibility
# All configuration has been moved to pyproject.toml

from setuptools import find_packages, setup

with open("README.md", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="agent-safeguards",
    version="0.1.0",
    author="Mason Team",
    author_email="dev@getmason.io",
    description="A comprehensive framework for implementing safety controls in AI agent systems",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/cirbuk/agent-safeguards",
    packages=find_packages(
        where="src",
        exclude=[
            "*.tests",
            "*.tests.*",
            "tests.*",
            "tests",
            "*.testing",
            "*.testing.*",
            "testing.*",
            "testing",
        ],
    ),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Security",
    ],
    python_requires=">=3.10",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.2.5",
            "pytest-asyncio>=0.16.0",
            "pytest-cov>=2.12.0",
            "black>=21.9b0",
            "isort>=5.9.0",
            "mypy>=0.910",
            "bandit>=1.7.0",
            "pre-commit>=2.15.0",
        ],
        "docs": [
            "sphinx>=6.0.0",
            "sphinx-rtd-theme>=1.2.0",
            "sphinx-autodoc-typehints>=1.23.0",
        ],
    },
)
