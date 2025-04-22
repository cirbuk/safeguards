# Installation Guide

## Prerequisites

Before installing Safeguards, make sure you have:

- Python 3.10 or newer
- pip package manager

## Basic Installation

Install the package using pip:

```bash
pip install agent-safeguards
```

This will install the core package with all required dependencies.

## Development Installation

For development work, install with additional development dependencies:

```bash
pip install agent-safeguards[dev]
```

This includes:
- Testing tools (pytest, pytest-asyncio, pytest-cov)
- Code formatting (black, isort)
- Linting (flake8)
- Type checking (mypy)

## Documentation Installation

To build documentation locally:

```bash
pip install agent-safeguards[docs]
```

This includes:
- Sphinx documentation generator
- Read the Docs theme
- Type hints support

## Verifying Installation

You can verify the installation by running:

```python
from safeguards import BudgetManager, ResourceMonitor
from safeguards.types import Agent

# Should not raise any ImportError
```

## System Dependencies

The package requires `psutil` for system resource monitoring. On some systems, you might need to install additional system packages:

### Ubuntu/Debian
```bash
sudo apt-get update
sudo apt-get install python3-dev
```

### CentOS/RHEL
```bash
sudo yum install python3-devel
```

### macOS
No additional system packages required.

### Windows
No additional system packages required.

## Troubleshooting

### Common Issues

1. ImportError: No module named 'psutil'
   ```bash
   pip install --no-cache-dir psutil
   ```

2. Build failures on Windows
   ```bash
   pip install --upgrade setuptools wheel
   ```

3. Permission errors during installation
   ```bash
   pip install --user agent-safeguards
   ```

For more issues, please check our [GitHub Issues](https://github.com/cirbuk/agent-safeguards/issues) page.
