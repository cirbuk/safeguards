# Installation Guide

## Requirements

- Python 3.8 or higher
- pip package manager

## Basic Installation

Install the package using pip:

```bash
pip install agent-safety
```

This will install the core package with all required dependencies.

## Development Installation

For development work, install with additional development dependencies:

```bash
pip install agent-safety[dev]
```

This includes:
- Testing tools (pytest, pytest-asyncio, pytest-cov)
- Code formatting (black, isort)
- Linting (flake8)
- Type checking (mypy)

## Documentation Installation

To build documentation locally:

```bash
pip install agent-safety[docs]
```

This includes:
- Sphinx documentation generator
- Read the Docs theme
- Type hints support

## Verifying Installation

You can verify the installation by running:

```python
from agent_safety import BudgetManager, ResourceMonitor
from agent_safety.types import Agent

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
   pip install --user agent-safety
   ```

For more issues, please check our [GitHub Issues](https://github.com/mason-ai/agent-safety/issues) page. 