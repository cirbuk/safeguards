# Agent Safety Framework

A comprehensive framework for implementing safety controls and monitoring for AI agents. This framework provides tools and utilities to ensure AI agents operate within defined constraints and maintain system stability.

## Features

### Budget Management
- Track and control agent resource usage
- Set and enforce budget limits
- Handle budget overrides and alerts
- Monitor hourly and daily spending

### Resource Monitoring
- Real-time system resource tracking
- CPU usage monitoring
- Memory usage monitoring
- Disk usage monitoring
- Configurable thresholds and alerts

### Safety Guardrails
- Implement pre-execution safety checks
- Post-execution validation
- Customizable safety rules
- Extensible guardrail system

### Notification System
- Multi-level alerting (INFO, WARNING, ERROR, CRITICAL)
- Agent-specific notifications
- Customizable notification handlers
- Historical notification tracking

## Installation

```bash
pip install agent-safety
```

For development installation:
```bash
pip install agent-safety[dev]
```

## Quick Start

```python
from agent_safety import BudgetManager, ResourceMonitor
from agent_safety.guardrails import BudgetGuardrail, ResourceGuardrail
from agent_safety.types import Agent

# Initialize safety components
budget_manager = BudgetManager(
    total_budget=1000,
    hourly_limit=100,
    daily_limit=500,
)

resource_monitor = ResourceMonitor(
    cpu_threshold=80.0,
    memory_threshold=85.0,
    disk_threshold=90.0,
)

# Create an agent with safety controls
agent = Agent(
    name="safe_agent",
    instructions="Your instructions here",
    guardrails=[
        BudgetGuardrail(budget_manager),
        ResourceGuardrail(resource_monitor),
    ]
)

# Run agent with safety controls
result = agent.run(input_data="Your input here")
```

## Documentation

### Installation & Setup
- [Basic Installation](docs/installation.md)
- [Development Setup](docs/development.md)
- [Configuration Options](docs/configuration.md)

### Usage Guides
- [Basic Usage](docs/usage/basic.md)
- [Budget Management](docs/usage/budget.md)
- [Resource Monitoring](docs/usage/resources.md)
- [Safety Guardrails](docs/usage/guardrails.md)
- [Notification System](docs/usage/notifications.md)

### API Reference
- [Agent Types](docs/api/types.md)
- [Budget Management](docs/api/budget.md)
- [Resource Monitoring](docs/api/monitoring.md)
- [Guardrails](docs/api/guardrails.md)
- [Notifications](docs/api/notifications.md)

## Development

1. Clone the repository:
```bash
git clone https://github.com/mason-ai/agent-safety.git
cd agent-safety
```

2. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate
```

3. Install development dependencies:
```bash
pip install -r requirements-dev.txt
```

4. Run tests:
```bash
pytest
```

5. Run linting and type checks:
```bash
black .
isort .
flake8
mypy .
```

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Security

For security concerns, please email security@mason.ai. For our security policy and reporting guidelines, see [SECURITY.md](SECURITY.md). 