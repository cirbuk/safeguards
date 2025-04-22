# Safeguards

A comprehensive framework for implementing safety measures in multi-agent systems, focusing on budget coordination, monitoring, and guardrails.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Security: Bandit](https://img.shields.io/badge/security-bandit-yellow.svg)](https://github.com/PyCQA/bandit)

## Overview

The Safeguards framework provides tools and infrastructure for ensuring safe and controlled operation of AI agent systems. It addresses key challenges in multi-agent environments:

- Resource management and budget enforcement
- Agent coordination and priority-based allocation
- Safety monitoring and violation detection
- Health assessment and alerting
- Dynamic resource adjustment based on operational needs

This framework is ideal for organizations deploying multiple AI agents that need to:
- Ensure predictable resource usage
- Prioritize critical operations
- Prevent runaway resource consumption
- Implement safe failure modes
- Monitor agent health and behavior

## Features

- **Budget Coordination System**
  - Direct transfer functionality between agents
  - Dynamic pool selection and priority-based allocation
  - Automatic pool scaling and load balancing
  - Emergency allocation handling
  - Priority levels (1-10) for agents and operations

- **Advanced Metrics Analysis**
  - Resource trend analysis
  - Usage pattern detection
  - Budget efficiency tracking
  - Anomaly detection
  - Health monitoring and recommendations

- **Safety Rules System**
  - Customizable rule definitions
  - Priority-based execution
  - Rule chain dependencies
  - Context-aware evaluation
  - Violation detection and reporting

- **API Contracts**
  - Versioned API interfaces
  - Budget management
  - Metrics tracking
  - Agent coordination
  - Configuration management

## Quick Start

### Installation

```bash
pip install agent-safeguards
```

### Basic Setup

```python
from decimal import Decimal
from safeguards.core.budget_coordination import BudgetCoordinator
from safeguards.core.notification_manager import NotificationManager
from safeguards.api import APIFactory, APIVersion
from safeguards.types.agent import Agent

# Create core components
notification_manager = NotificationManager()
budget_coordinator = BudgetCoordinator(notification_manager)
api_factory = APIFactory()

# Create APIs
budget_api = api_factory.create_budget_api(APIVersion.V1, budget_coordinator)
agent_api = api_factory.create_agent_api(APIVersion.V1, budget_coordinator)

# Create a budget pool
pool = budget_api.create_budget_pool(
    name="main_pool",
    initial_budget=Decimal("100.0"),
    priority=5
)

# Create an agent
agent = agent_api.create_agent(
    name="example_agent",
    initial_budget=Decimal("10.0"),
    priority=3
)

# Check agent budget
budget = budget_api.get_budget(agent.id)
print(f"Agent {agent.name} has budget: {budget}")
```

### Creating a Custom Agent

```python
from decimal import Decimal
from typing import Dict, Any
from safeguards.types.agent import Agent

class MyAgent(Agent):
    def __init__(self, name: str, cost_per_action: Decimal = Decimal("0.1")):
        super().__init__(name)
        self.cost_per_action = cost_per_action
        self.action_count = 0

    def run(self, **kwargs: Any) -> Dict[str, Any]:
        """Execute agent logic with cost tracking."""
        self.action_count += 1
        # Your agent implementation here
        return {
            "result": "Task completed",
            "action_count": self.action_count,
            "cost": self.cost_per_action,
        }

# Create and register agent
my_agent = MyAgent("custom_agent")
registered_agent = agent_api.create_agent(
    name=my_agent.name,
    initial_budget=Decimal("20.0"),
    priority=5
)

# Run agent and update budget
for _ in range(3):
    result = my_agent.run(input="Example task")
    current_budget = budget_api.get_budget(registered_agent.id)
    budget_api.update_budget(
        registered_agent.id,
        current_budget - result["cost"]
    )
```

For more detailed examples, see the [Quick Start Guide](docs/quickstart.md).

## Documentation

- [Core Concepts](docs/concepts.md) - Essential concepts and terminology
- [Installation Guide](docs/installation.md) - Detailed installation instructions
- [Quick Start Guide](docs/quickstart.md) - Get started with the framework
- [Component Status](COMPONENT_STATUS.md) - Current status of framework components
- [Budget Management](docs/guides/budget_management.md) - How to manage agent budgets
- [Safety Policies](docs/guides/safety_policies.md) - Implementing and enforcing safety policies
- [Safeguards](docs/guides/safeguards.md) - Safety features and guardrails
- [Monitoring](docs/guides/monitoring.md) - Metrics, visualization, and alerts
- [Agent Coordination](docs/guides/agent_coordination.md) - Multi-agent coordination
- [API Reference](docs/api/core.md) - Detailed API documentation
- [Architecture Overview](docs/development/architecture.md) - System design

For a complete documentation index, see the [Documentation README](docs/README.md).

## Use Cases

The Safeguards framework is designed for a variety of use cases:

- **Enterprise AI Systems**: Manage resource allocation across multiple AI services
- **Autonomous Systems**: Ensure safety constraints in autonomous operations
- **Research Environments**: Control experiment resource usage and monitor behavior
- **Agent Orchestration**: Coordinate multiple specialized agents working together
- **LLM Application Deployment**: Manage token budgets and processing resources

## Development

### Prerequisites

- Python 3.10 or higher
- pip package manager

### Setting Up Development Environment

```bash
# Clone the repository
git clone https://github.com/cirbuk/agent-safeguards.git
cd agent-safeguards

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install development dependencies
pip install -r requirements-dev.txt

# Install package in development mode
pip install -e .
```

### Running Tests

```bash
pytest tests/
```

### Code Style

The project uses:
- Black for code formatting
- isort for import sorting
- mypy for type checking
- flake8 and pylint for linting

Run formatters:
```bash
black .
isort .
```

Run type checking:
```bash
mypy src/
```

Run linters:
```bash
flake8 src/
pylint src/
```

## Contributing

Contributions are welcome! Please see our [Contributing Guide](CONTRIBUTING.md) for details on how to contribute to the project.

## Security

This framework implements several security measures:
- Pre-commit hooks for security scanning
- Automated security checks in CI/CD
- Regular dependency updates
- Code analysis tools

If you discover a security vulnerability, please report it to dev@getmason.io.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

For support, please open an issue on the GitHub repository or contact the Mason team at dev@getmason.io

## Acknowledgments

- Contributors and maintainers
- Security research community
- Open source projects that inspired this framework
