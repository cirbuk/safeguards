# Quick Start Guide

This guide will help you get started with the Safeguards, covering installation, basic setup, and common use cases.

## Prerequisites

Before you begin, ensure you have the following:

- Python 3.10 or higher
- pip package manager

### Install from PyPI

```bash
pip install agent-safeguards
```

### Install from Source

```bash
git clone https://github.com/cirbuk/agent-safeguards.git
cd agent-safeguards
pip install -e .
```

## Core Components Setup

The Safeguards consists of several components working together:

```python
from decimal import Decimal
from safeguards.core.budget_coordination import BudgetCoordinator
from safeguards.core.notification_manager import NotificationManager
from safeguards.core.violation_reporter import ViolationReporter
from safeguards.api import APIFactory, APIVersion

# Initialize core components
notification_manager = NotificationManager()
violation_reporter = ViolationReporter(notification_manager)
budget_coordinator = BudgetCoordinator(notification_manager)

# Create API factory
api_factory = APIFactory()

# Create APIs with specific versions
budget_api = api_factory.create_budget_api(APIVersion.V1, budget_coordinator)
agent_api = api_factory.create_agent_api(APIVersion.V1, budget_coordinator)
metrics_api = api_factory.create_metrics_api(APIVersion.V1, budget_coordinator)
```

## Budget Management

### Create Budget Pools

Budget pools are used to group and manage resources:

```python
# Create a high-priority pool for critical operations
critical_pool = budget_api.create_budget_pool(
    name="critical_operations",
    initial_budget=Decimal("500.0"),
    priority=8
)

# Create a medium-priority pool for regular operations
regular_pool = budget_api.create_budget_pool(
    name="regular_operations",
    initial_budget=Decimal("1000.0"),
    priority=5
)

# Create a low-priority pool for background tasks
background_pool = budget_api.create_budget_pool(
    name="background_tasks",
    initial_budget=Decimal("300.0"),
    priority=2
)
```

### Create Agents

Agents are registered with the framework and assigned initial budgets:

```python
# Create agents with different priorities
assistant_agent = agent_api.create_agent(
    name="assistant_agent",
    initial_budget=Decimal("100.0"),
    priority=7
)

research_agent = agent_api.create_agent(
    name="research_agent",
    initial_budget=Decimal("150.0"),
    priority=5
)

summarization_agent = agent_api.create_agent(
    name="summarization_agent",
    initial_budget=Decimal("50.0"),
    priority=3
)
```

### Check and Update Budgets

Monitor and modify agent budgets as needed:

```python
# Check current budget
assistant_budget = budget_api.get_budget(assistant_agent.id)
print(f"Assistant agent budget: {assistant_budget}")

# Update budget after usage
budget_api.update_budget(
    assistant_agent.id,
    assistant_budget - Decimal("10.0")
)

# Get updated budget
new_budget = budget_api.get_budget(assistant_agent.id)
print(f"Updated assistant agent budget: {new_budget}")
```

## Creating Custom Agents

Implement your own agent by extending the base `Agent` class:

```python
from typing import Dict, Any
from decimal import Decimal
from safeguards.types.agent import Agent

class CustomAssistantAgent(Agent):
    def __init__(self, name: str, model: str = "gpt-4"):
        super().__init__(name)
        self.model = model
        self.cost_per_token = Decimal("0.0001")
        self.history = []

    def run(self, **kwargs: Any) -> Dict[str, Any]:
        """Implement agent logic with resource tracking."""
        # Get input from kwargs
        user_input = kwargs.get("input", "")

        # Track conversation history
        self.history.append(f"User: {user_input}")

        # Simulate processing with your own logic here
        response = f"Response to: {user_input}"
        self.history.append(f"Assistant: {response}")

        # Calculate token usage (simplified example)
        input_tokens = len(user_input.split())
        output_tokens = len(response.split())
        total_tokens = input_tokens + output_tokens
        cost = self.cost_per_token * Decimal(total_tokens)

        return {
            "response": response,
            "token_count": total_tokens,
            "cost": cost
        }
```

### Using Your Custom Agent

Register and use your custom agent with the framework:

```python
# Create your custom agent
my_agent = CustomAssistantAgent("my_assistant", model="gpt-4")

# Register with the framework
registered_agent = agent_api.create_agent(
    name=my_agent.name,
    initial_budget=Decimal("50.0"),
    priority=6
)

# Use your agent and update its budget
for question in ["What is the weather?", "Tell me a joke", "Explain quantum physics"]:
    # Run the agent
    result = my_agent.run(input=question)

    # Get the current budget
    current_budget = budget_api.get_budget(registered_agent.id)

    # Update the budget
    new_budget = current_budget - result["cost"]
    budget_api.update_budget(registered_agent.id, new_budget)

    print(f"Response: {result['response']}")
    print(f"Cost: {result['cost']}")
    print(f"Remaining budget: {new_budget}")
```

## Multi-Agent Coordination

Coordinate multiple agents working together:

```python
from safeguards.types.enums import AgentPriority

def run_pipeline(input_text):
    """Run a multi-agent pipeline with budget awareness."""
    results = {}

    # Check if all agents have sufficient budget
    for agent_id in [research_agent.id, summarization_agent.id]:
        budget = budget_api.get_budget(agent_id)
        if budget < Decimal("5.0"):
            print(f"Agent {agent_id} has insufficient budget: {budget}")
            # Optional: request emergency budget allocation
            budget_api.request_emergency_allocation(agent_id, Decimal("10.0"))

    # Step 1: Research agent processes the input
    research_result = research_agent.run(input=input_text)
    research_cost = research_result.get("cost", Decimal("0"))
    budget_api.update_budget(
        research_agent.id,
        budget_api.get_budget(research_agent.id) - research_cost
    )
    results["research"] = research_result

    # Step 2: Summarization agent processes research output
    summary_result = summarization_agent.run(
        input=research_result.get("response", "")
    )
    summary_cost = summary_result.get("cost", Decimal("0"))
    budget_api.update_budget(
        summarization_agent.id,
        budget_api.get_budget(summarization_agent.id) - summary_cost
    )
    results["summary"] = summary_result

    return results
```

## Metrics and Monitoring

Access metrics for analysis and monitoring:

```python
# Get metrics for a specific agent
agent_metrics = metrics_api.get_agent_metrics(assistant_agent.id)
print(f"Agent metrics: {agent_metrics}")

# Get metrics for a budget pool
pool_metrics = metrics_api.get_pool_metrics(critical_pool.id)
print(f"Pool metrics: {pool_metrics}")

# Get all agent metrics for analysis
all_agent_metrics = metrics_api.get_all_agent_metrics()
for agent_id, metrics in all_agent_metrics.items():
    print(f"Agent {agent_id}: {metrics}")

# Get budget usage history for an agent
usage_history = metrics_api.get_agent_usage_history(
    assistant_agent.id,
    start_time="2023-01-01T00:00:00Z",
    end_time="2023-01-02T00:00:00Z"
)
```

## Handling Violations and Notifications

Set up violation handling and notifications:

```python
from safeguards.types.enums import AlertSeverity, ViolationType

# Setup notification callbacks
def budget_alert_callback(agent_id, alert_type, severity, message):
    print(f"ALERT: {severity} - {message} for agent {agent_id}")
    # Implement your handling logic here

# Register the callback with the notification manager
notification_manager.register_callback(
    "budget_alerts",
    budget_alert_callback
)

# Report a violation
violation_reporter.report_violation(
    agent_id=assistant_agent.id,
    violation_type=ViolationType.BUDGET_EXCEEDED,
    severity=AlertSeverity.HIGH,
    message="Agent has exceeded its allocated budget",
    details={
        "current_budget": Decimal("-10.0"),
        "initial_budget": Decimal("100.0"),
        "overage": Decimal("10.0")
    }
)
```

## Advanced Configuration

Configure the framework for your specific needs:

```python
from safeguards.config import SafetyConfig

# Create a custom configuration
config = SafetyConfig(
    enable_emergency_allocation=True,
    default_agent_priority=5,
    low_budget_threshold_percentage=10,
    enable_auto_rebalancing=True,
    rebalance_interval_seconds=3600,  # 1 hour
    default_notification_severity=AlertSeverity.MEDIUM
)

# Apply configuration
budget_coordinator.apply_config(config)
```

## Next Steps

Now that you've learned the basics, explore:

- [Budget Management Guide](guides/budget_management.md) for advanced budget techniques
- [API Reference](api/core.md) for detailed API documentation
- [Agent Coordination](guides/agent_coordination.md) for complex multi-agent scenarios
- [Safety Rules System](guides/safety_rules.md) for implementing guardrails

For complete examples, check the [examples directory](../examples/) in the source code.
