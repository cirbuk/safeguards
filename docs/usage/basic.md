# Basic Usage Guide

This guide covers the basic usage of the Safeguards.

## Core Components

The framework consists of four main components:

1. Budget Management
2. Resource Monitoring
3. Safety Guardrails
4. Notification System

## Basic Setup

```python
from safeguards import BudgetManager, ResourceMonitor, NotificationManager
from safeguards.guardrails import BudgetGuardrail, ResourceGuardrail
from safeguards.types import Agent

# Initialize managers
budget_manager = BudgetManager(
    total_budget=1000,
    hourly_limit=100,
    daily_limit=500,
    warning_threshold=75.0,
)

resource_monitor = ResourceMonitor(
    cpu_threshold=80.0,
    memory_threshold=85.0,
    disk_threshold=90.0,
)

notification_manager = NotificationManager()

# Create guardrails
guardrails = [
    BudgetGuardrail(budget_manager),
    ResourceGuardrail(resource_monitor),
]

# Create agent with safety controls
agent = Agent(
    name="safe_agent",
    instructions="Your instructions here",
    guardrails=guardrails,
)
```

## Running an Agent

```python
# Simple execution
result = agent.run(input_data="Your input here")

# With context
context = {
    "max_tokens": 1000,
    "temperature": 0.7,
}
result = agent.run(input_data="Your input here", **context)
```

## Monitoring Usage

```python
# Check budget usage
remaining_budget = budget_manager.get_remaining_budget(agent.id)
usage_percent = budget_manager.get_budget_usage_percent(agent.id)

# Check resource usage
metrics = resource_monitor.get_current_metrics()
print(resource_monitor.get_resource_usage_summary(metrics))

# Get notifications
notifications = notification_manager.get_notifications(agent_id=agent.id)
```

## Error Handling

```python
from safeguards.notifications import NotificationLevel

try:
    result = agent.run(input_data="Your input here")
except Exception as e:
    notification_manager.notify(
        level=NotificationLevel.ERROR,
        message=f"Agent execution failed: {str(e)}",
        agent_id=agent.id,
    )
```

## Best Practices

1. Always set appropriate thresholds for your environment
2. Monitor notification logs regularly
3. Handle budget overages gracefully
4. Implement proper error handling
5. Regular resource usage checks

## Example: Complete Workflow

```python
from safeguards import (
    BudgetManager,
    ResourceMonitor,
    NotificationManager,
    SafetyController,
)
from safeguards.types import Agent
from safeguards.notifications import NotificationLevel

# Initialize components
budget_manager = BudgetManager(total_budget=1000)
resource_monitor = ResourceMonitor()
notification_manager = NotificationManager()

# Create safety controller
controller = SafetyController(
    budget_manager=budget_manager,
    resource_monitor=resource_monitor,
    notification_manager=notification_manager,
)

# Create agent
agent = Agent(
    name="safe_agent",
    instructions="Your instructions here",
    controller=controller,
)

try:
    # Check resources before running
    if resource_monitor.has_exceeded_thresholds():
        notification_manager.notify(
            level=NotificationLevel.WARNING,
            message="High resource usage detected",
            agent_id=agent.id,
        )

    # Check budget
    if not budget_manager.has_sufficient_budget(agent.id):
        notification_manager.notify(
            level=NotificationLevel.ERROR,
            message="Insufficient budget",
            agent_id=agent.id,
        )
        raise ValueError("Insufficient budget")

    # Run agent
    result = agent.run(input_data="Your input here")

    # Record cost
    budget_manager.record_cost(agent.id, cost=10.0)

except Exception as e:
    notification_manager.notify(
        level=NotificationLevel.ERROR,
        message=f"Execution failed: {str(e)}",
        agent_id=agent.id,
    )
    raise
```

## Next Steps

- [Budget Management Guide](budget.md)
- [Resource Monitoring Guide](resources.md)
- [Safety Guardrails Guide](guardrails.md)
- [Notification System Guide](notifications.md)
