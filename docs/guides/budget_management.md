# Budget Management Guide

This guide provides a comprehensive overview of budget management features in the Safeguards.

## Core Concepts

### Budget Pools

Budget pools are containers for resources that can be allocated to agents. Each pool has:

- A unique identifier
- A name for easy reference
- An initial budget amount
- A priority level (1-10) that determines resource allocation priority
- Optional metadata for custom tracking

```python
from decimal import Decimal
from safeguards.core.budget_coordination import BudgetCoordinator
from safeguards.api import APIFactory, APIVersion

# Setup
notification_manager = NotificationManager()
budget_coordinator = BudgetCoordinator(notification_manager)
api_factory = APIFactory()
budget_api = api_factory.create_budget_api(APIVersion.V1, budget_coordinator)

# Create pools with different priorities
high_priority_pool = budget_api.create_budget_pool(
    name="critical_services",
    initial_budget=Decimal("1000.0"),
    priority=9
)

medium_priority_pool = budget_api.create_budget_pool(
    name="standard_services",
    initial_budget=Decimal("2000.0"),
    priority=5
)

low_priority_pool = budget_api.create_budget_pool(
    name="background_tasks",
    initial_budget=Decimal("500.0"),
    priority=2
)
```

### Agent Budgets

Agents consume resources from budget pools. The framework tracks:

- Initial budget allocation
- Current budget level
- Usage patterns over time
- Rate of consumption

```python
# Create agents with different priorities
critical_agent = agent_api.create_agent(
    name="critical_agent",
    initial_budget=Decimal("200.0"),
    priority=8
)

standard_agent = agent_api.create_agent(
    name="standard_agent",
    initial_budget=Decimal("300.0"),
    priority=5
)

# Check budgets
critical_budget = budget_api.get_budget(critical_agent.id)
standard_budget = budget_api.get_budget(standard_agent.id)

print(f"Critical agent budget: {critical_budget}")
print(f"Standard agent budget: {standard_budget}")
```

## Budget Allocation Strategies

### Fixed Allocation

The simplest approach is to assign fixed budgets to agents:

```python
agent = agent_api.create_agent(
    name="fixed_budget_agent",
    initial_budget=Decimal("100.0"),
    priority=5
)
```

### Dynamic Allocation

For more flexibility, implement dynamic allocation based on usage patterns:

```python
def allocate_dynamic_budget(agent_id, base_budget, usage_multiplier=1.2):
    """Allocate budget based on recent usage patterns."""
    # Get recent usage metrics
    usage_history = metrics_api.get_agent_usage_history(
        agent_id,
        start_time="2023-01-01T00:00:00Z",
        end_time="2023-01-02T00:00:00Z"
    )

    # Calculate average usage
    if usage_history:
        total_usage = sum(entry["amount"] for entry in usage_history)
        avg_usage = total_usage / len(usage_history)

        # Allocate budget based on usage pattern with a buffer
        new_budget = Decimal(avg_usage) * Decimal(usage_multiplier)

        # Ensure minimum base budget
        return max(new_budget, Decimal(base_budget))

    # Default to base budget if no history
    return Decimal(base_budget)

# Apply dynamic allocation
agent_id = standard_agent.id
new_budget = allocate_dynamic_budget(agent_id, "50.0")
budget_api.update_budget(agent_id, new_budget)
```

### Priority-Based Allocation

Prioritize critical agents during resource constraints:

```python
def allocate_by_priority(agents, total_available_budget):
    """Allocate budget based on agent priorities."""
    # Sort agents by priority (highest first)
    sorted_agents = sorted(agents, key=lambda a:
        budget_coordinator.get_agent_priority(a.id), reverse=True)

    remaining_budget = Decimal(total_available_budget)
    allocations = {}

    # First pass: ensure minimum allocations for critical agents
    for agent in sorted_agents:
        priority = budget_coordinator.get_agent_priority(agent.id)
        if priority >= 8:  # Critical priority
            min_allocation = Decimal("50.0")  # Minimum for critical agents
            allocations[agent.id] = min_allocation
            remaining_budget -= min_allocation

    # Second pass: allocate remaining budget proportionally by priority
    if remaining_budget > Decimal("0"):
        total_weights = sum(budget_coordinator.get_agent_priority(a.id)
                            for a in sorted_agents)

        for agent in sorted_agents:
            if agent.id not in allocations:
                # Non-critical agents get proportional allocation
                priority = budget_coordinator.get_agent_priority(agent.id)
                proportion = Decimal(priority) / Decimal(total_weights)
                allocation = remaining_budget * proportion
                allocations[agent.id] = allocation

    # Apply allocations
    for agent_id, allocation in allocations.items():
        budget_api.update_budget(agent_id, allocation)

    return allocations
```

## Budget Monitoring and Control

### Tracking Usage

Monitor budget consumption in real-time:

```python
def monitor_budget_usage(agent_id):
    """Monitor agent budget usage."""
    # Get current metrics
    metrics = metrics_api.get_agent_metrics(agent_id)

    # Calculate usage rate
    initial = metrics["initial_budget"]
    remaining = metrics["remaining_budget"]
    used = metrics["used_budget"]

    usage_percentage = (used / initial) * 100 if initial > 0 else 0

    print(f"Agent {agent_id} budget usage:")
    print(f"  Initial: {initial}")
    print(f"  Used: {used} ({usage_percentage:.2f}%)")
    print(f"  Remaining: {remaining}")

    # Check if budget is running low
    if usage_percentage > 80:
        print("  WARNING: Budget usage high (>80%)")

    return metrics
```

### Setting Spending Limits

Implement spending limits to prevent overconsumption:

```python
from safeguards.types import ViolationType, AlertSeverity

def set_spending_limit(agent_id, limit_amount):
    """Set a spending limit for an agent."""
    # Get current budget
    current_budget = budget_api.get_budget(agent_id)

    # Create a custom monitor function
    def monitor_spending(agent_id, amount):
        if amount > limit_amount:
            # Report violation
            violation_reporter.report_violation(
                agent_id=agent_id,
                violation_type=ViolationType.BUDGET_LIMIT_EXCEEDED,
                severity=AlertSeverity.MEDIUM,
                message=f"Agent exceeded spending limit of {limit_amount}",
                details={
                    "limit": limit_amount,
                    "spent": amount,
                    "overage": amount - limit_amount
                }
            )
            return False
        return True

    # Register monitor with the budget coordinator
    budget_coordinator.register_budget_monitor(agent_id, monitor_spending)

    return limit_amount
```

### Budget Alerts

Set up notifications for budget-related events:

```python
def setup_budget_alerts(threshold_percentage=15):
    """Set up budget alerts when agents approach depletion."""
    def low_budget_alert(agent_id, current_budget, initial_budget):
        # Calculate percentage remaining
        percentage = (current_budget / initial_budget) * 100 if initial_budget > 0 else 0

        if percentage <= threshold_percentage:
            # Generate alert
            violation_reporter.report_violation(
                agent_id=agent_id,
                violation_type=ViolationType.LOW_BUDGET,
                severity=AlertSeverity.MEDIUM,
                message=f"Agent budget below {threshold_percentage}% threshold",
                details={
                    "current_budget": current_budget,
                    "initial_budget": initial_budget,
                    "percentage_remaining": percentage
                }
            )
            return True
        return False

    # Register with each agent
    agent_ids = budget_coordinator.get_all_agent_ids()
    for agent_id in agent_ids:
        budget_coordinator.register_budget_monitor(
            agent_id,
            lambda a_id, amount: low_budget_alert(
                a_id,
                amount,
                budget_coordinator.get_agent_metrics(a_id)["initial_budget"]
            )
        )
```

## Advanced Budget Management

### Emergency Budget Allocation

Handle emergency budget needs:

```python
def handle_emergency_allocation(agent_id, requested_amount, reason):
    """Request emergency budget allocation for an agent."""
    current_metrics = metrics_api.get_agent_metrics(agent_id)
    agent_priority = budget_coordinator.get_agent_priority(agent_id)

    # Determine if emergency allocation is justified
    if agent_priority >= 7:  # High priority agent
        print(f"Approving emergency allocation for high-priority agent {agent_id}")
        # Approve full amount
        approved_amount = requested_amount
    elif "critical" in reason.lower():
        print(f"Approving partial emergency allocation due to critical reason")
        # Approve partial amount
        approved_amount = requested_amount * Decimal("0.5")
    else:
        print(f"Denying emergency allocation for low-priority non-critical request")
        # Deny request
        approved_amount = Decimal("0")

    if approved_amount > Decimal("0"):
        # Update budget
        new_budget = current_metrics["remaining_budget"] + approved_amount
        budget_api.update_budget(agent_id, new_budget)
        print(f"Emergency allocation complete: {approved_amount} added to agent {agent_id}")

    return approved_amount
```

### Budget Rebalancing

Implement automated budget rebalancing across agents:

```python
def rebalance_agent_budgets(pool_id):
    """Rebalance budgets among agents in a pool based on priorities and usage."""
    # Get all agents in the pool
    pool_agents = budget_coordinator.get_pool_agents(pool_id)

    # Get total remaining budget in the pool
    pool_metrics = metrics_api.get_pool_metrics(pool_id)
    total_remaining = pool_metrics["remaining_budget"]

    # Calculate priority-weighted allocation
    agent_priorities = {
        agent_id: budget_coordinator.get_agent_priority(agent_id)
        for agent_id in pool_agents
    }

    # Adjust based on recent usage patterns
    usage_weights = {}
    for agent_id in pool_agents:
        usage = metrics_api.get_agent_usage_history(
            agent_id,
            start_time="2023-01-01T00:00:00Z",
            end_time="2023-01-02T00:00:00Z"
        )

        if usage:
            # Calculate average hourly usage
            total_usage = sum(entry["amount"] for entry in usage)
            usage_weights[agent_id] = total_usage / len(usage)
        else:
            # Default weight if no usage history
            usage_weights[agent_id] = Decimal("1.0")

    # Combine priority and usage factors for final allocation
    allocation_weights = {}
    total_weight = Decimal("0")

    for agent_id in pool_agents:
        # Combine priority (70% weight) and usage (30% weight)
        weight = (Decimal(agent_priorities[agent_id]) * Decimal("0.7") +
                  usage_weights[agent_id] * Decimal("0.3"))
        allocation_weights[agent_id] = weight
        total_weight += weight

    # Calculate and apply new budgets
    new_allocations = {}
    for agent_id, weight in allocation_weights.items():
        proportion = weight / total_weight if total_weight > 0 else 0
        new_budget = total_remaining * proportion
        budget_api.update_budget(agent_id, new_budget)
        new_allocations[agent_id] = new_budget

    return new_allocations
```

## Best Practices

### Tracking and Logging

Implement comprehensive logging of budget changes:

```python
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("budget_management.log"),
        logging.StreamHandler()
    ]
)

def log_budget_change(agent_id, old_budget, new_budget, reason):
    """Log budget changes with detailed information."""
    change_amount = new_budget - old_budget
    agent_priority = budget_coordinator.get_agent_priority(agent_id)

    logging.info(
        f"Budget change for agent {agent_id} (priority {agent_priority}): "
        f"{old_budget} → {new_budget} ({change_amount:+}). "
        f"Reason: {reason}"
    )

    # Also store in database or metrics system
    metrics_api.record_budget_change(
        agent_id=agent_id,
        old_budget=old_budget,
        new_budget=new_budget,
        change_amount=change_amount,
        timestamp=datetime.now().isoformat(),
        reason=reason
    )
```

### Regular Budget Reviews

Implement systematic budget review processes:

```python
def scheduled_budget_review(frequency_hours=24):
    """Schedule regular budget reviews."""
    def review_budgets():
        # Get all agents
        all_agent_metrics = metrics_api.get_all_agent_metrics()

        # Identify potential issues
        for agent_id, metrics in all_agent_metrics.items():
            initial = metrics["initial_budget"]
            remaining = metrics["remaining_budget"]

            # Check for severely underused budgets
            if remaining > initial * Decimal("0.9"):
                logging.info(f"Agent {agent_id} using less than 10% of budget - consider reducing")

            # Check for nearly depleted budgets
            if remaining < initial * Decimal("0.1"):
                logging.warning(f"Agent {agent_id} budget nearly depleted - consider increasing")

        # Review budget pools
        all_pool_metrics = metrics_api.get_all_pool_metrics()
        for pool_id, metrics in all_pool_metrics.items():
            # Check pool health
            if metrics["remaining_budget"] < metrics["initial_budget"] * Decimal("0.2"):
                logging.warning(f"Pool {pool_id} below 20% remaining - consider rebalancing")

    # Schedule the review (implementation depends on scheduling system)
    # schedule.every(frequency_hours).hours.do(review_budgets)

    return "Budget review scheduled every {frequency_hours} hours"
```

### Graceful Degradation

Implement strategies for handling budget depletion:

```python
def setup_graceful_degradation(agent_id, service_levels):
    """Configure agent for graceful degradation as budget depletes."""
    # Service levels example:
    # {
    #   "full": {"min_percentage": 50, "features": ["high_quality", "all_capabilities"]},
    #   "standard": {"min_percentage": 20, "features": ["medium_quality", "core_capabilities"]},
    #   "minimal": {"min_percentage": 5, "features": ["low_quality", "basic_capabilities"]},
    #   "emergency": {"min_percentage": 0, "features": ["text_only", "critical_only"]}
    # }

    def determine_service_level(current_budget, initial_budget):
        if initial_budget <= 0:
            return "emergency"

        percentage = (current_budget / initial_budget) * 100

        for level, config in sorted(
            service_levels.items(),
            key=lambda x: x[1]["min_percentage"],
            reverse=True
        ):
            if percentage >= config["min_percentage"]:
                return level

        return "emergency"

    # Store the configuration with the agent
    budget_coordinator.store_agent_metadata(
        agent_id,
        {
            "service_levels": service_levels,
            "service_level_func": determine_service_level
        }
    )

    # Example usage in agent implementation:
    # def run(self, **kwargs):
    #     # Get current metrics
    #     metrics = metrics_api.get_agent_metrics(self.id)
    #
    #     # Get service level function
    #     metadata = budget_coordinator.get_agent_metadata(self.id)
    #     service_level_func = metadata.get("service_level_func")
    #
    #     if service_level_func:
    #         level = service_level_func(
    #             metrics["remaining_budget"],
    #             metrics["initial_budget"]
    #         )
    #         features = metadata["service_levels"][level]["features"]
    #
    #         # Adjust behavior based on available features
    #         # ...

    return f"Graceful degradation configured for agent {agent_id}"
```

## Conclusion

Effective budget management is critical for building safe, reliable agent systems. By implementing the strategies outlined in this guide, you can ensure that your agents operate within resource constraints while prioritizing high-value activities.

For more information, see the [API Reference](../api/core.md) and [Example Implementations](../examples/budget_management_examples.py).
