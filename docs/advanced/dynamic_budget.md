# Dynamic Budget Allocation

This guide covers advanced techniques for dynamic budget allocation in the Safeguards, enabling more adaptive and efficient resource management for your agent systems.

## Introduction to Dynamic Allocation

Dynamic budget allocation allows your system to:

- Adapt to changing workloads and priorities
- Optimize resource utilization
- Implement complex allocation policies
- Respond to emergent patterns in agent behavior
- Create self-tuning budget management

## Advanced Budget Pool Strategies

### Adaptive Pool Sizing

Automatically size budget pools based on usage patterns:

```python
from decimal import Decimal
from safeguards.core.budget_coordination import BudgetCoordinator
from safeguards.core.dynamic_budget import AdaptivePoolSizer
from safeguards.types.enums import AgentPriority

# Create core components
notification_manager = NotificationManager()
budget_coordinator = BudgetCoordinator(notification_manager)

# Create an adaptive pool sizer
pool_sizer = AdaptivePoolSizer(
    budget_coordinator=budget_coordinator,
    adjustment_interval_seconds=3600,  # Check every hour
    min_pool_budget=Decimal("100.0"),
    max_pool_budget=Decimal("10000.0")
)

# Register a budget pool for adaptive sizing
pool_id = budget_coordinator.create_budget_pool(
    name="adaptive_pool",
    initial_budget=Decimal("1000.0"),
    priority=7
)

# Configure adaptive sizing rules
pool_sizer.configure_pool(
    pool_id=pool_id,
    utilization_target=0.7,  # Target 70% utilization
    growth_rate=0.2,  # Grow by 20% when needed
    shrink_rate=0.1,  # Shrink by 10% when underutilized
    min_agents_for_adjustment=3  # Require at least 3 agents for statistical significance
)

# Start the adaptive sizer
pool_sizer.start()
```

### Priority-Based Reallocation

Dynamically reallocate budgets based on agent priorities:

```python
from safeguards.core.dynamic_budget import PriorityBasedReallocator
from safeguards.types.enums import AgentPriority
from decimal import Decimal

# Create a priority-based reallocator
reallocator = PriorityBasedReallocator(
    budget_coordinator=budget_coordinator,
    reallocation_interval_seconds=1800,  # Check every 30 minutes
    emergency_threshold=Decimal("5.0")  # Emergency when budget below 5.0
)

# Register agents with different priorities
agent_ids = []
priorities = [
    AgentPriority.CRITICAL,
    AgentPriority.HIGH,
    AgentPriority.MEDIUM,
    AgentPriority.LOW
]

for i, priority in enumerate(priorities):
    agent_id = budget_coordinator.register_agent(
        agent_id=f"agent_{i}",
        pool_id=pool_id,
        priority=priority.value,
        initial_budget=Decimal("50.0")
    )
    agent_ids.append(agent_id)

# Configure reallocation rules
reallocator.configure(
    min_budget_percentage=0.1,  # Ensure all agents have at least 10% of their initial budget
    reserve_percentage=0.2,  # Keep 20% in reserve for emergency allocations
    priority_weights={
        AgentPriority.CRITICAL.value: 4.0,
        AgentPriority.HIGH.value: 2.0,
        AgentPriority.MEDIUM.value: 1.0,
        AgentPriority.LOW.value: 0.5
    }
)

# Start the reallocator
reallocator.start()
```

## Time-Based Budget Control

### Budget Rate Limiting

Implement rate limits for budget consumption:

```python
from safeguards.budget.rate_limiter import BudgetRateLimiter
from datetime import timedelta
from decimal import Decimal

# Create a budget rate limiter
rate_limiter = BudgetRateLimiter(budget_coordinator)

# Configure rate limits for different agents
rate_limiter.set_rate_limit(
    agent_id="agent_0",  # Critical agent
    max_budget=Decimal("100.0"),
    time_window=timedelta(hours=1)
)

rate_limiter.set_rate_limit(
    agent_id="agent_1",  # High priority agent
    max_budget=Decimal("50.0"),
    time_window=timedelta(hours=1)
)

rate_limiter.set_rate_limit(
    agent_id="agent_2",  # Medium priority agent
    max_budget=Decimal("25.0"),
    time_window=timedelta(hours=1)
)

rate_limiter.set_rate_limit(
    agent_id="agent_3",  # Low priority agent
    max_budget=Decimal("10.0"),
    time_window=timedelta(hours=1)
)

# Check if an operation is within rate limits
def execute_agent_operation(agent_id, operation_cost):
    """Execute an agent operation with rate limiting."""
    if rate_limiter.check_rate_limit(agent_id, operation_cost):
        # Perform operation
        rate_limiter.record_usage(agent_id, operation_cost)
        return {"status": "success", "cost": operation_cost}
    else:
        # Rate limit exceeded
        return {
            "status": "error",
            "message": "Rate limit exceeded",
            "retry_after_seconds": rate_limiter.get_retry_after(agent_id)
        }
```

### Time-of-Day Budgeting

Adjust budgets based on time of day:

```python
from safeguards.budget.time_scheduler import BudgetTimeScheduler
from datetime import time, datetime, timedelta
from decimal import Decimal

# Create a time-based budget scheduler
scheduler = BudgetTimeScheduler(budget_coordinator)

# Configure time-based budget profiles for a pool
scheduler.add_time_profile(
    pool_id=pool_id,
    schedule={
        # Business hours (9 AM - 5 PM): High budget
        "business_hours": {
            "budget": Decimal("5000.0"),
            "start_time": time(9, 0),  # 9:00 AM
            "end_time": time(17, 0),   # 5:00 PM
            "days": [0, 1, 2, 3, 4]    # Monday to Friday
        },
        # Evening hours (5 PM - 10 PM): Medium budget
        "evening_hours": {
            "budget": Decimal("2000.0"),
            "start_time": time(17, 0),  # 5:00 PM
            "end_time": time(22, 0),    # 10:00 PM
            "days": [0, 1, 2, 3, 4]     # Monday to Friday
        },
        # Night hours (10 PM - 9 AM): Low budget
        "night_hours": {
            "budget": Decimal("500.0"),
            "start_time": time(22, 0),  # 10:00 PM
            "end_time": time(9, 0),     # 9:00 AM
            "days": [0, 1, 2, 3, 4]     # Monday to Friday
        },
        # Weekend: Medium budget all day
        "weekend": {
            "budget": Decimal("1000.0"),
            "start_time": time(0, 0),   # 12:00 AM
            "end_time": time(23, 59),   # 11:59 PM
            "days": [5, 6]              # Saturday and Sunday
        }
    }
)

# Apply the current time-based profile
scheduler.apply_current_profile(pool_id)

# Start the scheduler
scheduler.start()
```

## Budget Learning & Optimization

### Usage Pattern Learning

Learn from agent usage patterns to optimize budgets:

```python
from safeguards.budget.pattern_learner import BudgetPatternLearner
from datetime import datetime, timedelta
import numpy as np

# Create a budget pattern learner
learner = BudgetPatternLearner(
    budget_coordinator=budget_coordinator,
    learning_period_days=14,  # Learn from 2 weeks of data
    prediction_confidence=0.8,  # Require 80% confidence for predictions
    min_data_points=100  # Require at least 100 data points
)

# Start collecting usage data
learner.start_collection()

# After collecting data, analyze patterns
patterns = learner.analyze_patterns(
    agent_id="agent_0",
    time_bucket_minutes=60  # Analyze in 1-hour buckets
)

# Predict future budget needs
prediction = learner.predict_budget_needs(
    agent_id="agent_0",
    future_time=datetime.now() + timedelta(hours=24),  # Predict for tomorrow
    prediction_window_hours=24  # Predict for 24 hours
)

# Apply optimized budgets based on predictions
def apply_optimized_budgets():
    """Apply optimized budgets based on learned patterns."""
    for agent_id in agent_ids:
        next_day_prediction = learner.predict_budget_needs(
            agent_id=agent_id,
            future_time=datetime.now() + timedelta(hours=24),
            prediction_window_hours=24
        )

        if next_day_prediction and next_day_prediction.confidence >= 0.8:
            # Apply prediction with a 20% safety margin
            optimized_budget = next_day_prediction.predicted_budget * Decimal("1.2")

            # Update agent budget
            budget_coordinator.update_budget(
                agent_id=agent_id,
                budget=optimized_budget
            )

            print(f"Applied optimized budget of {optimized_budget} to {agent_id}")
```

### Reinforcement Learning for Budget Allocation

Use reinforcement learning to optimize budget allocation:

```python
from safeguards.budget.rl_optimizer import BudgetRLOptimizer
from safeguards.types.metrics import BudgetMetrics

# Create a reinforcement learning budget optimizer
rl_optimizer = BudgetRLOptimizer(
    budget_coordinator=budget_coordinator,
    learning_rate=0.01,
    discount_factor=0.9,
    exploration_rate=0.1,
    state_features=[
        "time_of_day",
        "day_of_week",
        "agent_priority",
        "recent_usage",
        "remaining_budget_percentage"
    ]
)

# Define reward function
def reward_function(state, action, next_state):
    """Calculate reward for budget allocation actions."""
    # Reward for high utilization without exhaustion
    utilization = next_state.get("utilization", 0)

    # Penalize budget exhaustion
    exhaustion_penalty = -100 if next_state.get("exhausted", False) else 0

    # Reward for completing tasks
    task_reward = next_state.get("completed_tasks", 0) * 10

    # Penalize large budget changes
    stability_penalty = -abs(action.get("budget_change", 0)) * 0.1

    return utilization * 50 + exhaustion_penalty + task_reward + stability_penalty

# Configure the optimizer
rl_optimizer.configure(
    reward_function=reward_function,
    training_episodes=1000,
    max_budget_change_percentage=0.3,  # Limit changes to 30%
    update_interval_minutes=60  # Update every hour
)

# Start the optimizer
rl_optimizer.start()

# Get budget allocation policy
allocation_policy = rl_optimizer.get_policy()
print(f"Learned allocation policy: {allocation_policy}")
```

## Multi-Dimensional Budgeting

### Resource Type Budgeting

Manage multiple resource types independently:

```python
from safeguards.budget.multi_resource import MultiResourceBudget
from decimal import Decimal

# Create a multi-resource budget manager
multi_resource = MultiResourceBudget(budget_coordinator)

# Define resource types
resource_types = [
    "api_calls",
    "compute_seconds",
    "storage_gb",
    "bandwidth_gb"
]

# Initialize multi-resource budgets for an agent
agent_id = "agent_0"
multi_resource.initialize_budgets(
    agent_id=agent_id,
    budgets={
        "api_calls": Decimal("1000"),
        "compute_seconds": Decimal("3600"),  # 1 hour
        "storage_gb": Decimal("10"),
        "bandwidth_gb": Decimal("20")
    }
)

# Check if operation is within budget
def check_multi_resource_budget(agent_id, resources_needed):
    """Check if an operation is within multi-resource budget limits."""
    for resource_type, amount in resources_needed.items():
        if not multi_resource.has_sufficient_budget(
            agent_id=agent_id,
            resource_type=resource_type,
            amount=amount
        ):
            return False, f"Insufficient {resource_type} budget"

    return True, "Operation within budget"

# Update multiple resource budgets after an operation
def update_multi_resource_budget(agent_id, resources_used):
    """Update multiple resource budgets after an operation."""
    for resource_type, amount in resources_used.items():
        current = multi_resource.get_budget(agent_id, resource_type)
        multi_resource.update_budget(
            agent_id=agent_id,
            resource_type=resource_type,
            budget=current - amount
        )

    # Get updated budgets
    updated_budgets = {
        rt: multi_resource.get_budget(agent_id, rt)
        for rt in resource_types
    }

    return updated_budgets
```

### Cost-Benefit Budgeting

Allocate budgets based on expected return on investment:

```python
from safeguards.budget.cost_benefit import CostBenefitAllocator
from decimal import Decimal

# Create a cost-benefit allocator
cb_allocator = CostBenefitAllocator(budget_coordinator)

# Register benefit metrics for different agent operations
cb_allocator.register_operation(
    agent_id="agent_0",
    operation_type="data_processing",
    average_cost=Decimal("5.0"),
    average_benefit=Decimal("15.0"),  # 3x ROI
    benefit_variability=0.2  # 20% variability
)

cb_allocator.register_operation(
    agent_id="agent_0",
    operation_type="content_generation",
    average_cost=Decimal("10.0"),
    average_benefit=Decimal("25.0"),  # 2.5x ROI
    benefit_variability=0.4  # 40% variability
)

cb_allocator.register_operation(
    agent_id="agent_1",
    operation_type="data_analysis",
    average_cost=Decimal("20.0"),
    average_benefit=Decimal("50.0"),  # 2.5x ROI
    benefit_variability=0.3  # 30% variability
)

# Optimize budget allocation to maximize benefit
optimized_allocation = cb_allocator.optimize_allocation(
    total_budget=Decimal("1000.0"),
    min_allocation_percentage=0.1,  # At least 10% for each agent
    risk_tolerance=0.5  # Balanced risk approach
)

# Apply optimized allocation
for agent_id, allocation in optimized_allocation.items():
    budget_coordinator.update_budget(
        agent_id=agent_id,
        budget=allocation
    )
    print(f"Allocated {allocation} to {agent_id} based on cost-benefit analysis")

# Evaluate allocation effectiveness
effectiveness = cb_allocator.evaluate_effectiveness(
    lookback_days=7,  # Evaluate last week's performance
    metric="roi"  # Return on investment
)

print(f"Allocation effectiveness: {effectiveness}")
```

## Budget Federation Across Systems

### Cross-System Budget Coordination

Coordinate budgets across multiple systems:

```python
from safeguards.budget.federation import BudgetFederation
import requests
from decimal import Decimal

# Create a budget federation coordinator
federation = BudgetFederation(
    local_system_id="system_a",
    local_coordinator=budget_coordinator
)

# Register remote systems
federation.register_remote_system(
    system_id="system_b",
    api_base_url="https://system-b.example.com/api/",
    api_key="system_b_api_key",
    trust_level=0.8  # High trust level
)

federation.register_remote_system(
    system_id="system_c",
    api_base_url="https://system-c.example.com/api/",
    api_key="system_c_api_key",
    trust_level=0.6  # Medium trust level
)

# Create a federated budget pool
federated_pool_id = federation.create_federated_pool(
    name="cross_system_pool",
    local_budget=Decimal("5000.0"),
    remote_allocations={
        "system_b": Decimal("2000.0"),
        "system_c": Decimal("1000.0")
    }
)

# Request budget from remote system
def request_remote_budget(agent_id, amount, remote_system_id):
    """Request budget from a remote system."""
    success, details = federation.request_remote_budget(
        agent_id=agent_id,
        remote_system_id=remote_system_id,
        amount=amount,
        request_metadata={
            "purpose": "critical_operation",
            "expected_duration_minutes": 30
        }
    )

    if success:
        print(f"Received {amount} from {remote_system_id} for {agent_id}")
        print(f"Transaction ID: {details.get('transaction_id')}")
        return True
    else:
        print(f"Failed to get budget from {remote_system_id}: {details.get('error')}")
        return False

# Synchronize budgets across systems
def synchronize_federation():
    """Synchronize budget information across federated systems."""
    sync_results = federation.synchronize()

    for system_id, result in sync_results.items():
        if result["success"]:
            print(f"Synchronized with {system_id}, last update: {result['last_sync']}")
        else:
            print(f"Failed to sync with {system_id}: {result['error']}")
```

## Best Practices

### Design Principles

1. **Start Conservative**: Begin with static allocations before implementing dynamic strategies
2. **Monitor Before Optimizing**: Collect usage data before implementing automatic adjustments
3. **Implement Safety Guardrails**: Add safeguards to prevent extreme budget changes
4. **Consider Time Horizons**: Balance short-term optimization with long-term sustainability
5. **Test Extreme Scenarios**: Verify behavior during unexpected spikes or drops in usage

### Implementation Tips

1. **Gradual Changes**: Make small, incremental budget adjustments rather than large swings
2. **Feedback Loops**: Collect performance metrics to evaluate allocation effectiveness
3. **Human Oversight**: Allow manual intervention for unexpected situations
4. **Transparency**: Make the reasoning behind budget decisions explainable
5. **Fallback Mechanisms**: Implement simple fallback rules if advanced strategies fail

## Conclusion

Dynamic budget allocation enables your agent system to adapt to changing conditions, optimize resource usage, and make intelligent decisions about resource allocation. By implementing these advanced strategies, you can create self-tuning systems that maximize value while maintaining safety constraints.

For more information, see:
- [Budget Management Basics](../guides/budget_management.md)
- [Agent Coordination](../guides/agent_coordination.md)
- [Monitoring Guide](../guides/monitoring.md)
