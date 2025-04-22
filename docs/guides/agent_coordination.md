# Agent Coordination Guide

This guide explains how to coordinate multiple agents within the Safeguards, covering communication patterns, resource sharing, and task allocation.

## Introduction to Agent Coordination

In multi-agent systems, coordination is essential for:
- Efficiently allocating resources among agents
- Managing dependencies between agent tasks
- Preventing conflicts and contention
- Enabling specialized agents to collaborate effectively
- Supporting graceful degradation when resources are constrained

The Safeguards provides several mechanisms to facilitate agent coordination.

## Basic Multi-Agent Setup

### Creating Multiple Agents

Start by creating different agents with appropriate priorities:

```python
from decimal import Decimal
from safeguards.core.budget_coordination import BudgetCoordinator
from safeguards.core.notification_manager import NotificationManager
from safeguards.api import APIFactory, APIVersion

# Setup core components
notification_manager = NotificationManager()
budget_coordinator = BudgetCoordinator(notification_manager)
api_factory = APIFactory()

# Create APIs
budget_api = api_factory.create_budget_api(APIVersion.V1, budget_coordinator)
agent_api = api_factory.create_agent_api(APIVersion.V1, budget_coordinator)

# Create agents with different roles and priorities
research_agent = agent_api.create_agent(
    name="research_agent",
    initial_budget=Decimal("100.0"),
    priority=7
)

analysis_agent = agent_api.create_agent(
    name="analysis_agent",
    initial_budget=Decimal("80.0"),
    priority=5
)

summarization_agent = agent_api.create_agent(
    name="summarization_agent",
    initial_budget=Decimal("50.0"),
    priority=3
)
```

### Creating Shared Budget Pools

For resource sharing, create budget pools that agents can draw from:

```python
# Create shared pools for different agent groups
high_priority_pool = budget_api.create_budget_pool(
    name="high_priority_tasks",
    initial_budget=Decimal("500.0"),
    priority=8
)

general_pool = budget_api.create_budget_pool(
    name="general_tasks",
    initial_budget=Decimal("1000.0"),
    priority=5
)
```

## Communication Patterns

### Event-Based Communication

Implement communication between agents using the notification system:

```python
from safeguards.types import AlertSeverity

def agent_communication_handler(notification):
    """Handle inter-agent communication."""
    if notification.agent_id and notification.metadata.get("message_type") == "agent_communication":
        target_agent_id = notification.metadata.get("target_agent_id")
        message = notification.metadata.get("message")

        print(f"Message from {notification.agent_id} to {target_agent_id}: {message}")

        # Process the message and take action
        # ...

        return True
    return False

# Register the handler
notification_manager.add_handler(agent_communication_handler)

# Send a message from one agent to another
notification_manager.send_alert(
    agent_id=research_agent.id,
    severity=AlertSeverity.INFORMATIONAL,
    message="Research results ready for analysis",
    metadata={
        "message_type": "agent_communication",
        "target_agent_id": analysis_agent.id,
        "message": "Research complete. Analysis required on data at path /tmp/research_data.json."
    }
)
```

### Shared State

For more direct coordination, implement a shared state service:

```python
from safeguards.coordination.shared_state import SharedStateManager

# Create a shared state manager
state_manager = SharedStateManager()

# Agent 1 updates state
state_manager.update_state(
    owner_id=research_agent.id,
    key="research_data",
    value={
        "status": "complete",
        "timestamp": "2023-07-26T15:30:00Z",
        "location": "/tmp/research_data.json"
    }
)

# Agent 2 reads state
research_data = state_manager.get_state(
    reader_id=analysis_agent.id,
    key="research_data"
)

if research_data and research_data.get("status") == "complete":
    print(f"Analysis agent processing data from {research_data.get('location')}")
    # Process the data
```

## Task Allocation Patterns

### Pipeline Pattern

Implement a sequential processing pipeline where agents perform tasks in order:

```python
def run_analysis_pipeline(input_data):
    """Execute a multi-stage pipeline of agent tasks."""
    results = {}

    # Stage 1: Research agent gathers information
    research_result = research_agent.run(input=input_data)
    research_cost = research_result.get("cost", Decimal("0"))

    # Update budget
    budget_api.update_budget(
        research_agent.id,
        budget_api.get_budget(research_agent.id) - research_cost
    )
    results["research"] = research_result

    # Stage 2: Analysis agent processes research data
    analysis_result = analysis_agent.run(
        input=research_result.get("output", "")
    )
    analysis_cost = analysis_result.get("cost", Decimal("0"))

    # Update budget
    budget_api.update_budget(
        analysis_agent.id,
        budget_api.get_budget(analysis_agent.id) - analysis_cost
    )
    results["analysis"] = analysis_result

    # Stage 3: Summarization agent creates final summary
    summary_result = summarization_agent.run(
        input=analysis_result.get("output", "")
    )
    summary_cost = summary_result.get("cost", Decimal("0"))

    # Update budget
    budget_api.update_budget(
        summarization_agent.id,
        budget_api.get_budget(summarization_agent.id) - summary_cost
    )
    results["summary"] = summary_result

    return results
```

### Fan-Out Pattern

Implement parallel processing for independent tasks:

```python
import concurrent.futures
from typing import List, Dict, Any

def run_parallel_tasks(task_inputs: List[str]) -> List[Dict[str, Any]]:
    """Execute multiple independent tasks in parallel."""
    results = []

    # Use a thread pool for concurrent execution
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        # Map tasks to agents
        futures = [
            executor.submit(research_agent.run, input=task)
            for task in task_inputs
        ]

        # Collect results
        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()
                results.append(result)

                # Update budget after task completion
                budget_api.update_budget(
                    research_agent.id,
                    budget_api.get_budget(research_agent.id) - result.get("cost", Decimal("0"))
                )
            except Exception as e:
                print(f"Task execution failed: {str(e)}")

    return results
```

## Resource Sharing

### Priority-Based Allocation

Implement resource allocation based on agent priorities:

```python
def allocate_resources_by_priority(agents, available_budget):
    """Allocate resources based on agent priorities."""
    # Sort agents by priority (highest first)
    sorted_agents = sorted(agents, key=lambda a:
        budget_coordinator.get_agent_priority(a.id), reverse=True)

    total_priority = sum(budget_coordinator.get_agent_priority(a.id) for a in sorted_agents)

    # Calculate proportional allocation
    allocations = {}
    for agent in sorted_agents:
        priority = budget_coordinator.get_agent_priority(agent.id)
        proportion = Decimal(priority) / Decimal(total_priority)
        allocation = available_budget * proportion
        allocations[agent.id] = allocation

        # Update agent budget
        budget_api.update_budget(agent.id, allocation)

    return allocations
```

### Dynamic Load Balancing

Implement dynamic reallocation based on agent needs:

```python
def balance_agent_resources(pool_id):
    """Balance resources among agents based on current workload."""
    # Get all agents in the pool
    agents = budget_coordinator.get_pool_agents(pool_id)

    # Calculate workload metrics
    agent_workloads = {}
    for agent_id in agents:
        metrics = metrics_api.get_agent_metrics(agent_id)
        recent_tasks = metrics.get("recent_task_count", 0)
        pending_tasks = metrics.get("pending_task_count", 0)

        # Higher value means higher workload
        workload_score = recent_tasks * 0.3 + pending_tasks * 0.7
        agent_workloads[agent_id] = workload_score

    # Get total pool budget
    pool_metrics = metrics_api.get_pool_metrics(pool_id)
    available_budget = pool_metrics["remaining_budget"]

    # Calculate new allocations
    total_workload = sum(agent_workloads.values()) or 1  # Avoid division by zero
    allocations = {}

    for agent_id, workload in agent_workloads.items():
        # Higher workload gets proportionally more budget
        proportion = Decimal(workload) / Decimal(total_workload)
        new_allocation = available_budget * proportion

        # Ensure minimum budget
        min_budget = Decimal("10.0")
        allocations[agent_id] = max(new_allocation, min_budget)

        # Apply new budget
        budget_api.update_budget(agent_id, allocations[agent_id])

    return allocations
```

## Dependency Management

### Task Dependencies

Manage dependencies between agent tasks:

```python
from safeguards.coordination.dependency_manager import DependencyManager

# Create a dependency manager
dependency_manager = DependencyManager()

# Define task dependencies
dependency_manager.add_dependency(
    task_id="analyze_data",
    depends_on="gather_data",
    agent_id=analysis_agent.id,
    required_resources=["data_file"]
)

# Check if dependencies are met
can_execute = dependency_manager.check_dependencies(
    task_id="analyze_data",
    available_resources=["data_file", "config"]
)

if can_execute:
    # Execute the task
    result = analysis_agent.run(task="analyze_data")
else:
    # Handle missing dependencies
    missing = dependency_manager.get_missing_dependencies("analyze_data")
    print(f"Cannot execute task due to missing dependencies: {missing}")
```

## Agent Coordination Patterns

### Supervisor Pattern

Implement a supervisor agent that coordinates other agents:

```python
from safeguards.types.agent import Agent
from typing import Dict, Any, List

class SupervisorAgent(Agent):
    def __init__(self, name, worker_agents=None):
        super().__init__(name)
        self.worker_agents = worker_agents or []

    def run(self, **kwargs) -> Dict[str, Any]:
        """Coordinate multiple worker agents."""
        task = kwargs.get("task", "")

        # Step 1: Decompose the task
        subtasks = self._decompose_task(task)

        # Step 2: Assign subtasks to workers
        assignments = self._assign_subtasks(subtasks)

        # Step 3: Monitor and collect results
        results = {}
        for agent_id, subtask in assignments.items():
            # Find the agent
            agent = next((a for a in self.worker_agents if a.id == agent_id), None)
            if agent:
                result = agent.run(task=subtask)
                results[agent_id] = result

                # Update budget
                current_budget = budget_api.get_budget(agent.id)
                cost = result.get("cost", Decimal("0"))
                budget_api.update_budget(agent.id, current_budget - cost)

        # Step 4: Combine results
        final_result = self._combine_results(results)

        return {
            "result": final_result,
            "subtask_count": len(subtasks),
            "worker_count": len(self.worker_agents)
        }

    def _decompose_task(self, task) -> List[str]:
        """Break a task into subtasks."""
        # Implementation depends on task type
        return [f"{task} - part {i}" for i in range(3)]

    def _assign_subtasks(self, subtasks) -> Dict[str, str]:
        """Assign subtasks to worker agents."""
        assignments = {}
        for i, subtask in enumerate(subtasks):
            if i < len(self.worker_agents):
                agent = self.worker_agents[i]
                assignments[agent.id] = subtask
        return assignments

    def _combine_results(self, results) -> Any:
        """Combine results from multiple agents."""
        # Implementation depends on result type
        combined = ""
        for agent_id, result in results.items():
            combined += f"{result.get('output', '')}\n"
        return combined
```

### Reactive Coordination

Implement event-driven coordination between agents:

```python
# Setup event subscriptions
agent_events = {
    "data_available": [],
    "analysis_complete": [],
    "error_reported": []
}

def subscribe_to_event(agent_id, event_type, callback):
    """Subscribe an agent to an event type."""
    if event_type in agent_events:
        agent_events[event_type].append({
            "agent_id": agent_id,
            "callback": callback
        })

def publish_event(source_agent_id, event_type, data):
    """Publish an event to all subscribers."""
    if event_type in agent_events:
        for subscriber in agent_events[event_type]:
            try:
                subscriber["callback"](source_agent_id, data)
            except Exception as e:
                print(f"Error in event handler: {str(e)}")

# Example event handler
def handle_data_available(source_agent_id, data):
    """Handle data availability events."""
    print(f"Data available from agent {source_agent_id}")

    # Trigger analysis agent
    analysis_result = analysis_agent.run(input=data)

    # Update budget
    current_budget = budget_api.get_budget(analysis_agent.id)
    cost = analysis_result.get("cost", Decimal("0"))
    budget_api.update_budget(analysis_agent.id, current_budget - cost)

    # Publish completion event
    publish_event(
        analysis_agent.id,
        "analysis_complete",
        analysis_result.get("output", "")
    )

# Subscribe analysis agent to data events
subscribe_to_event(
    analysis_agent.id,
    "data_available",
    handle_data_available
)

# Research agent publishes event when data is ready
publish_event(
    research_agent.id,
    "data_available",
    {"data": "Sample research data", "format": "json"}
)
```

## Best Practices

### Resource Efficiency

- **Prioritize Critical Agents**: Ensure critical agents have higher priority
- **Use Shared Pools**: Group related agents under shared budget pools
- **Monitor Resource Usage**: Track resource consumption across agents
- **Implement Graceful Degradation**: Plan for reduced functionality under resource constraints

### Communication Efficiency

- **Minimize Message Size**: Keep coordination messages concise
- **Use Appropriate Patterns**: Choose the right coordination pattern for your use case
- **Cache Common Data**: Avoid redundant data transfers between agents
- **Implement Timeouts**: Don't let agents wait indefinitely for responses

### Error Handling

- **Propagate Failures Appropriately**: Ensure errors in one agent don't silently break others
- **Implement Circuit Breakers**: Stop calling failing agents after repeated errors
- **Plan for Recovery**: Design agents to recover from coordination failures
- **Log Coordination Events**: Maintain logs for debugging multi-agent interactions

## Advanced Coordination

### Agent Teams

Create agent teams for specialized tasks:

```python
from safeguards.coordination.team import AgentTeam

# Create a research team
research_team = AgentTeam(
    name="research_team",
    agents=[research_agent, analysis_agent, summarization_agent],
    budget_pool_id=high_priority_pool.id
)

# Assign team task
team_result = research_team.execute_task(
    task="research quantum computing",
    coordination_strategy="pipeline"
)
```

### Dynamic Agent Discovery

Implement dynamic discovery of available agents:

```python
from safeguards.coordination.discovery import AgentDiscoveryService

# Create discovery service
discovery_service = AgentDiscoveryService(budget_coordinator)

# Register agent capabilities
discovery_service.register_capability(
    agent_id=research_agent.id,
    capability="data_retrieval",
    metadata={"formats": ["json", "xml"], "sources": ["web", "database"]}
)

discovery_service.register_capability(
    agent_id=analysis_agent.id,
    capability="data_analysis",
    metadata={"algorithms": ["regression", "classification"], "formats": ["json"]}
)

# Find agents with specific capabilities
analysis_agents = discovery_service.find_agents_by_capability(
    capability="data_analysis",
    required_metadata={"algorithms": ["regression"]}
)

if analysis_agents:
    # Use the first available agent
    agent_id = analysis_agents[0]
    print(f"Using agent {agent_id} for regression analysis")
```

## Conclusion

Effective agent coordination is essential for building robust multi-agent systems. By implementing appropriate coordination patterns, managing resource sharing, and handling dependencies correctly, you can create systems where agents collaborate effectively while respecting resource constraints.

For more information, see:
- [Budget Management Guide](budget_management.md)
- [Safeguards Guide](safeguards.md)
- [Monitoring Guide](monitoring.md)
- [API Reference](../api/core.md)
