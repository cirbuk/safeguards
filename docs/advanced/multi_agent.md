# Advanced Multi-Agent Coordination

This document covers advanced techniques for coordinating multiple agents within the Safeguards, focusing on complex coordination patterns, dynamic task allocation, and conflict resolution.

## Advanced Coordination Patterns

### Hierarchical Agent Organization

Implement a hierarchical agent structure for complex tasks:

```python
from safeguards.types.agent import Agent
from typing import Dict, Any, List
from decimal import Decimal

class ManagerAgent(Agent):
    """Manager agent that coordinates worker agents."""
    def __init__(self, name: str, worker_agents: List[Agent] = None):
        super().__init__(name)
        self.worker_agents = worker_agents or []
        self.task_assignments = {}

    def add_worker(self, worker: Agent) -> None:
        """Add a worker agent to be managed."""
        self.worker_agents.append(worker)

    def assign_tasks(self, tasks: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        """Assign tasks to worker agents based on capabilities."""
        assignments = {}

        # Example task assignment logic - customize based on your needs
        for i, task in enumerate(tasks):
            if i < len(self.worker_agents):
                worker = self.worker_agents[i]
                if worker.id not in assignments:
                    assignments[worker.id] = []
                assignments[worker.id].append(task["task_id"])
                self.task_assignments[task["task_id"]] = worker.id

        return assignments

    def run(self, **kwargs: Any) -> Dict[str, Any]:
        """Coordinate worker agents to complete a complex task."""
        tasks = kwargs.get("tasks", [])
        context = kwargs.get("context", {})

        # Step 1: Assign tasks to workers
        assignments = self.assign_tasks(tasks)

        # Step 2: Execute tasks and collect results
        results = {}
        for worker in self.worker_agents:
            worker_tasks = assignments.get(worker.id, [])
            if worker_tasks:
                # Get the full task definitions for this worker
                worker_task_defs = [t for t in tasks if t["task_id"] in worker_tasks]

                # Execute worker tasks
                worker_result = worker.run(
                    tasks=worker_task_defs,
                    context={**context, "manager_id": self.id}
                )

                # Store results
                results[worker.id] = worker_result

        # Step 3: Aggregate and process results
        aggregated_result = self._aggregate_results(results)

        return {
            "status": "completed",
            "worker_results": results,
            "aggregated_result": aggregated_result,
            "task_count": len(tasks)
        }

    def _aggregate_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Aggregate results from worker agents."""
        # Implement your specific aggregation logic
        combined_output = {}
        for worker_id, result in results.items():
            for key, value in result.items():
                if key not in combined_output:
                    combined_output[key] = []
                if isinstance(value, list):
                    combined_output[key].extend(value)
                else:
                    combined_output[key].append(value)

        return combined_output
```

### Federated Decision Making

Implement consensus-based decision making across agents:

```python
from typing import List, Dict, Any, Callable
from enum import Enum

class ConsensusMethod(Enum):
    MAJORITY_VOTE = "majority_vote"
    WEIGHTED_VOTE = "weighted_vote"
    UNANIMOUS = "unanimous"

class FederatedDecisionSystem:
    """Manages decision making across multiple agents."""

    def __init__(self, consensus_method: ConsensusMethod = ConsensusMethod.MAJORITY_VOTE):
        self.consensus_method = consensus_method
        self.agents = {}  # agent_id -> agent object
        self.weights = {}  # agent_id -> weight (for weighted voting)

    def register_agent(self, agent_id: str, agent: Any, weight: float = 1.0) -> None:
        """Register an agent with the federated system."""
        self.agents[agent_id] = agent
        self.weights[agent_id] = weight

    def make_decision(self, question: str, options: List[Any],
                      decision_fn: Callable = None) -> Dict[str, Any]:
        """Make a decision across all registered agents."""
        # Collect votes from each agent
        votes = {}
        for agent_id, agent in self.agents.items():
            if decision_fn:
                # Use custom decision function
                vote = decision_fn(agent, question, options)
            else:
                # Default: let agent run with question and options
                result = agent.run(
                    question=question,
                    options=options
                )
                vote = result.get("selected_option")

            votes[agent_id] = vote

        # Apply consensus method
        if self.consensus_method == ConsensusMethod.MAJORITY_VOTE:
            decision = self._apply_majority_vote(votes, options)
        elif self.consensus_method == ConsensusMethod.WEIGHTED_VOTE:
            decision = self._apply_weighted_vote(votes, options)
        elif self.consensus_method == ConsensusMethod.UNANIMOUS:
            decision = self._apply_unanimous(votes, options)
        else:
            raise ValueError(f"Unknown consensus method: {self.consensus_method}")

        return {
            "decision": decision,
            "votes": votes,
            "consensus_method": self.consensus_method.value,
            "question": question
        }

    def _apply_majority_vote(self, votes: Dict[str, Any], options: List[Any]) -> Any:
        """Apply simple majority voting."""
        # Count votes for each option
        vote_counts = {option: 0 for option in options}
        for vote in votes.values():
            if vote in vote_counts:
                vote_counts[vote] += 1

        # Find option with most votes
        return max(vote_counts.items(), key=lambda x: x[1])[0]

    def _apply_weighted_vote(self, votes: Dict[str, Any], options: List[Any]) -> Any:
        """Apply weighted voting."""
        # Count weighted votes for each option
        weighted_votes = {option: 0 for option in options}
        for agent_id, vote in votes.items():
            if vote in weighted_votes:
                weight = self.weights.get(agent_id, 1.0)
                weighted_votes[vote] += weight

        # Find option with highest weighted votes
        return max(weighted_votes.items(), key=lambda x: x[1])[0]

    def _apply_unanimous(self, votes: Dict[str, Any], options: List[Any]) -> Any:
        """Check for unanimous agreement."""
        # Get unique votes
        unique_votes = set(votes.values())

        # If all agents voted for the same option, return it
        if len(unique_votes) == 1:
            return next(iter(unique_votes))

        # Otherwise, no consensus
        return None
```

## Dynamic Task Allocation

### Workload-Based Allocation

Dynamically allocate tasks based on agent workload:

```python
from typing import Dict, List, Any
from decimal import Decimal
import heapq

class WorkloadBalancer:
    """Balances workload across multiple agents."""

    def __init__(self):
        self.agent_workloads = {}  # agent_id -> current workload
        self.agent_capacities = {}  # agent_id -> maximum capacity
        self.task_assignments = {}  # task_id -> agent_id

    def register_agent(self, agent_id: str, max_capacity: float) -> None:
        """Register an agent with the workload balancer."""
        self.agent_workloads[agent_id] = 0
        self.agent_capacities[agent_id] = max_capacity

    def assign_task(self, task_id: str, estimated_load: float) -> str:
        """Assign a task to the agent with the lowest workload."""
        if not self.agent_workloads:
            raise ValueError("No agents registered with the workload balancer")

        # Find agent with lowest workload percentage
        best_agent_id = None
        lowest_workload_pct = float('inf')

        for agent_id, current_load in self.agent_workloads.items():
            capacity = self.agent_capacities[agent_id]
            workload_pct = current_load / capacity if capacity > 0 else float('inf')

            if workload_pct < lowest_workload_pct:
                lowest_workload_pct = workload_pct
                best_agent_id = agent_id

        # Update workload and store assignment
        if best_agent_id:
            self.agent_workloads[best_agent_id] += estimated_load
            self.task_assignments[task_id] = best_agent_id

        return best_agent_id

    def complete_task(self, task_id: str, actual_load: float = None) -> None:
        """Mark a task as completed and update workload."""
        agent_id = self.task_assignments.get(task_id)
        if agent_id:
            # If actual load not provided, use the estimated load (workload difference)
            if actual_load is None:
                # Find how much this task contributed
                task_load = 0
                for tid, aid in self.task_assignments.items():
                    if tid == task_id and aid == agent_id:
                        task_load = self.agent_workloads[agent_id] / len(
                            [t for t, a in self.task_assignments.items() if a == agent_id]
                        )
                        break
                actual_load = task_load

            # Update workload
            self.agent_workloads[agent_id] = max(0, self.agent_workloads[agent_id] - actual_load)
            # Remove assignment
            self.task_assignments.pop(task_id, None)

    def get_agent_utilization(self) -> Dict[str, float]:
        """Get current utilization percentage for each agent."""
        utilization = {}
        for agent_id, load in self.agent_workloads.items():
            capacity = self.agent_capacities[agent_id]
            utilization[agent_id] = (load / capacity * 100) if capacity > 0 else 0
        return utilization

    def rebalance(self) -> Dict[str, str]:
        """Rebalance tasks across agents."""
        # Get all current tasks
        tasks = list(self.task_assignments.items())

        # Reset workloads
        for agent_id in self.agent_workloads:
            self.agent_workloads[agent_id] = 0

        # Clear assignments
        self.task_assignments = {}

        # Reassign tasks
        new_assignments = {}
        for task_id, _ in tasks:
            # Assign based on current workload
            new_agent = self.assign_task(task_id, 1.0)  # Simplified load of 1.0 for rebalancing
            new_assignments[task_id] = new_agent

        return new_assignments
```

### Skill-Based Routing

Route tasks to agents based on their capabilities:

```python
from typing import Dict, List, Set, Any

class AgentSkillRouter:
    """Routes tasks to agents based on their skills."""

    def __init__(self):
        self.agent_skills = {}  # agent_id -> set of skills
        self.skill_agents = {}  # skill -> list of agent_ids

    def register_agent(self, agent_id: str, skills: List[str]) -> None:
        """Register an agent with specified skills."""
        skill_set = set(skills)
        self.agent_skills[agent_id] = skill_set

        # Update skill -> agents mapping
        for skill in skill_set:
            if skill not in self.skill_agents:
                self.skill_agents[skill] = []
            self.skill_agents[skill].append(agent_id)

    def find_agents_with_skill(self, skill: str) -> List[str]:
        """Find all agents with a specific skill."""
        return self.skill_agents.get(skill, [])

    def find_agents_with_all_skills(self, required_skills: List[str]) -> List[str]:
        """Find agents that have all the required skills."""
        if not required_skills:
            return list(self.agent_skills.keys())

        skill_set = set(required_skills)
        qualified_agents = []

        for agent_id, agent_skill_set in self.agent_skills.items():
            if skill_set.issubset(agent_skill_set):
                qualified_agents.append(agent_id)

        return qualified_agents

    def route_task(self, task_id: str, required_skills: List[str],
                   load_balancer = None) -> str:
        """Route a task to the best agent based on skills and optionally load."""
        qualified_agents = self.find_agents_with_all_skills(required_skills)

        if not qualified_agents:
            return None

        if load_balancer:
            # Use load balancer to pick among qualified agents
            best_agent = None
            lowest_load = float('inf')

            for agent_id in qualified_agents:
                agent_load = load_balancer.agent_workloads.get(agent_id, 0)
                if agent_load < lowest_load:
                    lowest_load = agent_load
                    best_agent = agent_id

            if best_agent:
                # Update load balancer
                load_balancer.assign_task(task_id, 1.0)  # Simplified load of 1.0
                return best_agent

        # No load balancing - just return first qualified agent
        return qualified_agents[0]
```

## Conflict Resolution

### Resource Contention

Resolve conflicts when multiple agents need the same resources:

```python
from enum import Enum
from typing import Dict, List, Any
import time
from threading import Lock

class ResourceType(Enum):
    COMPUTE = "compute"
    MEMORY = "memory"
    STORAGE = "storage"
    API_CALL = "api_call"
    DATA = "data"

class ResourceContention:
    """Manages and resolves resource contentions between agents."""

    def __init__(self):
        self.resources = {}  # resource_id -> ResourceInfo
        self.locks = {}  # resource_id -> Lock
        self.reservations = {}  # resource_id -> {agent_id: priority}

    def register_resource(self, resource_id: str, resource_type: ResourceType,
                          capacity: float) -> None:
        """Register a resource with the contention manager."""
        self.resources[resource_id] = {
            "type": resource_type,
            "capacity": capacity,
            "allocated": 0.0,
            "allocations": {}  # agent_id -> amount
        }
        self.locks[resource_id] = Lock()

    def request_allocation(self, agent_id: str, resource_id: str,
                          amount: float, priority: int = 5) -> bool:
        """Request allocation of a resource amount to an agent."""
        if resource_id not in self.resources:
            raise ValueError(f"Unknown resource: {resource_id}")

        resource = self.resources[resource_id]

        with self.locks[resource_id]:
            # Check if there's enough capacity
            if resource["allocated"] + amount <= resource["capacity"]:
                # Allocate resource
                if agent_id not in resource["allocations"]:
                    resource["allocations"][agent_id] = 0

                resource["allocations"][agent_id] += amount
                resource["allocated"] += amount
                return True

            # Not enough capacity, add to reservations for future allocation
            if resource_id not in self.reservations:
                self.reservations[resource_id] = {}

            self.reservations[resource_id][agent_id] = priority
            return False

    def release_allocation(self, agent_id: str, resource_id: str,
                          amount: float = None) -> float:
        """Release allocated resources from an agent."""
        if resource_id not in self.resources:
            raise ValueError(f"Unknown resource: {resource_id}")

        resource = self.resources[resource_id]

        with self.locks[resource_id]:
            if agent_id not in resource["allocations"]:
                return 0.0

            # Determine how much to release
            current_allocation = resource["allocations"][agent_id]
            release_amount = amount if amount is not None else current_allocation
            release_amount = min(release_amount, current_allocation)

            # Update allocations
            resource["allocations"][agent_id] -= release_amount
            resource["allocated"] -= release_amount

            # Clean up if allocation is zero
            if resource["allocations"][agent_id] <= 0:
                resource["allocations"].pop(agent_id)

            # Process waiting reservations
            self._process_reservations(resource_id)

            return release_amount

    def _process_reservations(self, resource_id: str) -> None:
        """Process waiting reservations based on priority."""
        if resource_id not in self.reservations:
            return

        resource = self.resources[resource_id]
        reservations = self.reservations[resource_id]

        # Sort agents by priority (highest first)
        sorted_agents = sorted(
            reservations.items(),
            key=lambda x: x[1],
            reverse=True
        )

        # Try to allocate to each agent in priority order
        remaining_capacity = resource["capacity"] - resource["allocated"]
        updated_reservations = {}

        for agent_id, priority in sorted_agents:
            # Assume each reservation is for 1.0 unit (simplified)
            if remaining_capacity >= 1.0:
                # Allocate resource
                if agent_id not in resource["allocations"]:
                    resource["allocations"][agent_id] = 0

                resource["allocations"][agent_id] += 1.0
                resource["allocated"] += 1.0
                remaining_capacity -= 1.0
            else:
                # Keep in reservations
                updated_reservations[agent_id] = priority

        # Update reservations
        if updated_reservations:
            self.reservations[resource_id] = updated_reservations
        else:
            # No more reservations
            self.reservations.pop(resource_id, None)
```

### Task Priority Arbitration

Resolve conflicts between tasks with competing priorities:

```python
from enum import Enum
from typing import Dict, List, Any, Tuple
import time
import heapq

class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"

class TaskArbitrator:
    """Arbitrates between competing tasks based on priority."""

    def __init__(self, max_concurrent_tasks: int = 5):
        self.tasks = {}  # task_id -> task info
        self.running_tasks = set()  # set of running task_ids
        self.max_concurrent_tasks = max_concurrent_tasks

    def register_task(self, task_id: str, agent_id: str, priority: int,
                     estimated_duration: float = None) -> None:
        """Register a task with the arbitrator."""
        self.tasks[task_id] = {
            "agent_id": agent_id,
            "priority": priority,
            "status": TaskStatus.PENDING,
            "created_at": time.time(),
            "started_at": None,
            "completed_at": None,
            "estimated_duration": estimated_duration,
            "dependencies": []
        }

    def add_dependency(self, task_id: str, depends_on: str) -> None:
        """Add a dependency between tasks."""
        if task_id not in self.tasks or depends_on not in self.tasks:
            raise ValueError(f"Unknown task: {task_id} or {depends_on}")

        self.tasks[task_id]["dependencies"].append(depends_on)

    def start_task(self, task_id: str) -> bool:
        """Try to start a task, respecting priority and concurrency limits."""
        if task_id not in self.tasks:
            raise ValueError(f"Unknown task: {task_id}")

        task = self.tasks[task_id]

        # Check if already running
        if task["status"] == TaskStatus.RUNNING:
            return True

        # Check if task can be started
        if not self._can_start_task(task_id):
            return False

        # Check concurrency limit
        if len(self.running_tasks) >= self.max_concurrent_tasks:
            # Need to decide if we should preempt a running task
            lowest_priority_task = self._find_lowest_priority_running_task()

            if lowest_priority_task and self.tasks[lowest_priority_task]["priority"] < task["priority"]:
                # Preempt the lowest priority task
                self.pause_task(lowest_priority_task)
            else:
                # Can't start new task now
                return False

        # Start the task
        task["status"] = TaskStatus.RUNNING
        task["started_at"] = time.time()
        self.running_tasks.add(task_id)
        return True

    def pause_task(self, task_id: str) -> bool:
        """Pause a running task."""
        if task_id not in self.tasks:
            raise ValueError(f"Unknown task: {task_id}")

        task = self.tasks[task_id]

        if task["status"] != TaskStatus.RUNNING:
            return False

        task["status"] = TaskStatus.PAUSED
        self.running_tasks.remove(task_id)
        return True

    def complete_task(self, task_id: str, success: bool = True) -> None:
        """Mark a task as completed or failed."""
        if task_id not in self.tasks:
            raise ValueError(f"Unknown task: {task_id}")

        task = self.tasks[task_id]
        task["completed_at"] = time.time()

        if success:
            task["status"] = TaskStatus.COMPLETED
        else:
            task["status"] = TaskStatus.FAILED

        if task_id in self.running_tasks:
            self.running_tasks.remove(task_id)

        # Check if we can start any pending tasks
        self._start_pending_tasks()

    def _can_start_task(self, task_id: str) -> bool:
        """Check if a task's dependencies are satisfied."""
        task = self.tasks[task_id]

        # Check if task is already completed or failed
        if task["status"] in [TaskStatus.COMPLETED, TaskStatus.FAILED]:
            return False

        # Check dependencies
        for dep_id in task["dependencies"]:
            if dep_id not in self.tasks:
                return False

            dep_status = self.tasks[dep_id]["status"]
            if dep_status != TaskStatus.COMPLETED:
                return False

        return True

    def _find_lowest_priority_running_task(self) -> str:
        """Find the running task with lowest priority."""
        if not self.running_tasks:
            return None

        lowest_priority = float('inf')
        lowest_task_id = None

        for task_id in self.running_tasks:
            task = self.tasks[task_id]
            if task["priority"] < lowest_priority:
                lowest_priority = task["priority"]
                lowest_task_id = task_id

        return lowest_task_id

    def _start_pending_tasks(self) -> None:
        """Try to start pending tasks based on priority."""
        if len(self.running_tasks) >= self.max_concurrent_tasks:
            return

        # Create a priority queue of pending tasks
        pending_tasks = []
        for task_id, task in self.tasks.items():
            if task["status"] == TaskStatus.PENDING and self._can_start_task(task_id):
                # Higher priority first, then older tasks
                # Negate priority for max-heap behavior
                heapq.heappush(pending_tasks, (-task["priority"], task["created_at"], task_id))

        # Start tasks in priority order until we hit the concurrency limit
        while pending_tasks and len(self.running_tasks) < self.max_concurrent_tasks:
            _, _, task_id = heapq.heappop(pending_tasks)
            self.start_task(task_id)
```

## Best Practices

### Fault Tolerance

1. **Design for Partial Failure**: Assume any agent might fail and design your system to handle it
2. **Implement Timeouts**: Never wait indefinitely for an agent to respond
3. **Use Circuit Breakers**: Stop calling unreliable agents after repeated failures
4. **Maintain State Backups**: Keep snapshots of important coordination state
5. **Implement Fallback Strategies**: Define what happens when an agent is unavailable

### Performance Optimization

1. **Batch Related Tasks**: Group tasks that access similar data or resources
2. **Minimize Communication Overhead**: Use efficient message formats
3. **Implement Caching**: Cache frequently needed data across agents
4. **Use Asynchronous Processing**: Don't block when waiting for slow operations
5. **Profile and Optimize Hotspots**: Identify bottlenecks in your agent communication

### Coordination Patterns

1. **Start Simple**: Begin with basic coordination before implementing advanced patterns
2. **Use Consistent Interfaces**: Make agent APIs compatible for easier coordination
3. **Design Clear Protocols**: Define how agents should communicate and share data
4. **Implement Robust Logging**: Track coordination events for troubleshooting
5. **Test Coordination Scenarios**: Verify your coordination works under stress

## Conclusion

Advanced multi-agent coordination enables powerful, resilient systems by leveraging the strengths of specialized agents working together. By implementing hierarchical organization, federated decision making, dynamic task allocation, and conflict resolution strategies, you can build agent systems that efficiently handle complex tasks while maintaining safety constraints.

For more information, see:
- [Budget Management](../guides/budget_management.md)
- [Safety Guardrails](../guides/guardrails.md)
- [Agent Coordination Basics](../guides/agent_coordination.md)
