"""Swarm controller for managing budgets and safety across multiple coordinating agents."""

from decimal import Decimal
from uuid import uuid4

from agents import Agent, Runner

from ..budget import BudgetManager
from ..monitoring import ResourceMonitor
from ..types import (
    SwarmMetrics,
)
from .config import SwarmConfig


class SwarmController:
    """Controller for managing swarms of agents with coordinated budget and resource management.

    This class provides:
    1. Swarm-level budget and resource management
    2. Dynamic agent scaling and load balancing
    3. Inter-agent coordination and communication
    4. Failover and recovery mechanisms
    5. Centralized monitoring and control

    Example:
        ```python
        from agents import Agent
        from safeguards import SwarmController, SwarmConfig

        # Create swarm controller
        config = SwarmConfig(total_budget=Decimal("1000"))
        swarm = SwarmController(config)

        # Create agent templates
        researcher = Agent(name="researcher", instructions="Research task...")
        writer = Agent(name="writer", instructions="Write content...")

        # Register agent types
        swarm.register_agent_type("researcher", researcher, budget=Decimal("100"))
        swarm.register_agent_type("writer", writer, budget=Decimal("50"))

        # Run swarm task
        result = swarm.run_task(
            "Research quantum computing and write a summary",
            agent_types=["researcher", "writer"],
        )
        ```
    """

    def __init__(self, config: SwarmConfig):
        """Initialize the swarm controller.

        Args:
            config: Swarm configuration
        """
        self.config = config
        self.budget_manager = BudgetManager(
            total_budget=config.total_budget,
            hourly_limit=config.hourly_limit,
            daily_limit=config.daily_limit,
        )
        self.resource_monitor = ResourceMonitor()

        # Track agent types and instances
        self._agent_types: dict[str, Agent] = {}
        self._type_configs: dict[str, dict] = {}
        self._active_agents: dict[str, set[str]] = {}  # type -> instance_ids
        self._agent_tasks: dict[str, str] = {}  # agent_id -> task_id

        # Track swarm state
        self._task_agents: dict[str, set[str]] = {}  # task_id -> agent_ids
        self._task_status: dict[str, str] = {}
        self._task_results: dict[str, dict] = {}

    def register_agent_type(
        self,
        type_name: str,
        agent_template: Agent,
        budget: Decimal,
        min_instances: int = 1,
        max_instances: int = 5,
        priority: str = "MEDIUM",
        metadata: dict | None = None,
    ) -> None:
        """Register an agent type that can be instantiated in the swarm.

        Args:
            type_name: Identifier for this agent type
            agent_template: Template agent to clone
            budget: Budget per instance
            min_instances: Minimum number of instances
            max_instances: Maximum number of instances
            priority: Priority level
            metadata: Additional metadata
        """
        if type_name in self._agent_types:
            msg = f"Agent type {type_name} already registered"
            raise ValueError(msg)

        self._agent_types[type_name] = agent_template
        self._type_configs[type_name] = {
            "budget": budget,
            "min_instances": min_instances,
            "max_instances": max_instances,
            "priority": priority,
            "metadata": metadata or {},
        }
        self._active_agents[type_name] = set()

    def run_task(
        self,
        task: str,
        agent_types: list[str],
        coordination_type: str = "SEQUENTIAL",
        task_metadata: dict | None = None,
    ) -> dict:
        """Run a task using multiple coordinating agents.

        Args:
            task: Task description/prompt
            agent_types: Types of agents needed
            coordination_type: How agents should coordinate (SEQUENTIAL, PARALLEL, DAG)
            task_metadata: Additional task metadata

        Returns:
            Task results including all agent outputs
        """
        # Generate task ID
        task_id = str(uuid4())
        self._task_status[task_id] = "RUNNING"
        self._task_agents[task_id] = set()

        try:
            # Ensure required agent types exist
            for agent_type in agent_types:
                if agent_type not in self._agent_types:
                    msg = f"Unknown agent type: {agent_type}"
                    raise ValueError(msg)

            # Scale up agents as needed
            self._scale_agents(task_id, agent_types)

            # Run task based on coordination type
            if coordination_type == "SEQUENTIAL":
                results = self._run_sequential(task_id, task, agent_types)
            elif coordination_type == "PARALLEL":
                results = self._run_parallel(task_id, task, agent_types)
            else:
                msg = f"Unsupported coordination type: {coordination_type}"
                raise ValueError(msg)

            self._task_status[task_id] = "COMPLETED"
            self._task_results[task_id] = results
            return results

        except Exception as e:
            self._task_status[task_id] = "FAILED"
            self._task_results[task_id] = {"error": str(e)}
            raise

        finally:
            # Scale down if needed
            self._cleanup_task(task_id)

    def get_swarm_metrics(self) -> SwarmMetrics:
        """Get current metrics for the entire swarm.

        Returns:
            SwarmMetrics containing budget and resource usage
        """
        total_metrics = {
            "active_agents": sum(len(agents) for agents in self._active_agents.values()),
            "total_budget_used": self.budget_manager.get_total_usage(),
            "resource_usage": self.resource_monitor.get_metrics(),
            "agent_types": {
                type_name: {
                    "active_instances": len(instances),
                    "avg_budget_used": self._get_avg_budget_usage(type_name),
                }
                for type_name, instances in self._active_agents.items()
            },
        }
        return SwarmMetrics(**total_metrics)

    def _scale_agents(self, task_id: str, agent_types: list[str]) -> None:
        """Scale agent instances based on task needs.

        Args:
            task_id: Current task ID
            agent_types: Required agent types
        """
        for agent_type in agent_types:
            config = self._type_configs[agent_type]
            current_count = len(self._active_agents[agent_type])

            # Scale up to minimum if needed
            while current_count < config["min_instances"]:
                self._create_agent_instance(task_id, agent_type)
                current_count += 1

    def _create_agent_instance(self, task_id: str, agent_type: str) -> str:
        """Create a new instance of an agent type.

        Args:
            task_id: Current task ID
            agent_type: Type of agent to create

        Returns:
            ID of created agent instance
        """
        template = self._agent_types[agent_type]
        config = self._type_configs[agent_type]

        # Generate unique instance ID
        instance_id = f"{agent_type}_{len(self._active_agents[agent_type])}"

        # Clone template and customize
        Agent(
            name=instance_id,
            instructions=template.instructions,
            tools=template.tools,
        )

        # Register with budget manager
        self.budget_manager.register_agent(
            agent_id=instance_id,
            budget=config["budget"],
            priority=config["priority"],
            metadata=config["metadata"],
        )

        # Track instance
        self._active_agents[agent_type].add(instance_id)
        self._task_agents[task_id].add(instance_id)
        self._agent_tasks[instance_id] = task_id

        return instance_id

    def _run_sequential(
        self,
        task_id: str,
        task: str,
        agent_types: list[str],
    ) -> dict:
        """Run agents sequentially, passing outputs as inputs.

        Args:
            task_id: Current task ID
            task: Initial task description
            agent_types: Agent types in sequence

        Returns:
            Combined results from all agents
        """
        current_input = task
        results = {}

        for agent_type in agent_types:
            # Get available agent of this type
            agent_id = self._get_available_agent(agent_type)
            agent = self._agent_types[agent_type]

            # Run agent with budget tracking
            with self.budget_manager.track_agent(agent_id):
                trace = Runner.run_sync(agent, current_input)
                results[agent_type] = trace.final_output

            # Pass output to next agent
            current_input = trace.final_output

        return results

    def _run_parallel(
        self,
        task_id: str,
        task: str,
        agent_types: list[str],
    ) -> dict:
        """Run agents in parallel with shared context.

        Args:
            task_id: Current task ID
            task: Task description
            agent_types: Agent types to run

        Returns:
            Combined results from all agents
        """
        results = {}

        for agent_type in agent_types:
            # Get available agent
            agent_id = self._get_available_agent(agent_type)
            agent = self._agent_types[agent_type]

            # Run agent with budget tracking
            with self.budget_manager.track_agent(agent_id):
                trace = Runner.run_sync(agent, task)
                results[agent_type] = trace.final_output

        return results

    def _get_available_agent(self, agent_type: str) -> str:
        """Get ID of an available agent instance.

        Args:
            agent_type: Type of agent needed

        Returns:
            Agent instance ID

        Raises:
            RuntimeError: If no agents available
        """
        active_agents = self._active_agents[agent_type]
        if not active_agents:
            msg = f"No active agents of type {agent_type}"
            raise RuntimeError(msg)

        # Simple round-robin for now
        return next(iter(active_agents))

    def _cleanup_task(self, task_id: str) -> None:
        """Clean up after task completion.

        Args:
            task_id: Task to clean up
        """
        if task_id not in self._task_agents:
            return

        # Scale down agents if needed
        for agent_id in self._task_agents[task_id]:
            agent_type = agent_id.split("_")[0]
            config = self._type_configs[agent_type]

            if len(self._active_agents[agent_type]) > config["min_instances"]:
                self._active_agents[agent_type].remove(agent_id)
                del self._agent_tasks[agent_id]

        del self._task_agents[task_id]

    def _get_avg_budget_usage(self, agent_type: str) -> Decimal:
        """Calculate average budget usage for an agent type.

        Args:
            agent_type: Agent type to check

        Returns:
            Average budget usage
        """
        if not self._active_agents[agent_type]:
            return Decimal("0")

        total = sum(
            self.budget_manager.get_agent_usage(agent_id)
            for agent_id in self._active_agents[agent_type]
        )
        return total / len(self._active_agents[agent_type])
