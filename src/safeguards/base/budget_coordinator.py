"""Budget coordination system for managing multi-agent budgets."""

from decimal import Decimal

from pydantic import BaseModel

from safeguards.base.budget import BudgetConfig, BudgetMetrics
from safeguards.base.budget_impl import SimpleBudgetManager


class AgentAllocation(BaseModel):
    """Budget allocation for an agent."""

    agent_id: str
    budget_config: BudgetConfig
    priority: int = 0


class BudgetCoordinator:
    """Coordinates budget allocation and usage across multiple agents."""

    def __init__(self):
        """Initialize budget coordinator."""
        self._allocations: dict[str, AgentAllocation] = {}
        self._managers: dict[str, SimpleBudgetManager] = {}

    def register_agent(
        self,
        agent_id: str,
        config: BudgetConfig,
        priority: int = 0,
    ) -> None:
        """Register an agent with the coordinator.

        Args:
            agent_id: Agent identifier
            config: Budget configuration for the agent
            priority: Agent priority (higher priority agents get preference)
        """
        allocation = AgentAllocation(
            agent_id=agent_id,
            budget_config=config,
            priority=priority,
        )
        self._allocations[agent_id] = allocation
        self._managers[agent_id] = SimpleBudgetManager(config)

    def track_usage(self, agent_id: str, tokens_used: int) -> None:
        """Track token usage for an agent.

        Args:
            agent_id: Agent identifier
            tokens_used: Number of tokens used

        Raises:
            KeyError: If agent is not registered
        """
        if agent_id not in self._managers:
            msg = f"Agent {agent_id} not registered"
            raise KeyError(msg)

        manager = self._managers[agent_id]
        manager.record_usage(Decimal(tokens_used))

    def get_remaining_budget(self, agent_id: str) -> int:
        """Get remaining budget for an agent.

        Args:
            agent_id: Agent identifier

        Returns:
            int: Remaining budget

        Raises:
            KeyError: If agent is not registered
        """
        if agent_id not in self._managers:
            msg = f"Agent {agent_id} not registered"
            raise KeyError(msg)

        manager = self._managers[agent_id]
        metrics = manager.get_metrics()
        return int(metrics.total_budget - metrics.current_usage)

    def get_metrics(self, agent_id: str) -> BudgetMetrics:
        """Get budget metrics for an agent.

        Args:
            agent_id: Agent identifier

        Returns:
            BudgetMetrics: Budget metrics

        Raises:
            KeyError: If agent is not registered
        """
        if agent_id not in self._managers:
            msg = f"Agent {agent_id} not registered"
            raise KeyError(msg)

        manager = self._managers[agent_id]
        return manager.get_metrics()

    def reallocate_budgets(self) -> None:
        """Reallocate budgets based on usage patterns and priorities."""
        # Get all agents sorted by priority
        agents = sorted(
            self._allocations.values(),
            key=lambda x: x.priority,
            reverse=True,
        )

        # For each agent, check usage and adjust if needed
        for allocation in agents:
            manager = self._managers[allocation.agent_id]
            metrics = manager.get_metrics()

            # If usage is consistently high, consider increasing budget
            if metrics.usage_percentage > 0.9:  # 90%
                self._increase_budget(allocation.agent_id)
            # If usage is consistently low, consider decreasing budget
            elif metrics.usage_percentage < 0.3:  # 30%
                self._decrease_budget(allocation.agent_id)

    def _increase_budget(self, agent_id: str, increase_factor: float = 1.2) -> None:
        """Increase budget for an agent.

        Args:
            agent_id: Agent identifier
            increase_factor: Factor to increase budget by
        """
        allocation = self._allocations[agent_id]
        current_budget = allocation.budget_config.total_budget
        new_config = BudgetConfig(
            total_budget=current_budget * Decimal(str(increase_factor)),
            period=allocation.budget_config.period,
        )
        allocation.budget_config = new_config
        self._managers[agent_id] = SimpleBudgetManager(new_config)

    def _decrease_budget(self, agent_id: str, decrease_factor: float = 0.8) -> None:
        """Decrease budget for an agent.

        Args:
            agent_id: Agent identifier
            decrease_factor: Factor to decrease budget by
        """
        allocation = self._allocations[agent_id]
        current_budget = allocation.budget_config.total_budget
        new_config = BudgetConfig(
            total_budget=current_budget * Decimal(str(decrease_factor)),
            period=allocation.budget_config.period,
        )
        allocation.budget_config = new_config
        self._managers[agent_id] = SimpleBudgetManager(new_config)
