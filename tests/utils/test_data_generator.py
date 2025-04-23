"""Test data generator utilities for integration tests."""

import uuid
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any

from safeguards.core.budget_coordination import BudgetPool
from safeguards.monitoring.metrics import AgentMetrics, SystemMetrics
from safeguards.types.agent import Agent


class TestAgent(Agent):
    """Test agent implementation for integration tests."""

    def __init__(self, name: str, cost_per_action: Decimal = Decimal("0.1")):
        super().__init__(name)
        self.cost_per_action = cost_per_action
        self.action_count = 0

    def run(self, **kwargs: Any) -> dict[str, Any]:
        """Simulate agent execution with cost tracking."""
        self.action_count += 1
        return {
            "action_count": self.action_count,
            "cost": self.cost_per_action,
            "timestamp": datetime.now().isoformat(),
        }


class TestDataGenerator:
    """Generates test data for integration tests."""

    @staticmethod
    def generate_agent(
        name_prefix: str = "test_agent",
        cost_per_action: Decimal = Decimal("0.1"),
    ) -> TestAgent:
        """Generate a test agent.

        Args:
            name_prefix: Prefix for agent name
            cost_per_action: Cost per agent action

        Returns:
            Test agent instance
        """
        agent_id = f"{name_prefix}_{uuid.uuid4().hex[:8]}"
        return TestAgent(agent_id, cost_per_action)

    @staticmethod
    def generate_budget_pool(
        name_prefix: str = "test_pool",
        initial_budget: Decimal = Decimal("100.0"),
        priority: int = 0,
    ) -> BudgetPool:
        """Generate a test budget pool.

        Args:
            name_prefix: Prefix for pool name
            initial_budget: Initial budget amount
            priority: Pool priority

        Returns:
            Budget pool instance
        """
        pool_id = f"{name_prefix}_{uuid.uuid4().hex[:8]}"
        return BudgetPool(
            pool_id=pool_id,
            total_budget=initial_budget,
            priority=priority,
            used_budget=Decimal("0.0"),
        )

    @staticmethod
    def generate_agent_metrics(
        agent_id: str,
        budget_used: Decimal = Decimal("0.0"),
        action_count: int = 0,
    ) -> AgentMetrics:
        """Generate test agent metrics.

        Args:
            agent_id: ID of agent
            budget_used: Amount of budget used
            action_count: Number of actions performed

        Returns:
            Agent metrics instance
        """
        return AgentMetrics(
            agent_id=agent_id,
            name=f"Agent_{agent_id}",
            total_budget=Decimal("100.0"),
            used_budget=budget_used,
            remaining_budget=Decimal("100.0") - budget_used,
            budget_utilization=float(budget_used / Decimal("100.0")),
            action_count=action_count,
            error_count=0,
            average_latency=0.0,
        )

    @staticmethod
    def generate_system_metrics(
        total_agents: int = 5,
        total_budget: Decimal = Decimal("1000.0"),
        used_budget: Decimal = Decimal("0.0"),
    ) -> SystemMetrics:
        """Generate test system metrics.

        Args:
            total_agents: Number of agents in system
            total_budget: Total system budget
            used_budget: Total used budget

        Returns:
            System metrics instance
        """
        return SystemMetrics(
            total_agents=total_agents,
            active_agents=total_agents,
            total_budget=total_budget,
            used_budget=used_budget,
            budget_utilization=float(used_budget / total_budget),
            error_rate=0.0,
            last_updated=datetime.now(),
        )

    @classmethod
    def generate_multi_agent_scenario(
        cls,
        num_agents: int = 3,
        num_pools: int = 2,
        base_budget: Decimal = Decimal("100.0"),
    ) -> dict[str, Any]:
        """Generate a complete multi-agent test scenario.

        Args:
            num_agents: Number of agents to create
            num_pools: Number of budget pools
            base_budget: Base budget amount per pool

        Returns:
            Dictionary containing agents, pools, and initial metrics
        """
        agents = [cls.generate_agent() for _ in range(num_agents)]
        pools = [
            cls.generate_budget_pool(initial_budget=base_budget * (i + 1), priority=i)
            for i in range(num_pools)
        ]

        agent_metrics = {agent.id: cls.generate_agent_metrics(agent.id) for agent in agents}

        system_metrics = cls.generate_system_metrics(
            total_agents=len(agents),
            total_budget=sum(p.total_budget for p in pools),
        )

        return {
            "agents": agents,
            "pools": pools,
            "agent_metrics": agent_metrics,
            "system_metrics": system_metrics,
        }

    @classmethod
    def generate_usage_pattern(
        cls,
        agent: TestAgent,
        duration_days: int = 7,
        actions_per_day: int = 10,
    ) -> list[dict[str, Any]]:
        """Generate historical usage pattern for an agent.

        Args:
            agent: Agent to generate usage for
            duration_days: Number of days of history
            actions_per_day: Average actions per day

        Returns:
            List of usage records
        """
        usage_history = []
        start_date = datetime.now() - timedelta(days=duration_days)

        for day in range(duration_days):
            current_date = start_date + timedelta(days=day)

            for _ in range(actions_per_day):
                usage_history.append(
                    {
                        "agent_id": agent.id,
                        "timestamp": current_date.isoformat(),
                        "action_count": agent.action_count + 1,
                        "cost": agent.cost_per_action,
                    },
                )
                agent.action_count += 1

        return usage_history
