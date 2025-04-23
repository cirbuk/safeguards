"""Metrics analyzer for monitoring and analyzing agent and system metrics."""

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal


@dataclass
class AgentMetrics:
    """Metrics for individual agent performance and usage."""

    agent_id: str
    name: str
    total_budget: Decimal
    used_budget: Decimal
    remaining_budget: Decimal
    budget_utilization: float
    action_count: int = 0
    error_count: int = 0
    average_latency: float = 0.0
    last_updated: datetime = field(default_factory=datetime.now)
    metadata: dict = field(default_factory=dict)


@dataclass
class SystemMetrics:
    """System-wide metrics across all agents."""

    total_agents: int
    active_agents: int
    total_budget: Decimal
    used_budget: Decimal
    budget_utilization: float
    error_rate: float
    last_updated: datetime = field(default_factory=datetime.now)
    cpu_usage: float | None = None
    memory_usage: float | None = None
    metadata: dict = field(default_factory=dict)


class MetricsAnalyzer:
    """Analyzes and manages metrics for agents and system."""

    def __init__(self):
        """Initialize metrics analyzer."""
        self._agent_metrics: dict[str, dict] = {}
        self._system_metrics: dict = {
            "total_agents": 0,
            "active_agents": 0,
            "total_budget": Decimal("0"),
            "used_budget": Decimal("0"),
            "budget_utilization": 0.0,
            "error_rate": 0.0,
            "last_updated": datetime.now(),
        }

    def record_metrics(self, agent_id: str, metrics: dict) -> None:
        """Record new metrics for an agent.

        Args:
            agent_id: Agent to record metrics for
            metrics: Metrics data to record
        """
        if agent_id not in self._agent_metrics:
            self._agent_metrics[agent_id] = {}

        # Update agent metrics
        self._agent_metrics[agent_id].update(metrics)
        self._agent_metrics[agent_id]["last_updated"] = datetime.now()

        # Update system metrics
        self._update_system_metrics()

    def get_agent_metrics(self, agent_id: str) -> dict:
        """Get metrics for an agent.

        Args:
            agent_id: Agent to get metrics for

        Returns:
            Agent metrics
        """
        if agent_id not in self._agent_metrics:
            msg = f"No metrics found for agent {agent_id}"
            raise ValueError(msg)
        return self._agent_metrics[agent_id]

    def get_system_metrics(self) -> dict:
        """Get system-wide metrics.

        Returns:
            System metrics
        """
        return self._system_metrics

    def _update_system_metrics(self) -> None:
        """Update system-wide metrics based on agent metrics."""
        total_agents = len(self._agent_metrics)
        active_agents = sum(
            1
            for metrics in self._agent_metrics.values()
            if (datetime.now() - metrics["last_updated"]).total_seconds() < 300
        )

        total_budget = sum(
            Decimal(str(metrics.get("total_budget", 0))) for metrics in self._agent_metrics.values()
        )

        used_budget = sum(
            Decimal(str(metrics.get("used_budget", 0))) for metrics in self._agent_metrics.values()
        )

        budget_utilization = float(used_budget / total_budget) if total_budget else 0.0

        error_count = sum(metrics.get("error_count", 0) for metrics in self._agent_metrics.values())
        total_actions = sum(
            metrics.get("action_count", 0) for metrics in self._agent_metrics.values()
        )
        error_rate = error_count / total_actions if total_actions else 0.0

        self._system_metrics.update(
            {
                "total_agents": total_agents,
                "active_agents": active_agents,
                "total_budget": total_budget,
                "used_budget": used_budget,
                "budget_utilization": budget_utilization,
                "error_rate": error_rate,
                "last_updated": datetime.now(),
            },
        )
