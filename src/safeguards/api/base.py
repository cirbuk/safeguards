"""Base API layer for the Agent Safety Framework."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import Enum, auto
from typing import Any, Generic, TypeVar

from ..config import Config
from ..core.budget_coordination import BudgetPool
from ..monitoring.metrics import AgentMetrics, SystemMetrics
from ..types.agent import Agent

T = TypeVar("T")


class APIVersion(Enum):
    """API versions supported by the framework."""

    V1 = auto()
    V2 = auto()
    LATEST = auto()


@dataclass
class APIResponse(Generic[T]):
    """Standard API response format."""

    success: bool
    data: T | None = None
    error: dict[str, Any] | None = None
    meta: dict[str, Any] | None = None


class APIContract(ABC):
    """Base class for API contracts."""

    def __init__(self, version: APIVersion = APIVersion.LATEST):
        self.version = version
        self.validate_contract()

    @abstractmethod
    def validate_contract(self) -> None:
        """Validate the contract implementation."""
        pass

    def create_response(
        self,
        data: T | None = None,
        error: dict[str, Any] | None = None,
        meta: dict[str, Any] | None = None,
    ) -> APIResponse[T]:
        """Create a standardized API response."""
        return APIResponse(
            success=error is None,
            data=data,
            error=error,
            meta={
                **(meta or {}),
                "version": self.version.value,
                "timestamp": datetime.utcnow().isoformat(),
            },
        )


class BudgetAPIContract(APIContract):
    """Contract for budget-related API endpoints."""

    @abstractmethod
    def get_budget(self, agent_id: str) -> Decimal:
        """Get an agent's budget.

        Args:
            agent_id: ID of agent to get budget for

        Returns:
            Current budget amount
        """
        pass

    @abstractmethod
    def update_budget(self, agent_id: str, amount: Decimal) -> None:
        """Update an agent's budget.

        Args:
            agent_id: ID of agent to update budget for
            amount: New budget amount
        """
        pass

    @abstractmethod
    def get_budget_pools(self) -> list[BudgetPool]:
        """Get all budget pools.

        Returns:
            List of budget pools
        """
        pass

    @abstractmethod
    def create_budget_pool(
        self,
        name: str,
        initial_budget: Decimal,
        priority: int = 0,
    ) -> BudgetPool:
        """Create a new budget pool.

        Args:
            name: Name of pool
            initial_budget: Initial budget amount
            priority: Priority level (default 0)

        Returns:
            Created budget pool
        """
        pass


class MetricsAPIContract(APIContract):
    """Contract for metrics-related API endpoints."""

    @abstractmethod
    def get_agent_metrics(self, agent_id: str) -> AgentMetrics:
        """Get metrics for an agent.

        Args:
            agent_id: ID of agent to get metrics for

        Returns:
            Agent metrics
        """
        pass

    @abstractmethod
    def get_system_metrics(self) -> SystemMetrics:
        """Get system-wide metrics.

        Returns:
            System metrics
        """
        pass

    @abstractmethod
    def create_metrics(
        self,
        agent_id: str,
        metrics: AgentMetrics | SystemMetrics,
    ) -> None:
        """Create new metrics.

        Args:
            agent_id: ID of agent metrics are for
            metrics: Metrics to create
        """
        pass


class AgentAPIContract(APIContract):
    """Contract for agent-related API endpoints."""

    @abstractmethod
    def get_agent(self, agent_id: str) -> Agent:
        """Get an agent.

        Args:
            agent_id: ID of agent to get

        Returns:
            Agent instance
        """
        pass

    @abstractmethod
    def create_agent(
        self,
        name: str,
        initial_budget: Decimal,
        priority: int = 0,
    ) -> Agent:
        """Create a new agent.

        Args:
            name: Name of agent
            initial_budget: Initial budget amount
            priority: Priority level (default 0)

        Returns:
            Created agent
        """
        pass

    @abstractmethod
    def update_agent(self, agent: Agent) -> None:
        """Update an agent.

        Args:
            agent: Agent to update
        """
        pass

    @abstractmethod
    def delete_agent(self, agent_id: str) -> None:
        """Delete an agent.

        Args:
            agent_id: ID of agent to delete
        """
        pass


class ConfigAPIContract(APIContract):
    """Contract for configuration-related API endpoints."""

    @abstractmethod
    def get_config(self) -> Config:
        """Get current config.

        Returns:
            Current config
        """
        pass

    @abstractmethod
    def update_config(self, config: Config) -> None:
        """Update config.

        Args:
            config: New config
        """
        pass

    @abstractmethod
    def validate_config(self, config: Config) -> bool:
        """Validate a config.

        Args:
            config: Config to validate

        Returns:
            True if valid, False otherwise
        """
        pass
