"""V1 API implementations."""

from decimal import Decimal

from ..config import Config
from ..core.budget_coordination import BudgetCoordinator, BudgetPool
from ..exceptions import (
    AgentSafetyError,
    ErrorContext,
    InvalidConfigurationError,
)
from ..monitoring.metrics import AgentMetrics, MetricsAnalyzer, SystemMetrics
from ..types.agent import Agent
from .base import (
    AgentAPIContract,
    APIVersion,
    BudgetAPIContract,
    ConfigAPIContract,
    MetricsAPIContract,
)


class BudgetAPIV1(BudgetAPIContract):
    """V1 implementation of budget API."""

    def __init__(self, coordinator: BudgetCoordinator):
        """Initialize budget API.

        Args:
            coordinator: Budget coordinator instance
        """
        if not isinstance(coordinator, BudgetCoordinator):
            msg = "Invalid budget coordinator instance"
            raise InvalidConfigurationError(msg)
        self._coordinator = coordinator
        super().__init__(version=APIVersion.V1)

    def validate_contract(self) -> None:
        """Validate the contract implementation."""
        if not hasattr(self, "_coordinator"):
            msg = "Budget coordinator not initialized"
            raise InvalidConfigurationError(msg)

    def get_budget(self, agent_id: str) -> Decimal:
        """Get an agent's budget.

        Args:
            agent_id: ID of agent to get budget for

        Returns:
            Current budget amount

        Raises:
            AgentSafetyError: If agent not found or other error occurs
        """
        try:
            return self._coordinator.get_agent_budget(agent_id)
        except AgentSafetyError as e:
            raise e
        except Exception as e:
            context = ErrorContext(agent_id=agent_id)
            raise AgentSafetyError(str(e), "BUDGET_ERROR", context)

    def update_budget(self, agent_id: str, amount: Decimal) -> None:
        """Update an agent's budget.

        Args:
            agent_id: ID of agent to update budget for
            amount: New budget amount

        Raises:
            AgentSafetyError: If update fails or agent not found
        """
        try:
            self._coordinator.update_agent_budget(agent_id, amount)
        except AgentSafetyError as e:
            raise e
        except Exception as e:
            context = ErrorContext(agent_id=agent_id, details={"amount": str(amount)})
            raise AgentSafetyError(str(e), "BUDGET_UPDATE_ERROR", context)

    def get_budget_pools(self) -> list[BudgetPool]:
        """Get all budget pools.

        Returns:
            List of budget pools

        Raises:
            AgentSafetyError: If error occurs retrieving pools
        """
        try:
            return self._coordinator.get_pools()
        except AgentSafetyError as e:
            raise e
        except Exception as e:
            raise AgentSafetyError(str(e), "POOL_LIST_ERROR")

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

        Raises:
            AgentSafetyError: If pool creation fails
        """
        try:
            return self._coordinator.create_pool(name, initial_budget, priority)
        except AgentSafetyError as e:
            raise e
        except Exception as e:
            context = ErrorContext(
                details={
                    "name": name,
                    "initial_budget": str(initial_budget),
                    "priority": priority,
                },
            )
            raise AgentSafetyError(str(e), "POOL_CREATE_ERROR", context)


class MetricsAPIV1(MetricsAPIContract):
    """V1 implementation of metrics API."""

    def __init__(self, analyzer: MetricsAnalyzer):
        """Initialize metrics API.

        Args:
            analyzer: Metrics analyzer instance
        """
        if not isinstance(analyzer, MetricsAnalyzer):
            msg = "Invalid metrics analyzer instance"
            raise InvalidConfigurationError(msg)
        self._analyzer = analyzer
        super().__init__(version=APIVersion.V1)

    def validate_contract(self) -> None:
        """Validate the contract implementation."""
        if not hasattr(self, "_analyzer"):
            msg = "Metrics analyzer not initialized"
            raise InvalidConfigurationError(msg)

    def get_agent_metrics(self, agent_id: str) -> AgentMetrics:
        """Get metrics for an agent.

        Args:
            agent_id: ID of agent to get metrics for

        Returns:
            Agent metrics

        Raises:
            AgentSafetyError: If metrics retrieval fails
        """
        try:
            return self._analyzer.get_agent_metrics(agent_id)
        except AgentSafetyError as e:
            raise e
        except Exception as e:
            context = ErrorContext(agent_id=agent_id)
            raise AgentSafetyError(str(e), "METRICS_RETRIEVAL_ERROR", context)

    def get_system_metrics(self) -> SystemMetrics:
        """Get system-wide metrics.

        Returns:
            System metrics

        Raises:
            AgentSafetyError: If metrics retrieval fails
        """
        try:
            return self._analyzer.get_system_metrics()
        except AgentSafetyError as e:
            raise e
        except Exception as e:
            raise AgentSafetyError(str(e), "SYSTEM_METRICS_ERROR")

    def create_metrics(
        self,
        agent_id: str,
        metrics: AgentMetrics | SystemMetrics,
    ) -> None:
        """Create new metrics.

        Args:
            agent_id: ID of agent metrics are for
            metrics: Metrics to create

        Raises:
            AgentSafetyError: If metrics creation fails
        """
        try:
            self._analyzer.create_metrics(agent_id, metrics)
        except AgentSafetyError as e:
            raise e
        except Exception as e:
            context = ErrorContext(agent_id=agent_id)
            raise AgentSafetyError(str(e), "METRICS_CREATE_ERROR", context)


class AgentAPIV1(AgentAPIContract):
    """V1 implementation of agent API."""

    def __init__(self, coordinator: BudgetCoordinator):
        """Initialize agent API.

        Args:
            coordinator: Budget coordinator instance
        """
        if not isinstance(coordinator, BudgetCoordinator):
            msg = "Invalid budget coordinator instance"
            raise InvalidConfigurationError(msg)
        self._coordinator = coordinator
        super().__init__(version=APIVersion.V1)

    def validate_contract(self) -> None:
        """Validate the contract implementation."""
        if not hasattr(self, "_coordinator"):
            msg = "Budget coordinator not initialized"
            raise InvalidConfigurationError(msg)

    def get_agent(self, agent_id: str) -> Agent:
        """Get an agent.

        Args:
            agent_id: ID of agent to get

        Returns:
            Agent instance

        Raises:
            AgentSafetyError: If agent not found or retrieval fails
        """
        try:
            return self._coordinator.get_agent(agent_id)
        except AgentSafetyError as e:
            raise e
        except Exception as e:
            context = ErrorContext(agent_id=agent_id)
            raise AgentSafetyError(str(e), "AGENT_RETRIEVAL_ERROR", context)

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

        Raises:
            AgentSafetyError: If agent creation fails
        """
        try:
            return self._coordinator.create_agent(name, initial_budget, priority)
        except AgentSafetyError as e:
            raise e
        except Exception as e:
            context = ErrorContext(
                details={
                    "name": name,
                    "initial_budget": str(initial_budget),
                    "priority": priority,
                },
            )
            raise AgentSafetyError(str(e), "AGENT_CREATE_ERROR", context)

    def update_agent(self, agent: Agent) -> None:
        """Update an agent.

        Args:
            agent: Agent to update

        Raises:
            AgentSafetyError: If agent update fails
        """
        try:
            self._coordinator.update_agent(agent)
        except AgentSafetyError as e:
            raise e
        except Exception as e:
            context = ErrorContext(agent_id=agent.id)
            raise AgentSafetyError(str(e), "AGENT_UPDATE_ERROR", context)

    def delete_agent(self, agent_id: str) -> None:
        """Delete an agent.

        Args:
            agent_id: ID of agent to delete

        Raises:
            AgentSafetyError: If agent deletion fails
        """
        try:
            self._coordinator.delete_agent(agent_id)
        except AgentSafetyError as e:
            raise e
        except Exception as e:
            context = ErrorContext(agent_id=agent_id)
            raise AgentSafetyError(str(e), "AGENT_DELETE_ERROR", context)


class ConfigAPIV1(ConfigAPIContract):
    """V1 implementation of config API."""

    def __init__(self, config: Config):
        """Initialize config API.

        Args:
            config: Config instance
        """
        super().__init__(version=APIVersion.V1)
        if not isinstance(config, Config):
            msg = "Invalid config instance"
            raise InvalidConfigurationError(msg)
        self._config = config

    def validate_contract(self) -> None:
        """Validate the contract implementation."""
        if not hasattr(self, "_config"):
            msg = "Config not initialized"
            raise InvalidConfigurationError(msg)

    def get_config(self) -> Config:
        """Get current config.

        Returns:
            Current config

        Raises:
            AgentSafetyError: If config retrieval fails
        """
        try:
            return self._config
        except Exception as e:
            raise AgentSafetyError(str(e), "CONFIG_RETRIEVAL_ERROR")

    def update_config(self, config: Config) -> None:
        """Update config.

        Args:
            config: New config

        Raises:
            AgentSafetyError: If config update fails or validation fails
        """
        try:
            if not self.validate_config(config):
                msg = "Invalid configuration"
                raise InvalidConfigurationError(msg)
            self._config = config
        except AgentSafetyError as e:
            raise e
        except Exception as e:
            raise AgentSafetyError(str(e), "CONFIG_UPDATE_ERROR")

    def validate_config(self, config: Config) -> bool:
        """Validate a config.

        Args:
            config: Config to validate

        Returns:
            True if valid, False otherwise

        Raises:
            AgentSafetyError: If validation process fails
        """
        try:
            if not isinstance(config, Config):
                return False

            # Validate required fields
            required_fields = [
                "budget_limits",
                "resource_limits",
                "monitoring_settings",
                "alert_thresholds",
            ]

            for field in required_fields:
                if not hasattr(config, field):
                    return False

            # Validate budget limits
            if not isinstance(config.budget_limits, dict):
                return False

            # Validate resource limits
            if not isinstance(config.resource_limits, dict):
                return False

            # Validate monitoring settings
            if not isinstance(config.monitoring_settings, dict):
                return False

            # Validate alert thresholds
            return isinstance(config.alert_thresholds, dict)

        except Exception as e:
            raise AgentSafetyError(str(e), "CONFIG_VALIDATION_ERROR")
