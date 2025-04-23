"""API package for the Agent Safety Framework."""

from enum import Enum, auto

from .v1 import AgentAPIV1, BudgetAPIV1, MetricsAPIV1


class APIVersion(Enum):
    """API version enumeration."""

    V1 = auto()


class APIFactory:
    """Factory for creating API instances."""

    def create_budget_api(self, version: APIVersion, coordinator):
        """Create budget API instance.

        Args:
            version: API version to create
            coordinator: Budget coordinator instance

        Returns:
            Budget API instance
        """
        if version == APIVersion.V1:
            return BudgetAPIV1(coordinator)
        msg = f"Unsupported API version: {version}"
        raise ValueError(msg)

    def create_agent_api(self, version: APIVersion, coordinator):
        """Create agent API instance.

        Args:
            version: API version to create
            coordinator: Budget coordinator instance

        Returns:
            Agent API instance
        """
        if version == APIVersion.V1:
            return AgentAPIV1(coordinator)
        msg = f"Unsupported API version: {version}"
        raise ValueError(msg)

    def create_metrics_api(self, version: APIVersion, metrics_analyzer):
        """Create metrics API instance.

        Args:
            version: API version to create
            metrics_analyzer: Metrics analyzer instance

        Returns:
            Metrics API instance
        """
        if version == APIVersion.V1:
            return MetricsAPIV1(metrics_analyzer)
        msg = f"Unsupported API version: {version}"
        raise ValueError(msg)


__all__ = ["APIFactory", "APIVersion"]
