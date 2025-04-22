"""Factory for creating API instances."""

from typing import Dict, Type, Union

from .base import (
    APIVersion,
    BudgetAPIContract,
    MetricsAPIContract,
    AgentAPIContract,
    ConfigAPIContract,
)
from .v1 import BudgetAPIV1, MetricsAPIV1, AgentAPIV1, ConfigAPIV1
from ..core.budget_coordination import BudgetCoordinator
from ..monitoring.metrics import MetricsAnalyzer
from ..core.agent import Agent
from ..config import Config
from ..exceptions import InvalidConfigurationError, ErrorContext


class APIFactory:
    """Factory class for creating API instances."""

    def __init__(self):
        """Initialize the API factory."""
        self._budget_apis: Dict[APIVersion, Type[BudgetAPIContract]] = {
            APIVersion.V1: BudgetAPIV1
        }
        self._metrics_apis: Dict[APIVersion, Type[MetricsAPIContract]] = {
            APIVersion.V1: MetricsAPIV1
        }
        self._agent_apis: Dict[APIVersion, Type[AgentAPIContract]] = {
            APIVersion.V1: AgentAPIV1
        }
        self._config_apis: Dict[APIVersion, Type[ConfigAPIContract]] = {
            APIVersion.V1: ConfigAPIV1
        }

    def _resolve_version(self, version: Union[APIVersion, str]) -> APIVersion:
        """Resolve API version from string or enum.

        Args:
            version: Version string or enum

        Returns:
            Resolved APIVersion

        Raises:
            InvalidConfigurationError: If version is invalid
        """
        if isinstance(version, str):
            try:
                version = APIVersion[version.upper()]
            except KeyError:
                context = ErrorContext(details={"version": version})
                raise InvalidConfigurationError(
                    f"Invalid API version: {version}", context=context
                )

        if version == APIVersion.LATEST:
            version = max(v for v in APIVersion if v != APIVersion.LATEST)

        return version

    def create_budget_api(
        self, version: Union[APIVersion, str], budget_coordinator: BudgetCoordinator
    ) -> BudgetAPIContract:
        """Create a budget API instance.

        Args:
            version: API version to create
            budget_coordinator: Budget coordinator instance

        Returns:
            Budget API instance

        Raises:
            InvalidConfigurationError: If version is invalid or not supported
        """
        try:
            version = self._resolve_version(version)
            api_class = self._budget_apis[version]
            return api_class(budget_coordinator)
        except KeyError:
            context = ErrorContext(details={"version": str(version)})
            raise InvalidConfigurationError(
                f"Unsupported API version: {version}", context=context
            )

    def create_metrics_api(
        self, version: Union[APIVersion, str], metrics_analyzer: MetricsAnalyzer
    ) -> MetricsAPIContract:
        """Create a metrics API instance.

        Args:
            version: API version to create
            metrics_analyzer: Metrics analyzer instance

        Returns:
            Metrics API instance

        Raises:
            InvalidConfigurationError: If version is invalid or not supported
        """
        try:
            version = self._resolve_version(version)
            api_class = self._metrics_apis[version]
            return api_class(metrics_analyzer)
        except KeyError:
            context = ErrorContext(details={"version": str(version)})
            raise InvalidConfigurationError(
                f"Unsupported API version: {version}", context=context
            )

    def create_agent_api(
        self, version: Union[APIVersion, str], coordinator: BudgetCoordinator
    ) -> AgentAPIContract:
        """Create an agent API instance.

        Args:
            version: API version to create
            coordinator: Budget coordinator instance

        Returns:
            Agent API instance

        Raises:
            InvalidConfigurationError: If version is invalid or not supported
        """
        try:
            version = self._resolve_version(version)
            api_class = self._agent_apis[version]
            return api_class(coordinator)
        except KeyError:
            context = ErrorContext(details={"version": str(version)})
            raise InvalidConfigurationError(
                f"Unsupported API version: {version}", context=context
            )

    def create_config_api(
        self, version: Union[APIVersion, str], config: Config
    ) -> ConfigAPIContract:
        """Create a config API instance.

        Args:
            version: API version to create
            config: Config instance

        Returns:
            Config API instance

        Raises:
            InvalidConfigurationError: If version is invalid or not supported
        """
        try:
            version = self._resolve_version(version)
            api_class = self._config_apis[version]
            return api_class(config)
        except KeyError:
            context = ErrorContext(details={"version": str(version)})
            raise InvalidConfigurationError(
                f"Unsupported API version: {version}", context=context
            )
