"""Industry-specific safeguard plugins."""

from abc import abstractmethod
from typing import Any

from ..plugins import SafeguardPlugin
from ..types import AlertSeverity, SafetyAlert
from ..types.agent import Agent


class IndustrySafeguard(SafeguardPlugin):
    """Base class for industry-specific safeguards."""

    def __init__(self, industry_name: str):
        """Initialize the industry safeguard.

        Args:
            industry_name: Name of the industry this safeguard applies to
        """
        self._industry_name = industry_name
        self._initialized = False
        self._config: dict[str, Any] = {}
        self._monitored_agents: set[str] = set()

    @property
    def name(self) -> str:
        """Return the name of the plugin."""
        return f"{self._industry_name}_safeguard"

    @abstractmethod
    def validate_agent_action(
        self,
        agent: Agent,
        action_context: dict[str, Any],
    ) -> list[SafetyAlert]:
        """Validate an agent action against industry-specific rules.

        Args:
            agent: The agent performing the action
            action_context: Context information about the action

        Returns:
            List of safety alerts if any rules are violated, empty list otherwise
        """
        pass

    def monitor_agent(self, agent_id: str) -> None:
        """Add an agent to be monitored by this safeguard.

        Args:
            agent_id: ID of the agent to monitor
        """
        self._monitored_agents.add(agent_id)

    def stop_monitoring_agent(self, agent_id: str) -> None:
        """Remove an agent from being monitored by this safeguard.

        Args:
            agent_id: ID of the agent to stop monitoring
        """
        if agent_id in self._monitored_agents:
            self._monitored_agents.remove(agent_id)

    def is_monitoring_agent(self, agent_id: str) -> bool:
        """Check if an agent is being monitored by this safeguard.

        Args:
            agent_id: ID of the agent to check

        Returns:
            True if agent is being monitored, False otherwise
        """
        return agent_id in self._monitored_agents


class FinancialServicesSafeguard(IndustrySafeguard):
    """Safeguards specific to the financial services industry."""

    def __init__(self):
        """Initialize the financial services safeguard."""
        super().__init__("financial_services")
        self._restricted_actions = set()
        self._compliance_rules = {}
        self._transaction_limits = {}

    @property
    def version(self) -> str:
        """Return the version of the plugin."""
        return "1.0.0"

    def initialize(self, config: dict[str, Any]) -> None:
        """Initialize the plugin with configuration.

        Args:
            config: Plugin-specific configuration including:
                - restricted_actions: Set of actions not allowed
                - compliance_rules: Dict of compliance rules
                - transaction_limits: Dict of transaction limits
        """
        self._config = config
        self._restricted_actions = set(config.get("restricted_actions", []))
        self._compliance_rules = config.get("compliance_rules", {})
        self._transaction_limits = config.get("transaction_limits", {})
        self._initialized = True

    def shutdown(self) -> None:
        """Clean up resources when shutting down."""
        self._initialized = False
        self._monitored_agents.clear()

    def validate_agent_action(
        self,
        agent: Agent,
        action_context: dict[str, Any],
    ) -> list[SafetyAlert]:
        """Validate a financial services agent action.

        Args:
            agent: The agent performing the action
            action_context: Context information about the action

        Returns:
            List of safety alerts if any financial rules are violated
        """
        if not self._initialized:
            return [
                SafetyAlert(
                    title="Safeguard Not Initialized",
                    description="Financial services safeguard not properly initialized",
                    severity=AlertSeverity.ERROR,
                ),
            ]

        alerts = []

        # Check for restricted actions
        action_type = action_context.get("action_type", "")
        if action_type in self._restricted_actions:
            alerts.append(
                SafetyAlert(
                    title="Restricted Financial Action",
                    description=f"Agent attempted restricted action: {action_type}",
                    severity=AlertSeverity.ERROR,
                ),
            )

        # Check transaction limits
        if action_type == "transaction" and "amount" in action_context:
            amount = action_context["amount"]
            limit = self._transaction_limits.get(agent.id, float("inf"))
            if amount > limit:
                alerts.append(
                    SafetyAlert(
                        title="Transaction Limit Exceeded",
                        description=f"Transaction amount {amount} exceeds limit {limit}",
                        severity=AlertSeverity.WARNING,
                    ),
                )

        # Additional industry-specific validations can be added here

        return alerts


class HealthcareSafeguard(IndustrySafeguard):
    """Safeguards specific to the healthcare industry."""

    def __init__(self):
        """Initialize the healthcare safeguard."""
        super().__init__("healthcare")
        self._phi_patterns = set()
        self._restricted_operations = set()
        self._required_approvals = {}

    @property
    def version(self) -> str:
        """Return the version of the plugin."""
        return "1.0.0"

    def initialize(self, config: dict[str, Any]) -> None:
        """Initialize the plugin with configuration.

        Args:
            config: Plugin-specific configuration including:
                - phi_patterns: Patterns that might indicate PHI
                - restricted_operations: Operations requiring special handling
                - required_approvals: Dict of operations requiring approval
        """
        self._config = config
        self._phi_patterns = set(config.get("phi_patterns", []))
        self._restricted_operations = set(config.get("restricted_operations", []))
        self._required_approvals = config.get("required_approvals", {})
        self._initialized = True

    def shutdown(self) -> None:
        """Clean up resources when shutting down."""
        self._initialized = False
        self._monitored_agents.clear()

    def validate_agent_action(
        self,
        agent: Agent,
        action_context: dict[str, Any],
    ) -> list[SafetyAlert]:
        """Validate a healthcare agent action.

        Args:
            agent: The agent performing the action
            action_context: Context information about the action

        Returns:
            List of safety alerts if any healthcare rules are violated
        """
        if not self._initialized:
            return [
                SafetyAlert(
                    title="Safeguard Not Initialized",
                    description="Healthcare safeguard not properly initialized",
                    severity=AlertSeverity.ERROR,
                ),
            ]

        alerts = []

        # Check for potential PHI exposure
        content = action_context.get("content", "")
        for pattern in self._phi_patterns:
            if pattern in content:
                alerts.append(
                    SafetyAlert(
                        title="Potential PHI Exposure",
                        description="Content may contain protected health information",
                        severity=AlertSeverity.WARNING,
                    ),
                )
                break

        # Check for restricted operations
        operation = action_context.get("operation", "")
        if operation in self._restricted_operations:
            alerts.append(
                SafetyAlert(
                    title="Restricted Healthcare Operation",
                    description=f"Agent attempted restricted operation: {operation}",
                    severity=AlertSeverity.ERROR,
                ),
            )

        # Additional healthcare-specific validations can be added here

        return alerts
