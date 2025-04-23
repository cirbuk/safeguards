"""Core safety controller implementation."""

from decimal import Decimal

from ..base.budget import BudgetManager
from ..base.guardrails import Guardrail, ValidationResult
from ..base.monitoring import ResourceMonitor
from ..core.notification_manager import NotificationManager
from ..types import SafetyAlert, SafetyConfig, SafetyMetrics
from ..types.agent import Agent


class SafetyController:
    """Main controller for managing agent safety features.

    Provides centralized control over:
    - Budget management
    - Resource monitoring
    - Safety guardrails
    - Notifications and alerts
    """

    def __init__(self, config: SafetyConfig):
        """Initialize safety controller.

        Args:
            config: Safety configuration settings
        """
        self.config = config
        self.budget_manager = BudgetManager(
            total_budget=config.total_budget,
            hourly_limit=config.hourly_limit,
            daily_limit=config.daily_limit,
        )
        self.resource_monitor = ResourceMonitor()
        self.notification_manager = NotificationManager()

        # Track registered agents and their guardrails
        self._agents: dict[str, Agent] = {}
        self._agent_guardrails: dict[str, list[Guardrail]] = {}

    def register_agent(
        self,
        agent: Agent,
        budget: Decimal | None = None,
        guardrails: list[Guardrail] | None = None,
    ) -> None:
        """Register an agent for safety monitoring.

        Args:
            agent: Agent to register
            budget: Initial budget allocation
            guardrails: Safety guardrails to apply
        """
        if agent.id in self._agents:
            msg = f"Agent {agent.id} already registered"
            raise ValueError(msg)

        self._agents[agent.id] = agent
        if guardrails:
            self._agent_guardrails[agent.id] = guardrails

        if budget is not None:
            self.budget_manager.register_agent(agent.id, budget)

        # Start monitoring
        self.resource_monitor.start_monitoring(agent.id)

    def unregister_agent(self, agent_id: str) -> None:
        """Unregister an agent from safety monitoring.

        Args:
            agent_id: ID of agent to unregister
        """
        if agent_id not in self._agents:
            msg = f"Agent {agent_id} not registered"
            raise ValueError(msg)

        self.budget_manager.unregister_agent(agent_id)
        self.resource_monitor.stop_monitoring(agent_id)

        del self._agents[agent_id]
        if agent_id in self._agent_guardrails:
            del self._agent_guardrails[agent_id]

    def get_metrics(self, agent_id: str) -> SafetyMetrics:
        """Get current safety metrics for an agent.

        Args:
            agent_id: Agent to get metrics for

        Returns:
            Combined safety metrics
        """
        if agent_id not in self._agents:
            msg = f"Agent {agent_id} not registered"
            raise ValueError(msg)

        budget_metrics = self.budget_manager.get_metrics(agent_id)
        resource_metrics = self.resource_monitor.get_metrics(agent_id)
        alerts = self.notification_manager.get_alerts(agent_id)

        return SafetyMetrics(
            budget=budget_metrics,
            resources=resource_metrics,
            alerts=alerts,
        )

    def validate_action(self, agent_id: str, action_context: dict) -> ValidationResult:
        """Validate an agent action against safety guardrails.

        Args:
            agent_id: Agent performing the action
            action_context: Context for the action

        Returns:
            Validation result with any violations
        """
        if agent_id not in self._agents:
            msg = f"Agent {agent_id} not registered"
            raise ValueError(msg)

        # Check budget availability
        if not self.budget_manager.check_budget(agent_id):
            return ValidationResult(
                valid=False,
                violations=[
                    {
                        "type": "BUDGET_EXCEEDED",
                        "message": "Insufficient budget available",
                    },
                ],
            )

        # Check resource limits
        resource_status = self.resource_monitor.check_limits(agent_id)
        if not resource_status.within_limits:
            return ValidationResult(
                valid=False,
                violations=[
                    {
                        "type": "RESOURCE_LIMIT_EXCEEDED",
                        "message": f"Resource limits exceeded: {resource_status.details}",
                    },
                ],
            )

        # Apply guardrails
        if agent_id in self._agent_guardrails:
            for guardrail in self._agent_guardrails[agent_id]:
                result = guardrail.validate_input(action_context)
                if not result.valid:
                    return result

        return ValidationResult(valid=True)

    def record_action(
        self,
        agent_id: str,
        action_cost: Decimal,
        action_context: dict,
    ) -> None:
        """Record an agent action for monitoring.

        Args:
            agent_id: Agent that performed the action
            action_cost: Cost of the action
            action_context: Context of the action
        """
        if agent_id not in self._agents:
            msg = f"Agent {agent_id} not registered"
            raise ValueError(msg)

        # Update budget usage
        self.budget_manager.record_usage(agent_id, action_cost)

        # Check for concerning patterns
        metrics = self.get_metrics(agent_id)
        if metrics.budget.usage_percent > self.config.budget_warning_threshold:
            self.notification_manager.create_alert(
                SafetyAlert(
                    title="High Budget Usage",
                    description=f"Agent {agent_id} budget usage at {metrics.budget.usage_percent:.1f}%",
                    severity="WARNING",
                    metadata={"agent_id": agent_id},
                ),
            )

        if metrics.resources.cpu_percent > self.config.cpu_threshold:
            self.notification_manager.create_alert(
                SafetyAlert(
                    title="High CPU Usage",
                    description=f"Agent {agent_id} CPU usage at {metrics.resources.cpu_percent:.1f}%",
                    severity="WARNING",
                    metadata={"agent_id": agent_id},
                ),
            )
