"""Budget monitoring implementation for agent resource tracking."""

from datetime import datetime
from decimal import Decimal

from ..core.alert_types import AlertSeverity
from ..core.notification_manager import NotificationManager
from ..types import BudgetMonitor as BudgetMonitorInterface
from ..types import SafetyAlert


class BudgetMonitor(BudgetMonitorInterface):
    """Monitors agent budget usage and triggers alerts when thresholds are reached."""

    def __init__(
        self,
        notification_manager: NotificationManager | None = None,
        warning_threshold: float = 0.75,
        critical_threshold: float = 0.90,
    ):
        """Initialize the budget monitor.

        Args:
            notification_manager: Manager for sending alerts
            warning_threshold: Budget usage percentage that triggers a warning (default: 75%)
            critical_threshold: Budget usage percentage that triggers a critical alert (default: 90%)
        """
        super().__init__()
        self.notification_manager = notification_manager
        self.warning_threshold = Decimal(str(warning_threshold))
        self.critical_threshold = Decimal(str(critical_threshold))
        self._alerted_agents: dict[str, set[str]] = {
            "warning": set(),
            "critical": set(),
        }
        self._agent_usages: dict[str, dict[str, Decimal]] = {}

    def check_budget_usage(
        self,
        agent_id: str,
        used_budget: Decimal,
        total_budget: Decimal,
    ) -> None:
        """Check an agent's budget usage against thresholds and trigger alerts if needed.

        Args:
            agent_id: ID of the agent to check
            used_budget: Amount of budget used
            total_budget: Total budget allocated to the agent
        """
        # Skip if no notification manager
        if not self.notification_manager:
            return

        # Calculate usage percentage
        usage_ratio = Decimal("0") if total_budget == Decimal("0") else used_budget / total_budget

        # Store usage data
        if agent_id not in self._agent_usages:
            self._agent_usages[agent_id] = {}
        self._agent_usages[agent_id] = {
            "used_budget": used_budget,
            "total_budget": total_budget,
            "usage_ratio": usage_ratio,
            "last_check": datetime.now(),
        }

        # Check for critical threshold
        if (
            usage_ratio >= self.critical_threshold
            and agent_id not in self._alerted_agents["critical"]
        ):
            self._alerted_agents["critical"].add(agent_id)
            self.notification_manager.create_alert(
                SafetyAlert(
                    title="Critical Budget Usage",
                    description=f"Agent {agent_id} has used {usage_ratio * 100:.1f}% of its budget",
                    severity=AlertSeverity.CRITICAL,
                    timestamp=datetime.now(),
                    metadata={"agent_id": agent_id},
                ),
            )
        # Check for warning threshold
        elif (
            usage_ratio >= self.warning_threshold
            and agent_id not in self._alerted_agents["warning"]
        ):
            self._alerted_agents["warning"].add(agent_id)
            self.notification_manager.create_alert(
                SafetyAlert(
                    title="High Budget Usage",
                    description=f"Agent {agent_id} has used {usage_ratio * 100:.1f}% of its budget",
                    severity=AlertSeverity.WARNING,
                    timestamp=datetime.now(),
                    metadata={"agent_id": agent_id},
                ),
            )
        # Reset alerts if budget usage drops below thresholds
        elif usage_ratio < self.warning_threshold:
            if agent_id in self._alerted_agents["warning"]:
                self._alerted_agents["warning"].remove(agent_id)
            if agent_id in self._alerted_agents["critical"]:
                self._alerted_agents["critical"].remove(agent_id)

    def get_budget_status(self, agent_id: str) -> dict:
        """Get the budget status for an agent.

        Args:
            agent_id: ID of the agent to get status for

        Returns:
            Dictionary with budget status information
        """
        if agent_id not in self._agent_usages:
            return {
                "used_budget": Decimal("0"),
                "total_budget": Decimal("0"),
                "usage_ratio": Decimal("0"),
                "warning_alert": False,
                "critical_alert": False,
            }

        return {
            **self._agent_usages[agent_id],
            "warning_alert": agent_id in self._alerted_agents["warning"],
            "critical_alert": agent_id in self._alerted_agents["critical"],
        }

    def reset_agent_alerts(self, agent_id: str) -> None:
        """Reset alerts for an agent.

        Args:
            agent_id: ID of the agent to reset alerts for
        """
        if agent_id in self._alerted_agents["warning"]:
            self._alerted_agents["warning"].remove(agent_id)
        if agent_id in self._alerted_agents["critical"]:
            self._alerted_agents["critical"].remove(agent_id)

    def reset(self) -> None:
        """Reset the monitor state."""
        self._alerted_agents = {"warning": set(), "critical": set()}
        self._agent_usages = {}

    def get_all_budget_statuses(self) -> dict[str, dict]:
        """Get budget statuses for all tracked agents.

        Returns:
            Dictionary mapping agent IDs to their budget status
        """
        return {agent_id: self.get_budget_status(agent_id) for agent_id in self._agent_usages}

    def clear_all_alerts(self) -> None:
        """Clear all budget alerts."""
        self._alerted_agents = {"warning": set(), "critical": set()}
