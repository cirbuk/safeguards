"""Notification manager for handling alerts and notifications."""

from collections import defaultdict

from ..types import SafetyAlert


class NotificationManager:
    """Manages safety alerts and notifications."""

    def __init__(self):
        """Initialize notification manager."""
        self._alerts: dict[str, list[SafetyAlert]] = defaultdict(list)
        self._notifications: list[SafetyAlert] = []

    def create_alert(self, alert: SafetyAlert) -> None:
        """Create a new safety alert.

        Args:
            alert: Alert to create
        """
        self._notifications.append(alert)

        # If alert is associated with an agent, store in agent alerts
        if "agent_id" in alert.metadata:
            agent_id = alert.metadata["agent_id"]
            self._alerts[agent_id].append(alert)

    def send_alert(self, alert: SafetyAlert) -> None:
        """Send a safety alert.

        This is an alias for create_alert to maintain compatibility with test mocks.

        Args:
            alert: Alert to send
        """
        self.create_alert(alert)

    def get_alerts(self, agent_id: str | None = None) -> list[SafetyAlert]:
        """Get alerts for an agent or all alerts.

        Args:
            agent_id: Optional agent ID to filter alerts

        Returns:
            List of alerts
        """
        if agent_id:
            return self._alerts[agent_id]
        return self._notifications

    def get_notifications(self) -> list[SafetyAlert]:
        """Get all notifications.

        Returns:
            List of all notifications
        """
        return self._notifications

    def clear_alerts(self, agent_id: str | None = None) -> None:
        """Clear alerts for an agent or all alerts.

        Args:
            agent_id: Optional agent ID to clear alerts for
        """
        if agent_id:
            self._alerts[agent_id].clear()
        else:
            self._alerts.clear()
            self._notifications.clear()

    def get_active_alerts(
        self,
        agent_id: str | None = None,
        severity: str | None = None,
    ) -> list[SafetyAlert]:
        """Get active alerts filtered by agent and/or severity.

        Args:
            agent_id: Optional agent ID to filter alerts
            severity: Optional severity level to filter

        Returns:
            List of matching active alerts
        """
        alerts = self.get_alerts(agent_id)

        if severity:
            alerts = [a for a in alerts if a.severity == severity]

        return alerts
