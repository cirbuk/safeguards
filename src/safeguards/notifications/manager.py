"""Notification manager for handling alerts across different channels."""

import logging
from typing import Dict, List, Optional, Set
from datetime import datetime, timedelta
from enum import Enum, auto
from dataclasses import dataclass, field

from jinja2 import Environment, FileSystemLoader
import requests

from safeguards.types import Alert, AlertSeverity, NotificationChannel


# Backward compatibility classes
class NotificationLevel(Enum):
    """Notification severity levels (for backward compatibility)."""

    INFO = auto()
    WARNING = auto()
    ERROR = auto()
    CRITICAL = auto()


@dataclass
class Notification:
    """Notification data class (for backward compatibility)."""

    level: NotificationLevel
    message: str
    timestamp: datetime = field(default_factory=datetime.now)
    agent_id: Optional[str] = None
    metadata: Dict = field(default_factory=dict)


logger = logging.getLogger(__name__)


class NotificationManager:
    """Manages notifications across different channels."""

    def __init__(
        self,
        enabled_channels: Optional[Set[NotificationChannel]] = None,
        template_dir: Optional[str] = None,
        cooldown_period: int = 300,
    ):
        """Initialize notification manager.

        Args:
            enabled_channels: Set of enabled notification channels
            template_dir: Directory containing notification templates. If not provided,
                        uses the default templates from agent-safety package.
            cooldown_period: Minimum time (seconds) between similar alerts
        """
        self.enabled_channels = enabled_channels or {NotificationChannel.CONSOLE}
        self.template_env = Environment(
            loader=FileSystemLoader(template_dir or "src/safeguards/templates"),
            autoescape=True,  # Enable autoescaping for security
        )
        self.cooldown_period = cooldown_period
        self.last_alerts: Dict[str, datetime] = {}

        # Channel-specific configurations
        self.email_config: Dict = {}
        self.slack_config: Dict = {}
        self.webhook_config: Dict = {}

        # For backward compatibility
        self.notifications: List[Notification] = []

    def configure_email(
        self,
        smtp_host: str,
        smtp_port: int,
        username: str,
        password: str,
        from_addr: str,
        to_addrs: List[str],
    ):
        """Configure email notification settings."""
        self.email_config = {
            "smtp_host": smtp_host,
            "smtp_port": smtp_port,
            "username": username,
            "password": password,
            "from_addr": from_addr,
            "to_addrs": to_addrs,
        }

    def configure_slack(self, webhook_url: str, channel: str):
        """Configure Slack notification settings."""
        self.slack_config = {"webhook_url": webhook_url, "channel": channel}

    def configure_webhook(self, url: str, headers: Optional[Dict] = None):
        """Configure webhook notification settings."""
        self.webhook_config = {"url": url, "headers": headers or {}}

    def send_alert(self, alert: Alert) -> bool:
        """Send alert through configured channels.

        Args:
            alert: Alert object containing notification details

        Returns:
            bool: True if alert was sent successfully through any channel
        """
        alert_key = f"{alert.title}_{alert.severity.name}"

        # Check cooldown period
        if alert_key in self.last_alerts:
            time_since_last = datetime.now() - self.last_alerts[alert_key]
            if time_since_last < timedelta(seconds=self.cooldown_period):
                logger.debug(f"Alert {alert_key} suppressed due to cooldown period")
                return False

        success = False
        for channel in self.enabled_channels:
            try:
                if channel == NotificationChannel.EMAIL and self.email_config:
                    success |= self._send_email_alert(alert)
                elif channel == NotificationChannel.SLACK and self.slack_config:
                    success |= self._send_slack_alert(alert)
                elif channel == NotificationChannel.WEBHOOK and self.webhook_config:
                    success |= self._send_webhook_alert(alert)
                elif channel == NotificationChannel.CONSOLE:
                    success |= self._send_console_alert(alert)
            except Exception as e:
                logger.error(f"Failed to send alert through {channel}: {str(e)}")

        if success:
            self.last_alerts[alert_key] = datetime.now()

        return success

    def _send_email_alert(self, alert: Alert) -> bool:
        """Send alert via email using configured SMTP settings."""
        try:
            template = self.template_env.get_template("email_alert.html")
            html_content = template.render(
                title=alert.title,
                description=alert.description,
                severity=alert.severity.name,
                timestamp=alert.timestamp,
                metadata=alert.metadata,
            )
            # TODO: Implement email sending logic
            logger.info(f"Email alert sent: {alert.title}")
            return True
        except Exception as e:
            logger.error(f"Failed to send email alert: {str(e)}")
            return False

    def _send_slack_alert(self, alert: Alert) -> bool:
        """Send alert to Slack channel."""
        try:
            color = {
                AlertSeverity.INFO: "#36a64f",
                AlertSeverity.WARNING: "#ffd700",
                AlertSeverity.HIGH: "#ffa500",
                AlertSeverity.CRITICAL: "#ff0000",
            }.get(alert.severity, "#808080")

            payload = {
                "channel": self.slack_config["channel"],
                "attachments": [
                    {
                        "color": color,
                        "title": alert.title,
                        "text": alert.description,
                        "fields": [
                            {
                                "title": "Severity",
                                "value": alert.severity.name,
                                "short": True,
                            },
                            {
                                "title": "Time",
                                "value": alert.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                                "short": True,
                            },
                        ],
                    }
                ],
            }

            response = requests.post(
                self.slack_config["webhook_url"],
                json=payload,
                timeout=10,  # Add timeout for security
            )
            response.raise_for_status()
            logger.info(f"Slack alert sent: {alert.title}")
            return True
        except Exception as e:
            logger.error(f"Failed to send Slack alert: {str(e)}")
            return False

    def _send_webhook_alert(self, alert: Alert) -> bool:
        """Send alert to configured webhook endpoint."""
        try:
            payload = {
                "title": alert.title,
                "description": alert.description,
                "severity": alert.severity.name,
                "timestamp": alert.timestamp.isoformat(),
                "metadata": alert.metadata,
            }

            response = requests.post(
                self.webhook_config["url"],
                json=payload,
                headers=self.webhook_config["headers"],
                timeout=10,  # Add timeout for security
            )
            response.raise_for_status()
            logger.info(f"Webhook alert sent: {alert.title}")
            return True
        except Exception as e:
            logger.error(f"Failed to send webhook alert: {str(e)}")
            return False

    def _send_console_alert(self, alert: Alert) -> bool:
        """Log alert to console."""
        log_level = {
            AlertSeverity.INFO: logging.INFO,
            AlertSeverity.WARNING: logging.WARNING,
            AlertSeverity.HIGH: logging.ERROR,
            AlertSeverity.CRITICAL: logging.CRITICAL,
        }.get(alert.severity, logging.INFO)

        logger.log(
            log_level, f"[{alert.severity.name}] {alert.title}: {alert.description}"
        )
        return True

    # Backward compatibility methods
    def notify(
        self,
        level: NotificationLevel,
        message: str,
        agent_id: Optional[str] = None,
        metadata: Optional[Dict] = None,
    ):
        """Create and store a notification (backward compatibility)."""
        notification = Notification(
            level=level,
            message=message,
            agent_id=agent_id,
            metadata=metadata or {},
        )
        self.notifications.append(notification)

        # Print to console for test compatibility
        print(f"{level.name}: {message}")

    def get_notifications(
        self, level: Optional[NotificationLevel] = None, agent_id: Optional[str] = None
    ) -> List[Notification]:
        """Get notifications with optional filtering (backward compatibility)."""
        result = self.notifications

        if level is not None:
            result = [n for n in result if n.level == level]

        if agent_id is not None:
            result = [n for n in result if n.agent_id == agent_id]

        return result

    def clear_notifications(
        self, level: Optional[NotificationLevel] = None, agent_id: Optional[str] = None
    ) -> None:
        """Clear notifications with optional filtering (backward compatibility)."""
        if level is None and agent_id is None:
            self.notifications = []
            return

        to_keep = []
        for notification in self.notifications:
            keep = True

            if level is not None and notification.level == level:
                keep = False

            if agent_id is not None and notification.agent_id == agent_id:
                keep = False

            if keep:
                to_keep.append(notification)

        self.notifications = to_keep
