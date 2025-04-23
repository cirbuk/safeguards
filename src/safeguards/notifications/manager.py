"""Notification manager for handling alerts across different channels."""

import logging
import smtplib
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from enum import Enum, auto

import requests
from jinja2 import Environment, FileSystemLoader

from safeguards.core.resilience import RetryHandler, RetryStrategy
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
    agent_id: str | None = None
    metadata: dict = field(default_factory=dict)


logger = logging.getLogger(__name__)


class NotificationManager:
    """Manages notifications across different channels."""

    def __init__(
        self,
        enabled_channels: set[NotificationChannel] | None = None,
        template_dir: str | None = None,
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
        self.last_alerts: dict[str, datetime] = {}

        # Channel-specific configurations
        self.email_config: dict = {}
        self.slack_config: dict = {}
        self.webhook_config: dict = {}

        # For backward compatibility
        self.notifications: list[Notification] = []

        # Set default colors for different severity levels
        self.severity_colors = {
            AlertSeverity.INFO: "#0088cc",
            AlertSeverity.WARNING: "#ffcc00",
            AlertSeverity.ERROR: "#ffa500",
            AlertSeverity.CRITICAL: "#ff0000",
        }

        # Create retry handlers for different channels
        self.email_retry_handler = RetryHandler(
            max_attempts=3,
            strategy=RetryStrategy.EXPONENTIAL,
            base_delay=2.0,
            max_delay=30.0,
            jitter=0.2,
            retryable_exceptions=[
                ConnectionError,
                TimeoutError,
                smtplib.SMTPException,
                smtplib.SMTPServerDisconnected,
                smtplib.SMTPResponseException,
            ],
        )

        self.webhook_retry_handler = RetryHandler(
            max_attempts=3,
            strategy=RetryStrategy.EXPONENTIAL,
            base_delay=1.0,
            max_delay=15.0,
            jitter=0.1,
            retryable_exceptions=[
                ConnectionError,
                TimeoutError,
                requests.exceptions.RequestException,
            ],
        )

    def configure_email(
        self,
        smtp_host: str,
        smtp_port: int,
        username: str,
        password: str,
        from_addr: str,
        to_addrs: list[str],
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

    def configure_webhook(self, url: str, headers: dict | None = None):
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
                logger.error(f"Failed to send alert through {channel}: {e!s}")

        if success:
            self.last_alerts[alert_key] = datetime.now()

        return success

    @RetryHandler(
        max_attempts=3,
        strategy=RetryStrategy.EXPONENTIAL,
        base_delay=2.0,
        jitter=0.2,
        retryable_exceptions=[
            smtplib.SMTPException,
            smtplib.SMTPServerDisconnected,
            smtplib.SMTPResponseException,
            ConnectionError,
            TimeoutError,
        ],
    )
    def _send_email_alert(self, alert: Alert) -> bool:
        """Send alert via email using configured SMTP settings with automatic retry.

        Args:
            alert: Alert object to send

        Returns:
            bool: True if email was sent successfully

        Raises:
            Various SMTP exceptions that will trigger retries
        """
        # Render email template
        template = self.template_env.get_template("email_alert.html")
        html_content = template.render(
            alert=alert,  # Pass the entire alert object to template
            title=alert.title,
            description=alert.description,
            severity=alert.severity.name,
            timestamp=alert.timestamp,
            metadata=alert.metadata,
        )

        # Create email message
        msg = MIMEMultipart()
        msg["From"] = self.email_config["from_addr"]
        msg["Subject"] = f"[{alert.severity.name}] {alert.title}"

        # Add HTML content
        msg.attach(MIMEText(html_content, "html"))

        # Connect to SMTP server and send email
        with smtplib.SMTP(
            self.email_config["smtp_host"],
            self.email_config["smtp_port"],
        ) as server:
            # Use TLS if port is 587
            if self.email_config["smtp_port"] == 587:
                server.starttls()

            # Login if credentials are provided
            if self.email_config["username"] and self.email_config["password"]:
                server.login(
                    self.email_config["username"],
                    self.email_config["password"],
                )

            # Send email to all recipients
            for recipient in self.email_config["to_addrs"]:
                msg["To"] = recipient
                server.send_message(msg)
                logger.info(f"Email alert sent to {recipient}: {alert.title}")

        return True

    def _send_slack_alert(self, alert: Alert) -> bool:
        """Send alert to Slack channel."""
        try:
            color = self.severity_colors.get(alert.severity, "#808080")

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
                    },
                ],
            }

            # Use webhook retry handler for Slack notifications
            def send_slack_request():
                response = requests.post(
                    self.slack_config["webhook_url"],
                    json=payload,
                    timeout=10,  # Add timeout for security
                )
                response.raise_for_status()
                return response

            self.webhook_retry_handler(send_slack_request)()
            logger.info(f"Slack alert sent: {alert.title}")
            return True
        except Exception as e:
            logger.error(f"Failed to send Slack alert: {e!s}")
            return False

    def _send_webhook_alert(self, alert: Alert) -> bool:
        """Send alert to configured webhook endpoint with retry."""
        try:
            payload = {
                "title": alert.title,
                "description": alert.description,
                "severity": alert.severity.name,
                "timestamp": alert.timestamp.isoformat(),
                "metadata": alert.metadata,
            }

            # Use webhook retry handler
            def send_webhook_request():
                response = requests.post(
                    self.webhook_config["url"],
                    json=payload,
                    headers=self.webhook_config["headers"],
                    timeout=10,  # Add timeout for security
                )
                response.raise_for_status()
                return response

            self.webhook_retry_handler(send_webhook_request)()
            logger.info(f"Webhook alert sent: {alert.title}")
            return True
        except Exception as e:
            logger.error(f"Failed to send webhook alert: {e!s}")
            return False

    def _send_console_alert(self, alert: Alert) -> bool:
        """Log alert to console."""
        log_level = {
            AlertSeverity.INFO: logging.INFO,
            AlertSeverity.WARNING: logging.WARNING,
            AlertSeverity.ERROR: logging.ERROR,
            AlertSeverity.CRITICAL: logging.CRITICAL,
        }.get(alert.severity, logging.INFO)

        logger.log(
            log_level,
            f"[{alert.severity.name}] {alert.title}: {alert.description}",
        )
        return True

    # Backward compatibility methods
    def notify(
        self,
        level: NotificationLevel,
        message: str,
        agent_id: str | None = None,
        metadata: dict | None = None,
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
        self,
        level: NotificationLevel | None = None,
        agent_id: str | None = None,
    ) -> list[Notification]:
        """Get notifications with optional filtering (backward compatibility)."""
        result = self.notifications

        if level is not None:
            result = [n for n in result if n.level == level]

        if agent_id is not None:
            result = [n for n in result if n.agent_id == agent_id]

        return result

    def clear_notifications(
        self,
        level: NotificationLevel | None = None,
        agent_id: str | None = None,
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
