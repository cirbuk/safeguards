"""Notification system for FounderX.

This module provides functionality for:
- Slack notifications
- Email notifications
- Notification templates
- Alert severity levels
"""

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Literal

from founderx.config.settings import notification_settings
from founderx.core.alert_types import AlertSeverity, NotificationChannel
from founderx.core.cost_management import CostType
from pydantic import BaseModel, Field, validator

from safeguards.notifications import NotificationManager as BaseNotificationManager
from safeguards.types import (
    CostAlert,
    ResourceAlert,
    UsageReport,
)


@dataclass
class NotificationConfig:
    """Configuration for notifications."""

    # Slack settings
    slack_webhook_url: str | None = None
    slack_channel: str | None = None

    # Email settings (Resend)
    resend_api_key: str | None = None
    email_from: str | None = None
    email_to: list[str] = field(default_factory=list)

    # Global settings
    min_severity: AlertSeverity = AlertSeverity.WARNING
    enabled_channels: list[str] = field(default_factory=lambda: ["slack", "email"])

    @classmethod
    def from_settings(cls) -> "NotificationConfig":
        """Create NotificationConfig from settings."""
        return cls(
            slack_webhook_url=notification_settings.slack_webhook_url,
            slack_channel=notification_settings.slack_channel,
            resend_api_key=notification_settings.resend_api_key,
            email_from=notification_settings.email_from,
            email_to=notification_settings.email_to,
            min_severity=notification_settings.min_alert_severity,
            enabled_channels=notification_settings.enabled_notification_channels,
        )


class NotificationManager(BaseNotificationManager):
    """FounderX-specific notification manager implementation."""

    def __init__(
        self,
        enabled_channels: set[NotificationChannel] | None = None,
        template_dir: str | None = None,
        cooldown_period: int = 300,
    ):
        """Initialize FounderX notification manager.

        Args:
            enabled_channels: Set of enabled notification channels. If not provided,
                           uses channels from FounderX settings.
            template_dir: Directory containing notification templates. If not provided,
                        uses FounderX templates directory.
            cooldown_period: Minimum time (seconds) between similar alerts
        """
        # Use FounderX settings if no channels provided
        if enabled_channels is None:
            enabled_channels = set(notification_settings.enabled_channels)

        # Use FounderX templates if no directory provided
        if template_dir is None:
            template_dir = "src/founderx/templates"

        super().__init__(
            enabled_channels=enabled_channels,
            template_dir=template_dir,
            cooldown_period=cooldown_period,
        )

        # Configure channels from settings
        if notification_settings.email:
            self.configure_email(
                smtp_host=notification_settings.email.smtp_host,
                smtp_port=notification_settings.email.smtp_port,
                username=notification_settings.email.username,
                password=notification_settings.email.password,
                from_addr=notification_settings.email.from_addr,
                to_addrs=notification_settings.email.to_addrs,
            )

        if notification_settings.slack:
            self.configure_slack(
                webhook_url=notification_settings.slack.webhook_url,
                channel=notification_settings.slack.channel,
            )

        if notification_settings.webhook:
            self.configure_webhook(
                url=notification_settings.webhook.url,
                headers=notification_settings.webhook.headers,
            )

    def send_alert(
        self,
        alert: CostAlert | ResourceAlert,
        channels: list[str] | None = None,
    ) -> bool:
        """Send an alert through configured channels.

        Args:
            alert: Alert to send
            channels: Optional list of channels to use. If not provided, uses all enabled channels.

        Returns:
            True if alert was sent successfully through any channel
        """
        if alert.severity.value < notification_settings.min_alert_severity.value:
            return False

        return super().send_alert(alert, channels=channels)

    def send_report(
        self,
        report: UsageReport,
        channels: list[str] | None = None,
    ) -> bool:
        """Send a usage report through configured channels.

        Args:
            report: Report to send
            channels: Optional list of channels to use. If not provided, uses all enabled channels.

        Returns:
            True if report was sent successfully through any channel
        """
        return super().send_report(report, channels=channels)


class CostAlert(BaseModel):
    """Model representing a cost alert notification."""

    timestamp: datetime = Field(default_factory=datetime.now)
    agent_id: str
    cost_type: CostType
    current_usage: Decimal
    budget_limit: Decimal
    usage_percentage: float
    period: Literal["daily", "weekly", "monthly"]
    severity: AlertSeverity
    title: str
    message: str

    @validator("usage_percentage")
    def validate_percentage(self, v: float) -> float:
        """Validate that usage percentage is between 0 and 1."""
        if not 0 <= v <= 1:
            msg = "Usage percentage must be between 0 and 1"
            raise ValueError(msg)
        return v


class ResourceAlert(BaseModel):
    """Model representing a resource utilization alert."""

    timestamp: datetime = Field(default_factory=datetime.now)
    agent_id: str | None
    severity: AlertSeverity
    title: str
    message: str
    resource_type: str
    current_value: float
    threshold: float
    unit: str

    @validator("current_value", "threshold")
    def validate_positive(self, v: float) -> float:
        """Validate that values are positive."""
        if v < 0:
            msg = "Value must be positive"
            raise ValueError(msg)
        return v


class UsageReport(BaseModel):
    """Model representing a usage report."""

    timestamp: datetime = Field(default_factory=datetime.now)
    period: Literal["daily", "weekly", "monthly"]
    agent_id: str
    cost_breakdown: dict[str, Decimal]
    total_cost: Decimal
    budget_limit: Decimal
    usage_percentage: float

    @validator("usage_percentage")
    def validate_percentage(self, v: float) -> float:
        """Validate that usage percentage is between 0 and 1."""
        if not 0 <= v <= 1:
            msg = "Usage percentage must be between 0 and 1"
            raise ValueError(msg)
        return v

    @validator("total_cost")
    def validate_total_cost(self, v: Decimal, values: dict) -> Decimal:
        """Validate that total cost matches the sum of cost breakdown."""
        if "cost_breakdown" in values:
            total = sum(values["cost_breakdown"].values())
            if total != v:
                msg = f"Total cost ({v}) does not match sum of cost breakdown ({total})"
                raise ValueError(
                    msg,
                )
        return v
