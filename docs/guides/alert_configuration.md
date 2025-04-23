# Alert Configuration Guide

This guide explains how to set up and configure the alerting system in Safeguards, allowing you to receive notifications for important events, monitor agent activity, and respond to potential issues.

## Alert System Overview

The alerting system in Safeguards provides:

- Real-time notifications for important events
- Multiple severity levels to categorize alerts
- Configurable notification channels (email, Slack, webhooks)
- Alert throttling to prevent notification fatigue
- Customizable alert templates

## Alert Severity Levels

Safeguards uses four severity levels for alerts:

| Severity | Description | Use Cases |
|----------|-------------|-----------|
| `INFO` | Informational messages | Task completion, status updates, normal operations |
| `WARNING` | Potential issues that need attention | Budget approaching limits, resource usage growing |
| `ERROR` | Problems that require action | Failed operations, budget exceeded, resource limits reached |
| `CRITICAL` | Severe issues that need immediate attention | System instability, security breaches, severe resource exhaustion |

## Basic Alert Setup

### Creating the Notification Manager

Start by creating a notification manager, which is the central component of the alerting system:

```python
from safeguards.notifications.manager import NotificationManager
from safeguards.types import NotificationChannel

# Create notification manager with console logging enabled by default
notification_manager = NotificationManager(
    enabled_channels={NotificationChannel.CONSOLE},
    cooldown_period=300  # 5 minutes between similar alerts
)
```

### Sending Basic Alerts

Send alerts using the `send_alert` method:

```python
from safeguards.core.alert_types import Alert, AlertSeverity
from datetime import datetime

# Create and send a basic alert
notification_manager.send_alert(
    Alert(
        title="Agent Task Completed",
        description="Agent has successfully completed its assigned task",
        severity=AlertSeverity.INFO,
        timestamp=datetime.now(),
        metadata={"agent_id": "agent-123", "task_id": "task-456"}
    )
)

# Create and send a warning alert
notification_manager.send_alert(
    Alert(
        title="Budget Usage High",
        description="Agent is approaching its budget limit",
        severity=AlertSeverity.WARNING,
        metadata={
            "agent_id": "agent-123",
            "budget_used": 75.5,
            "budget_limit": 100.0,
            "usage_percent": 75.5
        }
    )
)
```

## Configuring Notification Channels

### Email Notifications

Configure email notifications to receive alerts via email:

```python
# Configure email notifications
notification_manager.configure_email(
    smtp_host="smtp.example.com",
    smtp_port=587,  # Use 587 for TLS, 465 for SSL
    username="alerts@example.com",
    password="your-secure-password",
    from_addr="alerts@example.com",
    to_addrs=["admin@example.com", "team@example.com"]
)

# Enable email channel
notification_manager.enabled_channels.add(NotificationChannel.EMAIL)
```

The email template is customizable and can be found at `src/safeguards/templates/email_alert.html`. You can provide your own template directory:

```python
# Use custom templates
notification_manager = NotificationManager(
    template_dir="/path/to/your/templates"
)
```

### Slack Notifications

Configure Slack notifications to receive alerts in your Slack workspace:

```python
# Configure Slack notifications
notification_manager.configure_slack(
    webhook_url="https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK",
    channel="#alerts"  # Channel name where alerts will be posted
)

# Enable Slack channel
notification_manager.enabled_channels.add(NotificationChannel.SLACK)
```

### Webhook Notifications

Configure webhook notifications to send alerts to your own API or service:

```python
# Configure webhook notifications
notification_manager.configure_webhook(
    url="https://your-service.example.com/api/alerts",
    headers={
        "Authorization": "Bearer your-api-key",
        "Content-Type": "application/json"
    }
)

# Enable webhook channel
notification_manager.enabled_channels.add(NotificationChannel.WEBHOOK)
```

## Alert Throttling

Alerts with the same title and severity are throttled to prevent notification fatigue. The default cooldown period is 5 minutes (300 seconds), but this can be configured:

```python
# Configure a shorter cooldown period (60 seconds)
notification_manager = NotificationManager(cooldown_period=60)

# Or update it after creation
notification_manager.cooldown_period = 120  # 2 minutes
```

## Integration with Budget Monitoring

A common use case is to send alerts when an agent's budget usage reaches certain thresholds:

```python
from safeguards.monitoring.budget_monitor import BudgetMonitor

# Create a budget monitor with the notification manager
budget_monitor = BudgetMonitor(
    notification_manager=notification_manager,
    warning_threshold=0.75,  # 75% of budget
    critical_threshold=0.90  # 90% of budget
)

# Check budget usage (typically called by the budget coordinator)
budget_monitor.check_budget_usage(
    agent_id="agent-123",
    used_budget=Decimal("75.50"),
    total_budget=Decimal("100.00")
)
```

When the budget usage exceeds the warning threshold, a `WARNING` alert is sent. When it exceeds the critical threshold, a `CRITICAL` alert is sent.

## Integration with Resource Monitoring

Similarly, alerts can be triggered when resource usage exceeds thresholds:

```python
from safeguards.monitoring.resource_monitor import ResourceMonitor

# Create a resource monitor with the notification manager
resource_monitor = ResourceMonitor(
    notification_manager=notification_manager,
    cpu_threshold=80.0,  # 80% CPU usage
    memory_threshold=75.0  # 75% memory usage
)

# Start monitoring (typically called by the safeguards system)
resource_monitor.start()
```

## Complete Example

Here's a complete example showing how to set up a comprehensive alerting system:

```python
from decimal import Decimal
from safeguards.notifications.manager import NotificationManager
from safeguards.types import NotificationChannel
from safeguards.monitoring.budget_monitor import BudgetMonitor
from safeguards.monitoring.resource_monitor import ResourceMonitor
from safeguards.core.alert_types import Alert, AlertSeverity

# 1. Create notification manager with all channels enabled
notification_manager = NotificationManager(
    enabled_channels={
        NotificationChannel.CONSOLE,
        NotificationChannel.EMAIL,
        NotificationChannel.SLACK
    },
    cooldown_period=300
)

# 2. Configure email notifications
notification_manager.configure_email(
    smtp_host="smtp.example.com",
    smtp_port=587,
    username="alerts@example.com",
    password="your-secure-password",
    from_addr="alerts@example.com",
    to_addrs=["admin@example.com"]
)

# 3. Configure Slack notifications
notification_manager.configure_slack(
    webhook_url="https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK",
    channel="#safeguards-alerts"
)

# 4. Set up budget monitoring
budget_monitor = BudgetMonitor(
    notification_manager=notification_manager,
    warning_threshold=0.75,
    critical_threshold=0.90
)

# 5. Set up resource monitoring
resource_monitor = ResourceMonitor(
    notification_manager=notification_manager,
    cpu_threshold=80.0,
    memory_threshold=75.0
)

# 6. Start resource monitoring
resource_monitor.start()

# 7. Custom alert for a system event
notification_manager.send_alert(
    Alert(
        title="System Initialization Complete",
        description="Safeguards system has been successfully initialized",
        severity=AlertSeverity.INFO,
        metadata={"version": "1.0.0"}
    )
)
```

## Best Practices

1. **Set Appropriate Thresholds**: Choose threshold values that balance between too many alerts and too few alerts.

2. **Use Descriptive Titles**: Make alert titles clear and descriptive to quickly identify issues.

3. **Include Relevant Metadata**: Add useful context in the metadata to help diagnose and resolve issues.

4. **Configure Alert Throttling**: Adjust the cooldown period to prevent alert fatigue while ensuring important notifications are received.

5. **Monitor Your Alert System**: Periodically review alert frequency and adjust thresholds as needed.

6. **Secure Credentials**: Use environment variables or a secret manager for sensitive information like SMTP passwords and API keys.

7. **Test Your Alert Setup**: Verify that alerts are being sent correctly for each enabled channel.

## Troubleshooting

If alerts are not being sent:

1. Check logs for any error messages related to the notification manager.
2. Verify that the appropriate notification channels are enabled.
3. Ensure that credentials for email, Slack, or webhooks are correct.
4. Check if throttling is preventing alerts from being sent too frequently.
5. Verify network connectivity to SMTP servers, Slack API, or webhook endpoints.
