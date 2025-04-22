# Notifications & Alerts Guide

This guide explains how to set up and customize the notification system in the Safeguards, enabling you to monitor agent activity, detect problems, and respond to events.

## Core Notification Concepts

The notification system in the Safeguards is built around:

- **Alerts**: Structured messages about significant events
- **Notification Channels**: Methods for delivering alerts (email, Slack, etc.)
- **Handlers**: Functions that process alerts and take action
- **Severity Levels**: Categorization of alerts by importance
- **Throttling**: Control over alert frequency

## Setting Up the Notification System

### Basic Notification Manager

Start by creating a notification manager:

```python
from safeguards.core.notification_manager import NotificationManager
from safeguards.types import AlertSeverity

# Create a notification manager
notification_manager = NotificationManager()
```

### Creating and Sending Alerts

Send alerts for important events:

```python
# Send a basic alert
notification_manager.send_alert(
    message="Agent has completed its task",
    severity=AlertSeverity.INFORMATIONAL
)

# Send an alert with additional context
notification_manager.send_alert(
    agent_id="agent123",
    message="Agent budget below 20% threshold",
    severity=AlertSeverity.WARNING,
    metadata={
        "current_budget": 18.5,
        "initial_budget": 100.0,
        "percentage": 18.5
    }
)

# Send a critical alert
notification_manager.send_alert(
    agent_id="agent456",
    message="Agent exceeded resource limit",
    severity=AlertSeverity.CRITICAL,
    metadata={
        "resource_type": "memory",
        "limit": "512MB",
        "actual_usage": "650MB",
        "overage_percentage": 27
    }
)
```

## Alert Handlers

### Basic Alert Handler

Implement a handler to process alerts:

```python
def basic_alert_handler(alert):
    """Process all alerts."""
    print(f"[{alert.severity.name}] {alert.timestamp}: {alert.message}")
    if alert.agent_id:
        print(f"Agent: {alert.agent_id}")
    if alert.metadata:
        print(f"Details: {alert.metadata}")
    return True  # Returning True indicates the alert was handled

# Register the handler
notification_manager.add_handler(basic_alert_handler)
```

### Filtering Alerts by Severity

Create handlers that only process certain severity levels:

```python
def critical_alert_handler(alert):
    """Handle only critical alerts."""
    if alert.severity == AlertSeverity.CRITICAL:
        print(f"CRITICAL ALERT: {alert.message}")
        # Take immediate action
        # ...
        return True
    return False  # Not handled, pass to other handlers

def warning_alert_handler(alert):
    """Handle only warning alerts."""
    if alert.severity == AlertSeverity.WARNING:
        print(f"Warning: {alert.message}")
        # Log warning
        # ...
        return True
    return False

# Register handlers in order of precedence
notification_manager.add_handler(critical_alert_handler)
notification_manager.add_handler(warning_alert_handler)
```

### Filtering Alerts by Agent

Create handlers that only process alerts for specific agents:

```python
def agent_specific_handler(alert):
    """Handle alerts only for a specific agent."""
    target_agent_id = "agent123"
    if alert.agent_id == target_agent_id:
        print(f"Alert for {target_agent_id}: {alert.message}")
        # Take agent-specific action
        # ...
        return True
    return False

# Register the handler
notification_manager.add_handler(agent_specific_handler)
```

## Notification Channels

### Email Notifications

Send alerts via email:

```python
from safeguards.notification.channels import EmailChannel

# Create an email channel
email_channel = EmailChannel(
    smtp_server="smtp.example.com",
    smtp_port=587,
    username="alerts@example.com",
    password="your-password",
    sender="alerts@example.com",
    recipients=["admin@example.com", "team@example.com"]
)

# Register the channel with minimum severity threshold
notification_manager.register_channel(
    channel=email_channel,
    min_severity=AlertSeverity.WARNING  # Only WARNING and above
)
```

### Slack Notifications

Send alerts to Slack:

```python
from safeguards.notification.channels import SlackChannel

# Create a Slack channel
slack_channel = SlackChannel(
    webhook_url="https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK",
    channel="#agent-alerts",
    username="Safeguards Monitor"
)

# Register the channel with custom formatting
notification_manager.register_channel(
    channel=slack_channel,
    min_severity=AlertSeverity.INFORMATIONAL,  # All alerts
    formatter=lambda alert: {
        "text": f"*{alert.severity.name}*: {alert.message}",
        "attachments": [{
            "fields": [
                {"title": "Agent", "value": alert.agent_id or "N/A", "short": True},
                {"title": "Time", "value": alert.timestamp, "short": True}
            ]
        }]
    }
)
```

### Webhook Notifications

Send alerts to custom webhooks:

```python
from safeguards.notification.channels import WebhookChannel

# Create a webhook channel
webhook_channel = WebhookChannel(
    url="https://your-service.example.com/api/alerts",
    headers={
        "Authorization": "Bearer your-api-key",
        "Content-Type": "application/json"
    }
)

# Register the channel
notification_manager.register_channel(
    channel=webhook_channel,
    min_severity=AlertSeverity.ERROR
)
```

### Custom Notification Channels

Implement a custom notification channel:

```python
from safeguards.notification.base import NotificationChannel
from safeguards.types.alerts import Alert

class CustomChannel(NotificationChannel):
    def __init__(self, config):
        self.config = config

    def send(self, alert: Alert) -> bool:
        """Send alert through custom channel."""
        # Custom implementation
        print(f"Sending through custom channel: {alert.message}")

        # You could implement any notification method here
        # - Database logging
        # - Message queue
        # - Mobile push notification
        # - Custom API integration

        return True  # Return True if successfully sent

# Create and register custom channel
custom_channel = CustomChannel(config={"param": "value"})
notification_manager.register_channel(custom_channel)
```

## Advanced Alert Handling

### Alert Throttling

Prevent alert floods with throttling:

```python
from safeguards.notification.throttling import ThrottlingManager

# Create a throttling manager
throttling_manager = ThrottlingManager()

# Create a throttled handler
def throttled_alert_handler(alert):
    """Handle alerts with throttling."""
    # Generate a key for this type of alert
    key = f"{alert.agent_id}:{alert.severity.name}:{alert.metadata.get('type', 'general')}"

    # Check if this alert type is throttled
    if throttling_manager.is_throttled(key):
        # Skip this alert
        return True

    # Process the alert
    print(f"Processing alert: {alert.message}")

    # Apply throttling for this alert type (no more alerts for 5 minutes)
    throttling_manager.throttle(key, duration_seconds=300)

    return True

# Register the handler
notification_manager.add_handler(throttled_alert_handler)
```

### Alert Aggregation

Aggregate similar alerts:

```python
from safeguards.notification.aggregation import AlertAggregator
from datetime import timedelta

# Create an alert aggregator
aggregator = AlertAggregator(
    window_size=timedelta(minutes=5),
    max_count=10
)

# Create an aggregating handler
def aggregating_handler(alert):
    """Handle alerts with aggregation."""
    # Determine aggregation key
    if alert.agent_id and alert.metadata.get("type") == "budget_warning":
        key = f"budget_warning:{alert.agent_id}"

        # Add alert to aggregator
        aggregated = aggregator.add(key, alert)

        if aggregated is not None:
            # We have enough alerts to trigger aggregated notification
            count = len(aggregated)
            first_alert = aggregated[0]
            last_alert = aggregated[-1]

            # Send aggregated alert
            notification_manager.send_alert(
                agent_id=first_alert.agent_id,
                message=f"Multiple budget warnings ({count}) for this agent",
                severity=AlertSeverity.WARNING,
                metadata={
                    "aggregated": True,
                    "count": count,
                    "time_span": (last_alert.timestamp - first_alert.timestamp).total_seconds(),
                    "first_message": first_alert.message,
                    "last_message": last_alert.message
                }
            )

            return True  # Handled by aggregation

    # Not handled by aggregation, pass to other handlers
    return False

# Register the handler
notification_manager.add_handler(aggregating_handler)
```

### Alert Escalation

Automatically escalate unhandled alerts:

```python
from safeguards.notification.escalation import EscalationManager
from datetime import timedelta

# Create an escalation manager
escalation_manager = EscalationManager()

# Configure escalation rules
escalation_manager.add_rule(
    name="critical_alerts",
    condition=lambda alert: alert.severity == AlertSeverity.CRITICAL,
    escalation_levels=[
        {
            "delay": timedelta(minutes=0),  # Immediate
            "action": lambda alert: notification_manager.send_via_channel(
                alert, channel_id="slack"
            )
        },
        {
            "delay": timedelta(minutes=5),  # After 5 minutes if not acknowledged
            "action": lambda alert: notification_manager.send_via_channel(
                alert, channel_id="email"
            )
        },
        {
            "delay": timedelta(minutes=15),  # After 15 minutes if still not acknowledged
            "action": lambda alert: notification_manager.send_via_channel(
                alert, channel_id="sms"
            )
        }
    ]
)

# Start the escalation manager
escalation_manager.start()

# Report an alert that will be escalated if not acknowledged
alert_id = notification_manager.send_alert(
    message="Critical system failure",
    severity=AlertSeverity.CRITICAL,
    metadata={"requires_acknowledgement": True}
)

# Later, to acknowledge the alert and stop escalation
escalation_manager.acknowledge_alert(alert_id)
```

## Integrating with Monitoring

### Budget Monitoring Alerts

Set up alerts for budget-related issues:

```python
from safeguards.types import ViolationType

def setup_budget_alerts(budget_coordinator, notification_manager):
    """Set up budget monitoring alerts."""

    def budget_monitor(agent_id, current_budget, initial_budget):
        """Monitor agent budget and trigger alerts."""
        # Calculate percentage
        if initial_budget > 0:
            percentage = (current_budget / initial_budget) * 100

            # Check thresholds
            if percentage <= 5:
                notification_manager.send_alert(
                    agent_id=agent_id,
                    message=f"Critical: Agent budget nearly depleted ({percentage:.1f}%)",
                    severity=AlertSeverity.CRITICAL,
                    metadata={
                        "type": "budget_warning",
                        "current_budget": float(current_budget),
                        "initial_budget": float(initial_budget),
                        "percentage": float(percentage)
                    }
                )
            elif percentage <= 20:
                notification_manager.send_alert(
                    agent_id=agent_id,
                    message=f"Warning: Agent budget below 20% ({percentage:.1f}%)",
                    severity=AlertSeverity.WARNING,
                    metadata={
                        "type": "budget_warning",
                        "current_budget": float(current_budget),
                        "initial_budget": float(initial_budget),
                        "percentage": float(percentage)
                    }
                )
            elif percentage <= 50:
                notification_manager.send_alert(
                    agent_id=agent_id,
                    message=f"Info: Agent budget below 50% ({percentage:.1f}%)",
                    severity=AlertSeverity.INFORMATIONAL,
                    metadata={
                        "type": "budget_warning",
                        "current_budget": float(current_budget),
                        "initial_budget": float(initial_budget),
                        "percentage": float(percentage)
                    }
                )

        return True  # Continue monitoring

    # Register budget monitor for all agents
    agent_ids = budget_coordinator.get_all_agent_ids()
    for agent_id in agent_ids:
        budget_coordinator.register_budget_monitor(agent_id, budget_monitor)
```

### Violation Alerts

Set up alerts for safety violations:

```python
from safeguards.monitoring.violation_reporter import ViolationReporter

def setup_violation_alerts(notification_manager):
    """Set up violation-to-alert conversion."""

    # Create a violation reporter
    violation_reporter = ViolationReporter(notification_manager)

    # Example of reporting a violation
    violation_reporter.report_violation(
        agent_id="agent123",
        violation_type=ViolationType.RESOURCE_LIMIT_EXCEEDED,
        severity=AlertSeverity.HIGH,
        message="Agent exceeded CPU usage limit",
        details={
            "limit": "2.0 cores",
            "actual": "3.2 cores",
            "duration": "45 seconds"
        }
    )

    return violation_reporter
```

### Periodic Status Alerts

Set up regular status notifications:

```python
import threading
import time
from datetime import datetime

def setup_periodic_status_alerts(notification_manager, interval_seconds=3600):
    """Set up periodic status notifications."""

    def send_status_update():
        """Send a status update notification."""
        # Gather system metrics
        system_metrics = {
            "active_agents": 5,
            "total_budget_used": 450.75,
            "total_remaining_budget": 1250.25,
            "violations_last_hour": 2
        }

        # Send status notification
        notification_manager.send_alert(
            message=f"Hourly system status update",
            severity=AlertSeverity.INFORMATIONAL,
            metadata={
                "type": "status_update",
                "timestamp": datetime.now().isoformat(),
                "metrics": system_metrics
            }
        )

    def status_thread():
        """Thread that sends periodic status updates."""
        while True:
            try:
                send_status_update()
            except Exception as e:
                print(f"Error sending status update: {str(e)}")

            # Sleep until next interval
            time.sleep(interval_seconds)

    # Start status thread
    thread = threading.Thread(target=status_thread, daemon=True)
    thread.start()

    return thread
```

## Creating a Notification Dashboard

Visualize alerts in a dashboard:

```python
from safeguards.notification.dashboard import AlertDashboard

# Create an alert dashboard
dashboard = AlertDashboard()

# Configure dashboard panels
dashboard.add_panel(
    title="Recent Alerts",
    type="alert_list",
    config={
        "columns": ["timestamp", "severity", "agent_id", "message"],
        "max_items": 10,
        "auto_refresh_seconds": 30
    }
)

dashboard.add_panel(
    title="Alert Distribution by Severity",
    type="pie_chart",
    config={
        "data_source": "alerts_by_severity",
        "time_range": "last_24h"
    }
)

dashboard.add_panel(
    title="Alert Timeline",
    type="time_series",
    config={
        "data_source": "alerts_by_time",
        "time_range": "last_24h",
        "group_by": "severity"
    }
)

# Start the dashboard (this could launch a web server)
dashboard_url = dashboard.start(host="0.0.0.0", port=8080)
print(f"Dashboard available at: {dashboard_url}")
```

## Best Practices

### Alert Prioritization

- **Use severity levels appropriately**:
  - `CRITICAL` - Reserved for immediate action items that might impact system stability
  - `HIGH` - Serious issues requiring prompt attention
  - `MEDIUM` - Important issues that should be addressed soon
  - `LOW` - Minor issues that should be reviewed when convenient
  - `INFORMATIONAL` - Status updates, non-issues

- **Filter alerts by environment**:
  - Production environments should have more strict filtering
  - Development environments can be more verbose

### Alert Content

- **Make messages actionable**: Include enough information to take action
- **Include context**: Add relevant metadata (agent ID, resource types, values)
- **Use consistent format**: Standardize alert messages for easier parsing
- **Include links**: Where appropriate, include links to dashboards or documentation

### Alert Management

- **Implement acknowledgement**: Track which alerts have been seen and addressed
- **Group related alerts**: Avoid flooding with repetitive notifications
- **Rotate on-call responsibilities**: Ensure 24/7 coverage for critical alerts
- **Review alert effectiveness**: Periodically audit which alerts are useful

## Advanced Topics

### Building an Alert Response Runbook

```python
from safeguards.notification.runbook import Runbook, RemediationStep

# Create a runbook for handling specific alerts
cpu_limit_runbook = Runbook(
    name="cpu_limit_exceeded",
    description="Steps to handle CPU limit exceeded alerts",
    applies_to=lambda alert: (
        alert.metadata.get("type") == "resource_limit" and
        alert.metadata.get("resource") == "cpu"
    )
)

# Add remediation steps
cpu_limit_runbook.add_step(
    RemediationStep(
        name="verify_usage",
        description="Verify current CPU usage",
        action=lambda alert: execute_command(
            f"check_agent_resource {alert.agent_id} cpu"
        )
    )
)

cpu_limit_runbook.add_step(
    RemediationStep(
        name="reduce_priority",
        description="Reduce agent task priority",
        action=lambda alert: execute_command(
            f"set_agent_priority {alert.agent_id} low"
        )
    )
)

cpu_limit_runbook.add_step(
    RemediationStep(
        name="suspend_if_persistent",
        description="Suspend agent if issue persists for over 5 minutes",
        action=lambda alert: execute_command(
            f"suspend_agent {alert.agent_id} --reason 'CPU limit exceeded'"
        ) if alert.metadata.get("duration_seconds", 0) > 300 else None
    )
)

# Register the runbook
notification_manager.register_runbook(cpu_limit_runbook)
```

### Alert Analytics

Implement alert analytics:

```python
from safeguards.notification.analytics import AlertAnalytics
from datetime import datetime, timedelta

# Create alert analytics
analytics = AlertAnalytics(notification_manager)

# Get alert frequency by type
alert_frequencies = analytics.get_frequency_by_type(
    start_time=datetime.now() - timedelta(days=7),
    end_time=datetime.now()
)

# Get noisiest agents
noisy_agents = analytics.get_top_alert_sources(
    limit=5,
    start_time=datetime.now() - timedelta(days=7),
    end_time=datetime.now()
)

# Get alert patterns by time of day
time_patterns = analytics.get_time_patterns(
    interval_minutes=60,
    start_time=datetime.now() - timedelta(days=7),
    end_time=datetime.now()
)

# Generate alert effectiveness report
effectiveness = analytics.get_effectiveness_report(
    start_time=datetime.now() - timedelta(days=30),
    end_time=datetime.now()
)

print("Alert Frequencies by Type:", alert_frequencies)
print("Noisiest Agents:", noisy_agents)
print("Time Patterns:", time_patterns)
print("Alert Effectiveness:", effectiveness)
```

## Conclusion

A well-configured notification system is essential for maintaining visibility into agent operations and responding quickly to issues. By properly setting up alert handlers, notification channels, and implementing best practices, you can ensure that the right people receive the right information at the right time.

For more information, see:
- [Monitoring Guide](monitoring.md)
- [Safeguards Guide](safeguards.md)
- [API Reference](../api/core.md)
