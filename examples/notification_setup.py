"""Example showing how to configure notification channels in the Agent Safety Framework.

This example demonstrates:
1. Setting up the NotificationManager with various channels
2. Configuring Email notifications
3. Configuring Slack notifications
4. Custom templates and formatting
5. Security best practices
"""

import logging
import os
from datetime import datetime
from decimal import Decimal
from typing import Any

from safeguards.api import APIFactory, APIVersion
from safeguards.core.budget_coordination import BudgetCoordinator
from safeguards.monitoring.violation_reporter import ViolationReporter, ViolationType
from safeguards.notifications.manager import NotificationManager
from safeguards.types import Alert, AlertSeverity, NotificationChannel
from safeguards.types.agent import Agent

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class SimpleAgent(Agent):
    """Simple agent implementation for demonstration."""

    def __init__(self, name: str, model: str = "gpt-4"):
        super().__init__(name)
        self.model = model
        self.cost_per_token = Decimal("0.0001")

    def run(self, **kwargs: Any) -> dict[str, Any]:
        """Run the agent with the given input."""
        # Simple implementation for the example
        input_text = kwargs.get("input", "")
        token_count = len(input_text.split())
        cost = self.cost_per_token * Decimal(token_count)

        return {
            "response": f"Processed: {input_text}",
            "token_count": token_count,
            "cost": cost,
        }


def load_env_var(name: str, default: str | None = None) -> str:
    """Safely load environment variable with optional default."""
    value = os.environ.get(name, default)
    if value is None:
        logger.warning(f"Environment variable {name} not set")
    return value


def setup_basic_notification_manager() -> NotificationManager:
    """Setup basic notification manager with console output only."""
    # Most basic setup - console output only
    notification_manager = NotificationManager()

    logger.info("Basic notification manager set up with console channel only")
    return notification_manager


def setup_slack_notification_manager() -> NotificationManager:
    """Setup notification manager with Slack integration."""
    # Create notification manager with Slack channel
    notification_manager = NotificationManager(
        enabled_channels={
            NotificationChannel.SLACK,
            NotificationChannel.CONSOLE,  # Always keep console for logging
        },
    )

    # Configure Slack
    # In a real application, get these from environment variables or secure configuration
    slack_webhook_url = load_env_var(
        "SLACK_WEBHOOK_URL",
        "https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK",
    )
    slack_channel = load_env_var("SLACK_CHANNEL", "#agent-safety-alerts")

    # Configure the Slack channel
    notification_manager.configure_slack(
        webhook_url=slack_webhook_url,
        channel=slack_channel,
    )

    logger.info("Notification manager configured with Slack channel")
    return notification_manager


def setup_email_notification_manager() -> NotificationManager:
    """Setup notification manager with Email integration."""
    # Create notification manager with Email channel
    notification_manager = NotificationManager(
        enabled_channels={
            NotificationChannel.EMAIL,
            NotificationChannel.CONSOLE,  # Always keep console for logging
        },
    )

    # Configure Email
    # In a real application, get these from environment variables or secure configuration
    smtp_host = load_env_var("SMTP_HOST", "smtp.example.com")
    smtp_port = int(load_env_var("SMTP_PORT", "587"))
    smtp_username = load_env_var("SMTP_USERNAME", "alerts@example.com")
    smtp_password = load_env_var("SMTP_PASSWORD", "your-password")
    email_from = load_env_var("EMAIL_FROM", "alerts@example.com")
    email_to = load_env_var("EMAIL_TO", "admin@example.com,team@example.com").split(",")

    # Configure the Email channel
    notification_manager.configure_email(
        smtp_host=smtp_host,
        smtp_port=smtp_port,
        username=smtp_username,
        password=smtp_password,
        from_addr=email_from,
        to_addrs=email_to,
    )

    logger.info("Notification manager configured with Email channel")
    return notification_manager


def setup_comprehensive_notification_manager() -> NotificationManager:
    """Setup notification manager with all channels and custom template directory."""
    # Create notification manager with all channels
    notification_manager = NotificationManager(
        enabled_channels={
            NotificationChannel.EMAIL,
            NotificationChannel.SLACK,
            NotificationChannel.WEBHOOK,
            NotificationChannel.CONSOLE,
        },
        template_dir="./custom_templates",  # Custom templates directory
        cooldown_period=60,  # Reduce default cooldown for demo purposes
    )

    # Configure Slack (using environment variables or defaults)
    slack_webhook_url = load_env_var(
        "SLACK_WEBHOOK_URL",
        "https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK",
    )
    slack_channel = load_env_var("SLACK_CHANNEL", "#agent-safety-alerts")

    notification_manager.configure_slack(
        webhook_url=slack_webhook_url,
        channel=slack_channel,
    )

    # Configure Email
    smtp_host = load_env_var("SMTP_HOST", "smtp.example.com")
    smtp_port = int(load_env_var("SMTP_PORT", "587"))
    smtp_username = load_env_var("SMTP_USERNAME", "alerts@example.com")
    smtp_password = load_env_var("SMTP_PASSWORD", "your-password")
    email_from = load_env_var("EMAIL_FROM", "alerts@example.com")
    email_to = load_env_var("EMAIL_TO", "admin@example.com,team@example.com").split(",")

    notification_manager.configure_email(
        smtp_host=smtp_host,
        smtp_port=smtp_port,
        username=smtp_username,
        password=smtp_password,
        from_addr=email_from,
        to_addrs=email_to,
    )

    # Configure Webhook
    webhook_url = load_env_var("WEBHOOK_URL", "https://example.com/api/webhook")
    notification_manager.configure_webhook(
        url=webhook_url,
        headers={
            "Authorization": f"Bearer {load_env_var('WEBHOOK_API_KEY', 'your-api-key')}",
            "Content-Type": "application/json",
        },
    )

    logger.info("Comprehensive notification manager set up with all channels")
    return notification_manager


def demonstrate_notifications(notification_manager: NotificationManager):
    """Demonstrate sending different types of alerts."""
    # Create alerts with different severity levels
    info_alert = Alert(
        title="Agent Task Completed",
        description="The agent has completed its assigned task successfully.",
        severity=AlertSeverity.INFO,
        timestamp=datetime.now(),
        metadata={
            "agent_id": "demo-agent-1",
            "task_id": "task-123",
            "duration_seconds": 45,
        },
    )

    warning_alert = Alert(
        title="Agent Budget Below 50%",
        description="The agent's budget is running low.",
        severity=AlertSeverity.WARNING,
        timestamp=datetime.now(),
        metadata={
            "agent_id": "demo-agent-2",
            "current_budget": 45.75,
            "initial_budget": 100.0,
            "percentage": 45.75,
        },
    )

    error_alert = Alert(
        title="Agent Encountered API Error",
        description="The agent encountered an error while calling an external API.",
        severity=AlertSeverity.HIGH,
        timestamp=datetime.now(),
        metadata={
            "agent_id": "demo-agent-3",
            "error_code": "API_TIMEOUT",
            "api_endpoint": "https://api.example.com/data",
            "retry_count": 3,
        },
    )

    critical_alert = Alert(
        title="Agent Budget Depleted",
        description="The agent has completely depleted its budget allocation.",
        severity=AlertSeverity.CRITICAL,
        timestamp=datetime.now(),
        metadata={
            "agent_id": "demo-agent-4",
            "current_budget": 0.0,
            "initial_budget": 100.0,
            "percentage": 0.0,
            "requires_immediate_action": True,
        },
    )

    # Send the alerts
    logger.info("Sending INFO alert")
    notification_manager.send_alert(info_alert)

    logger.info("Sending WARNING alert")
    notification_manager.send_alert(warning_alert)

    logger.info("Sending HIGH alert")
    notification_manager.send_alert(error_alert)

    logger.info("Sending CRITICAL alert")
    notification_manager.send_alert(critical_alert)


def setup_with_budget_coordinator():
    """Example of setting up notifications with the budget coordinator."""
    # Create notification manager
    notification_manager = setup_basic_notification_manager()

    # Create violation reporter
    violation_reporter = ViolationReporter(notification_manager)

    # Create budget coordinator with notification manager
    budget_coordinator = BudgetCoordinator(notification_manager)

    # Create API factory
    api_factory = APIFactory()

    # Create APIs
    budget_api = api_factory.create_budget_api(APIVersion.V1, budget_coordinator)
    agent_api = api_factory.create_agent_api(APIVersion.V1, budget_coordinator)

    # Create a budget pool
    pool = budget_api.create_budget_pool(
        name="example_pool",
        initial_budget=Decimal("1000.0"),
        priority=5,
    )

    # Create an agent
    agent = SimpleAgent("example_agent")
    registered_agent = agent_api.create_agent(
        name=agent.name,
        initial_budget=Decimal("100.0"),
        priority=5,
        pool_id=pool.id,
    )

    # Simulate a budget update that would trigger a notification
    current_budget = budget_api.get_budget(registered_agent.id)
    new_budget = current_budget * Decimal("0.1")  # Reduce to 10%
    budget_api.update_budget(registered_agent.id, new_budget)

    # Simulate a violation report
    violation_reporter.report_violation(
        agent_id=registered_agent.id,
        violation_type=ViolationType.BUDGET_EXCEEDED,
        severity=AlertSeverity.HIGH,
        message="Agent is approaching budget limit",
        details={
            "current_budget": float(new_budget),
            "initial_budget": float(current_budget),
            "percentage": 10.0,
        },
    )

    return budget_coordinator, agent, registered_agent.id


def main():
    """Run the notification examples."""
    logger.info("=== Agent Safety Notification Examples ===")

    # Example 1: Basic Notification Manager
    logger.info("\n1. Basic Notification Manager:")
    basic_manager = setup_basic_notification_manager()
    demonstrate_notifications(basic_manager)

    # Example 2: Slack Notification Manager
    logger.info("\n2. Slack Notification Manager:")
    try:
        slack_manager = setup_slack_notification_manager()
        # Only send one notification for this example
        info_alert = Alert(
            title="Slack Notification Example",
            description="This is a test alert sent to Slack.",
            severity=AlertSeverity.INFO,
            timestamp=datetime.now(),
        )
        slack_manager.send_alert(info_alert)
    except Exception as e:
        logger.error(f"Error setting up Slack notifications: {e!s}")

    # Example 3: Email Notification Manager
    logger.info("\n3. Email Notification Manager:")
    try:
        email_manager = setup_email_notification_manager()
        # Only send one notification for this example
        warning_alert = Alert(
            title="Email Notification Example",
            description="This is a test alert sent via email.",
            severity=AlertSeverity.WARNING,
            timestamp=datetime.now(),
        )
        email_manager.send_alert(warning_alert)
    except Exception as e:
        logger.error(f"Error setting up Email notifications: {e!s}")

    # Example 4: Integration with Budget Coordinator
    logger.info("\n4. Integration with Budget Coordinator:")
    try:
        budget_coordinator, agent, agent_id = setup_with_budget_coordinator()
        logger.info(f"Budget coordinator set up with agent {agent_id}")
    except Exception as e:
        logger.error(f"Error setting up budget coordinator: {e!s}")

    logger.info("\n=== Notification Examples Complete ===")
    logger.info("Set these environment variables to use real notification channels:")
    logger.info("  - SLACK_WEBHOOK_URL, SLACK_CHANNEL")
    logger.info("  - SMTP_HOST, SMTP_PORT, SMTP_USERNAME, SMTP_PASSWORD")
    logger.info("  - EMAIL_FROM, EMAIL_TO")
    logger.info("  - WEBHOOK_URL, WEBHOOK_API_KEY")


if __name__ == "__main__":
    main()
