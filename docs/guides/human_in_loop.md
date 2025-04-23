# Human-in-the-Loop Workflows

The Safeguards library provides robust support for human-in-the-loop (HITL) workflows, allowing agents to request human approval, feedback, or modifications before proceeding with sensitive or high-risk actions.

## Overview

Human-in-the-loop functionality in Safeguards consists of these key components:

1. **Notification Channels**: Different ways to communicate with humans (email, Slack, custom webhooks)
2. **Human Action Handling**: Workflow for requesting and processing human approvals
3. **Action Responses**: Processing approvals, rejections, or modifications from humans

## Setting Up Notification Channels

Before using HITL workflows, you need to set up appropriate notification channels:

```python
from safeguards.core.notification_manager import NotificationManager
from safeguards.notifications.channels import HumanInTheLoopChannel, SlackChannel, EmailChannel

# Create notification manager
notification_manager = NotificationManager()

# Set up a human-in-the-loop channel
hitl_channel = HumanInTheLoopChannel()
hitl_channel.initialize({
    "webhook_url": "https://your-approval-endpoint.com/api",
    "api_key": "your_api_key",
    "timeout_seconds": 300,  # 5 minutes timeout
    "poll_interval": 5  # Check for responses every 5 seconds
})

# Register the channel
notification_manager.register_channel("human_approvals", hitl_channel)

# You can also set up other channels
slack_channel = SlackChannel()
slack_channel.initialize({
    "webhook_url": "https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK",
    "channel": "#agent-alerts",
    "username": "SafeguardsBot"
})
notification_manager.register_channel("slack_alerts", slack_channel)
```

## Creating a Human Action Handler

Once you have the notification channels set up, you can create a Human Action Handler:

```python
from safeguards.human_action import HumanActionHandler

# Create the action handler with the HITL channel
action_handler = HumanActionHandler(hitl_channel)

# Set a default timeout (optional)
action_handler.set_timeout(120)  # 2 minute timeout
```

## Requesting Human Approval

Now you can request human approvals for agent actions:

```python
# Define action details
action = action_handler.request_action(
    title="Approve High-Value Transaction",
    description=(
        f"Agent financial_advisor is requesting approval for a "
        f"$2500.00 transaction:\n"
        f"Large investment purchase\n\n"
        f"This exceeds the agent's transaction limit of $1000.00."
    ),
    agent_id="financial_advisor",
    metadata={
        "action_type": "transaction",
        "amount": "2500.00",
        "description": "Large investment purchase"
    }
)

# Wait for human response (blocking)
status = action_handler.wait_for_action(action)

if status == ActionStatus.APPROVED:
    print("Action was approved!")
    # Execute the action
elif status == ActionStatus.MODIFIED:
    print(f"Action was modified: {action.response_data}")
    # Execute with modifications
elif status == ActionStatus.REJECTED:
    print(f"Action was rejected: {action.comments}")
    # Handle rejection
elif status == ActionStatus.TIMED_OUT:
    print("Action timed out waiting for human response")
    # Handle timeout
```

## Handling Human Responses

You can also use a non-blocking approach with callbacks:

```python
def handle_response(action):
    """Called when the human responds to the action."""
    if action.status == ActionStatus.APPROVED:
        print(f"Action {action.title} was approved!")
        # Execute the action
    elif action.status == ActionStatus.MODIFIED:
        print(f"Action {action.title} was modified: {action.response_data}")
        # Execute with modifications
    elif action.status == ActionStatus.REJECTED:
        print(f"Action {action.title} was rejected: {action.comments}")
        # Handle rejection
    elif action.status == ActionStatus.TIMED_OUT:
        print(f"Action {action.title} timed out")
        # Handle timeout

# Request with callback
action = action_handler.request_action(
    title="Approve Data Access",
    description="Agent is requesting access to sensitive customer data",
    agent_id="data_processor",
    metadata={"data_type": "customer_records"},
    callbacks=[handle_response]  # Will be called when human responds
)

# Continue execution without waiting
print("Request sent, continuing with other work...")
```

## Processing Human Modifications

When humans modify an action, they can include structured changes that your application can interpret:

```python
def apply_modifications(original_context, modifications):
    """Apply human modifications to the original context."""
    modified_context = original_context.copy()

    for key, value in modifications.items():
        if key in modified_context:
            # Handle type conversions if needed
            if key == "amount" and isinstance(original_context[key], Decimal):
                modified_context[key] = Decimal(value)
            else:
                modified_context[key] = value

    return modified_context

# Example usage
if action.status == ActionStatus.MODIFIED:
    # Original request context
    original_context = {
        "action_type": "transaction",
        "amount": Decimal("2500.00"),
        "description": "Large investment purchase"
    }

    # Apply the modifications from human
    modified_context = apply_modifications(original_context, action.response_data)

    # Execute with the modified context
    result = agent.run(**modified_context)
```

## Available Notification Channels

The Safeguards library includes several built-in notification channels:

### Logging Channel

Simple channel that logs alerts to Python's logging system:

```python
from safeguards.notifications.channels import LoggingChannel

logging_channel = LoggingChannel()
logging_channel.initialize({"log_level": "INFO"})
notification_manager.register_channel("logging", logging_channel)
```

### Email Channel

Sends alerts via email:

```python
from safeguards.notifications.channels import EmailChannel

email_channel = EmailChannel()
email_channel.initialize({
    "smtp_server": "smtp.gmail.com",
    "smtp_port": 587,
    "username": "your_email@gmail.com",
    "password": "your_password",
    "sender": "safeguards@yourcompany.com",
    "recipients": ["admin@yourcompany.com", "security@yourcompany.com"],
    "use_tls": True
})
notification_manager.register_channel("email", email_channel)
```

### Slack Channel

Sends alerts to a Slack channel:

```python
from safeguards.notifications.channels import SlackChannel

slack_channel = SlackChannel()
slack_channel.initialize({
    "webhook_url": "https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK",
    "channel": "#agent-alerts",
    "username": "SafeguardsBot",
    "icon_emoji": ":robot_face:"
})
notification_manager.register_channel("slack", slack_channel)
```

### Human-in-the-Loop Channel

Sends alerts that require human approval:

```python
from safeguards.notifications.channels import HumanInTheLoopChannel

hitl_channel = HumanInTheLoopChannel()
hitl_channel.initialize({
    "webhook_url": "https://your-approval-endpoint.com/api",
    "api_key": "your_api_key",
    "timeout_seconds": 300,
    "poll_interval": 5
})
notification_manager.register_channel("human_review", hitl_channel)
```

## Creating a Custom Notification Channel

You can create custom notification channels by implementing the `NotificationChannel` interface:

```python
from safeguards.notifications.channels import NotificationChannel
from safeguards.types import SafetyAlert

class CustomChannel(NotificationChannel):
    """Custom notification channel example."""

    def __init__(self):
        """Initialize the custom channel."""
        self._config = {}
        self._initialized = False

    def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize the channel with configuration."""
        self._config = config
        # Set up your channel
        self._initialized = True

    def send_notification(self, alert: SafetyAlert) -> bool:
        """Send a notification through this channel."""
        if not self._initialized:
            return False

        try:
            # Implement your notification logic here
            print(f"Custom notification: {alert.title} - {alert.description}")
            return True
        except Exception as e:
            print(f"Error sending notification: {e}")
            return False

    def shutdown(self) -> None:
        """Clean up resources when shutting down."""
        # Clean up any resources
        self._initialized = False
```

## Complete Example

Here's a complete example combining industry safeguards with human-in-the-loop workflows:

```python
from decimal import Decimal
from safeguards.core.notification_manager import NotificationManager
from safeguards.notifications.channels import HumanInTheLoopChannel, LoggingChannel
from safeguards.plugins.industry import FinancialServicesSafeguard
from safeguards.human_action import HumanActionHandler, ActionStatus
from safeguards.types.agent import Agent

# Create notification components
notification_manager = NotificationManager()

# Set up logging channel
logging_channel = LoggingChannel()
logging_channel.initialize({"log_level": "INFO"})
notification_manager.register_channel("logging", logging_channel)

# Set up human-in-the-loop channel
hitl_channel = HumanInTheLoopChannel()
hitl_channel.initialize({
    "webhook_url": "http://example.com/api/approvals",
    "api_key": "demo_key",
    "timeout_seconds": 60
})
notification_manager.register_channel("human_review", hitl_channel)

# Create human action handler
action_handler = HumanActionHandler(hitl_channel)

# Create financial agent
class FinancialAgent(Agent):
    def __init__(self, name, transaction_limit):
        super().__init__(name)
        self.transaction_limit = transaction_limit

    def run(self, **kwargs):
        # Agent implementation here
        return {"success": True, **kwargs}

# Create agent and safeguard
agent = FinancialAgent("financial_advisor", Decimal("1000.00"))

financial_safeguard = FinancialServicesSafeguard()
financial_safeguard.initialize({
    "restricted_actions": ["high_risk_investment"],
    "transaction_limits": {agent.id: Decimal("1000.00")}
})

# Start monitoring the agent
financial_safeguard.monitor_agent(agent.id)

# Create a transaction
transaction = {
    "action_type": "transaction",
    "amount": Decimal("2500.00"),
    "description": "Large investment purchase"
}

# Check if it needs approval
alerts = financial_safeguard.validate_agent_action(agent, transaction)
if alerts:
    # Request human approval
    action = action_handler.request_action(
        title="Approve High-Value Transaction",
        description=f"Transaction of ${transaction['amount']} requires approval",
        agent_id=agent.id,
        metadata=transaction
    )

    # Wait for response
    status = action_handler.wait_for_action(action)

    if status == ActionStatus.APPROVED:
        # Execute approved transaction
        result = agent.run(**transaction)
        print(f"Approved transaction executed: {result}")
    elif status == ActionStatus.MODIFIED:
        # Apply modifications
        modified_transaction = transaction.copy()
        for key, value in action.response_data.items():
            if key == "amount":
                modified_transaction[key] = Decimal(value)
            else:
                modified_transaction[key] = value

        # Execute modified transaction
        result = agent.run(**modified_transaction)
        print(f"Modified transaction executed: {result}")
    else:
        print(f"Transaction not executed. Status: {status}, Reason: {action.comments}")
else:
    # No alerts, proceed normally
    result = agent.run(**transaction)
    print(f"Transaction executed without approval: {result}")

# Clean up
financial_safeguard.stop_monitoring_agent(agent.id)
hitl_channel.shutdown()
```

## Integration with External Systems

For real-world HITL workflows, you'll typically need to integrate with external systems:

1. Set up an API endpoint to receive and process approval requests
2. Create a user interface for reviewers to approve, reject, or modify requests
3. Implement a callback mechanism to notify the Safeguards system of human decisions

The `HumanInTheLoopChannel` is designed to communicate with such external systems via webhooks. The implementation of the external system is left to you, but should include:

1. An endpoint to receive approval requests (`/request-approval`)
2. A UI to display pending requests to human reviewers
3. A mechanism to send responses back to the callback URL
4. Authentication to secure the communication
