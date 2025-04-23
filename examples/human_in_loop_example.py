#!/usr/bin/env python3
"""Example demonstrating human-in-the-loop workflows with industry-specific safeguards."""

import logging
import time
from decimal import Decimal
from typing import Any

from safeguards.core.budget_coordination import BudgetCoordinator
from safeguards.core.notification_manager import NotificationManager
from safeguards.human_action import ActionStatus, HumanActionHandler
from safeguards.notifications.channels import HumanInTheLoopChannel, LoggingChannel
from safeguards.plugins.industry import FinancialServicesSafeguard
from safeguards.types.agent import Agent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("human_in_loop_example")


class FinancialAgent(Agent):
    """Example financial services agent with transaction capabilities."""

    def __init__(self, name: str, transaction_limit: Decimal = Decimal("1000.00")):
        """Initialize the financial agent.

        Args:
            name: Name of the agent
            transaction_limit: Maximum transaction amount
        """
        super().__init__(name)
        self.transaction_limit = transaction_limit

    def run(self, **kwargs: Any) -> dict[str, Any]:
        """Execute a financial transaction.

        Args:
            **kwargs: Must include:
                - action_type: Type of financial action
                - amount: Transaction amount (if applicable)
                - description: Transaction description

        Returns:
            Result of the transaction
        """
        action_type = kwargs.get("action_type", "")
        amount = kwargs.get("amount", Decimal("0.00"))
        description = kwargs.get("description", "")

        logger.info(
            f"Agent {self.name} executing {action_type} of {amount} - {description}",
        )

        if action_type == "transaction" and amount > self.transaction_limit:
            logger.warning(
                f"Transaction amount {amount} exceeds agent limit {self.transaction_limit}",
            )

        return {
            "success": True,
            "action_type": action_type,
            "amount": amount,
            "description": description,
            "timestamp": time.time(),
        }


def handle_human_approval(action, agent, action_context):
    """Handle the result of human approval.

    Args:
        action: The human action with response
        agent: The agent that requested approval
        action_context: The original action context
    """
    if action.status == ActionStatus.APPROVED:
        logger.info(f"Human approved action: {action.title}")
        # Execute the action now that it's approved
        result = agent.run(**action_context)
        logger.info(f"Action executed with result: {result}")

    elif action.status == ActionStatus.MODIFIED:
        logger.info(f"Human modified action: {action.title}")
        logger.info(f"Modifications: {action.response_data}")

        # Apply modifications
        modified_context = action_context.copy()
        for key, value in action.response_data.items():
            if key in modified_context:
                # Convert amount back to Decimal if needed
                if key == "amount" and isinstance(modified_context[key], Decimal):
                    modified_context[key] = Decimal(value)
                else:
                    modified_context[key] = value

        # Execute with modifications
        result = agent.run(**modified_context)
        logger.info(f"Modified action executed with result: {result}")

    elif action.status == ActionStatus.REJECTED:
        logger.info(f"Human rejected action: {action.title}")
        logger.info(f"Rejection reason: {action.comments}")

    elif action.status == ActionStatus.TIMED_OUT:
        logger.warning(f"Action timed out waiting for human response: {action.title}")


def main():
    """Run the human-in-the-loop example."""
    # Create notification components
    notification_manager = NotificationManager()

    # Add logging channel for debugging
    logging_channel = LoggingChannel()
    logging_channel.initialize({"log_level": "INFO"})
    notification_manager.register_channel("logging", logging_channel)

    # Create human-in-the-loop channel
    # In a real application, you would configure with actual endpoints
    hitl_channel = HumanInTheLoopChannel()
    hitl_channel.initialize(
        {
            "webhook_url": "http://example.com/api/approvals",
            "api_key": "demo_key",
            "timeout_seconds": 60,  # Short timeout for the example
            "poll_interval": 1,
        },
    )
    notification_manager.register_channel("human_review", hitl_channel)

    # Create budget coordinator
    budget_coordinator = BudgetCoordinator(notification_manager)

    # Create human action handler
    action_handler = HumanActionHandler(hitl_channel)
    action_handler.set_timeout(60)  # 60 second timeout

    # Register a financial agent
    financial_agent = FinancialAgent("financial_advisor", Decimal("1000.00"))
    budget_coordinator.register_agent(
        name=financial_agent.name,
        initial_budget=Decimal("5000.00"),
        priority=5,
        agent=financial_agent,
    )

    # Create and configure financial services safeguard
    financial_safeguard = FinancialServicesSafeguard()
    financial_safeguard.initialize(
        {
            "restricted_actions": ["high_risk_investment", "unauthorized_withdrawal"],
            "compliance_rules": {"kyc_required": True, "aml_check": True},
            "transaction_limits": {financial_agent.id: Decimal("1000.00")},
        },
    )

    # Start monitoring the agent
    financial_safeguard.monitor_agent(financial_agent.id)

    # Simulate a standard transaction (under limit)
    logger.info("--- Standard Transaction Example ---")
    transaction_context = {
        "action_type": "transaction",
        "amount": Decimal("500.00"),
        "description": "Regular stock purchase",
    }

    # Validate before executing
    alerts = financial_safeguard.validate_agent_action(
        financial_agent,
        transaction_context,
    )
    if not alerts:
        # No alerts, proceed normally
        result = financial_agent.run(**transaction_context)
        logger.info(f"Transaction completed: {result}")
    else:
        logger.warning(f"Transaction blocked due to alerts: {alerts}")

    # Simulate a transaction exceeding limits
    logger.info("\n--- High-Value Transaction Example ---")
    high_value_context = {
        "action_type": "transaction",
        "amount": Decimal("2500.00"),
        "description": "Large investment purchase",
    }

    # Validate the high-value transaction
    alerts = financial_safeguard.validate_agent_action(
        financial_agent,
        high_value_context,
    )
    if alerts:
        logger.warning(f"High-value transaction requires approval: {alerts}")

        # Request human approval
        action = action_handler.request_action(
            title="Approve High-Value Transaction",
            description=(
                f"Agent {financial_agent.name} is requesting approval for a "
                f"${high_value_context['amount']} {high_value_context['action_type']}:\n"
                f"{high_value_context['description']}\n\n"
                f"This exceeds the agent's transaction limit of ${financial_agent.transaction_limit}."
            ),
            agent_id=financial_agent.id,
            metadata=high_value_context,
        )

        # For demo purposes, we'll simulate an approval
        # In a real application, this would come from a human through the webhook
        logger.info("Simulating human response...")

        # Uncomment one of these to simulate different responses:

        # 1. Approve as-is:
        hitl_channel.process_response(
            action.request_id,
            True,
            "Approved due to client priority",
        )

        # 2. Modify and approve:
        # hitl_channel.process_response(
        #    action.request_id,
        #    True,
        #    "amount: 1800.00\ndescription: Modified investment purchase - reduced amount"
        # )

        # 3. Reject:
        # hitl_channel.process_response(
        #     action.request_id,
        #     False,
        #     "Amount too high for current market conditions"
        # )

        # Wait for the response to be processed
        action_handler.wait_for_action(action)

        # Handle the result
        handle_human_approval(action, financial_agent, high_value_context)
    else:
        # No safeguard alerts (unlikely in this case)
        result = financial_agent.run(**high_value_context)
        logger.info(f"High-value transaction completed without approval: {result}")

    # Simulate a restricted action
    logger.info("\n--- Restricted Action Example ---")
    restricted_context = {
        "action_type": "high_risk_investment",
        "amount": Decimal("1200.00"),
        "description": "Speculative cryptocurrency investment",
    }

    # Validate the restricted action
    alerts = financial_safeguard.validate_agent_action(
        financial_agent,
        restricted_context,
    )
    if alerts:
        logger.warning(f"Restricted action requires approval: {alerts}")

        # Request human approval
        action = action_handler.request_action(
            title="Approve Restricted Investment Action",
            description=(
                f"Agent {financial_agent.name} is requesting approval for a "
                f"restricted action: {restricted_context['action_type']}\n"
                f"Amount: ${restricted_context['amount']}\n"
                f"Description: {restricted_context['description']}\n\n"
                f"This action is flagged as restricted and requires explicit approval."
            ),
            agent_id=financial_agent.id,
            metadata=restricted_context,
        )

        # Simulate a rejection for this case
        logger.info("Simulating human rejection...")
        hitl_channel.process_response(
            action.request_id,
            False,
            "This type of high-risk investment is not allowed per company policy",
        )

        # Wait for the response to be processed
        action_handler.wait_for_action(action)

        # Handle the result
        handle_human_approval(action, financial_agent, restricted_context)
    else:
        # No safeguard alerts (unlikely in this case)
        result = financial_agent.run(**restricted_context)
        logger.info(f"Restricted action completed without approval: {result}")

    # Clean up
    financial_safeguard.stop_monitoring_agent(financial_agent.id)
    hitl_channel.shutdown()
    logging_channel.shutdown()

    logger.info("Example completed.")


if __name__ == "__main__":
    main()
