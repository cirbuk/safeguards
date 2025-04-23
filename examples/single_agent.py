#!/usr/bin/env python
"""Example of using safety features with a single agent."""

import asyncio
import logging
import os
from decimal import Decimal

# Import OpenAI Agents SDK components
from agents import Agent, Runner

from safeguards.core import BudgetCoordinator
from safeguards.monitoring.metrics import MetricsAnalyzer
from safeguards.monitoring.violation_reporter import (
    ViolationContext,
    ViolationSeverity,
)

# Import Safeguards framework components
from safeguards.notifications import NotificationManager
from safeguards.types import NotificationChannel
from safeguards.violations import ViolationReporter, ViolationType

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Constants
DEFAULT_MODEL = "gpt-4o"
TOKEN_COST_PER_1K = {
    "gpt-4o": {"input": 0.01, "output": 0.03},
}


class SimpleAgentWrapper:
    """Simple wrapper for OpenAI agent to track resource usage."""

    def __init__(
        self,
        name: str,
        budget_coordinator: BudgetCoordinator,
        violation_reporter: ViolationReporter,
        model: str = DEFAULT_MODEL,
    ):
        self.name = name
        self.id = f"agent_{name.lower().replace(' ', '_')}"
        self.model = model
        self.budget_coordinator = budget_coordinator
        self.violation_reporter = violation_reporter

        # Create the OpenAI agent
        self.agent = Agent(
            name=name,
            instructions=f"You are {name}, a helpful assistant.",
            model=self.model,
        )

        # Track token usage
        self.total_input_tokens = 0
        self.total_output_tokens = 0

    async def run(self, query: str):
        """Run the agent with the provided query and track resource usage."""
        # Check if we're under budget before running
        try:
            metrics = self.budget_coordinator.get_agent_metrics(self.id)
            if metrics.get("remaining_budget", 0) <= 0:
                logger.warning(f"Agent {self.name} has no remaining budget")
                # Report budget violation
                context = ViolationContext(
                    agent_id=self.id,
                    pool_id=None,
                    current_balance=Decimal("0"),
                    violation_amount=Decimal("0"),
                )

                self.violation_reporter.report_violation(
                    violation_type=ViolationType.OVERSPEND,
                    severity=ViolationSeverity.HIGH,
                    context=context,
                    description=f"Agent {self.name} has no remaining budget",
                )
                return "Unable to process request due to budget limitations."
        except Exception as e:
            logger.error(f"Error checking budget: {e!s}")

        try:
            # Run the agent using Runner
            result = await Runner.run(self.agent, input=query)

            # Extract message content
            response_text = result.final_output

            # Extract token usage if available
            usage = getattr(result, "usage", None)
            if usage:
                prompt_tokens = getattr(usage, "prompt_tokens", 0)
                completion_tokens = getattr(usage, "completion_tokens", 0)

                self.total_input_tokens += prompt_tokens
                self.total_output_tokens += completion_tokens

                # Calculate cost
                cost = self._calculate_cost(prompt_tokens, completion_tokens)

                # Update budget
                current_budget = self.budget_coordinator.get_agent_budget(self.id)
                new_budget = current_budget - Decimal(str(cost))
                self.budget_coordinator.update_agent_budget(self.id, new_budget)

                # Log budget information
                metrics = self.budget_coordinator.get_agent_metrics(self.id)
                logger.info(
                    f"Agent {self.name} used ${cost:.6f}, "
                    f"remaining budget: ${metrics.get('remaining_budget', 0):.6f}",
                )

            return response_text

        except Exception as e:
            logger.error(f"Error running agent: {e!s}")
            return f"Error: {e!s}"

    def _calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Calculate the cost based on token usage."""
        if self.model not in TOKEN_COST_PER_1K:
            # Default to gpt-4o pricing if model not found
            model_costs = TOKEN_COST_PER_1K[DEFAULT_MODEL]
        else:
            model_costs = TOKEN_COST_PER_1K[self.model]

        input_cost = (input_tokens / 1000) * model_costs["input"]
        output_cost = (output_tokens / 1000) * model_costs["output"]

        return input_cost + output_cost


def setup_safety_framework():
    """Set up the core components of the Safeguards framework."""
    # Create notification manager with console notifications enabled
    notification_manager = NotificationManager(
        enabled_channels={NotificationChannel.CONSOLE},
    )

    # Optional: Configure webhook if needed
    # notification_manager.configure_webhook(url="http://localhost:8000/webhook")

    # Optional: Configure Slack if webhook URL is available
    slack_webhook_url = os.environ.get("SLACK_WEBHOOK_URL")
    if slack_webhook_url:
        notification_manager.configure_slack(
            webhook_url=slack_webhook_url,
            channel="#safeguards",
        )

    # Create violation reporter
    violation_reporter = ViolationReporter(notification_manager=notification_manager)

    # Create budget coordinator
    budget_coordinator = BudgetCoordinator(notification_manager=notification_manager)

    # Create metrics analyzer
    metrics_analyzer = MetricsAnalyzer()

    return {
        "notification_manager": notification_manager,
        "violation_reporter": violation_reporter,
        "budget_coordinator": budget_coordinator,
        "metrics_analyzer": metrics_analyzer,
    }


async def main():
    """Run single agent example."""
    # Check if OpenAI API key is set
    if not os.environ.get("OPENAI_API_KEY"):
        msg = "Please set the OPENAI_API_KEY environment variable"
        raise ValueError(msg)

    # Set up safety framework
    framework = setup_safety_framework()
    framework["notification_manager"]
    violation_reporter = framework["violation_reporter"]
    budget_coordinator = framework["budget_coordinator"]

    # Create a budget pool
    budget_coordinator.create_pool(
        pool_id="analyst_pool",
        total_budget=Decimal("1.0"),  # $1.00 initial budget
        priority=5,  # Medium priority
    )

    # Create and register the agent wrapper
    analyst_wrapper = SimpleAgentWrapper(
        name="Analyst",
        budget_coordinator=budget_coordinator,
        violation_reporter=violation_reporter,
    )

    # Register the agent with the budget coordinator
    agent = budget_coordinator.register_agent(
        name="Analyst",
        initial_budget=Decimal("0.5"),  # $0.50 initial budget
        priority=5,  # Medium priority
    )

    # Store the agent ID for the wrapper to use
    analyst_wrapper.id = agent.id

    # Run agent with a query
    logger.info("Running agent with a query...")
    response = await analyst_wrapper.run("Analyze the performance data for Q1 2024")
    logger.info(f"Response: {response}")

    # Get and display metrics
    metrics = budget_coordinator.get_agent_metrics(agent.id)
    logger.info(f"Budget usage: ${metrics.get('used_budget', 0):.4f}")
    logger.info(f"Remaining budget: ${metrics.get('remaining_budget', 0):.4f}")
    usage_percentage = (
        (metrics.get("used_budget", 0) / metrics.get("initial_budget", 1)) * 100
        if metrics.get("initial_budget", 0) > 0
        else 0
    )
    logger.info(f"Usage percentage: {usage_percentage:.1f}%")


if __name__ == "__main__":
    asyncio.run(main())
