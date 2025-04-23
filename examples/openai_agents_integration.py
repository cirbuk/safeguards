#!/usr/bin/env python
"""
Basic integration of the Agent Safety Framework with the OpenAI Agents SDK.

NOTE: This is a demonstration example that shows how to integrate the Agent Safety Framework
with an external Agents SDK. The OpenAI Agents SDK may need to be updated to match the
current OpenAI API version. If you encounter issues with empty responses, check the OpenAI
Agents SDK version or use the official OpenAI Python client library instead.

This example demonstrates how to:
1. Wrap an OpenAI agent to track resource usage
2. Set up a budget pool for the agent
3. Monitor and report on budget usage
4. Implement basic notification for budget alerts
"""

import asyncio
import logging
import os
from decimal import Decimal
from typing import Any

# Import OpenAI Agents SDK components
try:
    from agents import Agent, Runner
except ImportError:
    logging.warning("OpenAI Agents SDK not found. Install with 'pip install agents'")
    Agent = object
    Runner = object

# Import Agent Safety Framework components
from safeguards.core.budget_coordination import BudgetCoordinator
from safeguards.core.notification_manager import NotificationManager
from safeguards.monitoring.metrics import MetricsAnalyzer
from safeguards.monitoring.violation_reporter import (
    ViolationContext,
    ViolationReporter,
    ViolationSeverity,
    ViolationType,
)

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
    "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
}


class OpenAIAgentWrapper:
    """
    Wrapper for OpenAI agent to integrate with the Agent Safety Framework.
    """

    def __init__(
        self,
        name: str,
        description: str,
        model: str = DEFAULT_MODEL,
        budget_coordinator: BudgetCoordinator | None = None,
        violation_reporter: ViolationReporter | None = None,
    ):
        self.name = name
        self.description = description
        self.model = model
        self.budget_coordinator = budget_coordinator
        self.violation_reporter = violation_reporter
        self.id = f"agent_{name.lower().replace(' ', '_')}"

        # Create the OpenAI agent
        self.agent = Agent(name=name, model=model)

        # Track token usage
        self.total_input_tokens = 0
        self.total_output_tokens = 0

    async def run(self, query: str) -> dict[str, Any]:
        """
        Run the agent with the provided query and track resource usage.
        """
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
                return {
                    "response": "Unable to process request due to budget limitations.",
                }
        except Exception as e:
            logger.error(f"Error checking budget: {e!s}")

        try:
            # Run the agent
            response = await Runner.run(self.agent, query)

            # Debug logging to understand response structure
            logger.debug(f"Response type: {type(response)}")
            logger.debug(
                f"Response attributes: {dir(response) if hasattr(response, '__dir__') else 'No attributes'}",
            )

            # Extract content safely
            response_text = ""
            if hasattr(response, "message") and response.message:
                if hasattr(response.message, "content"):
                    response_text = response.message.content
            elif hasattr(response, "content"):
                response_text = response.content
            elif hasattr(response, "text"):
                response_text = response.text
            elif hasattr(response, "final_output"):
                response_text = response.final_output

            # Extract token usage
            usage = getattr(response, "usage", None)
            prompt_tokens = 0
            completion_tokens = 0

            if usage:
                prompt_tokens = getattr(usage, "prompt_tokens", 0)
                completion_tokens = getattr(usage, "completion_tokens", 0)

                self.total_input_tokens += prompt_tokens
                self.total_output_tokens += completion_tokens

                # If budget coordinator is available, update budget
                if self.budget_coordinator:
                    cost = self._calculate_cost(prompt_tokens, completion_tokens)
                    current_budget = self.budget_coordinator.get_agent_budget(self.id)
                    new_budget = current_budget - Decimal(str(cost))
                    self.budget_coordinator.update_agent_budget(self.id, new_budget)

                    # Log budget information
                    metrics = self.budget_coordinator.get_agent_metrics(self.id)
                    logger.info(
                        f"Agent {self.name} used ${cost:.6f}, remaining budget: ${metrics.get('remaining_budget', 0):.6f}",
                    )

            # Return the result
            return {
                "response": response_text,
                "tokens": {
                    "input": prompt_tokens,
                    "output": completion_tokens,
                },
                "model": self.model,
            }
        except Exception as e:
            logger.error(f"Error running agent: {e!s}")
            return {
                "response": f"Error: {e!s}",
                "tokens": {"input": 0, "output": 0},
                "model": self.model,
            }

    def _calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """
        Calculate the cost based on token usage.
        """
        if self.model not in TOKEN_COST_PER_1K:
            # Default to gpt-4o pricing if model not found
            model_costs = TOKEN_COST_PER_1K[DEFAULT_MODEL]
        else:
            model_costs = TOKEN_COST_PER_1K[self.model]

        input_cost = (input_tokens / 1000) * model_costs["input"]
        output_cost = (output_tokens / 1000) * model_costs["output"]

        return input_cost + output_cost


def setup_safety_framework():
    """
    Set up the core components of the Agent Safety Framework.
    """
    # Create notification manager
    notification_manager = NotificationManager()

    # Optional: Configure Slack notifications if webhook URL is available
    slack_webhook_url = os.environ.get("SLACK_WEBHOOK_URL")
    if slack_webhook_url:
        # Note: Update the actual implementation of Slack configuration if needed
        # This is a placeholder that assumes a method to configure Slack exists
        try:
            # For newer version that might have this method
            notification_manager.configure_slack(webhook_url=slack_webhook_url)
        except Exception as e:
            logger.warning(f"Could not configure Slack notifications: {e!s}")

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


async def main_async():
    """
    Main async function to set up and run the agent with safety framework.
    """
    # Check if OpenAI API key is set
    if not os.environ.get("OPENAI_API_KEY"):
        logger.warning(
            "OPENAI_API_KEY environment variable not set. This script may not work properly.",
        )

    # Set up safety framework
    framework = setup_safety_framework()
    framework["notification_manager"]
    violation_reporter = framework["violation_reporter"]
    budget_coordinator = framework["budget_coordinator"]

    # Create a budget pool
    pool_id = "research_pool"
    budget_coordinator.create_pool(
        pool_id=pool_id,
        total_budget=Decimal("1.0"),  # $1.00 initial budget
        priority=5,  # Medium priority
    )

    # Create and register the agent
    agent = OpenAIAgentWrapper(
        name="Research Assistant",
        description="An assistant that helps with research",
        model="gpt-4o",
        budget_coordinator=budget_coordinator,
        violation_reporter=violation_reporter,
    )

    # Register agent with budget coordinator - provide the agent object or register manually
    try:
        # Try to use the create_agent method, which is simpler
        registered_agent = budget_coordinator.create_agent(
            name=agent.name,
            initial_budget=Decimal("0.5"),  # $0.50 initial budget
            priority=5,  # Medium priority
        )
        agent.id = registered_agent.id
    except Exception as e:
        logger.warning(f"Could not use create_agent method: {e!s}")
        # Fall back to manual management
        budget_coordinator._agent_budgets[agent.id] = Decimal("0.5")
        budget_coordinator._initial_budgets = getattr(
            budget_coordinator,
            "_initial_budgets",
            {},
        )
        budget_coordinator._initial_budgets[agent.id] = Decimal("0.5")
        budget_coordinator._agent_pools[agent.id] = pool_id

    # Run a sample query
    logger.info("Running agent with a sample query...")
    try:
        result = await agent.run(
            "What are the key components of the Agent Safety Framework?",
        )

        # Display the result
        logger.info(f"Agent response: {result['response']}")
        logger.info(
            f"Token usage: {result['tokens']} (Note: Token counts may be inaccurate with the current SDK version)",
        )

        # Get and display budget metrics
        metrics = budget_coordinator.get_agent_metrics(agent.id)
        logger.info(f"Initial budget: ${metrics.get('initial_budget', 0):.2f}")
        logger.info(f"Used budget: ${metrics.get('used_budget', 0):.2f}")
        logger.info(f"Remaining budget: ${metrics.get('remaining_budget', 0):.2f}")

        # Example of reporting a violation when low on budget
        remaining_budget = metrics.get("remaining_budget", 0)
        initial_budget = metrics.get("initial_budget", 0)
        if (
            initial_budget > 0 and remaining_budget / initial_budget < 0.3
        ):  # Less than 30% remaining
            context = ViolationContext(
                agent_id=agent.id,
                pool_id=pool_id,
                current_balance=Decimal(str(remaining_budget)),
                violation_amount=Decimal("0"),
            )

            violation_reporter.report_violation(
                violation_type=ViolationType.RATE_LIMIT,
                severity=ViolationSeverity.MEDIUM,
                context=context,
                description=f"Agent {agent.name} is running low on budget (less than 30% remaining)",
            )

        # Run another query (to demonstrate continued usage)
        logger.info("\nRunning agent with another query...")
        result = await agent.run(
            "Explain how to implement budget tracking for an AI system",
        )

        # Display the result
        logger.info(f"Agent response: {result['response']}")
        logger.info(
            f"Token usage: {result['tokens']} (Note: Token counts may be inaccurate with the current SDK version)",
        )

        # Get and display updated budget metrics
        metrics = budget_coordinator.get_agent_metrics(agent.id)
        logger.info(f"Remaining budget: ${metrics.get('remaining_budget', 0):.2f}")

    except Exception as e:
        logger.error(f"Error in main execution: {e!s}")


def main():
    """
    Main entry point.
    """
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
