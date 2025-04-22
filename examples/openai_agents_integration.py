#!/usr/bin/env python
"""
Basic integration of the Agent Safety Framework with the OpenAI Agents SDK.

This example demonstrates how to:
1. Wrap an OpenAI agent to track resource usage
2. Set up a budget pool for the agent
3. Monitor and report on budget usage
4. Implement basic notification for budget alerts
"""

import os
import asyncio
import logging
from dataclasses import dataclass
from typing import Dict, Any, Optional, List

# Import OpenAI Agents SDK components
from agents_sdk import Agent, RunSpec, AgentResponse
from agents_sdk.models import CompletionRequest

# Import Agent Safety Framework components
from agent_safety.notifications import NotificationManager, NotificationLevel
from agent_safety.violations import ViolationReporter, ViolationType, Violation
from agent_safety.core import BudgetCoordinator, BudgetPool
from agent_safety.metrics import MetricsCollector
from agent_safety.types import Agent as SafetyAgent

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Constants
DEFAULT_MODEL = "gpt-4o"
TOKEN_COST_PER_1K = {
    "gpt-4o": {"input": 0.01, "output": 0.03},
    "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
}


class OpenAIAgentWrapper(SafetyAgent):
    """
    Wrapper for OpenAI agent to integrate with the Agent Safety Framework.
    """

    def __init__(
        self,
        name: str,
        description: str,
        model: str = DEFAULT_MODEL,
        budget_coordinator: Optional[BudgetCoordinator] = None,
        violation_reporter: Optional[ViolationReporter] = None,
    ):
        self.name = name
        self.description = description
        self.model = model
        self.budget_coordinator = budget_coordinator
        self.violation_reporter = violation_reporter

        # Create the OpenAI agent
        self.agent = Agent(name=name)

        # Track token usage
        self.total_input_tokens = 0
        self.total_output_tokens = 0

    async def run(self, query: str) -> Dict[str, Any]:
        """
        Run the agent with the provided query and track resource usage.
        """
        # Prepare the run specification
        run_spec = RunSpec(
            messages=[{"role": "user", "content": query}], model=self.model
        )

        # Run the agent
        response: AgentResponse = await self.agent.run(run_spec)

        # Extract token usage
        usage = response.usage
        if usage:
            self.total_input_tokens += usage.prompt_tokens
            self.total_output_tokens += usage.completion_tokens

            # If budget coordinator is available, update budget
            if self.budget_coordinator:
                cost = self._calculate_cost(
                    usage.prompt_tokens, usage.completion_tokens
                )
                self.budget_coordinator.update_usage(self.id, cost)

                # Log budget information
                remaining = self.budget_coordinator.get_agent_metrics(
                    self.id
                ).remaining_budget
                logger.info(
                    f"Agent {self.name} used ${cost:.6f}, remaining budget: ${remaining:.6f}"
                )

        # Return the result
        return {
            "response": response.message.content if response.message else "",
            "tokens": {
                "input": usage.prompt_tokens if usage else 0,
                "output": usage.completion_tokens if usage else 0,
            },
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

    # Configure console notifications (always enabled)
    notification_manager.configure_console()

    # Optional: Configure Slack notifications if webhook URL is available
    slack_webhook_url = os.environ.get("SLACK_WEBHOOK_URL")
    if slack_webhook_url:
        notification_manager.configure_slack(webhook_url=slack_webhook_url)

    # Create violation reporter
    violation_reporter = ViolationReporter(notification_manager=notification_manager)

    # Create budget coordinator
    budget_coordinator = BudgetCoordinator(violation_reporter=violation_reporter)

    # Create metrics collector
    metrics_collector = MetricsCollector()

    return {
        "notification_manager": notification_manager,
        "violation_reporter": violation_reporter,
        "budget_coordinator": budget_coordinator,
        "metrics_collector": metrics_collector,
    }


async def main_async():
    """
    Main async function to set up and run the agent with safety framework.
    """
    # Check if OpenAI API key is set
    if not os.environ.get("OPENAI_API_KEY"):
        raise ValueError("Please set the OPENAI_API_KEY environment variable")

    # Set up safety framework
    framework = setup_safety_framework()
    notification_manager = framework["notification_manager"]
    violation_reporter = framework["violation_reporter"]
    budget_coordinator = framework["budget_coordinator"]

    # Create a budget pool
    budget_pool = BudgetPool(
        name="research_pool",
        initial_budget=1.0,  # $1.00 initial budget
        low_budget_threshold=0.3,  # 30% threshold for low budget warning
        description="Budget pool for research agents",
    )
    budget_coordinator.create_budget_pool(budget_pool)

    # Create and register the agent
    agent = OpenAIAgentWrapper(
        name="Research Assistant",
        description="An assistant that helps with research",
        model="gpt-4o",
        budget_coordinator=budget_coordinator,
        violation_reporter=violation_reporter,
    )
    budget_coordinator.register_agent(
        agent_id=agent.id,
        pool_id=budget_pool.id,
        initial_budget=0.5,  # $0.50 initial budget
        priority=5,  # Medium priority
    )

    # Run a sample query
    logger.info("Running agent with a sample query...")
    result = await agent.run(
        "What are the key components of the Agent Safety Framework?"
    )

    # Display the result
    logger.info(f"Agent response: {result['response']}")
    logger.info(f"Token usage: {result['tokens']}")

    # Get and display budget metrics
    metrics = budget_coordinator.get_agent_metrics(agent.id)
    logger.info(f"Initial budget: ${metrics.initial_budget:.2f}")
    logger.info(f"Used budget: ${metrics.used_budget:.2f}")
    logger.info(f"Remaining budget: ${metrics.remaining_budget:.2f}")

    # Example of a budget violation (simulated)
    if (
        metrics.remaining_budget
        < budget_pool.low_budget_threshold * metrics.initial_budget
    ):
        violation = Violation(
            type=ViolationType.BUDGET,
            description=f"Agent {agent.name} is running low on budget",
            agent_id=agent.id,
            severity=NotificationLevel.WARNING,
        )
        violation_reporter.report(violation)

    # Run another query (to demonstrate continued usage)
    logger.info("\nRunning agent with another query...")
    result = await agent.run(
        "Explain how to implement budget tracking for an AI system"
    )

    # Display the result
    logger.info(f"Agent response: {result['response']}")
    logger.info(f"Token usage: {result['tokens']}")

    # Get and display updated budget metrics
    metrics = budget_coordinator.get_agent_metrics(agent.id)
    logger.info(f"Remaining budget: ${metrics.remaining_budget:.2f}")


def main():
    """
    Main entry point.
    """
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
