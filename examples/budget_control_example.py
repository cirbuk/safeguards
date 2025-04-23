#!/usr/bin/env python
"""Example of comprehensive budget control features in Agent Safety Framework."""

import asyncio
import logging
import os
from decimal import Decimal

# Import OpenAI Agents SDK components
from agents import Agent, Runner, function_tool

from safeguards.core import BudgetCoordinator
from safeguards.monitoring.metrics import MetricsAnalyzer
from safeguards.monitoring.violation_reporter import (
    ViolationContext,
    ViolationSeverity,
)

# Import Agent Safety Framework components
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
    "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
}


# Example tool for cost estimation
@function_tool
def estimate_cost(query_length: int, expected_response_length: int, model: str) -> dict:
    """Estimate the cost of an API call based on input and expected output length.

    Args:
        query_length: Length of the user query in characters
        expected_response_length: Expected length of response in characters
        model: Model to use for the calculation

    Returns:
        Dictionary with estimated tokens and cost
    """
    # Rough estimation: ~4 chars per token
    est_input_tokens = max(1, query_length // 4)
    est_output_tokens = max(1, expected_response_length // 4)

    model_costs = TOKEN_COST_PER_1K.get(model, TOKEN_COST_PER_1K[DEFAULT_MODEL])

    input_cost = (est_input_tokens / 1000) * model_costs["input"]
    output_cost = (est_output_tokens / 1000) * model_costs["output"]
    total_cost = input_cost + output_cost

    return {
        "model": model,
        "estimated_input_tokens": est_input_tokens,
        "estimated_output_tokens": est_output_tokens,
        "estimated_cost": total_cost,
    }


class BudgetAwareAgent:
    """Agent wrapper with budget awareness and monitoring."""

    def __init__(
        self,
        name: str,
        budget_coordinator: BudgetCoordinator,
        violation_reporter: ViolationReporter,
        model: str = DEFAULT_MODEL,
        initial_budget: Decimal = Decimal("1.0"),
        tools: list | None = None,
        pool_id: str | None = None,
    ):
        self.name = name
        self.model = model
        self.budget_coordinator = budget_coordinator
        self.violation_reporter = violation_reporter
        self.pool_id = pool_id

        # Create the OpenAI agent
        self.agent = Agent(
            name=name,
            instructions=f"You are {name}, a helpful assistant that is budget-conscious.",
            model=self.model,
            tools=tools or [estimate_cost],
        )

        # Register with budget coordinator
        agent_reg = budget_coordinator.register_agent(
            name=name,
            initial_budget=initial_budget,
            priority=5,  # Medium priority
        )
        self.id = agent_reg.id

        # If pool_id is provided, store it but don't try to add to pool
        if pool_id:
            self.pool_id = pool_id
            logger.info(
                f"Agent {self.id} associated with pool {pool_id} (not added to pool)",
            )
        else:
            self.pool_id = None

        # Track token usage
        self.total_input_tokens = 0
        self.total_output_tokens = 0

    async def run(self, query: str, cost_estimate: float | None = None):
        """Run the agent with budget checks and monitoring.

        Args:
            query: User query to process
            cost_estimate: Optional pre-estimated cost for this query

        Returns:
            Response text from the agent or an error message
        """
        # Pre-check budget before running
        try:
            metrics = self.budget_coordinator.get_agent_metrics(self.id)
            remaining_budget = metrics.get("remaining_budget", 0)

            if remaining_budget <= 0:
                logger.warning(f"Agent {self.name} has no remaining budget")
                self._report_budget_violation(
                    "No remaining budget",
                    current_balance=Decimal("0"),
                    violation_amount=Decimal("0"),
                )
                return "Unable to process request due to budget limitations."

            # If cost estimate provided, check if we can afford it
            if cost_estimate and Decimal(str(cost_estimate)) > remaining_budget:
                logger.warning(
                    f"Agent {self.name} has insufficient budget for estimated cost "
                    f"(${cost_estimate:.4f} > ${remaining_budget:.4f})",
                )
                self._report_budget_violation(
                    "Insufficient budget for estimated cost",
                    current_balance=remaining_budget,
                    violation_amount=Decimal(str(cost_estimate)),
                )
                return f"Unable to process request: estimated cost (${cost_estimate:.4f}) exceeds remaining budget (${remaining_budget:.4f})."

        except Exception as e:
            logger.error(f"Error checking budget: {e!s}")

        try:
            # Run the agent
            result = await Runner.run(self.agent, input=query)
            response_text = result.final_output

            # Process usage
            usage = getattr(result, "usage", None)
            if usage:
                prompt_tokens = getattr(usage, "prompt_tokens", 0)
                completion_tokens = getattr(usage, "completion_tokens", 0)

                self.total_input_tokens += prompt_tokens
                self.total_output_tokens += completion_tokens

                # Calculate and update budget
                cost = self._calculate_cost(prompt_tokens, completion_tokens)
                self._update_budget(cost)

            return response_text

        except Exception as e:
            logger.error(f"Error running agent: {e!s}")
            return f"Error: {e!s}"

    def _calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Calculate cost based on token usage."""
        model_costs = TOKEN_COST_PER_1K.get(
            self.model,
            TOKEN_COST_PER_1K[DEFAULT_MODEL],
        )

        input_cost = (input_tokens / 1000) * model_costs["input"]
        output_cost = (output_tokens / 1000) * model_costs["output"]

        return input_cost + output_cost

    def _update_budget(self, cost: float) -> None:
        """Update agent budget after usage."""
        try:
            # Get current budget
            current_budget = self.budget_coordinator.get_agent_budget(self.id)

            # Calculate new budget
            new_budget = current_budget - Decimal(str(cost))

            # Update budget
            self.budget_coordinator.update_agent_budget(self.id, new_budget)

            # Log budget information
            metrics = self.budget_coordinator.get_agent_metrics(self.id)
            logger.info(
                f"Agent {self.name} used ${cost:.6f}, "
                f"remaining budget: ${metrics.get('remaining_budget', 0):.6f}",
            )

            # Check for low budget warning
            if metrics.get("remaining_budget", 0) < metrics.get(
                "initial_budget",
                1,
            ) * Decimal("0.2"):
                logger.warning(f"Agent {self.name} budget is running low")

        except Exception as e:
            logger.error(f"Error updating budget: {e!s}")

    def _report_budget_violation(
        self,
        message: str,
        current_balance: Decimal,
        violation_amount: Decimal,
    ) -> None:
        """Report a budget violation."""
        context = ViolationContext(
            agent_id=self.id,
            pool_id=self.pool_id,
            current_balance=current_balance,
            violation_amount=violation_amount,
        )

        self.violation_reporter.report_violation(
            violation_type=ViolationType.OVERSPEND,
            severity=ViolationSeverity.HIGH,
            context=context,
            description=f"Agent {self.name}: {message}",
        )


def setup_framework():
    """Set up the Agent Safety Framework components."""
    # Create notification manager
    notification_manager = NotificationManager(
        enabled_channels={NotificationChannel.CONSOLE},
    )

    # Configure Slack if available
    slack_webhook_url = os.environ.get("SLACK_WEBHOOK_URL")
    if slack_webhook_url:
        notification_manager.configure_slack(
            webhook_url=slack_webhook_url,
            channel="#agent-safety",
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
    """Run the budget control example."""
    # Check for OpenAI API key
    if not os.environ.get("OPENAI_API_KEY"):
        msg = "Please set the OPENAI_API_KEY environment variable"
        raise ValueError(msg)

    # Set up framework
    framework = setup_framework()
    notification_manager = framework["notification_manager"]
    violation_reporter = framework["violation_reporter"]
    budget_coordinator = framework["budget_coordinator"]

    # Create budget pools with proper error handling
    try:
        research_pool = budget_coordinator.create_pool(
            pool_id="research_pool",
            total_budget=Decimal("2.0"),  # $2.00 initial budget
            priority=8,  # High priority
        )
        logger.info(f"Created research pool: {research_pool}")
    except Exception as e:
        logger.warning(f"Could not create research pool: {e!s}")

    try:
        support_pool = budget_coordinator.create_pool(
            pool_id="support_pool",
            total_budget=Decimal("1.0"),  # $1.00 initial budget
            priority=5,  # Medium priority
        )
        logger.info(f"Created support pool: {support_pool}")
    except Exception as e:
        logger.warning(f"Could not create support pool: {e!s}")

    # Create agents
    researcher = BudgetAwareAgent(
        name="Researcher",
        budget_coordinator=budget_coordinator,
        violation_reporter=violation_reporter,
        model="gpt-4o",
        initial_budget=Decimal("1.0"),
        pool_id="research_pool",
    )

    assistant = BudgetAwareAgent(
        name="Assistant",
        budget_coordinator=budget_coordinator,
        violation_reporter=violation_reporter,
        model="gpt-3.5-turbo",
        initial_budget=Decimal("0.5"),
        pool_id="support_pool",
    )

    # Run researcher with a query
    logger.info("Running researcher agent...")

    # First, get a cost estimate
    research_query = "Conduct a detailed analysis of recent advances in quantum computing and their potential applications in cryptography."
    estimate_result = await Runner.run(
        researcher.agent,
        input=f"Estimate the cost to answer this query: '{research_query}'. "
        f"Assume a detailed response of around 1000 words. "
        f"Use the model '{DEFAULT_MODEL}' for this calculation.",
    )

    # Extract estimated cost from response
    # For a real implementation, you'd parse this more carefully
    est_cost = 0.05  # Example estimated cost

    # Run the actual query with the cost estimate
    research_response = await researcher.run(research_query, cost_estimate=est_cost)
    logger.info(
        f"Researcher response: {research_response[:100]}...",
    )  # Show first 100 chars

    # Run assistant with follow-up
    logger.info("Running assistant agent...")
    assist_response = await assistant.run(
        f"Summarize this research in simple terms: {research_response[:500]}",
    )
    logger.info(
        f"Assistant response: {assist_response[:100]}...",
    )  # Show first 100 chars

    # Display pool metrics
    logger.info(
        f"Research pool ID: {research_pool.pool_id if 'research_pool' in locals() else 'not created'}",
    )
    logger.info(
        f"Support pool ID: {support_pool.pool_id if 'support_pool' in locals() else 'not created'}",
    )

    # Display agent metrics
    for agent in [researcher, assistant]:
        metrics = budget_coordinator.get_agent_metrics(agent.id)
        logger.info(f"{agent.name} usage: ${metrics.get('used_budget', 0):.4f}")
        logger.info(
            f"{agent.name} remaining: ${metrics.get('remaining_budget', 0):.4f}",
        )
        usage_percentage = (
            (metrics.get("used_budget", 0) / metrics.get("initial_budget", 1)) * 100
            if metrics.get("initial_budget", 0) > 0
            else 0
        )
        logger.info(f"{agent.name} usage percentage: {usage_percentage:.1f}%")


if __name__ == "__main__":
    asyncio.run(main())
