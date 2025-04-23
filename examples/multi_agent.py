#!/usr/bin/env python
"""Example of using safety features with multiple coordinating agents."""

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


# Example tool that requires budget monitoring
@function_tool
def analyze_data(data: str) -> str:
    """Analyze a large dataset."""
    # Simulate expensive operation
    return f"Analysis results for {data}"


class CoordinatedAgent:
    """Agent wrapper with coordination capabilities."""

    def __init__(
        self,
        name: str,
        budget_coordinator: BudgetCoordinator,
        violation_reporter: ViolationReporter,
        model: str = DEFAULT_MODEL,
        tools: list | None = None,
        instructions: str | None = None,
    ):
        self.name = name
        self.id = f"agent_{name.lower().replace(' ', '_')}"
        self.model = model
        self.budget_coordinator = budget_coordinator
        self.violation_reporter = violation_reporter

        # Create the OpenAI agent
        self.agent = Agent(
            name=name,
            instructions=instructions or f"You are {name}, a helpful assistant.",
            model=self.model,
            tools=tools or [],
        )

        # Track usage
        self.total_input_tokens = 0
        self.total_output_tokens = 0

    async def run(self, query: str):
        """Run the agent with the provided query."""
        # Check if we're under budget
        try:
            metrics = self.budget_coordinator.get_agent_metrics(self.id)
            if metrics.get("remaining_budget", 0) <= 0:
                logger.warning(f"Agent {self.name} has no remaining budget")
                self._report_budget_violation("No remaining budget")
                return "Unable to process request due to budget limitations."
        except Exception as e:
            logger.error(f"Error checking budget: {e!s}")

        try:
            # Run the agent
            result = await Runner.run(self.agent, input=query)

            # Update usage tracking (if available)
            usage = getattr(result, "usage", None)
            if usage:
                prompt_tokens = getattr(usage, "prompt_tokens", 0)
                completion_tokens = getattr(usage, "completion_tokens", 0)

                self.total_input_tokens += prompt_tokens
                self.total_output_tokens += completion_tokens

                # Update budget (simplified for example)
                current_budget = self.budget_coordinator.get_agent_budget(self.id)
                # Assume $0.01 per 1K tokens as a simplified cost model
                cost = (prompt_tokens + completion_tokens) / 1000 * 0.01
                new_budget = current_budget - Decimal(str(cost))
                self.budget_coordinator.update_agent_budget(self.id, new_budget)

                logger.info(
                    f"Agent {self.name} used ${cost:.4f}, remaining: ${new_budget:.4f}",
                )

            return result

        except Exception as e:
            logger.error(f"Error running agent: {e!s}")
            return f"Error: {e!s}"

    def _report_budget_violation(self, message: str):
        """Report a budget violation."""
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
    """Run multi-agent example."""
    # Check for OpenAI API key
    if not os.environ.get("OPENAI_API_KEY"):
        msg = "Please set the OPENAI_API_KEY environment variable"
        raise ValueError(msg)

    # Set up framework
    framework = setup_framework()
    framework["notification_manager"]
    violation_reporter = framework["violation_reporter"]
    budget_coordinator = framework["budget_coordinator"]

    # We'll skip pools for now since they're not working as expected

    # Create and register the agents
    researcher = CoordinatedAgent(
        name="Researcher",
        budget_coordinator=budget_coordinator,
        violation_reporter=violation_reporter,
        model=DEFAULT_MODEL,
        tools=[analyze_data],
        instructions="Research and gather data.",
    )

    analyst = CoordinatedAgent(
        name="Analyst",
        budget_coordinator=budget_coordinator,
        violation_reporter=violation_reporter,
        model=DEFAULT_MODEL,
        tools=[analyze_data],
        instructions="Analyze research findings.",
    )

    writer = CoordinatedAgent(
        name="Writer",
        budget_coordinator=budget_coordinator,
        violation_reporter=violation_reporter,
        model=DEFAULT_MODEL,
        instructions="Write reports based on analysis.",
    )

    # Register with budget coordinator
    researcher_reg = budget_coordinator.register_agent(
        name="Researcher",
        initial_budget=Decimal("8.0"),  # $8.00 initial budget
        priority=8,  # High priority
    )
    researcher.id = researcher_reg.id

    analyst_reg = budget_coordinator.register_agent(
        name="Analyst",
        initial_budget=Decimal("7.0"),  # $7.00 initial budget
        priority=8,  # High priority
    )
    analyst.id = analyst_reg.id

    writer_reg = budget_coordinator.register_agent(
        name="Writer",
        initial_budget=Decimal("5.0"),  # $5.00 initial budget
        priority=6,  # Medium-high priority
    )
    writer.id = writer_reg.id

    # Log the registered agent IDs
    logger.info(f"Registered agent IDs: {researcher.id}, {analyst.id}, {writer.id}")

    # Run coordinated analysis
    logger.info("Running Researcher agent...")
    research_result = await researcher.run("Research AI safety mechanisms")

    logger.info("Running Analyst agent...")
    analysis_result = await analyst.run(
        f"Analyze these findings: {research_result.final_output}",
    )

    logger.info("Running Writer agent...")
    await writer.run(
        f"Write a report about: {analysis_result.final_output}",
    )

    # Display final metrics
    logger.info("\nFinal metrics:")

    # Agent metrics
    for agent in [researcher, analyst, writer]:
        metrics = budget_coordinator.get_agent_metrics(agent.id)
        used = metrics.get("used_budget", Decimal("0"))
        remaining = metrics.get("remaining_budget", Decimal("0"))
        logger.info(f"{agent.name}: Used ${used:.2f}, Remaining ${remaining:.2f}")


if __name__ == "__main__":
    asyncio.run(main())
