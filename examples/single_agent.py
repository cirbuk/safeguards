"""Example of using safety features with a single agent."""

from decimal import Decimal
import asyncio

from agents import Agent, Runner, function_tool
from agent_safety import SafetyController, SafetyConfig


# Example tool that requires budget monitoring
@function_tool
def analyze_data(data: str) -> str:
    """Analyze a large dataset."""
    # Simulate expensive operation
    return f"Analysis results for {data}"


async def main():
    """Run single agent example."""

    # Configure safety controls
    config = SafetyConfig(
        total_budget=Decimal("1000"),
        hourly_limit=Decimal("100"),
        daily_limit=Decimal("500"),
        cpu_threshold=80.0,
        memory_threshold=80.0,
        budget_warning_threshold=75.0,
        require_human_approval=True,
    )

    # Create safety controller
    controller = SafetyController(config)

    # Create agent with safety controls
    agent = Agent(
        name="Analyst",
        instructions="You analyze data carefully and efficiently.",
        tools=[analyze_data],
        guardrails=[controller.budget_guardrail, controller.resource_guardrail],
    )

    # Register agent with controller
    controller.register_agent(agent, budget=Decimal("500"))

    # Run agent with safety monitoring
    result = await Runner.run(agent, "Analyze the performance data for Q1 2024")

    # Check metrics
    metrics = controller.get_metrics(agent.name)
    print(f"Budget usage: {metrics.budget.usage_percent:.1f}%")
    print(f"CPU usage: {metrics.resources.cpu_percent:.1f}%")


if __name__ == "__main__":
    asyncio.run(main())
