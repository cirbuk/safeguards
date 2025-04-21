"""Example of using safety features with multiple coordinating agents."""

from decimal import Decimal
import asyncio

from agents import Agent, Runner, function_tool
from agent_safety import SwarmController, SwarmConfig


# Example tool that requires budget monitoring
@function_tool
def analyze_data(data: str) -> str:
    """Analyze a large dataset."""
    # Simulate expensive operation
    return f"Analysis results for {data}"


async def main():
    """Run multi-agent example."""

    # Configure swarm safety
    config = SwarmConfig(
        total_budget=Decimal("2000"),
        max_concurrent_agents=3,
        coordination_strategy="COOPERATIVE",
    )

    # Create swarm controller
    swarm = SwarmController(config)

    # Create agents
    researcher = Agent(
        name="Researcher",
        instructions="Research and gather data.",
        tools=[analyze_data],
        guardrails=[swarm.get_agent_guardrails("Researcher")],
    )

    analyst = Agent(
        name="Analyst",
        instructions="Analyze research findings.",
        tools=[analyze_data],
        guardrails=[swarm.get_agent_guardrails("Analyst")],
    )

    writer = Agent(
        name="Writer",
        instructions="Write reports based on analysis.",
        guardrails=[swarm.get_agent_guardrails("Writer")],
    )

    # Register agents with swarm
    swarm.register_agent(researcher, budget=Decimal("800"), priority="HIGH")
    swarm.register_agent(analyst, budget=Decimal("700"), priority="HIGH")
    swarm.register_agent(writer, budget=Decimal("500"), priority="MEDIUM")

    # Run coordinated analysis
    research_result = await swarm.run_agent(researcher, "Research AI safety mechanisms")

    analysis_result = await swarm.run_agent(
        analyst, f"Analyze these findings: {research_result.final_output}"
    )

    report_result = await swarm.run_agent(
        writer, f"Write a report about: {analysis_result.final_output}"
    )

    # Check swarm metrics
    metrics = swarm.get_metrics()
    print(f"Total budget usage: {metrics.total_usage_percent:.1f}%")
    print(f"Active agents: {metrics.active_agents}")


if __name__ == "__main__":
    asyncio.run(main())
