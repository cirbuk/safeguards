"""Example script demonstrating real-time budget monitoring."""

import asyncio
import random
from decimal import Decimal

from safeguards.base.budget import BudgetPeriod
from safeguards.budget.cost_tracker import CostTracker
from safeguards.budget.dashboard_server import DashboardServer
from safeguards.budget.realtime_tracker import RealtimeBudgetTracker
from safeguards.budget.token_tracker import TokenTracker


async def simulate_agent_activity(
    realtime_tracker: RealtimeBudgetTracker,
    agent_id: str,
    budget: Decimal,
    duration: int = 60,
):
    """Simulate agent activity with random budget usage.

    Args:
        realtime_tracker: Real-time budget tracker instance
        agent_id: Agent ID to simulate
        budget: Total budget amount
        duration: Duration to run simulation in seconds
    """
    for _ in range(duration):
        # Simulate random usage between 1-5% of budget
        usage = budget * Decimal(str(random.uniform(0.01, 0.05)))
        await realtime_tracker.update_budget_usage(agent_id, usage, budget)
        await asyncio.sleep(1)


async def main():
    """Run the budget monitoring example."""
    # Initialize trackers
    token_tracker = TokenTracker(
        model_costs={"gpt-4": {"input": Decimal("0.03"), "output": Decimal("0.06")}},
        token_budget=1000000,
        cost_budget=Decimal("100.00"),
        period=BudgetPeriod.DAILY,
    )

    cost_tracker = CostTracker(
        token_tracker=token_tracker,
        api_tracker=None,
        storage_cost_per_gb=Decimal("0.10"),
        compute_cost_per_hour=Decimal("1.00"),
        total_budget=Decimal("1000.00"),
        period=BudgetPeriod.DAILY,
    )

    # Initialize real-time tracker
    realtime_tracker = RealtimeBudgetTracker(
        cost_tracker=cost_tracker,
        token_tracker=token_tracker,
        websocket_port=8765,
    )

    # Initialize dashboard server
    dashboard = DashboardServer(host="localhost", port=8080)

    # Start servers
    await dashboard.start()
    asyncio.create_task(realtime_tracker.start())

    print("Budget monitoring dashboard running at http://localhost:8080")
    print("WebSocket server running at ws://localhost:8765")

    # Simulate some agent activity
    try:
        await simulate_agent_activity(
            realtime_tracker=realtime_tracker,
            agent_id="test-agent-1",
            budget=Decimal("1000.00"),
            duration=60,
        )
    finally:
        # Cleanup
        await realtime_tracker.stop()
        await dashboard.stop()


if __name__ == "__main__":
    asyncio.run(main())
