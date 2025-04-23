#!/usr/bin/env python
"""
Example demonstrating budget monitoring for agents.

This example shows how to:
1. Set up a budget management system for multiple agents
2. Track and monitor resource usage of different operations
3. Apply budget constraints and guardrails
4. Handle budget violations and alerts
5. Generate usage reports
"""

import asyncio
import logging
from datetime import datetime
from decimal import Decimal
from typing import Any

from safeguards.base.budget import BudgetPeriod
from safeguards.budget.api_tracker import APITracker
from safeguards.budget.token_tracker import TokenTracker
from safeguards.notifications.manager import NotificationManager
from safeguards.types import Alert, AlertSeverity, NotificationChannel
from safeguards.types.agent import Agent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# Model costs per 1K tokens
MODEL_COSTS = {
    "gpt-4o": {"input": Decimal("0.01"), "output": Decimal("0.03")},
    "gpt-3.5-turbo": {"input": Decimal("0.0005"), "output": Decimal("0.0015")},
    "claude-3-opus": {"input": Decimal("0.015"), "output": Decimal("0.075")},
    "claude-3-sonnet": {"input": Decimal("0.003"), "output": Decimal("0.015")},
}

# API costs
API_COSTS = {
    "vision": Decimal("0.002"),  # per image
    "embeddings": Decimal("0.0001"),  # per record
    "audio": Decimal("0.006"),  # per minute
}


class BudgetManagedAgent(Agent):
    """Agent implementation with budget monitoring and enforcement."""

    def __init__(
        self,
        name: str,
        model: str,
        token_tracker: TokenTracker,
        api_tracker: APITracker,
        notification_manager: NotificationManager,
    ):
        """Initialize budget-managed agent.

        Args:
            name: Agent name
            model: Model identifier (e.g., "gpt-4o")
            token_tracker: Token usage tracker
            api_tracker: API usage tracker
            notification_manager: Notification manager for alerts
        """
        super().__init__(name)
        self.model = model
        self.token_tracker = token_tracker
        self.api_tracker = api_tracker
        self.notification_manager = notification_manager
        self.total_queries = 0
        self.successful_queries = 0
        self.session_start = datetime.now()

    async def run(self, **kwargs):
        """Implementation of the abstract run method required by Agent class.

        In a real implementation, this would contain the agent's core logic.
        For this example, we delegate to simulate_token_usage.

        Args:
            kwargs: Input parameters, should include 'query' and optional 'complexity'

        Returns:
            Dict with response and usage statistics
        """
        query = kwargs.get("query", "")
        complexity = kwargs.get("complexity", 1.0)
        return self.simulate_token_usage(query, complexity)

    def simulate_token_usage(
        self,
        query: str,
        complexity: float = 1.0,
    ) -> dict[str, Any]:
        """Simulate token usage and cost for demonstration purposes.

        In a real implementation, this would be replaced with actual token
        counts from the model API response.

        Args:
            query: User query
            complexity: Query complexity factor (affects token counts)

        Returns:
            Dict with response and usage statistics
        """
        self.total_queries += 1

        # Simulate token usage based on query length and complexity
        input_tokens = len(query.split()) * 4  # rough approximation
        input_tokens = int(input_tokens * complexity)

        # Simulate thinking and processing
        output_tokens = int(input_tokens * 1.5 * complexity)

        # Check if budget available before "spending"
        if not self.token_tracker.check_budget_available(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model_id=self.model,
        ):
            # Send budget alert
            self._send_budget_alert(
                "Budget limit reached",
                f"Agent {self.name} has reached its token budget limit.",
                AlertSeverity.ERROR,
            )

            return {
                "success": False,
                "response": "Unable to complete request due to budget constraints.",
                "usage": {
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0,
                    "cost": Decimal("0"),
                },
            }

        # Record the token usage
        usage = self.token_tracker.record_usage(
            model_id=self.model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

        # Simulate API usage if complex query
        if complexity > 1.0:
            # Simulate using the vision API for complex queries
            self.api_tracker.record_usage(
                endpoint="vision",
                data_transfer_mb=1.5
                * complexity,  # Simulate data transfer proportional to complexity
            )

        # Check if we're approaching budget threshold (75%)
        token_budget = self.token_tracker.token_budget
        tokens_used = self.token_tracker.get_total_tokens()
        percentage_used = tokens_used / token_budget * 100

        if percentage_used > 75 and percentage_used < 90:
            self._send_budget_alert(
                "Budget threshold warning",
                f"Agent {self.name} has used {percentage_used:.1f}% of its token budget.",
                AlertSeverity.WARNING,
            )
        elif percentage_used >= 90:
            self._send_budget_alert(
                "Critical budget warning",
                f"Agent {self.name} has used {percentage_used:.1f}% of its token budget.",
                AlertSeverity.CRITICAL,
            )

        self.successful_queries += 1

        # Generate simulated response
        response = (
            f"Processed query with {input_tokens} input tokens and {output_tokens} output tokens."
        )

        return {
            "success": True,
            "response": response,
            "usage": {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
                "cost": usage.cost,
            },
        }

    def get_usage_stats(self) -> dict[str, Any]:
        """Get detailed usage statistics for this agent."""
        end_time = datetime.now()
        session_duration = (end_time - self.session_start).total_seconds()

        token_stats = self.token_tracker.get_usage_stats()
        api_stats = self.api_tracker.get_usage_stats()

        return {
            "agent_id": self.id,
            "agent_name": self.name,
            "model": self.model,
            "session_duration_seconds": session_duration,
            "queries": {
                "total": self.total_queries,
                "successful": self.successful_queries,
                "failed": self.total_queries - self.successful_queries,
                "success_rate": (
                    (self.successful_queries / self.total_queries * 100)
                    if self.total_queries > 0
                    else 0
                ),
            },
            "tokens": token_stats,
            "api": api_stats,
            "budget": {
                "token_budget": self.token_tracker.token_budget,
                "tokens_used": self.token_tracker.get_total_tokens(),
                "tokens_remaining": self.token_tracker.token_budget
                - self.token_tracker.get_total_tokens(),
                "cost_budget": self.token_tracker.cost_budget,
                "token_cost": self.token_tracker.get_total_cost(),
                "api_cost": self.api_tracker.get_total_cost(),
                "total_cost": self.token_tracker.get_total_cost()
                + self.api_tracker.get_total_cost(),
                "cost_remaining": self.token_tracker.cost_budget
                - (self.token_tracker.get_total_cost() + self.api_tracker.get_total_cost()),
            },
        }

    def _send_budget_alert(self, title: str, description: str, severity: AlertSeverity):
        """Send a budget alert notification."""
        alert = Alert(
            title=title,
            description=description,
            severity=severity,
            timestamp=datetime.now(),
            metadata={
                "agent_id": self.id,
                "agent_name": self.name,
                "model": self.model,
                "tokens_used": self.token_tracker.get_total_tokens(),
                "token_budget": self.token_tracker.token_budget,
                "token_cost": float(self.token_tracker.get_total_cost()),
                "api_cost": float(self.api_tracker.get_total_cost()),
                "total_cost": float(
                    self.token_tracker.get_total_cost() + self.api_tracker.get_total_cost(),
                ),
                "cost_budget": float(self.token_tracker.cost_budget),
            },
        )

        self.notification_manager.send_alert(alert)


def print_usage_report(agent: BudgetManagedAgent):
    """Print a detailed usage report for an agent."""
    stats = agent.get_usage_stats()

    print("\n" + "=" * 50)
    print(f"USAGE REPORT: {stats['agent_name']} ({stats['agent_id']})")
    print("=" * 50)

    print(f"\nModel: {stats['model']}")
    print(f"Session duration: {stats['session_duration_seconds']:.1f} seconds")

    print("\nQUERIES:")
    print(f"  Total: {stats['queries']['total']}")
    print(f"  Successful: {stats['queries']['successful']}")
    print(f"  Failed: {stats['queries']['failed']}")
    print(f"  Success rate: {stats['queries']['success_rate']:.1f}%")

    print("\nTOKEN USAGE:")
    token_stats = stats["tokens"]
    for model, usage in token_stats.items():
        print(f"  {model}:")
        print(f"    Input tokens: {usage['input_tokens']}")
        print(f"    Output tokens: {usage['output_tokens']}")
        print(f"    Total tokens: {usage['total_tokens']}")

    print("\nAPI USAGE:")
    api_stats = stats["api"]
    for endpoint, usage in api_stats.items():
        print(f"  {endpoint}:")
        print(f"    Total calls: {usage.get('total_calls', 0)}")
        print(f"    Successful calls: {usage.get('successful_calls', 0)}")
        print(f"    Failed calls: {usage.get('failed_calls', 0)}")
        print(f"    Data transfer: {usage.get('data_transfer_mb', 0):.2f} MB")

    print("\nBUDGET STATUS:")
    budget = stats["budget"]
    print(f"  Token budget: {budget['token_budget']}")
    print(
        f"  Tokens used: {budget['tokens_used']} ({budget['tokens_used'] / budget['token_budget'] * 100:.1f}%)",
    )
    print(f"  Tokens remaining: {budget['tokens_remaining']}")
    print(f"  Cost budget: ${float(budget['cost_budget']):.4f}")
    print(f"  Token cost: ${float(budget['token_cost']):.4f}")
    print(f"  API cost: ${float(budget['api_cost']):.4f}")
    print(f"  Total cost: ${float(budget['total_cost']):.4f}")
    print(f"  Cost remaining: ${float(budget['cost_remaining']):.4f}")

    print("=" * 50 + "\n")


async def main():
    """Run the example."""
    print("=== Agent Budget Monitoring Example ===\n")

    # Set up notification manager
    notification_manager = NotificationManager(
        enabled_channels={NotificationChannel.CONSOLE},
    )

    # Configure token and API trackers for GPT-4o
    gpt4o_token_tracker = TokenTracker(
        model_costs=MODEL_COSTS,
        token_budget=100000,  # 100K tokens
        cost_budget=Decimal("10.0"),  # $10.00
        period=BudgetPeriod.DAILY,
    )

    gpt4o_api_tracker = APITracker(
        api_costs=API_COSTS,
        call_budget=100,  # 100 API calls
        cost_budget=Decimal("5.0"),  # $5.00
        period=BudgetPeriod.DAILY,
    )

    # Configure token and API trackers for Claude
    claude_token_tracker = TokenTracker(
        model_costs=MODEL_COSTS,
        token_budget=200000,  # 200K tokens
        cost_budget=Decimal("15.0"),  # $15.00
        period=BudgetPeriod.DAILY,
    )

    claude_api_tracker = APITracker(
        api_costs=API_COSTS,
        call_budget=200,  # 200 API calls
        cost_budget=Decimal("5.0"),  # $5.00
        period=BudgetPeriod.DAILY,
    )

    # Create agents
    gpt4o_agent = BudgetManagedAgent(
        name="GPT-4o Assistant",
        model="gpt-4o",
        token_tracker=gpt4o_token_tracker,
        api_tracker=gpt4o_api_tracker,
        notification_manager=notification_manager,
    )

    claude_agent = BudgetManagedAgent(
        name="Claude-3-Sonnet Assistant",
        model="claude-3-sonnet",
        token_tracker=claude_token_tracker,
        api_tracker=claude_api_tracker,
        notification_manager=notification_manager,
    )

    # Run some example queries with different complexity
    print("Running test queries on GPT-4o agent...")
    queries = [
        ("What is machine learning?", 0.8),
        ("Explain the theory of relativity in detail", 1.2),
        ("Write a short story about a robot learning to paint", 1.5),
        (
            "Compare and contrast 5 different sorting algorithms in terms of time complexity, space complexity, and real-world applications",
            2.0,
        ),
    ]

    for query, complexity in queries:
        print(f"\nProcessing query: '{query}' (complexity: {complexity})")
        result = gpt4o_agent.simulate_token_usage(query, complexity)
        print(f"Response: {result['response']}")
        print(
            f"Usage: {result['usage']['input_tokens']} input + {result['usage']['output_tokens']} output = {result['usage']['total_tokens']} tokens",
        )
        print(f"Cost: ${float(result['usage']['cost']):.6f}")

    # Run similar queries on Claude agent
    print("\nRunning test queries on Claude-3-Sonnet agent...")
    for query, complexity in queries:
        print(f"\nProcessing query: '{query}' (complexity: {complexity})")
        result = claude_agent.simulate_token_usage(query, complexity)
        print(f"Response: {result['response']}")
        print(
            f"Usage: {result['usage']['input_tokens']} input + {result['usage']['output_tokens']} output = {result['usage']['total_tokens']} tokens",
        )
        print(f"Cost: ${float(result['usage']['cost']):.6f}")

    # Print usage reports
    print_usage_report(gpt4o_agent)
    print_usage_report(claude_agent)

    # Demonstrate budget limit
    print("\nTesting budget limit with a large query...")
    large_query = "Write a comprehensive thesis on artificial intelligence, its history, current state, applications, ethical considerations, and future directions. Include detailed analysis of machine learning algorithms, neural networks, and natural language processing techniques."
    result = gpt4o_agent.simulate_token_usage(
        large_query,
        10.0,
    )  # Very high complexity to trigger budget limit
    print(f"Response: {result['response']}")
    if not result["success"]:
        print("Budget limit successfully enforced!")

    # Final usage report after budget limit test
    print_usage_report(gpt4o_agent)

    print("\n=== Example Complete ===")


if __name__ == "__main__":
    asyncio.run(main())
