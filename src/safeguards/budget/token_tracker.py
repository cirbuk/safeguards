"""Module for managing API token consumption and budget constraints.

This module provides classes for tracking token usage across different agents and models,
while enforcing budget constraints and providing usage metrics.
"""

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal

from safeguards.base.budget import BudgetPeriod


@dataclass
class TokenUsage:
    """Represents a single token usage entry."""

    timestamp: datetime
    model_id: str
    input_tokens: int
    output_tokens: int
    cost: Decimal


class TokenTracker:
    """Tracks token usage and manages budgets for API consumption."""

    def __init__(
        self,
        model_costs: dict[str, dict[str, Decimal]],
        token_budget: int,
        cost_budget: Decimal,
        period: BudgetPeriod,
    ):
        """Initialize the token tracker.

        Args:
            model_costs: Dictionary mapping model IDs to their input/output costs per 1K tokens
            token_budget: Maximum number of tokens allowed in the period
            cost_budget: Maximum cost allowed in the period
            period: Budget period (e.g., DAILY, WEEKLY)
        """
        self.model_costs = model_costs
        self.token_budget = token_budget
        self.cost_budget = cost_budget
        self.period = period
        self.usage_history: list[TokenUsage] = []

    def record_usage(
        self,
        model_id: str,
        input_tokens: int,
        output_tokens: int,
        timestamp: datetime | None = None,
    ) -> TokenUsage:
        """Record token usage.

        Args:
            model_id: Identifier for the model used
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            timestamp: Optional timestamp for the usage

        Returns:
            TokenUsage object representing the recorded usage

        Raises:
            ValueError: If model_id is not recognized
        """
        if model_id not in self.model_costs:
            msg = f"Unknown model: {model_id}"
            raise ValueError(msg)

        if timestamp is None:
            timestamp = datetime.now()

        # Calculate cost
        costs = self.model_costs[model_id]
        input_cost = Decimal(str(input_tokens)) * costs["input"] / Decimal("1000")
        output_cost = Decimal(str(output_tokens)) * costs["output"] / Decimal("1000")
        total_cost = input_cost + output_cost

        usage = TokenUsage(
            timestamp=timestamp,
            model_id=model_id,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost=total_cost,
        )

        self.usage_history.append(usage)
        return usage

    def get_usage_in_period(
        self,
        start_time: datetime,
        end_time: datetime,
    ) -> list[TokenUsage]:
        """Get usage records for a specific time period.

        Args:
            start_time: Start of the period
            end_time: End of the period

        Returns:
            List of TokenUsage records in the period
        """
        return [usage for usage in self.usage_history if start_time <= usage.timestamp < end_time]

    def get_total_tokens(self, model_id: str | None = None) -> int:
        """Get total tokens used.

        Args:
            model_id: Optional model ID to filter by

        Returns:
            Total number of tokens used
        """
        usages = (
            [u for u in self.usage_history if u.model_id == model_id]
            if model_id
            else self.usage_history
        )
        return sum(u.input_tokens + u.output_tokens for u in usages)

    def get_total_cost(self, model_id: str | None = None) -> Decimal:
        """Get total cost of usage.

        Args:
            model_id: Optional model ID to filter by

        Returns:
            Total cost of usage
        """
        usages = (
            [u for u in self.usage_history if u.model_id == model_id]
            if model_id
            else self.usage_history
        )
        return sum(u.cost for u in usages)

    def check_budget_available(
        self,
        input_tokens: int,
        output_tokens: int,
        model_id: str,
    ) -> bool:
        """Check if the usage would be within budget.

        Args:
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            model_id: Model identifier

        Returns:
            True if usage is within budget, False otherwise
        """
        # Check token budget
        total_new_tokens = input_tokens + output_tokens
        if self.get_total_tokens() + total_new_tokens > self.token_budget:
            return False

        # Calculate new cost
        if model_id not in self.model_costs:
            msg = f"Unknown model: {model_id}"
            raise ValueError(msg)

        costs = self.model_costs[model_id]
        input_cost = Decimal(str(input_tokens)) * costs["input"] / Decimal("1000")
        output_cost = Decimal(str(output_tokens)) * costs["output"] / Decimal("1000")
        new_cost = input_cost + output_cost

        # Check cost budget
        return self.get_total_cost() + new_cost <= self.cost_budget

    def get_usage_stats(self) -> dict[str, dict[str, int]]:
        """Get usage statistics by model.

        Returns:
            Dictionary mapping model IDs to their usage statistics
        """
        stats: dict[str, dict[str, int]] = {}
        for usage in self.usage_history:
            if usage.model_id not in stats:
                stats[usage.model_id] = {
                    "total_tokens": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                }
            model_stats = stats[usage.model_id]
            model_stats["total_tokens"] += usage.input_tokens + usage.output_tokens
            model_stats["input_tokens"] += usage.input_tokens
            model_stats["output_tokens"] += usage.output_tokens
        return stats

    def reset_tracking(self) -> None:
        """Reset all tracking data."""
        self.usage_history.clear()

    def get_remaining_budget(self) -> Decimal:
        """Get the remaining budget based on recorded usage.

        Returns:
            The remaining budget amount
        """
        return self.cost_budget - self.get_total_cost()

    def clear_history(self, model_id: str | None = None) -> None:
        """Clear usage history.

        Args:
            model_id: Optional model ID to clear history for. If None, clears all history.
        """
        if model_id:
            self.usage_history = [u for u in self.usage_history if u.model_id != model_id]
        else:
            self.usage_history.clear()
