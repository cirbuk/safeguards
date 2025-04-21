"""Budget manager module for tracking and controlling agent spending."""

from decimal import Decimal
from typing import Optional

from agent_safety.types import BudgetConfig


class BudgetManager:
    """Manages budget allocation and tracking for agents.

    This class handles budget-related operations including:
    - Tracking total spending
    - Checking budget limits
    - Calculating remaining budget
    - Monitoring usage percentages
    """

    def __init__(self, config: BudgetConfig):
        """Initialize the budget manager.

        Args:
            config: Budget configuration containing limits and thresholds
        """
        self.total_budget = config.total_budget
        self.hourly_limit = config.hourly_limit
        self.daily_limit = config.daily_limit
        self.warning_threshold = config.warning_threshold
        self.total_spent = Decimal("0")

    def has_sufficient_budget(self, cost: Decimal) -> bool:
        """Check if there is sufficient budget for a given cost.

        Args:
            cost: The cost to check against remaining budget

        Returns:
            bool: True if there is sufficient budget, False otherwise
        """
        return self.get_remaining_budget() >= cost

    def has_exceeded_budget(self) -> bool:
        """Check if total spending has exceeded the budget.

        Returns:
            bool: True if budget is exceeded, False otherwise
        """
        return self.total_spent > self.total_budget

    def record_cost(self, cost: Decimal) -> None:
        """Record a cost and update total spending.

        Args:
            cost: The cost to record
        """
        self.total_spent += cost

    def get_remaining_budget(self) -> Decimal:
        """Get the remaining budget.

        Returns:
            Decimal: The remaining budget amount
        """
        return self.total_budget - self.total_spent

    def get_budget_usage_percent(self) -> float:
        """Calculate the percentage of budget used.

        Returns:
            float: The percentage of total budget used
        """
        if self.total_budget == Decimal("0"):
            return 100.0
        return float(self.total_spent / self.total_budget * 100)

    def reset_budget(self) -> None:
        """Reset the budget tracking to initial state."""
        self.total_spent = Decimal("0")
