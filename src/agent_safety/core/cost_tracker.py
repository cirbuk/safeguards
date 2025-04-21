"""Cost tracking module for managing and monitoring costs across different time windows."""

from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, Optional

from founderx.core.cost_management import CostType


class CostTracker:
    """Tracks costs across different time windows and cost types."""

    def __init__(self) -> None:
        """Initialize the cost tracker."""
        self.costs: Dict[CostType, Decimal] = {ct: Decimal(0) for ct in CostType}
        self.cost_history: Dict[datetime, Dict[CostType, Decimal]] = {}

    def add_cost(self, cost_type: CostType, amount: Decimal) -> None:
        """
        Add a cost of the specified type.

        Args:
            cost_type: The type of cost being added
            amount: The cost amount to add
        """
        self.costs[cost_type] += amount
        current_time = datetime.now()
        if current_time not in self.cost_history:
            self.cost_history[current_time] = {ct: Decimal(0) for ct in CostType}
        self.cost_history[current_time][cost_type] += amount

    def get_total_cost(self) -> Decimal:
        """Get the total cost across all cost types."""
        return sum(self.costs.values())

    def get_cost_by_type(self, cost_type: CostType) -> Decimal:
        """
        Get the total cost for a specific cost type.

        Args:
            cost_type: The type of cost to get the total for

        Returns:
            The total cost for the specified type
        """
        return self.costs[cost_type]

    def get_costs_in_window(
        self,
        window_start: Optional[datetime] = None,
        window_end: Optional[datetime] = None,
    ) -> Dict[CostType, Decimal]:
        """
        Get costs within a specific time window.

        Args:
            window_start: Start of the time window (inclusive)
            window_end: End of the time window (exclusive)

        Returns:
            Dictionary mapping cost types to their total costs in the window
        """
        if window_start is None:
            window_start = datetime.min
        if window_end is None:
            window_end = datetime.max

        window_costs = {ct: Decimal(0) for ct in CostType}
        for timestamp, costs in self.cost_history.items():
            if window_start <= timestamp < window_end:
                for cost_type, amount in costs.items():
                    window_costs[cost_type] += amount

        return window_costs

    def get_hourly_costs(self) -> Dict[CostType, Decimal]:
        """Get costs from the last hour."""
        now = datetime.now()
        return self.get_costs_in_window(now - timedelta(hours=1), now)

    def get_daily_costs(self) -> Dict[CostType, Decimal]:
        """Get costs from the last 24 hours."""
        now = datetime.now()
        return self.get_costs_in_window(now - timedelta(days=1), now)

    def get_weekly_costs(self) -> Dict[CostType, Decimal]:
        """Get costs from the last 7 days."""
        now = datetime.now()
        return self.get_costs_in_window(now - timedelta(weeks=1), now)

    def get_monthly_costs(self) -> Dict[CostType, Decimal]:
        """Get costs from the last 30 days."""
        now = datetime.now()
        return self.get_costs_in_window(now - timedelta(days=30), now)

    def reset(self) -> None:
        """Reset all cost tracking data."""
        self.costs = {ct: Decimal(0) for ct in CostType}
        self.cost_history.clear()
