"""Cost tracking module for aggregating and monitoring all costs."""

from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal

from ..base.budget import BudgetPeriod
from .api_tracker import APITracker
from .token_tracker import TokenTracker


@dataclass
class CostBreakdown:
    """Detailed breakdown of costs by category."""

    llm_costs: dict[str, Decimal]  # model_id -> cost
    api_costs: dict[str, Decimal]  # endpoint -> cost
    storage_costs: Decimal
    compute_costs: Decimal
    total_cost: Decimal
    period_start: datetime
    period_end: datetime


class CostTracker:
    """Aggregates and tracks all costs across different services."""

    def __init__(
        self,
        token_tracker: TokenTracker,
        api_tracker: APITracker,
        storage_cost_per_gb: Decimal,
        compute_cost_per_hour: Decimal,
        total_budget: Decimal | None = None,
        period: BudgetPeriod = BudgetPeriod.DAILY,
        alert_threshold: float = 0.8,
    ):
        """Initialize cost tracker.

        Args:
            token_tracker: Token usage tracker instance
            api_tracker: API usage tracker instance
            storage_cost_per_gb: Cost per GB of storage
            compute_cost_per_hour: Cost per hour of compute
            total_budget: Optional total budget across all services
            period: Budget period for tracking
            alert_threshold: Alert threshold as percentage of budget
        """
        self.token_tracker = token_tracker
        self.api_tracker = api_tracker
        self.storage_cost_per_gb = storage_cost_per_gb
        self.compute_cost_per_hour = compute_cost_per_hour
        self.total_budget = total_budget
        self.period = period
        self.alert_threshold = alert_threshold

        self._storage_usage_gb = Decimal("0")
        self._compute_hours = Decimal("0")

    def record_storage_usage(self, size_gb: float | Decimal) -> None:
        """Record storage usage.

        Args:
            size_gb: Storage size in GB
        """
        self._storage_usage_gb = Decimal(str(size_gb))

    def record_compute_usage(self, hours: float | Decimal) -> None:
        """Record compute usage.

        Args:
            hours: Compute hours used
        """
        self._compute_hours = Decimal(str(hours))

    def get_storage_cost(self) -> Decimal:
        """Get current storage cost.

        Returns:
            Current storage cost
        """
        return self._storage_usage_gb * self.storage_cost_per_gb

    def get_compute_cost(self) -> Decimal:
        """Get current compute cost.

        Returns:
            Current compute cost
        """
        return self._compute_hours * self.compute_cost_per_hour

    def get_total_cost(self) -> Decimal:
        """Get total cost across all services.

        Returns:
            Total cost in decimal
        """
        return (
            self.token_tracker.get_total_cost()
            + self.api_tracker.get_total_cost()
            + self.get_storage_cost()
            + self.get_compute_cost()
        )

    def get_cost_breakdown(
        self,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
    ) -> CostBreakdown:
        """Get detailed cost breakdown.

        Args:
            start_time: Optional start time for period
            end_time: Optional end time for period

        Returns:
            Detailed cost breakdown
        """
        if not start_time or not end_time:
            start_time, end_time = self._get_current_period()

        # Get LLM costs by model
        token_usage = self.token_tracker.get_usage_in_period(start_time, end_time)
        llm_costs = {}
        for usage in token_usage:
            if usage.model_id not in llm_costs:
                llm_costs[usage.model_id] = Decimal("0")
            llm_costs[usage.model_id] += usage.cost

        # Get API costs by endpoint
        api_usage = self.api_tracker.get_usage_in_period(start_time, end_time)
        api_costs = {}
        for usage in api_usage:
            if usage.endpoint not in api_costs:
                api_costs[usage.endpoint] = Decimal("0")
            api_costs[usage.endpoint] += usage.cost

        # Calculate total cost
        total_cost = (
            sum(llm_costs.values())
            + sum(api_costs.values())
            + self.get_storage_cost()
            + self.get_compute_cost()
        )

        return CostBreakdown(
            llm_costs=llm_costs,
            api_costs=api_costs,
            storage_costs=self.get_storage_cost(),
            compute_costs=self.get_compute_cost(),
            total_cost=total_cost,
            period_start=start_time,
            period_end=end_time,
        )

    def check_budget_available(self, estimated_cost: Decimal) -> bool:
        """Check if budget is available for estimated cost.

        Args:
            estimated_cost: Estimated cost to check

        Returns:
            True if budget is available, False otherwise
        """
        if not self.total_budget:
            return True

        current_total = self.get_total_cost()
        return current_total + estimated_cost <= self.total_budget

    def is_approaching_budget(self) -> bool:
        """Check if approaching budget threshold.

        Returns:
            True if approaching budget threshold, False otherwise
        """
        if not self.total_budget:
            return False

        usage_percent = self.get_total_cost() / self.total_budget
        return usage_percent >= self.alert_threshold

    def get_budget_status(self) -> dict[str, Decimal | float]:
        """Get current budget status.

        Returns:
            Dict with budget status information
        """
        total_cost = self.get_total_cost()

        if self.total_budget:
            remaining_budget = self.total_budget - total_cost
            usage_percent = float(total_cost / self.total_budget)
        else:
            remaining_budget = None
            usage_percent = None

        return {
            "total_cost": total_cost,
            "total_budget": self.total_budget,
            "remaining_budget": remaining_budget,
            "usage_percent": usage_percent,
        }

    def _get_current_period(self) -> tuple[datetime, datetime]:
        """Get start and end times for current period.

        Returns:
            Tuple of (start_time, end_time)
        """
        now = datetime.now()

        if self.period == BudgetPeriod.HOURLY:
            start = now.replace(minute=0, second=0, microsecond=0)
            end = start.replace(minute=59, second=59, microsecond=999999)
        elif self.period == BudgetPeriod.DAILY:
            start = now.replace(hour=0, minute=0, second=0, microsecond=0)
            end = start.replace(hour=23, minute=59, second=59, microsecond=999999)
        elif self.period == BudgetPeriod.WEEKLY:
            start = now.replace(hour=0, minute=0, second=0, microsecond=0)
            start = start - timedelta(days=start.weekday())
            end = start + timedelta(
                days=6,
                hours=23,
                minutes=59,
                seconds=59,
                microseconds=999999,
            )
        elif self.period == BudgetPeriod.MONTHLY:
            start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            if now.month == 12:
                end = start.replace(year=now.year + 1, month=1) - timedelta(
                    microseconds=1,
                )
            else:
                end = start.replace(month=now.month + 1) - timedelta(microseconds=1)
        else:  # YEARLY
            start = now.replace(
                month=1,
                day=1,
                hour=0,
                minute=0,
                second=0,
                microsecond=0,
            )
            end = start.replace(year=now.year + 1) - timedelta(microseconds=1)

        return start, end

    def reset_tracking(self) -> None:
        """Reset all tracking data."""
        self.token_tracker.reset_tracking()
        self.api_tracker.reset_tracking()
        self._storage_usage_gb = Decimal("0")
        self._compute_hours = Decimal("0")
