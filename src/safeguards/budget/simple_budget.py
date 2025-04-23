"""Simple budget manager implementation."""

from datetime import datetime
from decimal import Decimal

from safeguards.base import (
    BudgetConfig,
    BudgetManager,
    BudgetMetrics,
    BudgetPeriod,
    BudgetStorage,
)


class SimpleBudgetManager(BudgetManager):
    """Simple implementation of budget management."""

    def __init__(
        self,
        config: BudgetConfig,
        storage: BudgetStorage | None = None,
    ):
        """Initialize simple budget manager.

        Args:
            config: Budget configuration
            storage: Optional budget storage implementation
        """
        super().__init__(config, storage)
        self._current_usage = Decimal(0)
        if storage:
            start, end = self.get_period_dates()
            usage_records = storage.get_usage(start, end)
            self._current_usage = sum(record["amount"] for record in usage_records)

    def check_budget(self, amount: Decimal) -> bool:
        """Check if amount is within budget.

        Args:
            amount: Amount to check

        Returns:
            True if amount is within budget
        """
        # Get current period usage
        if self.storage:
            start, end = self.get_period_dates()
            usage_records = self.storage.get_usage(start, end)
            current_usage = sum(record["amount"] for record in usage_records)
        else:
            current_usage = self._current_usage

        # Check against total budget
        if current_usage + amount > self.config.total_budget:
            return False

        # Check against period-specific limits
        if self.config.period == BudgetPeriod.HOURLY and self.config.hourly_limit:
            if current_usage + amount > self.config.hourly_limit:
                return False
        elif self.config.period == BudgetPeriod.DAILY and self.config.daily_limit:
            if current_usage + amount > self.config.daily_limit:
                return False
        elif self.config.period == BudgetPeriod.WEEKLY and self.config.weekly_limit:
            if current_usage + amount > self.config.weekly_limit:
                return False
        elif (
            self.config.period == BudgetPeriod.MONTHLY
            and self.config.monthly_limit
            and current_usage + amount > self.config.monthly_limit
        ):
            return False

        return True

    def record_usage(self, amount: Decimal) -> None:
        """Record usage amount.

        Args:
            amount: Amount to record
        """
        if amount < 0:
            msg = "Usage amount cannot be negative"
            raise ValueError(msg)

        self._current_usage += amount
        if self.storage:
            self.storage.store_usage(amount, datetime.now())

    def get_metrics(self) -> BudgetMetrics:
        """Get current budget metrics.

        Returns:
            Current budget metrics
        """
        # Get current period dates
        period_start, period_end = self.get_period_dates()

        # Get current usage
        if self.storage:
            usage_records = self.storage.get_usage(period_start, period_end)
            current_usage = sum(record["amount"] for record in usage_records)
        else:
            current_usage = self._current_usage

        # Calculate usage percentage
        usage_percentage = float(current_usage / self.config.total_budget)

        return BudgetMetrics(
            timestamp=datetime.now(),
            current_usage=current_usage,
            total_budget=self.config.total_budget,
            period_start=period_start,
            period_end=period_end,
            usage_percentage=usage_percentage,
        )
