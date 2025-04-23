"""Concrete implementation of budget management."""

from datetime import datetime
from decimal import Decimal

from safeguards.base.budget import (
    BudgetConfig,
    BudgetManager,
    BudgetMetrics,
    BudgetStorage,
)


class SimpleBudgetManager(BudgetManager):
    """Simple implementation of budget management."""

    def __init__(
        self,
        config: BudgetConfig,
        storage: BudgetStorage | None = None,
    ):
        """Initialize budget manager.

        Args:
            config: Budget configuration
            storage: Optional budget storage implementation
        """
        super().__init__(config, storage)
        self._current_usage = Decimal(0)

    def check_budget(self, amount: Decimal) -> bool:
        """Check if amount is within budget.

        Args:
            amount: Amount to check

        Returns:
            True if amount is within budget
        """
        current_usage = self._get_current_usage()
        return current_usage + amount <= self.config.total_budget

    def record_usage(self, amount: Decimal) -> None:
        """Record usage amount.

        Args:
            amount: Amount to record
        """
        if self.storage:
            self.storage.store_usage(amount, datetime.now())
        self._current_usage += amount

    def get_metrics(self) -> BudgetMetrics:
        """Get current budget metrics.

        Returns:
            Current budget metrics
        """
        current_usage = self._get_current_usage()
        period_start, period_end = self.get_period_dates()

        return BudgetMetrics(
            timestamp=datetime.now(),
            current_usage=current_usage,
            total_budget=self.config.total_budget,
            period_start=period_start,
            period_end=period_end,
            usage_percentage=float(current_usage / self.config.total_budget),
        )

    def _get_current_usage(self) -> Decimal:
        """Get current usage for the period.

        Returns:
            Current usage amount
        """
        if self.storage:
            period_start, _ = self.get_period_dates()
            return self.storage.get_current_period_usage(self.config.period)
        return self._current_usage
