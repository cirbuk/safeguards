"""Base interfaces and abstract classes for budget management."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum, auto
from typing import Protocol


class BudgetPeriod(Enum):
    """Time periods for budget tracking."""

    HOURLY = auto()
    DAILY = auto()
    WEEKLY = auto()
    MONTHLY = auto()
    QUARTERLY = auto()
    YEARLY = auto()


@dataclass
class BudgetConfig:
    """Budget configuration."""

    total_budget: Decimal
    period: BudgetPeriod
    hourly_limit: Decimal | None = None
    daily_limit: Decimal | None = None
    weekly_limit: Decimal | None = None
    monthly_limit: Decimal | None = None
    alert_threshold: float = 0.8  # Alert at 80% usage

    def __post_init__(self):
        """Validate budget configuration."""
        if self.total_budget <= 0:
            msg = "Total budget must be positive"
            raise ValueError(msg)
        if self.alert_threshold <= 0 or self.alert_threshold > 1:
            msg = "Alert threshold must be between 0 and 1"
            raise ValueError(msg)

        # Validate period-specific limits
        if self.hourly_limit is not None and self.hourly_limit <= 0:
            msg = "Hourly limit must be positive"
            raise ValueError(msg)
        if self.daily_limit is not None and self.daily_limit <= 0:
            msg = "Daily limit must be positive"
            raise ValueError(msg)
        if self.weekly_limit is not None and self.weekly_limit <= 0:
            msg = "Weekly limit must be positive"
            raise ValueError(msg)
        if self.monthly_limit is not None and self.monthly_limit <= 0:
            msg = "Monthly limit must be positive"
            raise ValueError(msg)


@dataclass
class BudgetMetrics:
    """Budget usage metrics."""

    timestamp: datetime
    current_usage: Decimal
    total_budget: Decimal
    period_start: datetime
    period_end: datetime
    usage_percentage: float

    def __post_init__(self):
        """Validate budget metrics."""
        if self.current_usage < 0:
            msg = "Current usage cannot be negative"
            raise ValueError(msg)
        if self.total_budget <= 0:
            msg = "Total budget must be positive"
            raise ValueError(msg)
        if self.usage_percentage < 0 or self.usage_percentage > 1:
            msg = "Usage percentage must be between 0 and 1"
            raise ValueError(msg)
        if self.period_end <= self.period_start:
            msg = "Period end must be after period start"
            raise ValueError(msg)

    @property
    def usage_percent(self) -> float:
        """For backward compatibility with existing tests."""
        return self.usage_percentage * 100

    @property
    def remaining(self) -> Decimal:
        """Get remaining budget."""
        return self.total_budget - self.current_usage


class BudgetStorage(Protocol):
    """Protocol for budget storage implementations."""

    def store_usage(self, amount: Decimal, timestamp: datetime) -> None:
        """Store usage amount."""
        ...

    def get_usage(
        self,
        start_time: datetime,
        end_time: datetime,
    ) -> list[dict[str, Decimal]]:
        """Get usage for time period."""
        ...

    def get_current_period_usage(self, period: BudgetPeriod) -> Decimal:
        """Get total usage for current period."""
        ...


class BudgetManager(ABC):
    """Abstract base class for budget management."""

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
        self.config = config
        self.storage = storage

    @abstractmethod
    def check_budget(self, amount: Decimal) -> bool:
        """Check if amount is within budget.

        Args:
            amount: Amount to check

        Returns:
            True if amount is within budget
        """
        ...

    @abstractmethod
    def record_usage(self, amount: Decimal) -> None:
        """Record usage amount.

        Args:
            amount: Amount to record
        """
        ...

    @abstractmethod
    def get_metrics(self) -> BudgetMetrics:
        """Get current budget metrics.

        Returns:
            Current budget metrics
        """
        ...

    def get_period_dates(self) -> tuple[datetime, datetime]:
        """Get start and end dates for current period.

        Returns:
            Tuple of (start_date, end_date)
        """
        now = datetime.now()
        if self.config.period == BudgetPeriod.HOURLY:
            start = now.replace(minute=0, second=0, microsecond=0)
            end = start + timedelta(hours=1)
        elif self.config.period == BudgetPeriod.DAILY:
            start = now.replace(hour=0, minute=0, second=0, microsecond=0)
            end = start + timedelta(days=1)
        elif self.config.period == BudgetPeriod.WEEKLY:
            start = now.replace(hour=0, minute=0, second=0, microsecond=0)
            start = start - timedelta(days=start.weekday())
            end = start + timedelta(weeks=1)
        elif self.config.period == BudgetPeriod.MONTHLY:
            start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            if now.month == 12:
                end = start.replace(year=now.year + 1, month=1)
            else:
                end = start.replace(month=now.month + 1)
        elif self.config.period == BudgetPeriod.QUARTERLY:
            quarter = (now.month - 1) // 3
            start = now.replace(
                month=quarter * 3 + 1,
                day=1,
                hour=0,
                minute=0,
                second=0,
                microsecond=0,
            )
            if quarter == 3:
                end = start.replace(year=now.year + 1, month=1)
            else:
                end = start.replace(month=start.month + 3)
        else:  # YEARLY
            start = now.replace(
                month=1,
                day=1,
                hour=0,
                minute=0,
                second=0,
                microsecond=0,
            )
            end = start.replace(year=now.year + 1)

        return start, end
