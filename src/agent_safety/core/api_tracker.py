"""API usage tracking and cost calculation for external services.

This module provides functionality for:
- Tracking API call counts and rates
- Managing rate limits
- Calculating API usage costs
- Monitoring quota utilization
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Tuple

from founderx.config.cost_config import CostConfigLoader
from founderx.core.cost_management import CostType
from founderx.core.cost_tracker import CostTracker


@dataclass
class APIRateLimit:
    """Rate limit configuration for an API."""

    calls_per_second: Optional[int] = None
    calls_per_minute: Optional[int] = None
    calls_per_hour: Optional[int] = None
    calls_per_day: Optional[int] = None
    calls_per_month: Optional[int] = None


@dataclass
class APIConfig:
    """Configuration for an API service."""

    cost_per_call: Decimal
    rate_limits: APIRateLimit
    quota_period: str  # 'hourly', 'daily', 'monthly'
    quota_limit: int


class CallWindow:
    """Tracks API calls within a time window."""

    def __init__(self, window_size: timedelta):
        """Initialize call window.

        Args:
            window_size: Size of the time window
        """
        self.window_size = window_size
        self.calls: List[datetime] = []

    def add_call(self) -> None:
        """Record a new API call."""
        now = datetime.now()
        self.calls = [t for t in self.calls if now - t < self.window_size]
        self.calls.append(now)

    def get_count(self) -> int:
        """Get number of calls in current window.

        Returns:
            Number of calls
        """
        now = datetime.now()
        self.calls = [t for t in self.calls if now - t < self.window_size]
        return len(self.calls)


class APIUsageTracker:
    """Tracks API usage and enforces rate limits."""

    DEFAULT_CONFIGS = {
        "linkedin": APIConfig(
            cost_per_call=Decimal("0.001"),
            rate_limits=APIRateLimit(
                calls_per_second=1,
                calls_per_minute=60,
                calls_per_hour=1200,
                calls_per_day=10000,
            ),
            quota_period="monthly",
            quota_limit=100000,
        ),
        "twitter": APIConfig(
            cost_per_call=Decimal("0.0005"),
            rate_limits=APIRateLimit(
                calls_per_second=1,
                calls_per_minute=50,
                calls_per_hour=1500,
                calls_per_day=15000,
            ),
            quota_period="monthly",
            quota_limit=150000,
        ),
        "instagram": APIConfig(
            cost_per_call=Decimal("0.002"),
            rate_limits=APIRateLimit(
                calls_per_second=1,
                calls_per_minute=30,
                calls_per_hour=500,
                calls_per_day=5000,
            ),
            quota_period="monthly",
            quota_limit=50000,
        ),
    }

    def __init__(
        self,
        cost_tracker: CostTracker,
        config_loader: Optional[CostConfigLoader] = None,
        api_configs: Optional[Dict[str, APIConfig]] = None,
    ):
        """Initialize API tracker.

        Args:
            cost_tracker: Cost tracker instance
            config_loader: Configuration loader instance
            api_configs: Optional custom API configurations
        """
        self.cost_tracker = cost_tracker
        self.config = config_loader or CostConfigLoader()
        self.api_configs = api_configs or self.DEFAULT_CONFIGS

        # Initialize tracking windows
        self.windows: Dict[str, Dict[str, CallWindow]] = {}
        for api in self.api_configs:
            self.windows[api] = {
                "second": CallWindow(timedelta(seconds=1)),
                "minute": CallWindow(timedelta(minutes=1)),
                "hour": CallWindow(timedelta(hours=1)),
                "day": CallWindow(timedelta(days=1)),
                "month": CallWindow(timedelta(days=30)),  # Simplified
            }

        # Initialize counters
        self.call_counts: Dict[str, Dict[str, int]] = {
            api: {"total": 0, "success": 0, "failed": 0} for api in self.api_configs
        }

    def can_make_call(self, api: str) -> Tuple[bool, Optional[str]]:
        """Check if an API call can be made within rate limits.

        Args:
            api: Name of the API service

        Returns:
            Tuple of (can_call, error_message)

        Raises:
            ValueError: If API is not configured
        """
        if api not in self.api_configs:
            raise ValueError(f"Unknown API: {api}")

        config = self.api_configs[api]
        windows = self.windows[api]

        # Check rate limits
        if (
            config.rate_limits.calls_per_second
            and windows["second"].get_count() >= config.rate_limits.calls_per_second
        ):
            return False, "Rate limit exceeded: too many calls per second"

        if (
            config.rate_limits.calls_per_minute
            and windows["minute"].get_count() >= config.rate_limits.calls_per_minute
        ):
            return False, "Rate limit exceeded: too many calls per minute"

        if (
            config.rate_limits.calls_per_hour
            and windows["hour"].get_count() >= config.rate_limits.calls_per_hour
        ):
            return False, "Rate limit exceeded: too many calls per hour"

        if (
            config.rate_limits.calls_per_day
            and windows["day"].get_count() >= config.rate_limits.calls_per_day
        ):
            return False, "Rate limit exceeded: too many calls per day"

        # Check quota
        quota_window = windows["month" if config.quota_period == "monthly" else "day"]
        if quota_window.get_count() >= config.quota_limit:
            return False, f"Quota exceeded for {config.quota_period} period"

        return True, None

    def record_call(
        self, api: str, success: bool = True, record_cost: bool = True
    ) -> Tuple[bool, Optional[str]]:
        """Record an API call and update costs.

        Args:
            api: Name of the API service
            success: Whether the call was successful
            record_cost: Whether to record the cost

        Returns:
            Tuple of (success, error_message)

        Raises:
            ValueError: If API is not configured
        """
        if api not in self.api_configs:
            raise ValueError(f"Unknown API: {api}")

        # Check rate limits first
        can_call, error = self.can_make_call(api)
        if not can_call:
            self.call_counts[api]["failed"] += 1
            return False, error

        # Update windows
        for window in self.windows[api].values():
            window.add_call()

        # Update counters
        self.call_counts[api]["total"] += 1
        if success:
            self.call_counts[api]["success"] += 1
        else:
            self.call_counts[api]["failed"] += 1

        # Record cost if successful
        if success and record_cost:
            cost = self.api_configs[api].cost_per_call
            cost_type = getattr(CostType, f"{api.upper()}_API")
            self.cost_tracker.add_cost(cost_type, cost)

        return True, None

    def get_usage_stats(self, api: str) -> Dict[str, int]:
        """Get usage statistics for an API.

        Args:
            api: Name of the API service

        Returns:
            Dictionary containing usage statistics

        Raises:
            ValueError: If API is not configured
        """
        if api not in self.api_configs:
            raise ValueError(f"Unknown API: {api}")

        windows = self.windows[api]
        return {
            "current": {
                "second": windows["second"].get_count(),
                "minute": windows["minute"].get_count(),
                "hour": windows["hour"].get_count(),
                "day": windows["day"].get_count(),
                "month": windows["month"].get_count(),
            },
            "total": self.call_counts[api]["total"],
            "success": self.call_counts[api]["success"],
            "failed": self.call_counts[api]["failed"],
        }

    def estimate_cost(self, api: str, num_calls: int) -> Decimal:
        """Estimate cost for a number of API calls.

        Args:
            api: Name of the API service
            num_calls: Number of calls to estimate for

        Returns:
            Estimated cost

        Raises:
            ValueError: If API is not configured
        """
        if api not in self.api_configs:
            raise ValueError(f"Unknown API: {api}")

        return self.api_configs[api].cost_per_call * Decimal(num_calls)
