"""API tracking module for monitoring API usage and costs."""

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Dict, List, Optional

from ..base.budget import BudgetPeriod


@dataclass
class APIUsage:
    """Represents a single API call usage."""

    endpoint: str
    timestamp: datetime
    cost: Decimal
    data_transfer_mb: Optional[float] = None
    status_code: Optional[int] = None
    error: Optional[str] = None


class APITracker:
    """Tracks API usage and costs."""

    def __init__(
        self,
        api_costs: Dict[str, Decimal],  # endpoint -> cost per call
        call_budget: Optional[int] = None,
        cost_budget: Optional[Decimal] = None,
        period: BudgetPeriod = BudgetPeriod.DAILY,
        data_transfer_cost_per_gb: Optional[Decimal] = None,
    ):
        """Initialize API tracker.

        Args:
            api_costs: Mapping of API endpoints to their costs per call
            call_budget: Optional maximum number of API calls
            cost_budget: Optional maximum cost budget
            period: Budget period for tracking
            data_transfer_cost_per_gb: Optional cost per GB of data transfer
        """
        self.api_costs = api_costs
        self.call_budget = call_budget
        self.cost_budget = cost_budget
        self.period = period
        self.data_transfer_cost_per_gb = data_transfer_cost_per_gb
        self.usage_history: List[APIUsage] = []
        self._total_calls = 0
        self._total_cost = Decimal("0")

    def record_usage(
        self,
        endpoint: str,
        timestamp: Optional[datetime] = None,
        data_transfer_mb: Optional[float] = None,
        status_code: Optional[int] = None,
        error: Optional[str] = None,
    ) -> APIUsage:
        """Record an API call.

        Args:
            endpoint: API endpoint identifier
            timestamp: Optional timestamp (defaults to now)
            data_transfer_mb: Optional data transfer size in MB
            status_code: Optional HTTP status code
            error: Optional error message

        Returns:
            APIUsage record

        Raises:
            ValueError: If endpoint is not recognized
        """
        if endpoint not in self.api_costs:
            raise ValueError(f"Unknown API endpoint: {endpoint}")

        timestamp = timestamp or datetime.now()

        # Calculate cost
        base_cost = self.api_costs[endpoint]
        total_cost = base_cost

        # Add data transfer cost if applicable
        if data_transfer_mb and self.data_transfer_cost_per_gb:
            data_cost = (
                Decimal(str(data_transfer_mb / 1024))  # Convert to GB
                * self.data_transfer_cost_per_gb
            )
            total_cost += data_cost

        usage = APIUsage(
            endpoint=endpoint,
            timestamp=timestamp,
            cost=total_cost,
            data_transfer_mb=data_transfer_mb,
            status_code=status_code,
            error=error,
        )

        self.usage_history.append(usage)
        self._total_calls += 1
        self._total_cost += total_cost

        return usage

    def get_usage_in_period(
        self,
        start_time: datetime,
        end_time: datetime,
    ) -> List[APIUsage]:
        """Get API usage records for a specific time period.

        Args:
            start_time: Start of the period
            end_time: End of the period

        Returns:
            List of API usage records in the period
        """
        return [
            usage
            for usage in self.usage_history
            if start_time <= usage.timestamp <= end_time
        ]

    def get_total_calls(self, endpoint: Optional[str] = None) -> int:
        """Get total number of API calls.

        Args:
            endpoint: Optional endpoint to filter by

        Returns:
            Total call count
        """
        if endpoint:
            return sum(1 for usage in self.usage_history if usage.endpoint == endpoint)
        return self._total_calls

    def get_total_cost(self, endpoint: Optional[str] = None) -> Decimal:
        """Get total cost of API usage.

        Args:
            endpoint: Optional endpoint to filter by

        Returns:
            Total cost in decimal
        """
        if endpoint:
            return sum(
                usage.cost for usage in self.usage_history if usage.endpoint == endpoint
            )
        return self._total_cost

    def check_budget_available(
        self,
        endpoint: str,
        data_transfer_mb: Optional[float] = None,
    ) -> bool:
        """Check if call/cost budget is available.

        Args:
            endpoint: API endpoint to check
            data_transfer_mb: Optional data transfer size to include in cost calculation

        Returns:
            True if budget is available, False otherwise
        """
        # Check call budget if set
        if self.call_budget and self._total_calls + 1 > self.call_budget:
            return False

        # Check cost budget if set
        if self.cost_budget:
            new_cost = self.api_costs[endpoint]
            if data_transfer_mb and self.data_transfer_cost_per_gb:
                new_cost += (
                    Decimal(str(data_transfer_mb / 1024))
                    * self.data_transfer_cost_per_gb
                )
            if self._total_cost + new_cost > self.cost_budget:
                return False

        return True

    def get_usage_stats(self) -> Dict[str, Dict[str, int]]:
        """Get usage statistics broken down by endpoint.

        Returns:
            Dict mapping endpoints to their usage stats
        """
        stats = {}
        for endpoint in self.api_costs.keys():
            endpoint_usage = [u for u in self.usage_history if u.endpoint == endpoint]
            successful_calls = sum(
                1
                for u in endpoint_usage
                if u.status_code and 200 <= u.status_code < 300
            )
            failed_calls = sum(
                1
                for u in endpoint_usage
                if u.status_code and (u.status_code < 200 or u.status_code >= 300)
            )

            stats[endpoint] = {
                "total_calls": len(endpoint_usage),
                "successful_calls": successful_calls,
                "failed_calls": failed_calls,
                "data_transfer_mb": sum(
                    u.data_transfer_mb or 0 for u in endpoint_usage
                ),
            }
        return stats

    def get_error_stats(self) -> Dict[str, List[str]]:
        """Get error statistics by endpoint.

        Returns:
            Dict mapping endpoints to lists of error messages
        """
        errors = {}
        for usage in self.usage_history:
            if usage.error:
                if usage.endpoint not in errors:
                    errors[usage.endpoint] = []
                errors[usage.endpoint].append(usage.error)
        return errors

    def reset_tracking(self) -> None:
        """Reset all tracking data."""
        self.usage_history = []
        self._total_calls = 0
        self._total_cost = Decimal("0")
