"""Tests for the API tracking module."""

from datetime import datetime, timedelta
from decimal import Decimal
import pytest

from agent_safety.base.budget import BudgetPeriod
from agent_safety.budget.api_tracker import APITracker, APIUsage


@pytest.fixture
def api_tracker():
    """Create an API tracker instance for testing."""
    api_costs = {
        "search": Decimal("0.01"),  # $0.01 per call
        "embedding": Decimal("0.0004"),  # $0.0004 per call
        "image-gen": Decimal("0.02"),  # $0.02 per call
    }
    return APITracker(
        api_costs=api_costs,
        call_budget=10000,  # 10K calls
        cost_budget=Decimal("100.0"),  # $100
        period=BudgetPeriod.DAILY,
        data_transfer_cost_per_gb=Decimal("0.10"),  # $0.10 per GB
    )


def test_record_usage(api_tracker):
    """Test recording API usage."""
    # Test basic usage
    usage = api_tracker.record_usage(
        endpoint="search",
        data_transfer_mb=100,
        status_code=200,
    )

    assert isinstance(usage, APIUsage)
    assert usage.endpoint == "search"
    assert usage.data_transfer_mb == 100
    assert usage.status_code == 200
    assert usage.cost > Decimal("0")

    # Test with error
    usage = api_tracker.record_usage(
        endpoint="search",
        status_code=500,
        error="Internal server error",
    )
    assert usage.error == "Internal server error"

    # Test unknown endpoint
    with pytest.raises(ValueError):
        api_tracker.record_usage(endpoint="unknown-endpoint")


def test_get_usage_in_period(api_tracker):
    """Test getting usage for a specific period."""
    now = datetime.now()
    yesterday = now - timedelta(days=1)
    tomorrow = now + timedelta(days=1)

    # Record usage for different times
    api_tracker.record_usage(
        endpoint="search",
        timestamp=yesterday,
    )
    api_tracker.record_usage(
        endpoint="search",
        timestamp=now,
    )
    api_tracker.record_usage(
        endpoint="search",
        timestamp=tomorrow,
    )

    # Get usage for today
    today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    today_end = today_start + timedelta(days=1)
    usage = api_tracker.get_usage_in_period(today_start, today_end)

    assert len(usage) == 1  # Should only include today's usage


def test_get_total_calls(api_tracker):
    """Test getting total call counts."""
    # Record usage for different endpoints
    api_tracker.record_usage(endpoint="search")
    api_tracker.record_usage(endpoint="search")
    api_tracker.record_usage(endpoint="embedding")

    # Test total calls
    assert api_tracker.get_total_calls() == 3

    # Test endpoint-specific calls
    assert api_tracker.get_total_calls("search") == 2
    assert api_tracker.get_total_calls("embedding") == 1


def test_get_total_cost(api_tracker):
    """Test getting total costs."""
    # Record usage with different costs
    api_tracker.record_usage(
        endpoint="search",
        data_transfer_mb=100,
    )
    api_tracker.record_usage(
        endpoint="image-gen",
        data_transfer_mb=500,
    )

    # Test total cost
    total_cost = api_tracker.get_total_cost()
    assert total_cost > Decimal("0")

    # Test endpoint-specific costs
    search_cost = api_tracker.get_total_cost("search")
    image_cost = api_tracker.get_total_cost("image-gen")
    assert image_cost > search_cost  # Image generation should be more expensive


def test_check_budget_available(api_tracker):
    """Test budget availability checking."""
    # Test within call budget
    assert api_tracker.check_budget_available("search")

    # Fill up most of the budget
    for _ in range(9995):
        api_tracker.record_usage("search")

    # Test approaching call budget
    assert not api_tracker.check_budget_available("search", data_transfer_mb=1000)

    # Test within cost budget
    assert api_tracker.check_budget_available("embedding")

    # Test exceeding cost budget with large data transfer
    assert not api_tracker.check_budget_available(
        "image-gen", data_transfer_mb=1000000
    )  # 1TB


def test_get_usage_stats(api_tracker):
    """Test getting usage statistics."""
    # Record successful and failed calls
    api_tracker.record_usage(
        endpoint="search",
        status_code=200,
        data_transfer_mb=100,
    )
    api_tracker.record_usage(
        endpoint="search",
        status_code=500,
        error="Server error",
    )
    api_tracker.record_usage(
        endpoint="embedding",
        status_code=200,
        data_transfer_mb=50,
    )

    stats = api_tracker.get_usage_stats()

    # Check stats for each endpoint
    assert "search" in stats
    assert "embedding" in stats
    assert stats["search"]["total_calls"] == 2
    assert stats["search"]["successful_calls"] == 1
    assert stats["search"]["failed_calls"] == 1
    assert stats["search"]["data_transfer_mb"] == 100
    assert stats["embedding"]["total_calls"] == 1
    assert stats["embedding"]["successful_calls"] == 1


def test_get_error_stats(api_tracker):
    """Test getting error statistics."""
    # Record some errors
    api_tracker.record_usage(
        endpoint="search",
        status_code=500,
        error="Internal server error",
    )
    api_tracker.record_usage(
        endpoint="search",
        status_code=429,
        error="Rate limit exceeded",
    )

    errors = api_tracker.get_error_stats()

    assert "search" in errors
    assert len(errors["search"]) == 2
    assert "Internal server error" in errors["search"]
    assert "Rate limit exceeded" in errors["search"]


def test_reset_tracking(api_tracker):
    """Test resetting tracking data."""
    # Record some usage
    api_tracker.record_usage(
        endpoint="search",
        data_transfer_mb=100,
    )

    # Verify usage is recorded
    assert api_tracker.get_total_calls() > 0
    assert api_tracker.get_total_cost() > Decimal("0")

    # Reset tracking
    api_tracker.reset_tracking()

    # Verify everything is reset
    assert api_tracker.get_total_calls() == 0
    assert api_tracker.get_total_cost() == Decimal("0")
    assert len(api_tracker.usage_history) == 0
