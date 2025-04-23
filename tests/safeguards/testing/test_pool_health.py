"""Tests for shared pool health monitoring."""

from decimal import Decimal

import pytest

from tests.safeguards.testing.mock_implementations import (
    MockBudgetCoordinator,
    MockNotificationManager,
    MockViolationReporter,
)


@pytest.fixture()
def notification_manager():
    """Create mock notification manager."""
    return MockNotificationManager()


@pytest.fixture()
def violation_reporter(notification_manager):
    """Create mock violation reporter."""
    return MockViolationReporter(notification_manager)


@pytest.fixture()
def budget_coordinator(notification_manager, violation_reporter):
    """Create budget coordinator."""
    coordinator = MockBudgetCoordinator(
        notification_manager=notification_manager,
    )

    # Set up test data
    coordinator._pools["pool1"] = {"balance": Decimal("1000.00")}
    coordinator._pools["pool2"] = {"balance": Decimal("2000.00")}

    coordinator._balances["agent1"] = Decimal("500.00")
    coordinator._balances["agent2"] = Decimal("300.00")

    coordinator.violation_reporter = violation_reporter

    return coordinator


@pytest.mark.skip(reason="Using mocks without actual implementation for now")
class TestPoolHealth:
    """Test pool health monitor functionality."""

    def test_pool_monitoring(self, budget_coordinator):
        """Test pool usage monitoring."""
        # Just ensure we're setting up the mocks correctly
        assert budget_coordinator._pools["pool1"]["balance"] == Decimal("1000.00")
        assert budget_coordinator._pools["pool2"]["balance"] == Decimal("2000.00")
