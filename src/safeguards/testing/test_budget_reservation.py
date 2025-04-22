"""Tests for budget reservation system."""

import pytest
from datetime import datetime, timedelta
from decimal import Decimal
from uuid import UUID

from safeguards.core.budget_reservation import (
    ReservationManager,
    ReservationType,
    ReservationStatus,
    BudgetReservation,
)
from safeguards.testing.mock_implementations import (
    MockNotificationManager,
    MockViolationReporter,
)


@pytest.fixture
def notification_manager():
    """Create mock notification manager."""
    return MockNotificationManager()


@pytest.fixture
def violation_reporter(notification_manager):
    """Create mock violation reporter."""
    return MockViolationReporter(notification_manager)


@pytest.fixture
def reservation_manager(notification_manager, violation_reporter):
    """Create reservation manager with mock dependencies."""
    return ReservationManager(
        notification_manager=notification_manager,
        violation_reporter=violation_reporter,
    )


class TestBudgetReservation:
    """Test cases for budget reservation system."""

    def test_create_reservation(self, reservation_manager):
        """Test creating a basic reservation."""
        # Arrange
        pool_id = "test_pool"
        agent_id = "test_agent"
        amount = Decimal("100.00")

        # Act
        reservation = reservation_manager.create_reservation(
            pool_id=pool_id,
            agent_id=agent_id,
            amount=amount,
            reservation_type=ReservationType.SCHEDULED,
        )

        # Assert
        assert isinstance(reservation.id, UUID)
        assert reservation.pool_id == pool_id
        assert reservation.agent_id == agent_id
        assert reservation.amount == amount
        assert reservation.status == ReservationStatus.PENDING
        assert reservation.reservation_type == ReservationType.SCHEDULED

    def test_emergency_reservation(self, reservation_manager):
        """Test emergency reservation creation and handling."""
        # Arrange
        pool_id = "emergency_pool"
        agent_id = "critical_agent"
        amount = Decimal("500.00")
        reason = "Critical system maintenance"

        # Act
        reservation = reservation_manager.handle_emergency_request(
            pool_id=pool_id,
            agent_id=agent_id,
            amount=amount,
            reason=reason,
        )

        # Assert
        assert reservation is not None
        assert reservation.status == ReservationStatus.ACTIVE
        assert reservation.reservation_type == ReservationType.EMERGENCY
        assert reservation.priority == 100
        assert reservation.metadata["reason"] == reason

    def test_reservation_expiry(self, reservation_manager):
        """Test reservation expiration handling."""
        # Arrange
        pool_id = "test_pool"
        agent_id = "test_agent"
        amount = Decimal("100.00")
        expires_in = timedelta(seconds=1)  # Short expiry for testing

        # Create reservation
        reservation = reservation_manager.create_reservation(
            pool_id=pool_id,
            agent_id=agent_id,
            amount=amount,
            reservation_type=ReservationType.SCHEDULED,
            expires_in=expires_in,
        )
        reservation.status = ReservationStatus.ACTIVE

        # Wait for expiration
        import time

        time.sleep(1.1)  # Wait just over 1 second

        # Act
        reservation_manager.cleanup_expired_reservations()

        # Assert
        updated_reservation = reservation_manager.get_reservation(reservation.id)
        assert updated_reservation.status == ReservationStatus.EXPIRED

    def test_reservation_release(self, reservation_manager):
        """Test releasing a reservation."""
        # Arrange
        pool_id = "test_pool"
        agent_id = "test_agent"
        amount = Decimal("100.00")

        # Create and activate reservation
        reservation = reservation_manager.create_reservation(
            pool_id=pool_id,
            agent_id=agent_id,
            amount=amount,
            reservation_type=ReservationType.SCHEDULED,
        )
        reservation.status = ReservationStatus.ACTIVE

        # Act
        reservation_manager.release_reservation(reservation.id)

        # Assert
        assert reservation.status == ReservationStatus.RELEASED
        assert reservation_manager.get_pool_reserved_amount(pool_id) == Decimal("0")

    def test_multiple_reservations(self, reservation_manager):
        """Test handling multiple reservations for same pool."""
        # Arrange
        pool_id = "test_pool"
        reservations = [
            ("agent1", Decimal("100.00"), ReservationType.SCHEDULED),
            ("agent2", Decimal("200.00"), ReservationType.PRIORITY),
            ("agent3", Decimal("150.00"), ReservationType.MAINTENANCE),
        ]

        # Act
        created_reservations = []
        for agent_id, amount, res_type in reservations:
            reservation = reservation_manager.create_reservation(
                pool_id=pool_id,
                agent_id=agent_id,
                amount=amount,
                reservation_type=res_type,
            )
            created_reservations.append(reservation)

        # Assert
        total_reserved = reservation_manager.get_pool_reserved_amount(pool_id)
        assert total_reserved == Decimal("450.00")

        # Check individual type amounts
        scheduled_amount = reservation_manager.get_pool_reserved_amount(
            pool_id, ReservationType.SCHEDULED
        )
        assert scheduled_amount == Decimal("100.00")

    def test_emergency_fund_freeing(self, reservation_manager):
        """Test freeing up funds for emergency use."""
        # Arrange
        pool_id = "test_pool"

        # Create some existing reservations
        low_priority = reservation_manager.create_reservation(
            pool_id=pool_id,
            agent_id="agent1",
            amount=Decimal("100.00"),
            reservation_type=ReservationType.SCHEDULED,
            priority=1,
        )
        low_priority.status = ReservationStatus.ACTIVE

        high_priority = reservation_manager.create_reservation(
            pool_id=pool_id,
            agent_id="agent2",
            amount=Decimal("200.00"),
            reservation_type=ReservationType.PRIORITY,
            priority=5,
        )
        high_priority.status = ReservationStatus.ACTIVE

        # Act
        freed_amount = reservation_manager._free_up_emergency_funds(
            pool_id, Decimal("150.00")
        )

        # Assert
        assert freed_amount == Decimal(
            "100.00"
        )  # Should free up low priority reservation
        assert low_priority.status == ReservationStatus.RELEASED
        assert high_priority.status == ReservationStatus.ACTIVE

    def test_get_agent_reservations(self, reservation_manager):
        """Test retrieving agent reservations with filters."""
        # Arrange
        agent_id = "test_agent"
        pool_id = "test_pool"

        # Create various reservations
        active_res = reservation_manager.create_reservation(
            pool_id=pool_id,
            agent_id=agent_id,
            amount=Decimal("100.00"),
            reservation_type=ReservationType.SCHEDULED,
        )
        active_res.status = ReservationStatus.ACTIVE

        expired_res = reservation_manager.create_reservation(
            pool_id=pool_id,
            agent_id=agent_id,
            amount=Decimal("150.00"),
            reservation_type=ReservationType.MAINTENANCE,
        )
        expired_res.status = ReservationStatus.EXPIRED

        # Act
        all_reservations = reservation_manager.get_agent_reservations(agent_id)
        active_only = reservation_manager.get_agent_reservations(
            agent_id, status=ReservationStatus.ACTIVE
        )
        scheduled_only = reservation_manager.get_agent_reservations(
            agent_id, reservation_type=ReservationType.SCHEDULED
        )

        # Assert
        assert len(all_reservations) == 2
        assert len(active_only) == 1
        assert len(scheduled_only) == 1
        assert active_only[0].status == ReservationStatus.ACTIVE
        assert scheduled_only[0].reservation_type == ReservationType.SCHEDULED

    def test_invalid_reservation_operations(self, reservation_manager):
        """Test handling of invalid reservation operations."""
        # Test invalid amount
        with pytest.raises(ValueError):
            reservation_manager.create_reservation(
                pool_id="test_pool",
                agent_id="test_agent",
                amount=Decimal("-100.00"),
                reservation_type=ReservationType.SCHEDULED,
            )

        # Test releasing non-existent reservation
        with pytest.raises(ValueError):
            reservation_manager.release_reservation(
                UUID("00000000-0000-0000-0000-000000000000")
            )

        # Test releasing already released reservation
        reservation = reservation_manager.create_reservation(
            pool_id="test_pool",
            agent_id="test_agent",
            amount=Decimal("100.00"),
            reservation_type=ReservationType.SCHEDULED,
        )
        reservation.status = ReservationStatus.RELEASED

        with pytest.raises(ValueError):
            reservation_manager.release_reservation(reservation.id)
