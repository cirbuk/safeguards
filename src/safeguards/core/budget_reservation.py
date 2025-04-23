"""Budget reservation system for managing reserved and emergency funds.

This module provides functionality for:
- Managing reserved budget pools
- Handling emergency budget requests
- Tracking reservation status
- Managing reservation priorities
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum, auto
from uuid import UUID, uuid4

from safeguards.core.notification_manager import NotificationManager
from safeguards.monitoring.violation_reporter import (
    ViolationReporter,
    ViolationSeverity,
    ViolationType,
)


class ReservationType(Enum):
    """Types of budget reservations."""

    EMERGENCY = auto()  # Emergency fund reservation
    SCHEDULED = auto()  # Scheduled task reservation
    PRIORITY = auto()  # High-priority task reservation
    MAINTENANCE = auto()  # System maintenance reservation


class ReservationStatus(Enum):
    """Status of a budget reservation."""

    PENDING = auto()  # Reservation requested but not confirmed
    ACTIVE = auto()  # Reservation is currently active
    RELEASED = auto()  # Reservation was released back to pool
    CONSUMED = auto()  # Reservation was used
    EXPIRED = auto()  # Reservation expired without use
    DENIED = auto()  # Reservation request was denied


@dataclass
class BudgetReservation:
    """Represents a budget reservation."""

    pool_id: str
    agent_id: str
    amount: Decimal
    reservation_type: ReservationType
    id: UUID = field(default_factory=uuid4)
    status: ReservationStatus = ReservationStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    expires_at: datetime | None = None
    priority: int = 0
    metadata: dict = field(default_factory=dict)


class ReservationManager:
    """Manages budget reservations and emergency allocations."""

    def __init__(
        self,
        notification_manager: NotificationManager,
        violation_reporter: ViolationReporter,
        emergency_threshold: float = 0.2,  # 20% of pool for emergencies
        scheduled_threshold: float = 0.3,  # 30% of pool for scheduled tasks
        default_expiry: timedelta = timedelta(hours=24),
    ):
        """Initialize the reservation manager.

        Args:
            notification_manager: For sending reservation-related alerts
            violation_reporter: For reporting reservation violations
            emergency_threshold: Percentage of pool to reserve for emergencies
            scheduled_threshold: Percentage of pool to reserve for scheduled tasks
            default_expiry: Default expiration time for reservations
        """
        self.notification_manager = notification_manager
        self.violation_reporter = violation_reporter
        self.emergency_threshold = emergency_threshold
        self.scheduled_threshold = scheduled_threshold
        self.default_expiry = default_expiry

        # Track reservations
        self._reservations: dict[UUID, BudgetReservation] = {}
        self._pool_reservations: dict[str, set[UUID]] = {}
        self._agent_reservations: dict[str, set[UUID]] = {}

        # Track reserved amounts by pool and type
        self._pool_reserved: dict[str, dict[ReservationType, Decimal]] = {}

    def create_reservation(
        self,
        pool_id: str,
        agent_id: str,
        amount: Decimal,
        reservation_type: ReservationType,
        priority: int = 0,
        expires_in: timedelta | None = None,
        metadata: dict | None = None,
    ) -> BudgetReservation:
        """Create a new budget reservation.

        Args:
            pool_id: Pool to reserve from
            agent_id: Agent requesting reservation
            amount: Amount to reserve
            reservation_type: Type of reservation
            priority: Reservation priority (higher = more important)
            expires_in: Optional custom expiration time
            metadata: Optional metadata for the reservation

        Returns:
            Created reservation

        Raises:
            ValueError: If reservation cannot be created
        """
        # Validate reservation amount
        if amount <= 0:
            msg = "Reservation amount must be positive"
            raise ValueError(msg)

        # Create reservation
        reservation = BudgetReservation(
            pool_id=pool_id,
            agent_id=agent_id,
            amount=amount,
            reservation_type=reservation_type,
            priority=priority,
            expires_at=datetime.now() + (expires_in or self.default_expiry),
            metadata=metadata or {},
        )

        # Track reservation
        self._reservations[reservation.id] = reservation
        self._pool_reservations.setdefault(pool_id, set()).add(reservation.id)
        self._agent_reservations.setdefault(agent_id, set()).add(reservation.id)

        # Update pool reserved amounts
        pool_reserved = self._pool_reserved.setdefault(pool_id, {})
        pool_reserved[reservation_type] = pool_reserved.get(reservation_type, Decimal(0)) + amount

        return reservation

    def get_reservation(self, reservation_id: UUID) -> BudgetReservation | None:
        """Get a reservation by ID.

        Args:
            reservation_id: Reservation ID

        Returns:
            Reservation if found, None otherwise
        """
        return self._reservations.get(reservation_id)

    def release_reservation(self, reservation_id: UUID) -> None:
        """Release a reservation back to its pool.

        Args:
            reservation_id: Reservation to release

        Raises:
            ValueError: If reservation not found or already released
        """
        reservation = self._reservations.get(reservation_id)
        if not reservation:
            msg = f"Reservation {reservation_id} not found"
            raise ValueError(msg)

        if reservation.status != ReservationStatus.ACTIVE:
            msg = f"Reservation {reservation_id} is not active"
            raise ValueError(msg)

        # Update status
        reservation.status = ReservationStatus.RELEASED

        # Update pool reserved amounts
        pool_reserved = self._pool_reserved[reservation.pool_id]
        pool_reserved[reservation.reservation_type] -= reservation.amount

        # Send notification
        self.notification_manager.send_notification(
            agent_id=reservation.agent_id,
            message=f"Released reservation {reservation_id} for {reservation.amount}",
            severity="INFO",
        )

    def consume_reservation(self, reservation_id: UUID) -> None:
        """Mark a reservation as consumed.

        Args:
            reservation_id: Reservation to consume

        Raises:
            ValueError: If reservation not found or not active
        """
        reservation = self._reservations.get(reservation_id)
        if not reservation:
            msg = f"Reservation {reservation_id} not found"
            raise ValueError(msg)

        if reservation.status != ReservationStatus.ACTIVE:
            msg = f"Reservation {reservation_id} is not active"
            raise ValueError(msg)

        # Update status
        reservation.status = ReservationStatus.CONSUMED

        # Update pool reserved amounts
        pool_reserved = self._pool_reserved[reservation.pool_id]
        pool_reserved[reservation.reservation_type] -= reservation.amount

    def get_pool_reserved_amount(
        self,
        pool_id: str,
        reservation_type: ReservationType | None = None,
    ) -> Decimal:
        """Get total reserved amount for a pool.

        Args:
            pool_id: Pool to check
            reservation_type: Optional specific type to check

        Returns:
            Total reserved amount
        """
        pool_reserved = self._pool_reserved.get(pool_id, {})
        if reservation_type:
            return pool_reserved.get(reservation_type, Decimal(0))
        return sum(pool_reserved.values())

    def get_agent_reservations(
        self,
        agent_id: str,
        status: ReservationStatus | None = None,
        reservation_type: ReservationType | None = None,
    ) -> list[BudgetReservation]:
        """Get reservations for an agent.

        Args:
            agent_id: Agent to check
            status: Optional status filter
            reservation_type: Optional type filter

        Returns:
            List of matching reservations
        """
        if agent_id not in self._agent_reservations:
            return []

        reservations = [self._reservations[rid] for rid in self._agent_reservations[agent_id]]

        if status:
            reservations = [r for r in reservations if r.status == status]
        if reservation_type:
            reservations = [r for r in reservations if r.reservation_type == reservation_type]

        return reservations

    def cleanup_expired_reservations(self) -> None:
        """Clean up expired reservations."""
        now = datetime.now()
        for reservation in self._reservations.values():
            if (
                reservation.status == ReservationStatus.ACTIVE
                and reservation.expires_at
                and now >= reservation.expires_at
            ):
                # Mark as expired
                reservation.status = ReservationStatus.EXPIRED

                # Update pool reserved amounts
                pool_reserved = self._pool_reserved[reservation.pool_id]
                pool_reserved[reservation.reservation_type] -= reservation.amount

                # Send notification
                self.notification_manager.send_notification(
                    agent_id=reservation.agent_id,
                    message=f"Reservation {reservation.id} expired",
                    severity="WARNING",
                )

    def handle_emergency_request(
        self,
        pool_id: str,
        agent_id: str,
        amount: Decimal,
        reason: str,
    ) -> BudgetReservation | None:
        """Handle an emergency budget request.

        This method will:
        1. Check if emergency funds are available
        2. If not, attempt to free up funds from lower priority reservations
        3. Create an emergency reservation if possible

        Args:
            pool_id: Pool to request from
            agent_id: Agent making request
            amount: Amount needed
            reason: Reason for emergency request

        Returns:
            Created reservation if successful, None if request cannot be satisfied
        """
        try:
            # Create emergency reservation
            reservation = self.create_reservation(
                pool_id=pool_id,
                agent_id=agent_id,
                amount=amount,
                reservation_type=ReservationType.EMERGENCY,
                priority=100,  # Highest priority
                metadata={"reason": reason},
            )

            # Activate immediately
            reservation.status = ReservationStatus.ACTIVE

            # Send alert
            self.notification_manager.send_notification(
                agent_id=agent_id,
                message=f"Emergency reservation created: {reason}",
                severity="HIGH",
            )

            return reservation

        except ValueError:
            # Try to free up funds
            freed_amount = self._free_up_emergency_funds(pool_id, amount)
            if freed_amount >= amount:
                return self.handle_emergency_request(
                    pool_id=pool_id,
                    agent_id=agent_id,
                    amount=amount,
                    reason=reason,
                )

            # Report violation if cannot satisfy
            self.violation_reporter.report_violation(
                type=ViolationType.OVERSPEND,
                severity=ViolationSeverity.HIGH,
                agent_id=agent_id,
                description=f"Could not satisfy emergency request for {amount}",
            )

            return None

    def _free_up_emergency_funds(self, pool_id: str, needed_amount: Decimal) -> Decimal:
        """Attempt to free up funds for emergency use.

        Args:
            pool_id: Pool to free up funds from
            needed_amount: Amount needed

        Returns:
            Amount freed up
        """
        freed_amount = Decimal(0)

        # Get all active reservations for pool, sorted by priority (lowest first)
        pool_reservations = sorted(
            [
                r
                for r in self._reservations.values()
                if r.pool_id == pool_id
                and r.status == ReservationStatus.ACTIVE
                and r.reservation_type != ReservationType.EMERGENCY
            ],
            key=lambda x: x.priority,
        )

        # Try to free up funds from lowest priority reservations
        for reservation in pool_reservations:
            if freed_amount >= needed_amount:
                break

            # Release this reservation
            self.release_reservation(reservation.id)
            freed_amount += reservation.amount

            # Send notification
            self.notification_manager.send_notification(
                agent_id=reservation.agent_id,
                message=f"Reservation {reservation.id} released for emergency funds",
                severity="WARNING",
            )

        return freed_amount
