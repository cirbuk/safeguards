"""Budget override system for FounderX.

This module provides functionality for manual budget overrides and emergency allocations:
- Manual override protocol
- Emergency resource allocation
- Override audit logging
- Override-specific alerts
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum, auto
from uuid import UUID, uuid4

from .alert_types import Alert, AlertSeverity
from .dynamic_budget import (
    AgentPriority,
    BudgetAdjustmentTrigger,
    DynamicBudgetManager,
)
from .notification_manager import NotificationManager


class OverrideType(Enum):
    """Types of budget overrides."""

    TEMPORARY = auto()  # Temporary increase for specific duration
    PERMANENT = auto()  # Permanent budget adjustment
    EMERGENCY = auto()  # Emergency resource allocation
    ONE_TIME = auto()  # One-time budget boost


class OverrideStatus(Enum):
    """Status of a budget override."""

    PENDING = auto()  # Awaiting approval
    ACTIVE = auto()  # Currently in effect
    EXPIRED = auto()  # Override duration ended
    CANCELLED = auto()  # Manually cancelled
    REJECTED = auto()  # Override request rejected


@dataclass
class OverrideRequest:
    """Budget override request details."""

    agent_id: str = field()
    override_type: OverrideType = field()
    requested_amount: Decimal = field()
    current_allocation: Decimal = field()
    justification: str = field()
    requester: str = field()
    request_id: UUID = field(default_factory=uuid4)
    duration: timedelta | None = None
    priority_override: AgentPriority | None = None
    created_at: datetime = field(default_factory=datetime.now)
    approved_by: str | None = None
    approved_at: datetime | None = None
    status: OverrideStatus = field(default=OverrideStatus.PENDING)
    metadata: dict = field(default_factory=dict)


@dataclass
class OverrideAuditLog:
    """Audit log entry for budget overrides."""

    override_id: UUID
    timestamp: datetime
    action: str
    actor: str
    details: str
    previous_state: dict | None = None
    new_state: dict | None = None
    metadata: dict = field(default_factory=dict)


class BudgetOverrideManager:
    """Manages budget overrides and emergency allocations."""

    def __init__(
        self,
        budget_manager: DynamicBudgetManager,
        notification_manager: NotificationManager,
        auto_approve_threshold: Decimal | None = None,
        emergency_reserve_ratio: float = 0.2,
        max_override_duration: timedelta = timedelta(days=7),
    ):
        """Initialize the override manager.

        Args:
            budget_manager: Dynamic budget manager instance
            notification_manager: For sending override-related alerts
            auto_approve_threshold: Amount below which overrides are auto-approved
            emergency_reserve_ratio: Ratio of pool budget reserved for emergencies
            max_override_duration: Maximum duration for temporary overrides
        """
        self.budget_manager = budget_manager
        self.notification_manager = notification_manager
        self.auto_approve_threshold = auto_approve_threshold
        self.emergency_reserve_ratio = emergency_reserve_ratio
        self.max_override_duration = max_override_duration

        # Track override requests and audit logs
        self._override_requests: dict[UUID, OverrideRequest] = {}
        self._audit_logs: list[OverrideAuditLog] = []

        # Track active overrides by agent
        self._active_overrides: dict[str, set[UUID]] = {}

    def request_override(
        self,
        agent_id: str,
        requested_amount: Decimal,
        override_type: OverrideType,
        justification: str,
        requester: str,
        duration: timedelta | None = None,
        priority_override: AgentPriority | None = None,
        metadata: dict | None = None,
    ) -> UUID:
        """Request a budget override.

        Args:
            agent_id: Agent requesting override
            requested_amount: Desired budget amount
            override_type: Type of override requested
            justification: Reason for override
            requester: Identity of the requester
            duration: Duration for temporary overrides
            priority_override: Optional priority change
            metadata: Additional override metadata

        Returns:
            UUID of the created override request
        """
        # Validate agent exists
        if agent_id not in self.budget_manager.agent_profiles:
            msg = f"Unknown agent {agent_id}"
            raise ValueError(msg)

        # Get current allocation
        profile = self.budget_manager.agent_profiles[agent_id]
        current_allocation = profile.current_allocation

        # Validate duration for temporary overrides
        if override_type == OverrideType.TEMPORARY:
            if not duration:
                msg = "Duration required for temporary overrides"
                raise ValueError(msg)
            if duration > self.max_override_duration:
                msg = f"Override duration exceeds maximum of {self.max_override_duration}"
                raise ValueError(
                    msg,
                )

        # Create override request
        request = OverrideRequest(
            agent_id=agent_id,
            override_type=override_type,
            requested_amount=requested_amount,
            current_allocation=current_allocation,
            justification=justification,
            requester=requester,
            duration=duration,
            priority_override=priority_override,
            metadata=metadata or {},
        )

        # Store request
        self._override_requests[request.request_id] = request

        # Log creation
        self._add_audit_log(
            override_id=request.request_id,
            action="CREATE",
            actor=requester,
            details=f"Override request created for {agent_id}",
            new_state=self._request_to_dict(request),
        )

        # Check for auto-approval
        if self._should_auto_approve(request):
            self.approve_override(
                override_id=request.request_id,
                approver="SYSTEM",
                auto_approved=True,
            )
        else:
            # Send notification for manual approval
            self._send_approval_request_alert(request)

        return request.request_id

    def approve_override(
        self,
        override_id: UUID,
        approver: str,
        auto_approved: bool = False,
    ) -> None:
        """Approve a budget override request.

        Args:
            override_id: ID of override to approve
            approver: Identity of the approver
            auto_approved: Whether this was auto-approved
        """
        if override_id not in self._override_requests:
            msg = f"Unknown override request {override_id}"
            raise ValueError(msg)

        request = self._override_requests[override_id]
        if request.status != OverrideStatus.PENDING:
            msg = f"Override {override_id} is not pending approval"
            raise ValueError(msg)

        # Record approval
        old_state = self._request_to_dict(request)
        request.approved_by = approver
        request.approved_at = datetime.now()
        request.status = OverrideStatus.ACTIVE

        # Track active override
        if request.agent_id not in self._active_overrides:
            self._active_overrides[request.agent_id] = set()
        self._active_overrides[request.agent_id].add(override_id)

        # Apply override
        self._apply_override(request)

        # Log approval
        self._add_audit_log(
            override_id=override_id,
            action="APPROVE",
            actor=approver,
            details=f"Override approved{' (auto)' if auto_approved else ''}",
            previous_state=old_state,
            new_state=self._request_to_dict(request),
        )

        # Send notification
        self._send_override_alert(
            request,
            "Budget Override Approved",
            f"Override for {request.agent_id} has been {'auto-' if auto_approved else ''}approved",
            AlertSeverity.INFO,
        )

    def reject_override(self, override_id: UUID, rejector: str, reason: str) -> None:
        """Reject a budget override request.

        Args:
            override_id: ID of override to reject
            rejector: Identity of the rejector
            reason: Reason for rejection
        """
        if override_id not in self._override_requests:
            msg = f"Unknown override request {override_id}"
            raise ValueError(msg)

        request = self._override_requests[override_id]
        if request.status != OverrideStatus.PENDING:
            msg = f"Override {override_id} is not pending approval"
            raise ValueError(msg)

        # Update status
        old_state = self._request_to_dict(request)
        request.status = OverrideStatus.REJECTED
        request.metadata["rejection_reason"] = reason

        # Log rejection
        self._add_audit_log(
            override_id=override_id,
            action="REJECT",
            actor=rejector,
            details=f"Override rejected: {reason}",
            previous_state=old_state,
            new_state=self._request_to_dict(request),
        )

        # Send notification
        self._send_override_alert(
            request,
            "Budget Override Rejected",
            f"Override for {request.agent_id} has been rejected: {reason}",
            AlertSeverity.WARNING,
        )

    def cancel_override(self, override_id: UUID, canceller: str, reason: str) -> None:
        """Cancel an active override.

        Args:
            override_id: ID of override to cancel
            canceller: Identity of the canceller
            reason: Reason for cancellation
        """
        if override_id not in self._override_requests:
            msg = f"Unknown override request {override_id}"
            raise ValueError(msg)

        request = self._override_requests[override_id]
        if request.status != OverrideStatus.ACTIVE:
            msg = f"Override {override_id} is not active"
            raise ValueError(msg)

        # Update status
        old_state = self._request_to_dict(request)
        request.status = OverrideStatus.CANCELLED
        request.metadata["cancellation_reason"] = reason

        # Remove from active overrides
        self._active_overrides[request.agent_id].remove(override_id)
        if not self._active_overrides[request.agent_id]:
            del self._active_overrides[request.agent_id]

        # Revert override
        self._revert_override(request)

        # Log cancellation
        self._add_audit_log(
            override_id=override_id,
            action="CANCEL",
            actor=canceller,
            details=f"Override cancelled: {reason}",
            previous_state=old_state,
            new_state=self._request_to_dict(request),
        )

        # Send notification
        self._send_override_alert(
            request,
            "Budget Override Cancelled",
            f"Override for {request.agent_id} has been cancelled: {reason}",
            AlertSeverity.INFO,
        )

    def get_override_status(self, override_id: UUID) -> OverrideRequest:
        """Get current status of an override request.

        Args:
            override_id: ID of override to check

        Returns:
            Current override request state
        """
        if override_id not in self._override_requests:
            msg = f"Unknown override request {override_id}"
            raise ValueError(msg)
        return self._override_requests[override_id]

    def get_agent_overrides(
        self,
        agent_id: str,
        include_inactive: bool = False,
    ) -> list[OverrideRequest]:
        """Get all overrides for an agent.

        Args:
            agent_id: Agent to check
            include_inactive: Whether to include non-active overrides

        Returns:
            List of override requests for the agent
        """
        return [
            req
            for req in self._override_requests.values()
            if req.agent_id == agent_id
            and (include_inactive or req.status == OverrideStatus.ACTIVE)
        ]

    def get_override_audit_logs(self, override_id: UUID) -> list[OverrideAuditLog]:
        """Get audit logs for an override.

        Args:
            override_id: Override to get logs for

        Returns:
            List of audit log entries
        """
        return [log for log in self._audit_logs if log.override_id == override_id]

    def _should_auto_approve(self, request: OverrideRequest) -> bool:
        """Check if an override should be auto-approved.

        Args:
            request: Override request to check

        Returns:
            True if request should be auto-approved
        """
        if not self.auto_approve_threshold:
            return False

        # Only auto-approve increases up to threshold
        if request.requested_amount <= request.current_allocation:
            return False

        increase_amount = request.requested_amount - request.current_allocation
        return increase_amount <= self.auto_approve_threshold

    def _apply_override(self, request: OverrideRequest) -> None:
        """Apply an approved override.

        Args:
            request: Override request to apply
        """
        # Get agent profile
        profile = self.budget_manager.agent_profiles[request.agent_id]

        # Apply changes based on override type
        if request.override_type == OverrideType.EMERGENCY:
            # Use emergency reserve for allocation
            pool = self.budget_manager.budget_pools[
                self.budget_manager.agent_pool_mapping[request.agent_id]
            ]
            emergency_budget = pool.total_budget * Decimal(
                str(self.emergency_reserve_ratio),
            )
            profile.current_allocation = min(request.requested_amount, emergency_budget)
        else:
            # Standard override
            profile.current_allocation = request.requested_amount

        # Apply priority override if specified
        if request.priority_override:
            profile.priority = request.priority_override

        # Update pool allocations
        pool_id = self.budget_manager.agent_pool_mapping[request.agent_id]
        pool = self.budget_manager.budget_pools[pool_id]
        pool.agent_allocations[request.agent_id] = profile.current_allocation
        pool.allocated_budget = sum(pool.agent_allocations.values())

    def _revert_override(self, request: OverrideRequest) -> None:
        """Revert an override's changes.

        Args:
            request: Override request to revert
        """
        # Reset to original allocation
        profile = self.budget_manager.agent_profiles[request.agent_id]
        profile.current_allocation = request.current_allocation

        # Update pool allocations
        pool_id = self.budget_manager.agent_pool_mapping[request.agent_id]
        pool = self.budget_manager.budget_pools[pool_id]
        pool.agent_allocations[request.agent_id] = profile.current_allocation
        pool.allocated_budget = sum(pool.agent_allocations.values())

        # Trigger reallocation to rebalance
        self.budget_manager._reallocate_pool_budget(
            pool,
            BudgetAdjustmentTrigger.MANUAL,
        )

    def _add_audit_log(
        self,
        override_id: UUID,
        action: str,
        actor: str,
        details: str,
        previous_state: dict | None = None,
        new_state: dict | None = None,
        metadata: dict | None = None,
    ) -> None:
        """Add an audit log entry.

        Args:
            override_id: Override being logged
            action: Action being performed
            actor: Identity performing the action
            details: Description of the action
            previous_state: Optional previous override state
            new_state: Optional new override state
            metadata: Optional additional metadata
        """
        log = OverrideAuditLog(
            override_id=override_id,
            timestamp=datetime.now(),
            action=action,
            actor=actor,
            details=details,
            previous_state=previous_state,
            new_state=new_state,
            metadata=metadata or {},
        )
        self._audit_logs.append(log)

    def _request_to_dict(self, request: OverrideRequest) -> dict:
        """Convert override request to dictionary for logging.

        Args:
            request: Request to convert

        Returns:
            Dictionary representation of request
        """
        return {
            "request_id": str(request.request_id),
            "agent_id": request.agent_id,
            "override_type": request.override_type.name,
            "requested_amount": str(request.requested_amount),
            "current_allocation": str(request.current_allocation),
            "justification": request.justification,
            "requester": request.requester,
            "duration": str(request.duration) if request.duration else None,
            "priority_override": (
                request.priority_override.name if request.priority_override else None
            ),
            "created_at": request.created_at.isoformat(),
            "approved_by": request.approved_by,
            "approved_at": (request.approved_at.isoformat() if request.approved_at else None),
            "status": request.status.name,
            "metadata": request.metadata,
        }

    def _send_override_alert(
        self,
        request: OverrideRequest,
        title: str,
        description: str,
        severity: AlertSeverity,
        metadata: dict | None = None,
    ) -> None:
        """Send an override-related alert.

        Args:
            request: Related override request
            title: Alert title
            description: Alert description
            severity: Alert severity
            metadata: Optional additional metadata
        """
        alert_metadata = {
            "override_id": str(request.request_id),
            "agent_id": request.agent_id,
            "override_type": request.override_type.name,
            "requested_amount": str(request.requested_amount),
            "current_allocation": str(request.current_allocation),
            "requester": request.requester,
            "status": request.status.name,
        }
        if metadata:
            alert_metadata.update(metadata)

        alert = Alert(
            title=title,
            description=description,
            severity=severity,
            metadata=alert_metadata,
        )
        self.notification_manager.send_alert(alert)

    def _send_approval_request_alert(self, request: OverrideRequest) -> None:
        """Send alert for override approval request.

        Args:
            request: Override request needing approval
        """
        description = (
            f"Override requested for {request.agent_id}\n"
            f"Type: {request.override_type.name}\n"
            f"Current allocation: {request.current_allocation}\n"
            f"Requested amount: {request.requested_amount}\n"
            f"Justification: {request.justification}\n"
            f"Requester: {request.requester}"
        )
        if request.duration:
            description += f"\nDuration: {request.duration}"
        if request.priority_override:
            description += f"\nPriority override: {request.priority_override.name}"

        self._send_override_alert(
            request=request,
            title="Budget Override Approval Required",
            description=description,
            severity=AlertSeverity.ERROR,
        )
