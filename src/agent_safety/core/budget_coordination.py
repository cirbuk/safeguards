"""Budget coordination system for multi-agent resource management."""

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum, auto
from typing import Dict, List, Optional, Set
from uuid import UUID, uuid4

from .alert_types import Alert, AlertSeverity
from .budget_override import OverrideRequest, OverrideStatus, OverrideType
from .dynamic_budget import AgentBudgetProfile, AgentPriority, BudgetPool


class TransferStatus(Enum):
    """Status of a budget transfer."""

    PENDING = auto()  # Transfer is awaiting approval
    APPROVED = auto()  # Transfer is approved but not executed
    EXECUTED = auto()  # Transfer has been completed
    FAILED = auto()  # Transfer failed during execution
    REJECTED = auto()  # Transfer was rejected
    ROLLED_BACK = auto()  # Transfer was rolled back


class TransferType(Enum):
    """Types of budget transfers."""

    DIRECT = auto()  # Direct transfer between agents
    POOL_DEPOSIT = auto()  # Agent depositing to shared pool
    POOL_WITHDRAW = auto()  # Agent withdrawing from shared pool


@dataclass
class TransferRequest:
    """Budget transfer request details."""

    source_id: str  # Source agent/pool ID
    target_id: str  # Target agent/pool ID
    amount: Decimal
    transfer_type: TransferType
    justification: str
    requester: str
    request_id: UUID = field(default_factory=uuid4)
    priority: AgentPriority = AgentPriority.MEDIUM
    created_at: datetime = field(default_factory=datetime.now)
    status: TransferStatus = TransferStatus.PENDING
    executed_at: Optional[datetime] = None
    metadata: Dict = field(default_factory=dict)


@dataclass
class SharedPool:
    """Shared resource pool for multi-agent coordination."""

    pool_id: str
    total_budget: Decimal
    allocated_budget: Decimal = Decimal("0")
    min_balance: Decimal = Decimal("0")
    priority: AgentPriority = AgentPriority.MEDIUM
    agent_allocations: Dict[str, Decimal] = field(default_factory=dict)
    active_agents: Set[str] = field(default_factory=set)
    metadata: Dict = field(default_factory=dict)


class BudgetCoordinator:
    """Manages multi-agent budget coordination and transfers."""

    def __init__(self, notification_manager=None):
        """Initialize the budget coordinator.

        Args:
            notification_manager: For sending coordination-related alerts
        """
        self.notification_manager = notification_manager
        self._shared_pools: Dict[str, SharedPool] = {}
        self._transfer_requests: Dict[UUID, TransferRequest] = {}
        self._agent_pools: Dict[str, Set[str]] = {}  # Agent -> Pool memberships

    def create_shared_pool(
        self,
        pool_id: str,
        total_budget: Decimal,
        min_balance: Decimal = Decimal("0"),
        priority: AgentPriority = AgentPriority.MEDIUM,
        metadata: Optional[Dict] = None,
    ) -> SharedPool:
        """Create a new shared resource pool.

        Args:
            pool_id: Unique identifier for the pool
            total_budget: Total budget allocation for the pool
            min_balance: Minimum balance to maintain
            priority: Pool priority level
            metadata: Additional pool metadata

        Returns:
            Created shared pool

        Raises:
            ValueError: If pool ID already exists
        """
        if pool_id in self._shared_pools:
            raise ValueError(f"Pool {pool_id} already exists")

        pool = SharedPool(
            pool_id=pool_id,
            total_budget=total_budget,
            min_balance=min_balance,
            priority=priority,
            metadata=metadata or {},
        )
        self._shared_pools[pool_id] = pool
        return pool

    def add_agent_to_pool(
        self,
        agent_id: str,
        pool_id: str,
        initial_allocation: Optional[Decimal] = None,
    ) -> None:
        """Add an agent to a shared pool.

        Args:
            agent_id: Agent to add
            pool_id: Pool to add agent to
            initial_allocation: Initial budget allocation

        Raises:
            ValueError: If pool doesn't exist or insufficient pool budget
        """
        if pool_id not in self._shared_pools:
            raise ValueError(f"Pool {pool_id} does not exist")

        pool = self._shared_pools[pool_id]

        if initial_allocation:
            available = pool.total_budget - pool.allocated_budget
            if initial_allocation > available:
                raise ValueError(
                    f"Insufficient pool budget. Available: {available}, Requested: {initial_allocation}"
                )
            pool.agent_allocations[agent_id] = initial_allocation
            pool.allocated_budget += initial_allocation

        pool.active_agents.add(agent_id)

        if agent_id not in self._agent_pools:
            self._agent_pools[agent_id] = set()
        self._agent_pools[agent_id].add(pool_id)

    def remove_agent_from_pool(self, agent_id: str, pool_id: str) -> None:
        """Remove an agent from a shared pool.

        Args:
            agent_id: Agent to remove
            pool_id: Pool to remove from

        Raises:
            ValueError: If pool doesn't exist or agent not in pool
        """
        if pool_id not in self._shared_pools:
            raise ValueError(f"Pool {pool_id} does not exist")

        pool = self._shared_pools[pool_id]
        if agent_id not in pool.active_agents:
            raise ValueError(f"Agent {agent_id} not in pool {pool_id}")

        if agent_id in pool.agent_allocations:
            pool.allocated_budget -= pool.agent_allocations[agent_id]
            del pool.agent_allocations[agent_id]

        pool.active_agents.remove(agent_id)
        self._agent_pools[agent_id].remove(pool_id)
        if not self._agent_pools[agent_id]:
            del self._agent_pools[agent_id]

    def request_transfer(
        self,
        source_id: str,
        target_id: str,
        amount: Decimal,
        transfer_type: TransferType,
        justification: str,
        requester: str,
        priority: AgentPriority = AgentPriority.MEDIUM,
        metadata: Optional[Dict] = None,
    ) -> UUID:
        """Request a budget transfer.

        Args:
            source_id: Source agent/pool ID
            target_id: Target agent/pool ID
            amount: Amount to transfer
            transfer_type: Type of transfer
            justification: Reason for transfer
            requester: Identity of requester
            priority: Transfer priority
            metadata: Additional transfer metadata

        Returns:
            Transfer request ID

        Raises:
            ValueError: If invalid source/target or insufficient funds
        """
        # Validate source has sufficient funds
        if transfer_type == TransferType.DIRECT:
            if source_id not in self._agent_pools:
                raise ValueError(f"Source agent {source_id} not found")
            # Check source agent's total allocation across pools
            total_allocation = sum(
                self._shared_pools[pool_id].agent_allocations.get(
                    source_id, Decimal("0")
                )
                for pool_id in self._agent_pools[source_id]
            )
            if amount > total_allocation:
                raise ValueError(
                    f"Insufficient funds. Available: {total_allocation}, Requested: {amount}"
                )

        elif transfer_type == TransferType.POOL_WITHDRAW:
            if source_id not in self._shared_pools:
                raise ValueError(f"Source pool {source_id} not found")
            pool = self._shared_pools[source_id]
            available = pool.total_budget - pool.allocated_budget
            if amount > available:
                raise ValueError(
                    f"Insufficient pool funds. Available: {available}, Requested: {amount}"
                )

        # Create transfer request
        request = TransferRequest(
            source_id=source_id,
            target_id=target_id,
            amount=amount,
            transfer_type=transfer_type,
            justification=justification,
            requester=requester,
            priority=priority,
            metadata=metadata or {},
        )

        self._transfer_requests[request.request_id] = request

        # Send notification if configured
        if self.notification_manager:
            self._send_transfer_alert(
                request,
                "Budget Transfer Request",
                f"New budget transfer request from {source_id} to {target_id}",
                AlertSeverity.INFO,
            )

        return request.request_id

    def approve_transfer(
        self,
        request_id: UUID,
        approver: str,
        execute: bool = True,
    ) -> None:
        """Approve a budget transfer request.

        Args:
            request_id: Transfer to approve
            approver: Identity of approver
            execute: Whether to execute transfer immediately

        Raises:
            ValueError: If request not found or invalid status
        """
        if request_id not in self._transfer_requests:
            raise ValueError(f"Transfer request {request_id} not found")

        request = self._transfer_requests[request_id]
        if request.status != TransferStatus.PENDING:
            raise ValueError(f"Transfer {request_id} is not pending approval")

        request.status = TransferStatus.APPROVED
        request.metadata["approver"] = approver
        request.metadata["approved_at"] = datetime.now()

        if execute:
            self.execute_transfer(request_id)

    def execute_transfer(self, request_id: UUID) -> None:
        """Execute an approved transfer.

        Args:
            request_id: Transfer to execute

        Raises:
            ValueError: If request not found or not approved
        """
        if request_id not in self._transfer_requests:
            raise ValueError(f"Transfer request {request_id} not found")

        request = self._transfer_requests[request_id]
        if request.status != TransferStatus.APPROVED:
            raise ValueError(f"Transfer {request_id} is not approved")

        try:
            if request.transfer_type == TransferType.DIRECT:
                self._execute_direct_transfer(request)
            elif request.transfer_type == TransferType.POOL_DEPOSIT:
                self._execute_pool_deposit(request)
            elif request.transfer_type == TransferType.POOL_WITHDRAW:
                self._execute_pool_withdraw(request)

            request.status = TransferStatus.EXECUTED
            request.executed_at = datetime.now()

            if self.notification_manager:
                self._send_transfer_alert(
                    request,
                    "Budget Transfer Executed",
                    f"Transfer of {request.amount} completed successfully",
                    AlertSeverity.INFO,
                )

        except Exception as e:
            request.status = TransferStatus.FAILED
            request.metadata["error"] = str(e)

            if self.notification_manager:
                self._send_transfer_alert(
                    request,
                    "Budget Transfer Failed",
                    f"Transfer failed: {str(e)}",
                    AlertSeverity.HIGH,
                )

            raise

    def reject_transfer(self, request_id: UUID, rejector: str, reason: str) -> None:
        """Reject a transfer request.

        Args:
            request_id: Transfer to reject
            rejector: Identity of rejector
            reason: Reason for rejection

        Raises:
            ValueError: If request not found or invalid status
        """
        if request_id not in self._transfer_requests:
            raise ValueError(f"Transfer request {request_id} not found")

        request = self._transfer_requests[request_id]
        if request.status != TransferStatus.PENDING:
            raise ValueError(f"Transfer {request_id} is not pending approval")

        request.status = TransferStatus.REJECTED
        request.metadata["rejector"] = rejector
        request.metadata["rejection_reason"] = reason
        request.metadata["rejected_at"] = datetime.now()

        if self.notification_manager:
            self._send_transfer_alert(
                request,
                "Budget Transfer Rejected",
                f"Transfer rejected: {reason}",
                AlertSeverity.WARNING,
            )

    def _execute_direct_transfer(self, request: TransferRequest) -> None:
        """Execute a direct transfer between agents."""
        # Implementation will depend on how agent budgets are stored/managed
        pass

    def _execute_pool_deposit(self, request: TransferRequest) -> None:
        """Execute a deposit to a shared pool."""
        pool = self._shared_pools[request.target_id]
        pool.total_budget += request.amount

    def _execute_pool_withdraw(self, request: TransferRequest) -> None:
        """Execute a withdrawal from a shared pool."""
        pool = self._shared_pools[request.source_id]
        available = pool.total_budget - pool.allocated_budget

        if request.amount > available:
            raise ValueError(
                f"Insufficient pool funds. Available: {available}, Requested: {request.amount}"
            )

        pool.total_budget -= request.amount

    def _send_transfer_alert(
        self,
        request: TransferRequest,
        title: str,
        description: str,
        severity: AlertSeverity,
    ) -> None:
        """Send a transfer-related alert."""
        if not self.notification_manager:
            return

        alert = Alert(
            title=title,
            description=description,
            severity=severity,
            metadata={
                "request_id": str(request.request_id),
                "source_id": request.source_id,
                "target_id": request.target_id,
                "amount": str(request.amount),
                "transfer_type": request.transfer_type.name,
                "requester": request.requester,
                "status": request.status.name,
            },
        )
        self.notification_manager.send_alert(alert)
