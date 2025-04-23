"""Budget coordination system for multi-agent resource management."""

import copy
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum, auto
from uuid import UUID, uuid4

from safeguards.core.alert_types import Alert, AlertSeverity
from safeguards.core.dynamic_budget import (
    AgentPriority,
    BudgetPool,
)
from safeguards.core.transaction import (
    Transactional,
    TransactionContext,
    TransactionError,
    TransactionManager,
    TransactionStatus,
)
from safeguards.monitoring.budget_monitor import BudgetMonitor
from safeguards.monitoring.violation_reporter import (
    ViolationContext,
    ViolationReporter,
    ViolationSeverity,
    ViolationType,
)

from ..exceptions import AgentSafetyError, BudgetError
from ..types import SafetyAlert
from ..types.agent import Agent
from ..types.default_agent import DefaultAgent


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
    ALLOCATION = auto()  # Initial allocation from pool
    REALLOCATION = auto()  # Moving budget between agents
    RETURN = auto()  # Return unused budget to pool


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
    executed_at: datetime | None = None
    metadata: dict = field(default_factory=dict)
    notes: str = ""


@dataclass
class SharedPool:
    """Shared resource pool for multi-agent coordination."""

    pool_id: str
    total_budget: Decimal
    allocated_budget: Decimal = Decimal("0")
    min_balance: Decimal = Decimal("0")
    priority: AgentPriority = AgentPriority.MEDIUM
    agent_allocations: dict[str, Decimal] = field(default_factory=dict)
    active_agents: set[str] = field(default_factory=set)
    metadata: dict = field(default_factory=dict)


@dataclass
class BudgetState:
    """Snapshot of budget state."""

    balances: dict[str, Decimal]
    pending_transfers: dict[UUID, TransferRequest]


class BudgetCoordinator(Transactional[BudgetState]):
    """Manages multi-agent budget coordination and transfers."""

    def __init__(
        self,
        notification_manager=None,
        initial_pool_budget: Decimal = Decimal("0"),
    ):
        """Initialize the budget coordinator.

        Args:
            notification_manager: For sending coordination-related alerts
            initial_pool_budget: Initial budget in the pool
        """
        self.notification_manager = notification_manager
        self.violation_reporter = ViolationReporter(notification_manager)
        self._shared_pools: dict[str, SharedPool] = {}
        self._transfer_requests: dict[UUID, TransferRequest] = {}
        self._agent_pools: dict[str, set[str]] = {}  # Agent -> Pool memberships
        self._balances: dict[str, Decimal] = {
            f"pool_{len(self._shared_pools) + 1}": initial_pool_budget,
        }
        self._pending_transfers: dict[UUID, TransferRequest] = {}
        self._registered_agents: set[str] = set()
        self._transaction_manager = TransactionManager()
        self._pools: dict[str, BudgetPool] = {}
        self._agents: dict[str, Agent] = {}
        self._agent_budgets: dict[str, Decimal] = {}
        self._agent_pools: dict[str, str] = {}  # agent_id -> pool_id
        self._initial_budgets: dict[str, Decimal] = {}

        # Initialize budget monitor
        self.budget_monitor = BudgetMonitor(notification_manager)

    def begin_transaction(self) -> TransactionContext:
        """Begin a new transaction for budget operations.

        Creates a transaction context and captures the current state for potential rollback.

        Returns:
            New transaction context
        """
        ctx = TransactionContext()
        ctx.changes["snapshot"] = self.get_snapshot()

        # Log the transaction start
        if self.notification_manager:
            self.notification_manager.send_alert(
                Alert(
                    title="Budget Transaction Started",
                    description=f"Transaction {ctx.transaction_id} initiated",
                    severity=AlertSeverity.INFO,
                    metadata={"transaction_id": str(ctx.transaction_id)},
                ),
            )

        return ctx

    def commit_transaction(self, ctx: TransactionContext) -> None:
        """Commit a budget transaction.

        Finalizes changes made during the transaction.

        Args:
            ctx: Transaction context to commit

        Raises:
            TransactionError: If commit fails
        """
        # Mark the transaction as committed
        ctx.status = TransactionStatus.COMMITTED

        # Log the transaction commit
        if self.notification_manager:
            self.notification_manager.send_alert(
                Alert(
                    title="Budget Transaction Committed",
                    description=f"Transaction {ctx.transaction_id} committed successfully",
                    severity=AlertSeverity.INFO,
                    metadata={
                        "transaction_id": str(ctx.transaction_id),
                        "changes": {k: str(v) for k, v in ctx.changes.items() if k != "snapshot"},
                    },
                ),
            )

    def rollback_transaction(self, ctx: TransactionContext) -> None:
        """Rollback a budget transaction.

        Reverts all changes made during the transaction.

        Args:
            ctx: Transaction context to rollback

        Raises:
            TransactionError: If rollback fails
        """
        try:
            # Restore the snapshot
            if "snapshot" in ctx.changes:
                self.restore_snapshot(ctx.changes["snapshot"])

            # Mark the transaction as rolled back
            ctx.status = TransactionStatus.ROLLED_BACK

            # Log the transaction rollback
            if self.notification_manager:
                self.notification_manager.send_alert(
                    Alert(
                        title="Budget Transaction Rolled Back",
                        description=f"Transaction {ctx.transaction_id} rolled back",
                        severity=AlertSeverity.WARNING,
                        metadata={"transaction_id": str(ctx.transaction_id)},
                    ),
                )
        except Exception as e:
            ctx.status = TransactionStatus.FAILED
            msg = f"Rollback failed: {e!s}"
            raise TransactionError(msg)

    def get_snapshot(self) -> BudgetState:
        """Get a snapshot of the current budget state.

        Creates a complete copy of the current budget state for potential rollback.

        Returns:
            Snapshot of current state
        """
        return BudgetState(
            balances=self._balances.copy(),
            pending_transfers={k: copy.deepcopy(v) for k, v in self._pending_transfers.items()},
        )

    def restore_snapshot(self, snapshot: BudgetState) -> None:
        """Restore state from a snapshot.

        Args:
            snapshot: State snapshot to restore

        Raises:
            TransactionError: If restore fails
        """
        try:
            self._balances = snapshot.balances.copy()
            self._pending_transfers = {
                k: copy.deepcopy(v) for k, v in snapshot.pending_transfers.items()
            }
        except Exception as e:
            msg = f"Failed to restore state from snapshot: {e!s}"
            raise TransactionError(msg)

    def create_shared_pool(
        self,
        pool_id: str,
        total_budget: Decimal,
        min_balance: Decimal = Decimal("0"),
        priority: AgentPriority = AgentPriority.MEDIUM,
        metadata: dict | None = None,
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
            msg = f"Pool {pool_id} already exists"
            raise ValueError(msg)

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
        initial_allocation: Decimal | None = None,
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
            msg = f"Pool {pool_id} does not exist"
            raise ValueError(msg)

        pool = self._shared_pools[pool_id]

        if initial_allocation:
            available = pool.total_budget - pool.allocated_budget
            if initial_allocation > available:
                msg = f"Insufficient pool budget. Available: {available}, Requested: {initial_allocation}"
                raise ValueError(
                    msg,
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
            msg = f"Pool {pool_id} does not exist"
            raise ValueError(msg)

        pool = self._shared_pools[pool_id]
        if agent_id not in pool.active_agents:
            msg = f"Agent {agent_id} not in pool {pool_id}"
            raise ValueError(msg)

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
        metadata: dict | None = None,
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
                msg = f"Source agent {source_id} not found"
                raise ValueError(msg)
            # Check source agent's total allocation across pools
            total_allocation = sum(
                self._shared_pools[pool_id].agent_allocations.get(
                    source_id,
                    Decimal("0"),
                )
                for pool_id in self._agent_pools[source_id]
            )
            if amount > total_allocation:
                msg = f"Insufficient funds. Available: {total_allocation}, Requested: {amount}"
                raise ValueError(
                    msg,
                )

        elif transfer_type == TransferType.POOL_WITHDRAW:
            if source_id not in self._shared_pools:
                msg = f"Source pool {source_id} not found"
                raise ValueError(msg)
            pool = self._shared_pools[source_id]
            available = pool.total_budget - pool.allocated_budget
            if amount > available:
                msg = f"Insufficient pool funds. Available: {available}, Requested: {amount}"
                raise ValueError(
                    msg,
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
            msg = f"Transfer request {request_id} not found"
            raise ValueError(msg)

        request = self._transfer_requests[request_id]
        if request.status != TransferStatus.PENDING:
            msg = f"Transfer {request_id} is not pending approval"
            raise ValueError(msg)

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
            msg = f"Transfer request {request_id} not found"
            raise ValueError(msg)

        request = self._transfer_requests[request_id]
        if request.status != TransferStatus.APPROVED:
            msg = f"Transfer {request_id} is not approved"
            raise ValueError(msg)

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
                    f"Transfer failed: {e!s}",
                    AlertSeverity.ERROR,
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
            msg = f"Transfer request {request_id} not found"
            raise ValueError(msg)

        request = self._transfer_requests[request_id]
        if request.status != TransferStatus.PENDING:
            msg = f"Transfer {request_id} is not pending approval"
            raise ValueError(msg)

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

    def _check_and_report_violation(
        self,
        agent_id: str,
        pool_id: str | None,
        current_balance: Decimal,
        required_amount: Decimal,
        violation_type: ViolationType,
    ) -> None:
        """Check for and report budget violations.

        Args:
            agent_id: Agent to check
            pool_id: Optional pool ID
            current_balance: Current balance
            required_amount: Amount being requested
            violation_type: Type of potential violation
        """
        if required_amount > current_balance:
            context = ViolationContext(
                agent_id=agent_id,
                pool_id=pool_id,
                current_balance=current_balance,
                violation_amount=required_amount - current_balance,
            )

            severity = (
                ViolationSeverity.CRITICAL
                if violation_type
                in (
                    ViolationType.UNAUTHORIZED,
                    ViolationType.POOL_BREACH,
                )
                else ViolationSeverity.HIGH
            )

            self.violation_reporter.report_violation(
                violation_type=violation_type,
                severity=severity,
                context=context,
                description=(
                    f"Agent {agent_id} attempted to use {required_amount} but only has "
                    f"{current_balance} available."
                ),
            )

    def _execute_direct_transfer(self, request: TransferRequest) -> None:
        """Execute a direct transfer between agents.

        This method handles the actual transfer of budget between two agents by:
        1. Validating both agents exist and have pool memberships
        2. Checking source agent has sufficient funds across their pools
        3. Executing the transfer by updating pool allocations
        4. Maintaining pool constraints and balance

        Args:
            request: Transfer request details

        Raises:
            ValueError: If agents not found, insufficient funds, or invalid pool state
        """
        # Start a transaction for atomic operation
        tx_ctx = self.begin_transaction()

        try:
            # Validate both agents exist and have pool memberships
            if request.source_id not in self._agent_pools:
                msg = f"Source agent {request.source_id} not found"
                raise ValueError(msg)
            if request.target_id not in self._agent_pools:
                msg = f"Target agent {request.target_id} not found"
                raise ValueError(msg)

            # Get all pools where source agent has allocations
            source_pools = self._agent_pools[request.source_id]
            source_allocations = {
                pool_id: self._shared_pools[pool_id].agent_allocations.get(
                    request.source_id,
                    Decimal("0"),
                )
                for pool_id in source_pools
            }

            # Save original allocations for potential rollback
            request.metadata["original_allocations"] = {
                pool_id: dict(self._shared_pools[pool_id].agent_allocations.items())
                for pool_id in set(source_pools) | self._agent_pools[request.target_id]
            }

            # Store the original pool allocated budgets
            request.metadata["original_pool_budgets"] = {
                pool_id: self._shared_pools[pool_id].allocated_budget
                for pool_id in set(source_pools) | self._agent_pools[request.target_id]
            }

            # Verify total source allocation is sufficient
            total_source_allocation = sum(source_allocations.values())
            if request.amount > total_source_allocation:
                # Report violation before raising error
                self._check_and_report_violation(
                    agent_id=request.source_id,
                    pool_id=None,
                    current_balance=total_source_allocation,
                    required_amount=request.amount,
                    violation_type=ViolationType.OVERSPEND,
                )
                msg = (
                    f"Insufficient total allocation. Available: {total_source_allocation}, "
                    f"Requested: {request.amount}"
                )
                raise ValueError(
                    msg,
                )

            # Find pools where target agent is active
            target_pools = self._agent_pools[request.target_id]
            if not target_pools:
                msg = f"Target agent {request.target_id} has no active pools"
                raise ValueError(
                    msg,
                )

            # Choose the highest priority pool for the target
            target_pool_id = max(
                target_pools,
                key=lambda pid: (
                    self._shared_pools[pid].priority.value
                    if hasattr(self._shared_pools[pid], "priority")
                    else 0
                ),
            )
            target_pool = self._shared_pools[target_pool_id]

            # Transfer the budget
            remaining_amount = request.amount
            for pool_id, allocation in sorted(
                source_allocations.items(),
                key=lambda x: (
                    (
                        self._shared_pools[x[0]].priority.value
                        if hasattr(self._shared_pools[x[0]], "priority")
                        else 0
                    ),
                    x[1],
                ),
            ):
                if allocation <= 0:
                    continue

                source_pool = self._shared_pools[pool_id]
                transfer_amount = min(remaining_amount, allocation)

                # Check for pool minimum balance violation
                if (
                    source_pool.total_budget - source_pool.allocated_budget - transfer_amount
                ) < source_pool.min_balance:
                    self._check_and_report_violation(
                        agent_id=request.source_id,
                        pool_id=pool_id,
                        current_balance=source_pool.total_budget - source_pool.allocated_budget,
                        required_amount=transfer_amount,
                        violation_type=ViolationType.POOL_BREACH,
                    )

                # Update source pool
                source_pool.agent_allocations[request.source_id] -= transfer_amount
                source_pool.allocated_budget -= transfer_amount

                # Update target pool
                target_pool.agent_allocations[request.target_id] = (
                    target_pool.agent_allocations.get(request.target_id, Decimal("0"))
                    + transfer_amount
                )
                target_pool.allocated_budget += transfer_amount

                remaining_amount -= transfer_amount
                if remaining_amount <= 0:
                    break

            # Verify transfer completed
            if remaining_amount > 0:
                # Report violation before rollback
                self._check_and_report_violation(
                    agent_id=request.source_id,
                    pool_id=None,
                    current_balance=total_source_allocation - (request.amount - remaining_amount),
                    required_amount=remaining_amount,
                    violation_type=ViolationType.OVERSPEND,
                )
                # Roll back the transaction
                self.rollback_transaction(tx_ctx)
                msg = f"Failed to transfer full amount. Remaining: {remaining_amount}"
                raise ValueError(
                    msg,
                )

            # Commit the successful transaction
            self.commit_transaction(tx_ctx)

        except Exception:
            # Roll back on any error
            self.rollback_transaction(tx_ctx)
            raise

    def _execute_pool_deposit(self, request: TransferRequest) -> None:
        """Execute a deposit to a shared pool."""
        # Start a transaction for atomic operation
        tx_ctx = self.begin_transaction()

        try:
            pool = self._shared_pools[request.target_id]

            # Store original state for potential rollback
            request.metadata["original_pool_budget"] = pool.total_budget

            # Execute the deposit
            pool.total_budget += request.amount

            # Commit the successful transaction
            self.commit_transaction(tx_ctx)

        except Exception:
            # Rollback on any error
            self.rollback_transaction(tx_ctx)
            raise

    def _execute_pool_withdraw(self, request: TransferRequest) -> None:
        """Execute a withdrawal from a shared pool."""
        # Start a transaction for atomic operation
        tx_ctx = self.begin_transaction()

        try:
            pool = self._shared_pools[request.source_id]
            available = pool.total_budget - pool.allocated_budget

            # Store original state for potential rollback
            request.metadata["original_pool_budget"] = pool.total_budget

            if request.amount > available:
                # Report violation before raising error
                self._check_and_report_violation(
                    agent_id=request.target_id,
                    pool_id=request.source_id,
                    current_balance=available,
                    required_amount=request.amount,
                    violation_type=ViolationType.POOL_BREACH,
                )
                msg = (
                    f"Insufficient pool funds. Available: {available}, Requested: {request.amount}"
                )
                raise ValueError(
                    msg,
                )

            # Check minimum balance
            if (available - request.amount) < pool.min_balance:
                self._check_and_report_violation(
                    agent_id=request.target_id,
                    pool_id=request.source_id,
                    current_balance=available,
                    required_amount=request.amount,
                    violation_type=ViolationType.POOL_BREACH,
                )

            # Execute the withdrawal
            pool.total_budget -= request.amount

            # Commit the successful transaction
            self.commit_transaction(tx_ctx)

        except Exception:
            # Rollback on any error
            self.rollback_transaction(tx_ctx)
            raise

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

    def select_optimal_pool(self, agent_id: str, required_budget: Decimal) -> str:
        """Select the optimal pool for an agent based on budget requirements and priority.

        This method implements dynamic pool selection by considering:
        1. Available budget in each pool
        2. Agent's priority level
        3. Current pool utilization
        4. Pool priority levels

        Args:
            agent_id: Agent needing allocation
            required_budget: Minimum budget needed

        Returns:
            ID of the selected pool

        Raises:
            ValueError: If no suitable pool is found
        """
        candidate_pools = []

        for pool_id, pool in self._shared_pools.items():
            available_budget = pool.total_budget - pool.allocated_budget

            # Skip pools with insufficient budget
            if available_budget < required_budget:
                continue

            # Calculate pool score based on multiple factors
            utilization = pool.allocated_budget / pool.total_budget
            priority_score = pool.priority.value

            # Higher score is better
            pool_score = (
                available_budget * 0.4  # Weight available budget
                + (1 - utilization) * 0.3  # Prefer less utilized pools
                + priority_score * 0.3  # Consider pool priority
            )

            candidate_pools.append((pool_id, pool_score))

        if not candidate_pools:
            msg = f"No suitable pool found for agent {agent_id} requiring {required_budget}"
            raise ValueError(
                msg,
            )

        # Select pool with highest score
        return max(candidate_pools, key=lambda x: x[1])[0]

    def optimize_pool_allocations(self) -> None:
        """Optimize budget allocations across all pools based on priorities and usage.

        This method implements the core budget allocation strategy by:
        1. Analyzing current pool utilization
        2. Identifying overloaded and underutilized pools
        3. Redistributing budgets based on agent priorities
        4. Maintaining minimum balance requirements
        """
        # Get all agents sorted by priority
        all_agents = []
        for pool in self._shared_pools.values():
            for agent_id in pool.active_agents:
                all_agents.append(
                    (agent_id, pool.agent_allocations.get(agent_id, Decimal("0"))),
                )

        # Sort by priority (higher priority first)
        all_agents.sort(
            key=lambda x: self._shared_pools[self._agent_pools[x[0]].pop()].priority.value,
            reverse=True,
        )

        # Redistribute budgets
        for agent_id, current_allocation in all_agents:
            current_pool_id = next(iter(self._agent_pools[agent_id]))
            current_pool = self._shared_pools[current_pool_id]

            # Check if agent should be moved to a better pool
            try:
                optimal_pool_id = self.select_optimal_pool(agent_id, current_allocation)
                if optimal_pool_id != current_pool_id:
                    # Move agent to better pool
                    self.remove_agent_from_pool(agent_id, current_pool_id)
                    self.add_agent_to_pool(
                        agent_id,
                        optimal_pool_id,
                        current_allocation,
                    )
            except ValueError:
                # No better pool found, optimize within current pool
                self._optimize_agent_allocation(agent_id, current_pool)

    def _optimize_agent_allocation(self, agent_id: str, pool: SharedPool) -> None:
        """Optimize an agent's allocation within its current pool.

        Args:
            agent_id: Agent to optimize
            pool: Current pool
        """
        current_allocation = pool.agent_allocations.get(agent_id, Decimal("0"))
        pool_priority = pool.priority

        # Calculate optimal allocation based on priority and available budget
        available_budget = pool.total_budget - pool.allocated_budget + current_allocation
        priority_weight = pool.priority_weights.get(pool_priority, 0.5)

        # Higher priority agents get larger share of available budget
        optimal_allocation = available_budget * Decimal(str(priority_weight))

        # Ensure we maintain minimum balance
        optimal_allocation = min(optimal_allocation, pool.total_budget - pool.min_balance)

        # Update allocation
        pool.agent_allocations[agent_id] = optimal_allocation
        pool.allocated_budget = sum(pool.agent_allocations.values())

    def handle_emergency_allocation(
        self,
        agent_id: str,
        required_budget: Decimal,
    ) -> None:
        """Handle emergency budget allocation for critical operations.

        This method provides immediate budget allocation for high-priority operations by:
        1. Identifying available emergency reserves
        2. Temporarily reducing allocations of lower-priority agents
        3. Moving budgets between pools if necessary

        Args:
            agent_id: Agent requiring emergency allocation
            required_budget: Budget amount needed

        Raises:
            ValueError: If emergency allocation cannot be satisfied
        """
        if agent_id not in self._agent_pools:
            msg = f"Agent {agent_id} not registered"
            raise ValueError(msg)

        # Try to allocate from current pool first
        current_pool_id = self._agent_pools[agent_id]
        current_pool = self._pools[current_pool_id]

        # Check if we can reallocate within current pool
        available_budget = current_pool.total_budget - current_pool.used_budget
        if available_budget >= required_budget:
            # Update allocation
            self._agent_budgets[agent_id] = required_budget
            current_pool.used_budget += required_budget
            return

        # If not enough in current pool, try to free up budget
        needed = required_budget - available_budget
        freed_budget = self._free_up_budget(current_pool, needed)

        if freed_budget >= needed:
            # Update allocation
            self._agent_budgets[agent_id] = required_budget
            current_pool.used_budget = sum(self._agent_budgets[aid] for aid in current_pool.agents)
            return

        # If still not enough, try other pools
        try:
            new_pool_id = self.select_optimal_pool(agent_id, required_budget)
            if new_pool_id != current_pool_id:
                # Move agent to better pool
                current_pool.agents.remove(agent_id)
                current_pool.used_budget -= self._agent_budgets[agent_id]

                new_pool = self._pools[new_pool_id]
                new_pool.agents.add(agent_id)
                self._agent_pools[agent_id] = new_pool_id
                self._agent_budgets[agent_id] = required_budget
                new_pool.used_budget += required_budget
            else:
                msg = f"Could not satisfy emergency allocation of {required_budget} for agent {agent_id}"
                raise ValueError(
                    msg,
                )
        except ValueError:
            msg = (
                f"Could not satisfy emergency allocation of {required_budget} for agent {agent_id}"
            )
            raise ValueError(
                msg,
            )

    def _free_up_budget(self, pool: SharedPool, needed: Decimal) -> Decimal:
        """Attempt to free up budget in a pool by reducing lower priority allocations.

        Args:
            pool: Pool to free up budget from
            needed: Amount of budget needed

        Returns:
            Amount of budget freed up
        """
        freed = Decimal("0")

        # Get all agents in pool sorted by priority (lowest first)
        agents = sorted(
            [(aid, pool.agent_allocations[aid]) for aid in pool.active_agents],
            key=lambda x: self._shared_pools[self._agent_pools[x[0]].pop()].priority.value,
        )

        # Reduce allocations of lower priority agents
        for agent_id, allocation in agents:
            if freed >= needed:
                break

            # Can reduce up to 50% of current allocation
            reduction = min(allocation * Decimal("0.5"), needed - freed)
            pool.agent_allocations[agent_id] -= reduction
            freed += reduction

        pool.allocated_budget = sum(pool.agent_allocations.values())
        return freed

    def auto_scale_pools(self) -> None:
        """Automatically scale pools based on utilization and demand.

        This method implements dynamic pool scaling by:
        1. Monitoring pool utilization
        2. Creating new pools when needed
        3. Merging underutilized pools
        4. Adjusting pool budgets based on demand
        """
        # Calculate overall system utilization
        total_budget = sum(pool.total_budget for pool in self._shared_pools.values())
        total_allocated = sum(pool.allocated_budget for pool in self._shared_pools.values())
        system_utilization = total_allocated / total_budget if total_budget > 0 else 1

        # Handle high utilization - might need new pools
        if system_utilization > 0.8:  # 80% threshold
            self._handle_high_utilization()
        # Handle low utilization - might need to merge pools
        elif system_utilization < 0.3:  # 30% threshold
            self._handle_low_utilization()

        # Balance load across pools
        self._balance_pool_load()

    def _handle_high_utilization(self) -> None:
        """Handle high system utilization by creating new pools or expanding existing ones."""
        # Find most constrained pool
        constrained_pool = max(
            self._shared_pools.values(),
            key=lambda p: p.allocated_budget / p.total_budget,
        )

        if constrained_pool.allocated_budget / constrained_pool.total_budget > 0.9:
            # Create new pool with similar configuration
            new_pool_id = f"pool_{len(self._shared_pools) + 1}"
            new_pool = self.create_shared_pool(
                pool_id=new_pool_id,
                total_budget=constrained_pool.total_budget,
                min_balance=constrained_pool.min_balance,
                priority=constrained_pool.priority,
            )

            # Move some agents to new pool
            self._redistribute_agents(constrained_pool, new_pool)

    def _handle_low_utilization(self) -> None:
        """Handle low system utilization by merging underutilized pools."""
        # Find pools that could be merged
        underutilized = [
            pool_id
            for pool_id, pool in self._shared_pools.items()
            if pool.allocated_budget / pool.total_budget < 0.3
        ]

        while len(underutilized) > 1:
            pool1_id = underutilized.pop()
            pool2_id = underutilized.pop()

            pool1 = self._shared_pools[pool1_id]
            pool2 = self._shared_pools[pool2_id]

            # Merge into pool with higher priority
            target_pool = pool1 if pool1.priority.value > pool2.priority.value else pool2
            source_pool = pool2 if target_pool == pool1 else pool1

            # Move all agents from source to target
            for agent_id in list(source_pool.active_agents):
                allocation = source_pool.agent_allocations[agent_id]
                self.remove_agent_from_pool(agent_id, source_pool.pool_id)
                self.add_agent_to_pool(agent_id, target_pool.pool_id, allocation)

            # Remove empty pool
            del self._shared_pools[source_pool.pool_id]

    def _balance_pool_load(self) -> None:
        """Balance load across pools to optimize resource usage."""
        # Calculate average utilization
        utilizations = [
            (pool_id, pool.allocated_budget / pool.total_budget)
            for pool_id, pool in self._shared_pools.items()
        ]
        avg_utilization = sum(u for _, u in utilizations) / len(utilizations)

        # Identify overloaded and underloaded pools
        overloaded = [
            pool_id
            for pool_id, util in utilizations
            if util > avg_utilization * 1.2  # 20% above average
        ]
        underloaded = [
            pool_id
            for pool_id, util in utilizations
            if util < avg_utilization * 0.8  # 20% below average
        ]

        # Balance by moving agents from overloaded to underloaded pools
        for over_id in overloaded:
            over_pool = self._shared_pools[over_id]

            # Sort agents by priority (lowest first)
            agents = sorted(
                [(aid, over_pool.agent_allocations[aid]) for aid in over_pool.active_agents],
                key=lambda x: self._shared_pools[self._agent_pools[x[0]].pop()].priority.value,
            )

            # Try to move lower priority agents to underloaded pools
            for agent_id, allocation in agents:
                if not underloaded:
                    break

                for under_id in underloaded[:]:
                    under_pool = self._shared_pools[under_id]
                    if under_pool.total_budget - under_pool.allocated_budget >= allocation:
                        # Move agent
                        self.remove_agent_from_pool(agent_id, over_id)
                        self.add_agent_to_pool(agent_id, under_id, allocation)

                        # Update utilizations
                        if (
                            under_pool.allocated_budget / under_pool.total_budget
                            > avg_utilization * 0.8
                        ):
                            underloaded.remove(under_id)
                        break

    def _redistribute_agents(
        self,
        source_pool: SharedPool,
        target_pool: SharedPool,
    ) -> None:
        """Redistribute agents from source pool to target pool.

        Args:
            source_pool: Pool to move agents from
            target_pool: Pool to move agents to
        """
        # Calculate how much to move
        move_budget = source_pool.allocated_budget * Decimal(
            "0.4",
        )  # Move 40% of allocation
        moved_budget = Decimal("0")

        # Sort agents by priority (lowest first)
        agents = sorted(
            [(aid, source_pool.agent_allocations[aid]) for aid in source_pool.active_agents],
            key=lambda x: self._shared_pools[self._agent_pools[x[0]].pop()].priority.value,
        )

        # Move agents until we've moved enough budget
        for agent_id, allocation in agents:
            if moved_budget >= move_budget:
                break

            self.remove_agent_from_pool(agent_id, source_pool.pool_id)
            self.add_agent_to_pool(agent_id, target_pool.pool_id, allocation)
            moved_budget += allocation

    def create_pool(
        self,
        pool_id: str,
        total_budget: Decimal,
        priority: int = 0,
    ) -> BudgetPool:
        """Create a new budget pool.

        Args:
            pool_id: Unique identifier for the pool
            total_budget: Total budget allocation for the pool
            priority: Pool priority level

        Returns:
            Created budget pool

        Raises:
            ValueError: If pool ID already exists
        """
        if pool_id in self._pools:
            msg = f"Pool {pool_id} already exists"
            raise ValueError(msg)

        pool = BudgetPool(
            pool_id=pool_id,
            total_budget=total_budget,
            priority=priority,
        )
        self._pools[pool_id] = pool
        return pool

    def register_agent(
        self,
        name: str,
        initial_budget: Decimal,
        priority: int = 0,
        agent: Agent | None = None,
    ) -> Agent:
        """Register an agent with the budget coordinator.

        Args:
            name: Agent name
            initial_budget: Initial budget allocation
            priority: Agent priority level (higher = more important)
            agent: Optional Agent instance

        Returns:
            Registered agent instance

        Raises:
            ValueError: If agent is already registered or validation fails
        """
        # Validate inputs
        if initial_budget <= Decimal("0"):
            msg = "Initial budget must be positive"
            raise ValueError(msg)

        if not name or not isinstance(name, str):
            msg = "Agent name must be a non-empty string"
            raise ValueError(msg)

        # Validate priority for backward compatibility with tests
        if priority < 0 or priority > 10:
            msg = "Priority must be between 1 and 10"
            raise ValueError(msg)

        agent_id = name  # For simplicity, use name as ID

        if agent_id in self._registered_agents:
            msg = f"Agent {agent_id} is already registered"
            raise ValueError(msg)

        # Create agent if not provided
        if agent is None:
            agent = DefaultAgent(name=name)
        # Check for name mismatch for backward compatibility with tests
        elif agent.name != name:
            msg = f"Agent name mismatch: {agent.name} != {name}"
            raise ValueError(msg)

        # Create default pool if needed
        default_pool_id = "default_pool"
        if default_pool_id not in self._pools:
            # Start with twice the initial budget to have room for growth
            pool_budget = max(initial_budget * Decimal("2"), Decimal("1000"))
            self.create_pool(default_pool_id, pool_budget, priority)

        # Get pool
        pool = self._pools[default_pool_id]

        # Check if pool has enough budget
        if initial_budget > pool.remaining_budget:
            msg = f"Pool {default_pool_id} does not have enough budget for agent {name}"
            raise ValueError(
                msg,
            )

        # Add agent to pool
        self._agents[agent.id] = agent
        self._agent_budgets[agent.id] = initial_budget

        # Track initial budget for usage calculation
        self._initial_budgets[agent.id] = initial_budget
        self._agent_pools[agent.id] = pool.pool_id
        pool.agents.add(agent.id)
        self._registered_agents.add(agent.id)

        # Initialize budget monitoring
        if initial_budget > Decimal("0"):
            self.budget_monitor.check_budget_usage(
                agent.id,
                Decimal("0"),
                initial_budget,
            )

        # Log the registration
        if self.notification_manager:
            self.notification_manager.send_alert(
                Alert(
                    title="Agent Registered",
                    description=f"Agent {agent_id} registered with initial budget {initial_budget}",
                    severity=AlertSeverity.INFO,
                    metadata={
                        "agent_id": agent_id,
                        "initial_budget": str(initial_budget),
                    },
                ),
            )

        return agent

    def unregister_agent(self, agent_id: str) -> None:
        """Unregister an agent.

        Args:
            agent_id: ID of agent to unregister

        Raises:
            AgentSafetyError: If agent not found
        """
        if agent_id not in self._agents:
            msg = f"Agent {agent_id} not found"
            raise AgentSafetyError(msg)

        # Get pool and remove agent
        pool_id = self._agent_pools[agent_id]
        pool = self._pools[pool_id]
        pool.agents.remove(agent_id)

        # Clean up budget
        budget = self._agent_budgets[agent_id]
        pool.used_budget -= budget

        # Clean up agent data
        del self._agents[agent_id]
        del self._agent_budgets[agent_id]
        del self._agent_pools[agent_id]
        if agent_id in self._initial_budgets:
            del self._initial_budgets[agent_id]

        # Reset budget monitor for this agent
        self.budget_monitor.reset_agent_alerts(agent_id)

    def get_agent(self, agent_id: str) -> Agent:
        """Get an agent.

        Args:
            agent_id: ID of agent to get

        Returns:
            Agent instance
        """
        if agent_id not in self._agents:
            msg = f"Agent {agent_id} not found"
            raise AgentSafetyError(msg)
        return self._agents[agent_id]

    def get_agent_budget(self, agent_id: str) -> Decimal:
        """Get current budget for an agent.

        Args:
            agent_id: Agent to get budget for

        Returns:
            Current budget amount
        """
        if agent_id not in self._agent_budgets:
            msg = f"No budget found for agent {agent_id}"
            raise BudgetError(msg)
        return self._agent_budgets[agent_id]

    def update_agent_budget(self, agent_id: str, new_budget: Decimal) -> None:
        """Update budget for an agent.

        Args:
            agent_id: Agent to update budget for
            new_budget: New budget amount

        Raises:
            BudgetError: If update fails
        """
        if agent_id not in self._agent_budgets:
            msg = f"No budget found for agent {agent_id}"
            raise BudgetError(msg)

        current_budget = self._agent_budgets[agent_id]
        pool = self._pools[self._agent_pools[agent_id]]

        # Calculate budget change
        budget_change = new_budget - current_budget

        # Check if enough budget in pool
        if budget_change > 0 and budget_change > pool.remaining_budget:
            msg = "Insufficient budget in pool"
            raise BudgetError(msg)

        # Calculate original initial budget and used budget
        initial_budget = self._initial_budgets.get(agent_id, current_budget)
        used_budget = initial_budget - new_budget

        # Update budgets
        self._agent_budgets[agent_id] = new_budget
        pool.used_budget += budget_change
        pool.last_update = datetime.now()

        # Check budget usage and trigger alerts if necessary
        self.budget_monitor.check_budget_usage(agent_id, used_budget, initial_budget)

        # Create high usage budget alert if necessary
        usage_ratio = (
            used_budget / initial_budget if initial_budget != Decimal("0") else Decimal("0")
        )
        if usage_ratio >= Decimal("0.75") and self.notification_manager:
            self.notification_manager.create_alert(
                SafetyAlert(
                    title="High Budget Usage",
                    description=f"Agent {agent_id} has used {usage_ratio * 100:.1f}% of its budget",
                    severity=AlertSeverity.WARNING,
                    timestamp=datetime.now(),
                    metadata={"agent_id": agent_id},
                ),
            )

    def get_pool(self, pool_id: str) -> BudgetPool:
        """Get a budget pool.

        Args:
            pool_id: ID of pool to get

        Returns:
            Budget pool
        """
        if pool_id not in self._pools:
            msg = f"Pool {pool_id} not found"
            raise BudgetError(msg)
        return self._pools[pool_id]

    def get_agent_pool(self, agent_id: str) -> BudgetPool:
        """Get pool for an agent.

        Args:
            agent_id: Agent to get pool for

        Returns:
            Agent's budget pool
        """
        if agent_id not in self._agent_pools:
            msg = f"No pool found for agent {agent_id}"
            raise BudgetError(msg)
        return self._pools[self._agent_pools[agent_id]]

    def get_pools(self) -> list[BudgetPool]:
        """Get all budget pools.

        Returns:
            List of all budget pools
        """
        return list(self._pools.values())

    def create_agent(
        self,
        name: str,
        initial_budget: Decimal,
        priority: int = 0,
    ) -> Agent:
        """Create and register a new agent.

        Args:
            name: Agent name
            initial_budget: Initial budget allocation
            priority: Agent priority level (higher = more important)

        Returns:
            Created agent instance
        """
        agent = DefaultAgent(name=name)
        return self.register_agent(
            name=name,
            initial_budget=initial_budget,
            priority=priority,
            agent=agent,
        )

    def get_agent_metrics(self, agent_id: str) -> dict:
        """Get budget metrics for a specific agent.

        Args:
            agent_id: The ID of the agent to get metrics for

        Returns:
            Dict containing budget metrics including:
            - initial_budget: Initial budget allocated
            - used_budget: Amount of budget used
            - remaining_budget: Current remaining budget
            - last_update: Timestamp of last budget update
        """
        agent = self.get_agent(agent_id)
        if not agent:
            msg = f"No agent found with ID {agent_id}"
            raise ValueError(msg)

        pool = self.get_agent_pool(agent_id)
        if not pool:
            msg = f"No pool found for agent {agent_id}"
            raise ValueError(msg)

        # Get the agent's initial and current budget allocation
        if not hasattr(self, "_initial_budgets"):
            self._initial_budgets = {}
            self._initial_budgets[agent_id] = self._agent_budgets[agent_id]

        initial_budget = self._initial_budgets.get(
            agent_id,
            self._agent_budgets[agent_id],
        )
        current_budget = self._agent_budgets[agent_id]
        used_budget = initial_budget - current_budget

        return {
            "initial_budget": initial_budget,
            "used_budget": used_budget,
            "remaining_budget": current_budget,
            "last_update": pool.last_update,
        }
