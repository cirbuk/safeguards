"""Transaction management for atomic budget operations."""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Protocol, TypeVar
from uuid import UUID, uuid4

T = TypeVar("T")


class TransactionStatus(Enum):
    """Status of a transaction."""

    PENDING = auto()
    EXECUTING = auto()
    COMMITTED = auto()
    ROLLED_BACK = auto()
    FAILED = auto()


class TransactionError(Exception):
    """Base class for transaction-related errors."""

    pass


class TransactionConflictError(TransactionError):
    """Error raised when a transaction conflicts with another."""

    pass


class TransactionTimeoutError(TransactionError):
    """Error raised when a transaction times out."""

    pass


@dataclass
class TransactionLog:
    """Log entry for a transaction operation."""

    operation: str
    timestamp: datetime
    data: dict[str, Any]
    status: str


@dataclass
class TransactionContext:
    """Context for a transaction."""

    transaction_id: UUID = field(default_factory=uuid4)
    start_time: datetime = field(default_factory=datetime.now)
    status: TransactionStatus = TransactionStatus.PENDING
    logs: list[TransactionLog] = field(default_factory=list)
    locks: set[str] = field(default_factory=set)
    changes: dict[str, Any] = field(default_factory=dict)


class Transactional(Protocol[T]):
    """Protocol for objects that support transactions."""

    def begin_transaction(self) -> TransactionContext:
        """Begin a new transaction."""
        ...

    def commit_transaction(self, ctx: TransactionContext) -> None:
        """Commit a transaction."""
        ...

    def rollback_transaction(self, ctx: TransactionContext) -> None:
        """Rollback a transaction."""
        ...

    def get_snapshot(self) -> T:
        """Get a snapshot of the current state."""
        ...

    def restore_snapshot(self, snapshot: T) -> None:
        """Restore state from a snapshot."""
        ...


class TransactionManager:
    """Manages atomic transactions across multiple resources."""

    def __init__(self, lock_timeout: float = 5.0):
        """Initialize transaction manager.

        Args:
            lock_timeout: Timeout in seconds for acquiring locks
        """
        self.lock_timeout = lock_timeout
        self._active_transactions: dict[UUID, TransactionContext] = {}
        self._resource_locks: dict[str, UUID] = {}
        self._lock = asyncio.Lock()

    async def begin(self) -> TransactionContext:
        """Begin a new transaction.

        Returns:
            New transaction context

        Raises:
            TransactionError: If transaction cannot be started
        """
        async with self._lock:
            ctx = TransactionContext()
            self._active_transactions[ctx.transaction_id] = ctx
            return ctx

    async def commit(self, ctx: TransactionContext) -> None:
        """Commit a transaction.

        Args:
            ctx: Transaction context to commit

        Raises:
            TransactionError: If commit fails
        """
        if ctx.transaction_id not in self._active_transactions:
            msg = "Transaction not found"
            raise TransactionError(msg)

        try:
            # Release all locks
            async with self._lock:
                for resource_id in ctx.locks:
                    if self._resource_locks.get(resource_id) == ctx.transaction_id:
                        del self._resource_locks[resource_id]

                ctx.status = TransactionStatus.COMMITTED
                del self._active_transactions[ctx.transaction_id]

        except Exception as e:
            ctx.status = TransactionStatus.FAILED
            msg = f"Commit failed: {e!s}"
            raise TransactionError(msg)

    async def rollback(self, ctx: TransactionContext) -> None:
        """Rollback a transaction.

        Args:
            ctx: Transaction context to rollback

        Raises:
            TransactionError: If rollback fails
        """
        if ctx.transaction_id not in self._active_transactions:
            msg = "Transaction not found"
            raise TransactionError(msg)

        try:
            # Release all locks
            async with self._lock:
                for resource_id in ctx.locks:
                    if self._resource_locks.get(resource_id) == ctx.transaction_id:
                        del self._resource_locks[resource_id]

                ctx.status = TransactionStatus.ROLLED_BACK
                del self._active_transactions[ctx.transaction_id]

        except Exception as e:
            ctx.status = TransactionStatus.FAILED
            msg = f"Rollback failed: {e!s}"
            raise TransactionError(msg)

    async def acquire_lock(self, ctx: TransactionContext, resource_id: str) -> None:
        """Acquire a lock on a resource.

        Args:
            ctx: Transaction context
            resource_id: Resource to lock

        Raises:
            TransactionTimeoutError: If lock cannot be acquired within timeout
            TransactionError: If lock acquisition fails
        """
        start_time = datetime.now()
        while True:
            async with self._lock:
                # Check if resource is already locked by this transaction
                if self._resource_locks.get(resource_id) == ctx.transaction_id:
                    return

                # Check if resource is locked by another transaction
                if resource_id not in self._resource_locks:
                    self._resource_locks[resource_id] = ctx.transaction_id
                    ctx.locks.add(resource_id)
                    return

            # Check timeout
            if (datetime.now() - start_time).total_seconds() > self.lock_timeout:
                msg = f"Timeout acquiring lock for resource {resource_id}"
                raise TransactionTimeoutError(
                    msg,
                )

            # Wait before retrying
            await asyncio.sleep(0.1)

    async def release_lock(self, ctx: TransactionContext, resource_id: str) -> None:
        """Release a lock on a resource.

        Args:
            ctx: Transaction context
            resource_id: Resource to unlock

        Raises:
            TransactionError: If lock release fails
        """
        async with self._lock:
            if (
                resource_id in self._resource_locks
                and self._resource_locks[resource_id] == ctx.transaction_id
            ):
                del self._resource_locks[resource_id]
                ctx.locks.remove(resource_id)

    def add_log(
        self,
        ctx: TransactionContext,
        operation: str,
        data: dict[str, Any],
    ) -> None:
        """Add a log entry to a transaction.

        Args:
            ctx: Transaction context
            operation: Operation being logged
            data: Operation data to log
        """
        log = TransactionLog(
            operation=operation,
            timestamp=datetime.now(),
            data=data,
            status=ctx.status.name,
        )
        ctx.logs.append(log)

    @property
    def active_transactions(self) -> dict[UUID, TransactionContext]:
        """Get all active transactions.

        Returns:
            Dict of transaction ID to context
        """
        return self._active_transactions.copy()

    @property
    def locked_resources(self) -> dict[str, UUID]:
        """Get all locked resources.

        Returns:
            Dict of resource ID to locking transaction ID
        """
        return self._resource_locks.copy()
