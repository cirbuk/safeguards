"""Type definitions for the agent safety framework."""

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum, auto


class AlertSeverity(Enum):
    """Alert severity levels."""

    INFO = auto()
    WARNING = auto()
    ERROR = auto()
    CRITICAL = auto()


@dataclass
class SafetyAlert:
    """Safety alert notification."""

    title: str
    description: str
    severity: AlertSeverity
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: dict = field(default_factory=dict)


@dataclass
class BudgetMetrics:
    """Budget usage metrics."""

    current_usage: Decimal
    total_budget: Decimal
    remaining: Decimal
    usage_percent: float
    last_update: datetime = field(default_factory=datetime.now)


@dataclass
class ResourceMetrics:
    """Resource usage metrics."""

    cpu_percent: float
    memory_percent: float
    disk_percent: float
    network_bytes_sent: int
    network_bytes_recv: int
    last_update: datetime = field(default_factory=datetime.now)


@dataclass
class SafetyMetrics:
    """Combined safety metrics."""

    budget: BudgetMetrics
    resources: ResourceMetrics
    alerts: list[SafetyAlert] | None = None


@dataclass
class BudgetOverride:
    """Budget override request."""

    agent_id: str
    amount: Decimal
    reason: str
    status: str  # PENDING, APPROVED, REJECTED
    requester: str
    timestamp: datetime = field(default_factory=datetime.now)
    approver: str | None = None
    approval_time: datetime | None = None
