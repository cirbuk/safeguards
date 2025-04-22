"""Type definitions for the Agent Safety Framework."""

from dataclasses import dataclass, field
from decimal import Decimal
from datetime import datetime
from typing import Dict, List, Optional
from enum import Enum, auto

from .agent import Agent
from .guardrail import Guardrail, RunContext, ResourceUsage


class AlertSeverity(Enum):
    """Severity levels for alerts."""

    INFO = auto()
    WARNING = auto()
    HIGH = auto()
    CRITICAL = auto()


class NotificationChannel(Enum):
    """Available notification channels."""

    CONSOLE = auto()
    EMAIL = auto()
    SLACK = auto()
    WEBHOOK = auto()


@dataclass
class Alert:
    """Alert notification."""

    title: str
    description: str
    severity: AlertSeverity
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict = field(default_factory=dict)


@dataclass
class SafetyConfig:
    """Configuration for safety controls."""

    total_budget: Decimal
    hourly_limit: Optional[Decimal] = None
    daily_limit: Optional[Decimal] = None
    cpu_threshold: float = 80.0
    memory_threshold: float = 80.0
    budget_warning_threshold: float = 75.0
    require_human_approval: bool = False


@dataclass
class BudgetConfig:
    """Configuration for budget management."""

    total_budget: Decimal
    hourly_limit: Optional[Decimal] = None
    daily_limit: Optional[Decimal] = None
    warning_threshold: float = 75.0


@dataclass
class ResourceConfig:
    """Configuration for resource monitoring."""

    cpu_limit: float = 80.0
    memory_limit: float = 80.0
    disk_limit: float = 90.0
    network_limit: float = 100.0
    monitor_interval_seconds: int = 60
    alert_threshold: float = 75.0
    # Backward compatibility for tests
    check_interval: int = None

    def __post_init__(self):
        """Initialize with backward compatibility."""
        # Handle backward compatibility
        if hasattr(self, "cpu_threshold"):
            self.cpu_limit = self.cpu_threshold
        if hasattr(self, "memory_threshold"):
            self.memory_limit = self.memory_threshold
        if hasattr(self, "disk_threshold"):
            self.disk_limit = self.disk_threshold
        if self.check_interval is not None:
            self.monitor_interval_seconds = self.check_interval

    def __init__(
        self,
        cpu_limit: float = 80.0,
        memory_limit: float = 80.0,
        disk_limit: float = 90.0,
        network_limit: float = 100.0,
        monitor_interval_seconds: int = 60,
        alert_threshold: float = 75.0,
        # Backward compatibility parameters
        cpu_threshold: float = None,
        memory_threshold: float = None,
        disk_threshold: float = None,
        check_interval: int = None,
    ):
        """Initialize resource configuration with backward compatibility.

        Args:
            cpu_limit: CPU usage limit (percentage)
            memory_limit: Memory usage limit (percentage)
            disk_limit: Disk usage limit (percentage)
            network_limit: Network bandwidth limit (Mbps)
            monitor_interval_seconds: Monitoring interval in seconds
            alert_threshold: Alert threshold (percentage of limit)

            # Backward compatibility parameters
            cpu_threshold: Legacy name for cpu_limit
            memory_threshold: Legacy name for memory_limit
            disk_threshold: Legacy name for disk_limit
            check_interval: Legacy name for monitor_interval_seconds
        """
        self.cpu_limit = cpu_threshold if cpu_threshold is not None else cpu_limit
        self.memory_limit = (
            memory_threshold if memory_threshold is not None else memory_limit
        )
        self.disk_limit = disk_threshold if disk_threshold is not None else disk_limit
        self.network_limit = network_limit
        self.monitor_interval_seconds = (
            check_interval if check_interval is not None else monitor_interval_seconds
        )
        self.alert_threshold = alert_threshold
        self.check_interval = self.monitor_interval_seconds


@dataclass
class ResourceThresholds:
    """Thresholds for resource usage."""

    cpu_percent: float = 80.0
    memory_percent: float = 80.0
    disk_percent: float = 90.0
    network_mbps: float = 100.0


@dataclass
class BudgetMetrics:
    """Budget usage metrics."""

    total_budget: Decimal
    used_budget: Decimal
    remaining_budget: Decimal
    usage_percent: float
    last_usage: datetime
    period_usage: Dict[str, Decimal]  # Usage by time period


@dataclass
class ResourceMetrics:
    """Resource usage metrics."""

    cpu_percent: float
    memory_percent: float
    disk_percent: float
    network_usage: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    last_updated: datetime = None
    memory_used: Optional[int] = None
    memory_total: Optional[int] = None
    disk_used: Optional[int] = None
    disk_total: Optional[int] = None
    network_sent: Optional[int] = None
    network_received: Optional[int] = None
    network_speed: Optional[float] = None
    process_count: int = 0
    open_files: int = 0

    def __init__(
        self,
        cpu_percent: float,
        memory_percent: float,
        disk_percent: float,
        network_usage: float = 0.0,
        timestamp: datetime = None,
        last_updated: datetime = None,
        memory_used: Optional[int] = None,
        memory_total: Optional[int] = None,
        disk_used: Optional[int] = None,
        disk_total: Optional[int] = None,
        network_sent: Optional[int] = None,
        network_received: Optional[int] = None,
        network_speed: Optional[float] = None,
        process_count: int = 0,
        open_files: int = 0,
        # Backward compatibility parameters
        cpu_usage: float = None,
        memory_usage: float = None,
        disk_usage: float = None,
        network_mbps: float = None,
    ):
        """Initialize resource metrics with backward compatibility support.

        Args:
            cpu_percent: CPU usage percentage
            memory_percent: Memory usage percentage
            disk_percent: Disk usage percentage
            network_usage: Network usage (deprecated)
            timestamp: Time when metrics were collected
            last_updated: Last update time
            memory_used: Memory used in bytes
            memory_total: Total memory in bytes
            disk_used: Disk used in bytes
            disk_total: Total disk space in bytes
            network_sent: Network bytes sent
            network_received: Network bytes received
            network_speed: Network speed in Mbps
            process_count: Number of processes
            open_files: Number of open files

            # Backward compatibility parameters
            cpu_usage: Alias for cpu_percent
            memory_usage: Alias for memory_percent
            disk_usage: Alias for disk_percent
            network_mbps: Network speed in Mbps
        """
        self.cpu_percent = cpu_usage if cpu_usage is not None else cpu_percent
        self.memory_percent = (
            memory_usage if memory_usage is not None else memory_percent
        )
        self.disk_percent = disk_usage if disk_usage is not None else disk_percent
        self.network_usage = network_usage
        self.timestamp = timestamp or datetime.now()
        self.last_updated = last_updated or self.timestamp
        self.memory_used = memory_used
        self.memory_total = memory_total
        self.disk_used = disk_used
        self.disk_total = disk_total
        self.network_sent = network_sent
        self.network_received = network_received
        self.network_speed = network_speed or network_mbps
        self.process_count = process_count
        self.open_files = open_files

        # Aliases for backward compatibility
        self.cpu_usage = self.cpu_percent
        self.memory_usage = self.memory_percent
        self.disk_usage = self.disk_percent
        self.network_mbps = (
            self.network_speed if self.network_speed is not None else 0.0
        )


@dataclass
class SafetyMetrics:
    """Combined safety metrics."""

    budget: BudgetMetrics
    resources: ResourceMetrics
    alerts: Optional[List["SafetyAlert"]] = None


@dataclass
class SafetyAlert:
    """Safety alert notification."""

    title: str
    description: str
    severity: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict = field(default_factory=dict)


__all__ = [
    "Agent",
    "Guardrail",
    "RunContext",
    "ResourceUsage",
    "SafetyConfig",
    "BudgetConfig",
    "ResourceConfig",
    "ResourceThresholds",
    "BudgetMetrics",
    "ResourceMetrics",
    "SafetyMetrics",
    "SafetyAlert",
    "Alert",
    "AlertSeverity",
    "NotificationChannel",
]
