"""Base interfaces and abstract classes for resource monitoring."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Protocol


@dataclass
class ResourceMetrics:
    """Resource metrics data structure."""

    timestamp: datetime = field(default_factory=datetime.now)
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    disk_percent: float = 0.0
    network_mbps: float = 0.0
    process_count: int = 0
    open_files: int = 0

    def __init__(
        self,
        timestamp: datetime | None = None,
        cpu_percent: float = 0.0,
        memory_percent: float = 0.0,
        disk_percent: float = 0.0,
        network_mbps: float = 0.0,
        process_count: int = 0,
        open_files: int = 0,
        # Backward compatibility parameters
        cpu_usage: float | None = None,
        memory_usage: float | None = None,
        disk_usage: float | None = None,
        network_usage: float | None = None,
        memory_used: int | None = None,
        memory_total: int | None = None,
        disk_used: int | None = None,
        disk_total: int | None = None,
        network_sent: int | None = None,
        network_received: int | None = None,
        network_speed: float | None = None,
        last_updated: datetime | None = None,
    ):
        """Initialize resource metrics with support for backward compatibility.

        Args:
            timestamp: Time when metrics were collected
            cpu_percent: CPU usage percentage (0-100)
            memory_percent: Memory usage percentage (0-100)
            disk_percent: Disk usage percentage (0-100)
            network_mbps: Network usage in megabits per second
            process_count: Number of processes
            open_files: Number of open files

            # Backward compatibility parameters
            cpu_usage: Alias for cpu_percent
            memory_usage: Alias for memory_percent
            disk_usage: Alias for disk_percent
            network_usage: Network usage (deprecated)
            memory_used: Memory used in bytes
            memory_total: Total memory in bytes
            disk_used: Disk used in bytes
            disk_total: Total disk space in bytes
            network_sent: Network bytes sent
            network_received: Network bytes received
            network_speed: Network speed in Mbps
            last_updated: Timestamp for last update
        """
        self.timestamp = timestamp or datetime.now()
        self.cpu_percent = cpu_usage if cpu_usage is not None else cpu_percent
        self.memory_percent = memory_usage if memory_usage is not None else memory_percent
        self.disk_percent = disk_usage if disk_usage is not None else disk_percent
        self.network_mbps = network_speed if network_speed is not None else network_mbps
        self.process_count = process_count
        self.open_files = open_files

        # Store additional backward compatibility fields
        self.cpu_usage = self.cpu_percent
        self.memory_usage = self.memory_percent
        self.disk_usage = self.disk_percent
        self.network_usage = network_usage
        self.memory_used = memory_used
        self.memory_total = memory_total
        self.disk_used = disk_used
        self.disk_total = disk_total
        self.network_sent = network_sent
        self.network_received = network_received
        self.network_speed = network_speed
        self.last_updated = last_updated or self.timestamp

        # Explicitly run validation
        self.__post_init__()

    def __post_init__(self):
        """Validate metrics are within valid ranges."""
        for field_name in ["cpu_percent", "memory_percent", "disk_percent"]:
            value = getattr(self, field_name)
            if not 0 <= value <= 100:
                msg = f"{field_name} must be between 0 and 100"
                raise ValueError(msg)

        if self.network_mbps < 0:
            msg = "network_mbps must be non-negative"
            raise ValueError(msg)
        if self.process_count < 0:
            msg = "process_count must be non-negative"
            raise ValueError(msg)
        if self.open_files < 0:
            msg = "open_files must be non-negative"
            raise ValueError(msg)


@dataclass
class ResourceThresholds:
    """Resource threshold configuration."""

    cpu_percent: float = 80.0
    memory_percent: float = 85.0
    disk_percent: float = 90.0
    network_mbps: float = 1000.0
    process_count: int = 1000
    open_files: int = 1000

    def __post_init__(self):
        """Validate thresholds are within valid ranges."""
        for field_name in ["cpu_percent", "memory_percent", "disk_percent"]:
            value = getattr(self, field_name)
            if not 0 <= value <= 100:
                msg = f"{field_name} must be between 0 and 100"
                raise ValueError(msg)

        if self.network_mbps < 0:
            msg = "network_mbps must be non-negative"
            raise ValueError(msg)
        if self.process_count < 0:
            msg = "process_count must be non-negative"
            raise ValueError(msg)
        if self.open_files < 0:
            msg = "open_files must be non-negative"
            raise ValueError(msg)


class MetricsStorage(Protocol):
    """Protocol for metrics storage implementations."""

    def store_metrics(self, metrics: ResourceMetrics) -> None:
        """Store resource metrics."""
        ...

    def get_metrics(
        self,
        start_time: datetime,
        end_time: datetime,
    ) -> list[ResourceMetrics]:
        """Retrieve metrics for a time range."""
        ...

    def cleanup_old_metrics(self, older_than: datetime) -> None:
        """Remove metrics older than specified time."""
        ...


class ResourceMonitor(ABC):
    """Abstract base class for resource monitoring."""

    def __init__(
        self,
        thresholds: ResourceThresholds | None = None,
        history_retention_days: int = 7,
        metrics_storage: MetricsStorage | None = None,
    ):
        """Initialize resource monitor.

        Args:
            thresholds: Resource thresholds configuration
            history_retention_days: Days to retain metrics history
            metrics_storage: Optional metrics storage implementation
        """
        self.thresholds = thresholds or ResourceThresholds()
        self.history_retention_days = history_retention_days
        self.metrics_storage = metrics_storage

    @abstractmethod
    def collect_metrics(self) -> ResourceMetrics:
        """Collect current resource metrics.

        Returns:
            Current resource metrics
        """
        ...

    @abstractmethod
    def check_thresholds(self, metrics: ResourceMetrics) -> dict[str, bool]:
        """Check if metrics exceed thresholds.

        Args:
            metrics: Resource metrics to check

        Returns:
            Dict mapping metric names to boolean indicating if threshold exceeded
        """
        ...

    def store_metrics(self, metrics: ResourceMetrics) -> None:
        """Store metrics if storage is configured."""
        if self.metrics_storage:
            self.metrics_storage.store_metrics(metrics)

    def get_historical_metrics(
        self,
        start_time: datetime,
        end_time: datetime,
    ) -> list[ResourceMetrics]:
        """Get historical metrics for time range.

        Args:
            start_time: Start of time range
            end_time: End of time range

        Returns:
            List of metrics in time range

        Raises:
            RuntimeError if no metrics storage configured
        """
        if not self.metrics_storage:
            msg = "No metrics storage configured"
            raise RuntimeError(msg)
        return self.metrics_storage.get_metrics(start_time, end_time)

    def cleanup_old_metrics(self) -> None:
        """Clean up metrics older than retention period."""
        if self.metrics_storage:
            cutoff = datetime.now().timestamp() - (self.history_retention_days * 24 * 60 * 60)
            self.metrics_storage.cleanup_old_metrics(datetime.fromtimestamp(cutoff))
