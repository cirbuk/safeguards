"""Resource utilization monitoring for agent safety.

This module provides functionality for:
- CPU usage tracking
- Memory utilization monitoring
- Disk space monitoring
- Network usage tracking
- Resource usage alerts
"""

import logging
from datetime import datetime, timedelta

import psutil

from safeguards.types import (
    ResourceMetrics as BaseResourceMetrics,
)
from safeguards.types import (
    ResourceThresholds,
)

logger = logging.getLogger(__name__)


class ResourceMetrics(BaseResourceMetrics):
    """Extended resource metrics data structure."""

    def __init__(
        self,
        cpu_percent: float = 0.0,
        memory_percent: float = 0.0,
        disk_percent: float = 0.0,
        network_mbps: float = 0.0,
        process_count: int = 0,
        open_files: int = 0,
        timestamp: datetime | None = None,
        # Additional fields
        memory_used: int | None = None,
        memory_total: int | None = None,
        disk_used: int | None = None,
        disk_total: int | None = None,
        network_sent: int | None = None,
        network_received: int | None = None,
        network_speed: float | None = None,
        # Backward compatibility
        cpu_usage: float | None = None,
        memory_usage: float | None = None,
        disk_usage: float | None = None,
    ):
        """Initialize resource metrics.

        Args:
            cpu_percent: CPU usage percentage
            memory_percent: Memory usage percentage
            disk_percent: Disk usage percentage
            network_mbps: Network usage in mbps
            process_count: Number of processes
            open_files: Number of open files
            timestamp: Time when metrics were collected
            memory_used: Memory used in bytes
            memory_total: Total memory in bytes
            disk_used: Disk used in bytes
            disk_total: Total disk space in bytes
            network_sent: Bytes sent
            network_received: Bytes received
            network_speed: Alias for network_mbps
            cpu_usage: Alias for cpu_percent (backward compatibility)
            memory_usage: Alias for memory_percent (backward compatibility)
            disk_usage: Alias for disk_percent (backward compatibility)
        """
        # Handle backward compatibility
        cpu = cpu_usage if cpu_usage is not None else cpu_percent
        memory = memory_usage if memory_usage is not None else memory_percent
        disk = disk_usage if disk_usage is not None else disk_percent
        network = network_speed if network_speed is not None else network_mbps

        super().__init__(
            timestamp=timestamp,
            cpu_percent=cpu,
            memory_percent=memory,
            disk_percent=disk,
            network_mbps=network,
            process_count=process_count,
            open_files=open_files,
        )

        # Additional fields
        self.memory_used = memory_used
        self.memory_total = memory_total
        self.disk_used = disk_used
        self.disk_total = disk_total
        self.network_sent = network_sent
        self.network_received = network_received
        self.network_speed = network

        # For backward compatibility
        self.cpu_usage = self.cpu_percent
        self.memory_usage = self.memory_percent
        self.disk_usage = self.disk_percent


class ResourceMonitor:
    """Monitors and tracks system resource utilization."""

    def __init__(
        self,
        thresholds: ResourceThresholds | None = None,
        history_retention_days: int = 7,
        # Backward compatibility parameters
        cpu_threshold: float | None = None,
        memory_threshold: float | None = None,
        disk_threshold: float | None = None,
        check_interval: int | None = None,
    ):
        """Initialize resource monitor.

        Args:
            thresholds: Optional resource thresholds. If not provided,
                       will be loaded from environment variables.
            history_retention_days: Days to retain metrics history

            # Backward compatibility parameters
            cpu_threshold: CPU usage threshold (percentage)
            memory_threshold: Memory usage threshold (percentage)
            disk_threshold: Disk usage threshold (percentage)
            check_interval: Check interval in seconds
        """
        # Set backward compatibility attributes
        self.cpu_threshold = cpu_threshold or 80.0
        self.memory_threshold = memory_threshold or 85.0
        self.disk_threshold = disk_threshold or 90.0
        self.check_interval = check_interval or 5

        # Create thresholds if needed
        if thresholds is None and (cpu_threshold or memory_threshold or disk_threshold):
            thresholds = ResourceThresholds(
                cpu_percent=self.cpu_threshold,
                memory_percent=self.memory_threshold,
                disk_percent=self.disk_threshold,
            )

        self.thresholds = thresholds or ResourceThresholds()
        self.history_retention_days = history_retention_days
        self.metrics_history: list[ResourceMetrics] = []

        # Initialize network I/O baseline
        net_io = psutil.net_io_counters()
        self._last_net_io = (net_io.bytes_sent, net_io.bytes_recv)
        self._last_net_time = datetime.now()

    # Add backward compatibility methods
    def has_exceeded_thresholds(self, metrics: ResourceMetrics) -> bool:
        """Check if any metrics have exceeded thresholds.

        Args:
            metrics: Resource metrics to check

        Returns:
            True if any threshold is exceeded
        """
        exceeded = self.check_thresholds(metrics)
        return any(exceeded.values())

    def get_resource_usage_summary(self, metrics: ResourceMetrics) -> str:
        """Get a formatted summary of resource usage.

        Args:
            metrics: Resource metrics to summarize

        Returns:
            Formatted summary string
        """
        return (
            f"CPU Usage: {metrics.cpu_percent:.1f}%\n"
            f"Memory Usage: {metrics.memory_percent:.1f}%\n"
            f"Disk Usage: {metrics.disk_percent:.1f}%\n"
        )

    def get_current_metrics(self) -> ResourceMetrics:
        """Get current resource utilization metrics.

        Returns:
            Current resource metrics
        """
        # Get CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)

        # Get memory usage
        memory = psutil.virtual_memory()
        memory_used = memory.used
        memory_total = memory.total
        memory_percent = memory.percent

        # Get disk usage
        disk = psutil.disk_usage("/")
        disk_used = disk.used
        disk_total = disk.total
        disk_percent = disk.percent

        # Get network usage
        current_net_io = psutil.net_io_counters()
        current_time = datetime.now()
        time_diff = (current_time - self._last_net_time).total_seconds()

        # Calculate network speed in Mbps
        bytes_sent = current_net_io.bytes_sent - self._last_net_io[0]
        bytes_received = current_net_io.bytes_recv - self._last_net_io[1]
        network_speed = ((bytes_sent + bytes_received) * 8) / (time_diff * 1_000_000)  # Mbps

        # Update network tracking
        self._last_net_io = (current_net_io.bytes_sent, current_net_io.bytes_recv)
        self._last_net_time = current_time

        metrics = ResourceMetrics(
            timestamp=current_time,
            cpu_percent=cpu_percent,
            memory_used=memory_used,
            memory_total=memory_total,
            memory_percent=memory_percent,
            disk_used=disk_used,
            disk_total=disk_total,
            disk_percent=disk_percent,
            network_sent=bytes_sent,
            network_received=bytes_received,
            network_mbps=network_speed,
        )

        self._add_to_history(metrics)
        return metrics

    def check_thresholds(self, metrics: ResourceMetrics) -> dict[str, bool]:
        """Check if metrics exceed thresholds.

        Args:
            metrics: Resource metrics to check

        Returns:
            Dictionary of metric names to boolean indicating if threshold exceeded
        """
        return {
            "cpu": metrics.cpu_percent > self.cpu_threshold,
            "memory": metrics.memory_percent > self.memory_threshold,
            "disk": metrics.disk_percent > self.disk_threshold,
        }

    def get_average_metrics(self, hours: int = 1) -> ResourceMetrics:
        """Get average metrics over specified time period.

        Args:
            hours: Number of hours to average over

        Returns:
            Average resource metrics
        """
        cutoff_time = datetime.now() - timedelta(hours=hours)
        relevant_metrics = [m for m in self.metrics_history if m.timestamp >= cutoff_time]

        if not relevant_metrics:
            return self.get_current_metrics()

        return ResourceMetrics(
            timestamp=datetime.now(),
            cpu_percent=sum(m.cpu_percent for m in relevant_metrics) / len(relevant_metrics),
            memory_used=sum(m.memory_used for m in relevant_metrics) // len(relevant_metrics),
            memory_total=relevant_metrics[0].memory_total,
            memory_percent=sum(m.memory_percent for m in relevant_metrics) / len(relevant_metrics),
            disk_used=sum(m.disk_used for m in relevant_metrics) // len(relevant_metrics),
            disk_total=relevant_metrics[0].disk_total,
            disk_percent=sum(m.disk_percent for m in relevant_metrics) / len(relevant_metrics),
            network_sent=sum(m.network_sent for m in relevant_metrics),
            network_received=sum(m.network_received for m in relevant_metrics),
            network_speed=sum(m.network_speed for m in relevant_metrics) / len(relevant_metrics),
        )

    def _add_to_history(self, metrics: ResourceMetrics) -> None:
        """Add metrics to history and cleanup old entries.

        Args:
            metrics: Metrics to add to history
        """
        self.metrics_history.append(metrics)

        # Cleanup old metrics
        cutoff_time = datetime.now() - timedelta(days=self.history_retention_days)
        self.metrics_history = [m for m in self.metrics_history if m.timestamp >= cutoff_time]
