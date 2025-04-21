"""Resource utilization monitoring for agent safety.

This module provides functionality for:
- CPU usage tracking
- Memory utilization monitoring
- Disk space monitoring
- Network usage tracking
- Resource usage alerts
"""

import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from decimal import Decimal

import psutil

from agent_safety.types import ResourceThresholds, ResourceMetrics


class ResourceMonitor:
    """Monitors and tracks system resource utilization."""

    def __init__(
        self,
        thresholds: Optional[ResourceThresholds] = None,
        history_retention_days: int = 7,
    ):
        """Initialize resource monitor.

        Args:
            thresholds: Optional resource thresholds. If not provided,
                       will be loaded from environment variables.
            history_retention_days: Number of days to retain metrics history
        """
        self.thresholds = thresholds or ResourceThresholds()
        self.history_retention_days = history_retention_days
        self.metrics_history: List[ResourceMetrics] = []

        # Initialize network I/O baseline
        net_io = psutil.net_io_counters()
        self._last_net_io = (net_io.bytes_sent, net_io.bytes_recv)
        self._last_net_time = datetime.now()

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
        network_speed = ((bytes_sent + bytes_received) * 8) / (
            time_diff * 1_000_000
        )  # Mbps

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
            network_speed=network_speed,
        )

        self._add_to_history(metrics)
        return metrics

    def check_thresholds(self) -> List[Tuple[str, float, float]]:
        """Check if current metrics exceed thresholds.

        Returns:
            List of (resource_name, current_value, threshold) for exceeded thresholds
        """
        metrics = self.get_current_metrics()
        alerts = []

        if metrics.cpu_percent > self.thresholds.cpu_percent:
            alerts.append(("CPU", metrics.cpu_percent, self.thresholds.cpu_percent))

        if metrics.memory_percent > self.thresholds.memory_percent:
            alerts.append(
                ("Memory", metrics.memory_percent, self.thresholds.memory_percent)
            )

        if metrics.disk_percent > self.thresholds.disk_percent:
            alerts.append(("Disk", metrics.disk_percent, self.thresholds.disk_percent))

        if metrics.network_speed > self.thresholds.network_mbps:
            alerts.append(
                ("Network", metrics.network_speed, self.thresholds.network_mbps)
            )

        return alerts

    def get_average_metrics(self, hours: int = 1) -> ResourceMetrics:
        """Get average metrics over specified time period.

        Args:
            hours: Number of hours to average over

        Returns:
            Average resource metrics
        """
        cutoff_time = datetime.now() - timedelta(hours=hours)
        relevant_metrics = [
            m for m in self.metrics_history if m.timestamp >= cutoff_time
        ]

        if not relevant_metrics:
            return self.get_current_metrics()

        avg_metrics = ResourceMetrics(
            timestamp=datetime.now(),
            cpu_percent=sum(m.cpu_percent for m in relevant_metrics)
            / len(relevant_metrics),
            memory_used=sum(m.memory_used for m in relevant_metrics)
            // len(relevant_metrics),
            memory_total=relevant_metrics[0].memory_total,
            memory_percent=sum(m.memory_percent for m in relevant_metrics)
            / len(relevant_metrics),
            disk_used=sum(m.disk_used for m in relevant_metrics)
            // len(relevant_metrics),
            disk_total=relevant_metrics[0].disk_total,
            disk_percent=sum(m.disk_percent for m in relevant_metrics)
            / len(relevant_metrics),
            network_sent=sum(m.network_sent for m in relevant_metrics),
            network_received=sum(m.network_received for m in relevant_metrics),
            network_speed=sum(m.network_speed for m in relevant_metrics)
            / len(relevant_metrics),
        )

        return avg_metrics

    def _add_to_history(self, metrics: ResourceMetrics) -> None:
        """Add metrics to history and cleanup old entries.

        Args:
            metrics: Metrics to add to history
        """
        self.metrics_history.append(metrics)

        # Cleanup old metrics
        cutoff_time = datetime.now() - timedelta(days=self.history_retention_days)
        self.metrics_history = [
            m for m in self.metrics_history if m.timestamp >= cutoff_time
        ]
