"""System resource monitoring implementation using psutil."""

import asyncio
from datetime import datetime, timedelta

import psutil

from safeguards.base.monitoring import (
    MetricsStorage,
    ResourceMetrics,
    ResourceMonitor,
    ResourceThresholds,
)


class SystemResourceMonitor(ResourceMonitor):
    """Resource monitor implementation using psutil."""

    def __init__(
        self,
        thresholds: ResourceThresholds | None = None,
        history_retention_days: int = 7,
        metrics_storage: MetricsStorage | None = None,
        process_filter: str | None = None,
    ):
        """Initialize system resource monitor.

        Args:
            thresholds: Resource thresholds configuration
            history_retention_days: Days to retain metrics history
            metrics_storage: Optional metrics storage implementation
            process_filter: Optional process name filter for monitoring
        """
        super().__init__(
            thresholds=thresholds,
            history_retention_days=history_retention_days,
            metrics_storage=metrics_storage,
        )
        self.process_filter = process_filter
        self.metrics_history: list[ResourceMetrics] = []
        self.history_retention_days = history_retention_days

    def collect_metrics(self) -> ResourceMetrics:
        """Collect current system resource metrics.

        Returns:
            Current resource metrics
        """
        # Get CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)

        # Get memory usage
        memory = psutil.virtual_memory()
        memory_percent = memory.percent

        # Get disk usage
        disk = psutil.disk_usage("/")
        disk_percent = disk.percent

        # Get network I/O
        net_io = psutil.net_io_counters()
        network_mbps = (net_io.bytes_sent + net_io.bytes_recv) / (1024 * 1024)  # Convert to MB

        # Get process info
        processes = list(psutil.process_iter(["name", "open_files"]))
        if self.process_filter:
            processes = [
                p for p in processes if self.process_filter.lower() in p.info["name"].lower()
            ]
        process_count = len(processes)

        # Get open files count
        try:
            open_files = len(psutil.Process().open_files())
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            open_files = 0

        metrics = ResourceMetrics(
            timestamp=datetime.now(),
            cpu_percent=cpu_percent,
            memory_percent=memory_percent,
            disk_percent=disk_percent,
            network_mbps=network_mbps,
            process_count=process_count,
            open_files=open_files,
        )

        # Add to history and remove old entries
        self.metrics_history.append(metrics)
        cutoff_time = datetime.now() - timedelta(days=self.history_retention_days)
        self.metrics_history = [m for m in self.metrics_history if m.timestamp >= cutoff_time]

        return metrics

    async def collect_metrics_async(self) -> ResourceMetrics:
        """Collect metrics asynchronously.

        Returns:
            Current resource metrics
        """
        # Use run_in_executor for CPU-bound operations
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.collect_metrics)

    def check_thresholds(self, metrics: ResourceMetrics) -> dict[str, bool]:
        """Check if metrics exceed configured thresholds.

        Args:
            metrics: Resource metrics to check

        Returns:
            Dict mapping metric names to boolean indicating if threshold exceeded
        """
        return {
            "cpu_percent": metrics.cpu_percent > self.thresholds.cpu_percent,
            "memory_percent": metrics.memory_percent > self.thresholds.memory_percent,
            "disk_percent": metrics.disk_percent > self.thresholds.disk_percent,
            "network_mbps": metrics.network_mbps > self.thresholds.network_mbps,
            "process_count": metrics.process_count > self.thresholds.process_count,
            "open_files": metrics.open_files > self.thresholds.open_files,
        }
