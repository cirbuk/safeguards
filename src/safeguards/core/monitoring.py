"""FounderX-specific resource monitoring extensions."""

from safeguards.monitoring import ResourceMonitor as BaseResourceMonitor
from safeguards.types import ResourceThresholds

from .config.settings import resource_monitor_settings


class ResourceMonitor(BaseResourceMonitor):
    """FounderX-specific resource monitor implementation."""

    def __init__(
        self,
        thresholds: ResourceThresholds | None = None,
        history_retention_days: int = 7,
    ):
        """Initialize FounderX resource monitor.

        Args:
            thresholds: Optional resource thresholds. If not provided,
                       will be loaded from FounderX settings.
            history_retention_days: Number of days to retain metrics history
        """
        # Use FounderX settings if no thresholds provided
        if thresholds is None:
            thresholds = ResourceThresholds(
                cpu_percent=resource_monitor_settings.cpu_threshold,
                memory_percent=resource_monitor_settings.memory_threshold,
                disk_percent=resource_monitor_settings.disk_threshold,
                network_mbps=resource_monitor_settings.network_threshold,
            )

        super().__init__(
            thresholds=thresholds,
            history_retention_days=history_retention_days,
        )
