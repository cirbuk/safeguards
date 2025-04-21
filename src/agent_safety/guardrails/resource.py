"""Resource monitoring guardrail implementation."""

from typing import Any, Optional

from ..monitoring import ResourceMonitor
from ..types.guardrail import Guardrail, RunContext


class ResourceGuardrail(Guardrail):
    """Guardrail for monitoring and enforcing resource limits.

    This guardrail:
    1. Monitors resource usage (CPU, memory, disk)
    2. Enforces resource limits
    3. Alerts on resource thresholds

    Example:
        ```python
        from agent_safety import ResourceMonitor, ResourceGuardrail
        from agent_safety.types import Agent

        resource_monitor = ResourceMonitor()
        agent = Agent(
            name="Assistant",
            instructions="You are a helpful assistant",
            guardrails=[ResourceGuardrail(resource_monitor)]
        )
        ```
    """

    def __init__(
        self,
        resource_monitor: ResourceMonitor,
        cpu_threshold: float = 80.0,
        memory_threshold: float = 80.0,
        disk_threshold: float = 80.0,
    ):
        """Initialize the resource guardrail.

        Args:
            resource_monitor: Resource monitor instance
            cpu_threshold: CPU usage threshold (percentage)
            memory_threshold: Memory usage threshold (percentage)
            disk_threshold: Disk usage threshold (percentage)
        """
        self.resource_monitor = resource_monitor
        self.cpu_threshold = cpu_threshold
        self.memory_threshold = memory_threshold
        self.disk_threshold = disk_threshold

    async def run(self, context: RunContext) -> Optional[str]:
        """Run resource checks before agent execution.

        Args:
            context: Run context with agent info

        Returns:
            Error message if resources exceeded, None otherwise
        """
        metrics = self.resource_monitor.get_metrics()

        # Check CPU usage
        if metrics.cpu_percent > self.cpu_threshold:
            return f"CPU usage ({metrics.cpu_percent}%) exceeds threshold ({self.cpu_threshold}%)"

        # Check memory usage
        if metrics.memory_percent > self.memory_threshold:
            return f"Memory usage ({metrics.memory_percent}%) exceeds threshold ({self.memory_threshold}%)"

        # Check disk usage
        if metrics.disk_percent > self.disk_threshold:
            return f"Disk usage ({metrics.disk_percent}%) exceeds threshold ({self.disk_threshold}%)"

        return None

    async def validate(self, context: RunContext, result: Any) -> Optional[str]:
        """Validate resource usage after agent execution.

        Args:
            context: Run context with agent info
            result: Result from agent execution

        Returns:
            Error message if validation fails, None otherwise
        """
        metrics = self.resource_monitor.get_metrics()

        # Check if execution caused resource spikes
        if metrics.cpu_percent > self.cpu_threshold:
            return f"Agent execution caused CPU spike ({metrics.cpu_percent}%)"

        if metrics.memory_percent > self.memory_threshold:
            return f"Agent execution caused memory spike ({metrics.memory_percent}%)"

        if metrics.disk_percent > self.disk_threshold:
            return f"Agent execution caused disk usage spike ({metrics.disk_percent}%)"

        return None
