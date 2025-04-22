"""Tests for resource guardrail."""

import pytest
from unittest.mock import MagicMock

from agent_safety.monitoring.resource_monitor import ResourceMonitor, ResourceMetrics
from agent_safety.guardrails.resource import ResourceGuardrail
from agent_safety.types import Agent, RunContext


class TestAgent(Agent):
    """Test agent implementation."""

    def run(self, **kwargs):
        """Mock implementation."""
        return {"status": "success"}


@pytest.fixture
def test_agent() -> Agent:
    """Create a test agent."""
    return TestAgent(name="test_agent")


@pytest.fixture
def resource_monitor() -> ResourceMonitor:
    """Create a mock resource monitor."""
    monitor = MagicMock(spec=ResourceMonitor)

    # Default metrics for tests
    default_metrics = ResourceMetrics(
        cpu_usage=50.0,
        memory_usage=60.0,
        disk_usage=70.0,
        process_count=10,
        open_files=20,
    )
    monitor.get_current_metrics.return_value = default_metrics

    # Set up check_thresholds to return appropriate values based on metrics
    def check_thresholds_mock(metrics):
        result = {}
        if hasattr(metrics, "cpu_usage"):
            result["cpu_percent"] = metrics.cpu_usage > 80.0
        elif hasattr(metrics, "cpu_percent"):
            result["cpu_percent"] = metrics.cpu_percent > 80.0

        if hasattr(metrics, "memory_usage"):
            result["memory_percent"] = metrics.memory_usage > 85.0
        elif hasattr(metrics, "memory_percent"):
            result["memory_percent"] = metrics.memory_percent > 85.0

        if hasattr(metrics, "disk_usage"):
            result["disk_percent"] = metrics.disk_usage > 90.0
        elif hasattr(metrics, "disk_percent"):
            result["disk_percent"] = metrics.disk_percent > 90.0

        return result

    monitor.check_thresholds.side_effect = check_thresholds_mock

    # Also mock the collect_metrics method
    monitor.collect_metrics = monitor.get_current_metrics

    return monitor


@pytest.fixture
def resource_guardrail(resource_monitor: ResourceMonitor) -> ResourceGuardrail:
    """Create a resource guardrail instance."""
    return ResourceGuardrail(
        resource_monitor,
        cpu_threshold=80.0,
        memory_threshold=85.0,
        disk_threshold=90.0,
    )


@pytest.fixture
def run_context(test_agent: Agent) -> RunContext:
    """Create a test run context."""
    return RunContext(
        agent=test_agent,
        inputs={"test": "input"},
        metadata={"test": "metadata"},
    )


@pytest.mark.asyncio
async def test_run_within_limits(
    resource_guardrail: ResourceGuardrail,
    run_context: RunContext,
):
    """Test run with resources within limits."""
    result = await resource_guardrail.run(run_context)
    assert result is None


@pytest.mark.asyncio
async def test_run_cpu_exceeded(
    resource_guardrail: ResourceGuardrail,
    resource_monitor: ResourceMonitor,
    run_context: RunContext,
):
    """Test run with CPU usage exceeded."""
    resource_monitor.get_current_metrics.return_value = ResourceMetrics(
        cpu_usage=85.0,
        memory_usage=60.0,
        disk_usage=70.0,
        process_count=10,
        open_files=20,
    )

    result = await resource_guardrail.run(run_context)
    assert result is not None
    assert "cpu_percent" in result
    assert "exceeded threshold" in result


@pytest.mark.asyncio
async def test_run_memory_exceeded(
    resource_guardrail: ResourceGuardrail,
    resource_monitor: ResourceMonitor,
    run_context: RunContext,
):
    """Test run with memory usage exceeded."""
    resource_monitor.get_current_metrics.return_value = ResourceMetrics(
        cpu_usage=50.0,
        memory_usage=90.0,
        disk_usage=70.0,
        process_count=10,
        open_files=20,
    )

    result = await resource_guardrail.run(run_context)
    assert result is not None
    assert "memory_percent" in result
    assert "exceeded threshold" in result


@pytest.mark.asyncio
async def test_run_disk_exceeded(
    resource_guardrail: ResourceGuardrail,
    resource_monitor: ResourceMonitor,
    run_context: RunContext,
):
    """Test run with disk usage exceeded."""
    resource_monitor.get_current_metrics.return_value = ResourceMetrics(
        cpu_usage=50.0,
        memory_usage=60.0,
        disk_usage=95.0,
        process_count=10,
        open_files=20,
    )

    result = await resource_guardrail.run(run_context)
    assert result is not None
    assert "disk_percent" in result
    assert "exceeded threshold" in result


@pytest.mark.asyncio
async def test_validate_within_limits(
    resource_guardrail: ResourceGuardrail,
    run_context: RunContext,
):
    """Test validation with resources within limits."""
    result = await resource_guardrail.validate(run_context, {"test": "result"})
    assert result is None


@pytest.mark.asyncio
async def test_validate_resource_spike(
    resource_guardrail: ResourceGuardrail,
    resource_monitor: ResourceMonitor,
    run_context: RunContext,
):
    """Test validation with resource spike."""
    resource_monitor.get_current_metrics.return_value = ResourceMetrics(
        cpu_usage=90.0,
        memory_usage=60.0,
        disk_usage=70.0,
        process_count=10,
        open_files=20,
    )

    result = await resource_guardrail.validate(run_context, {"test": "result"})
    assert result is not None
    assert "cpu_percent" in result
    assert "exceeded threshold" in result
