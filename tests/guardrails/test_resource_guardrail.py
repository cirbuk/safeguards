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
    monitor.get_current_metrics.return_value = ResourceMetrics(
        cpu_usage=50.0,
        memory_usage=60.0,
        disk_usage=70.0,
    )
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
    )

    result = await resource_guardrail.run(run_context)
    assert "CPU usage" in result
    assert "exceeds threshold" in result


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
    )

    result = await resource_guardrail.run(run_context)
    assert "Memory usage" in result
    assert "exceeds threshold" in result


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
    )

    result = await resource_guardrail.run(run_context)
    assert "Disk usage" in result
    assert "exceeds threshold" in result


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
    )

    result = await resource_guardrail.validate(run_context, {"test": "result"})
    assert "CPU spike" in result
