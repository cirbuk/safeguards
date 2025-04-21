"""Tests for resource monitor."""

import pytest
from unittest.mock import patch, MagicMock

from agent_safety.monitoring.resource_monitor import ResourceMonitor, ResourceMetrics


@pytest.fixture
def resource_monitor() -> ResourceMonitor:
    """Create a resource monitor instance."""
    return ResourceMonitor(
        cpu_threshold=80.0,
        memory_threshold=85.0,
        disk_threshold=90.0,
        check_interval=5,
    )


def test_resource_monitor_initialization(resource_monitor: ResourceMonitor):
    """Test resource monitor initialization."""
    assert resource_monitor.cpu_threshold == 80.0
    assert resource_monitor.memory_threshold == 85.0
    assert resource_monitor.disk_threshold == 90.0
    assert resource_monitor.check_interval == 5


@patch("psutil.cpu_percent")
@patch("psutil.virtual_memory")
@patch("psutil.disk_usage")
def test_get_current_metrics(
    mock_disk_usage: MagicMock,
    mock_virtual_memory: MagicMock,
    mock_cpu_percent: MagicMock,
    resource_monitor: ResourceMonitor,
):
    """Test getting current metrics."""
    # Setup mock returns
    mock_cpu_percent.return_value = 50.0
    mock_virtual_memory.return_value.percent = 60.0
    mock_disk_usage.return_value.percent = 70.0

    metrics = resource_monitor.get_current_metrics()
    assert isinstance(metrics, ResourceMetrics)
    assert metrics.cpu_usage == 50.0
    assert metrics.memory_usage == 60.0
    assert metrics.disk_usage == 70.0


def test_has_exceeded_thresholds(resource_monitor: ResourceMonitor):
    """Test threshold checking."""
    # Test when all metrics are below thresholds
    metrics = ResourceMetrics(
        cpu_usage=70.0,
        memory_usage=75.0,
        disk_usage=80.0,
    )
    assert not resource_monitor.has_exceeded_thresholds(metrics)

    # Test when CPU exceeds threshold
    metrics = ResourceMetrics(
        cpu_usage=85.0,
        memory_usage=75.0,
        disk_usage=80.0,
    )
    assert resource_monitor.has_exceeded_thresholds(metrics)

    # Test when memory exceeds threshold
    metrics = ResourceMetrics(
        cpu_usage=70.0,
        memory_usage=90.0,
        disk_usage=80.0,
    )
    assert resource_monitor.has_exceeded_thresholds(metrics)

    # Test when disk exceeds threshold
    metrics = ResourceMetrics(
        cpu_usage=70.0,
        memory_usage=75.0,
        disk_usage=95.0,
    )
    assert resource_monitor.has_exceeded_thresholds(metrics)


def test_get_resource_usage_summary(resource_monitor: ResourceMonitor):
    """Test resource usage summary formatting."""
    metrics = ResourceMetrics(
        cpu_usage=50.0,
        memory_usage=60.0,
        disk_usage=70.0,
    )

    summary = resource_monitor.get_resource_usage_summary(metrics)
    assert "CPU Usage: 50.0%" in summary
    assert "Memory Usage: 60.0%" in summary
    assert "Disk Usage: 70.0%" in summary
