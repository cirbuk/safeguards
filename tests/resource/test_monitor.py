"""Tests for the resource monitor module."""

from unittest.mock import MagicMock, patch

import pytest

from safeguards.resource.monitor import ResourceMonitor
from safeguards.types import ResourceConfig, ResourceMetrics


@pytest.fixture()
def resource_config() -> ResourceConfig:
    """Create a test resource configuration."""
    return ResourceConfig(
        cpu_threshold=80.0,
        memory_threshold=85.0,
        disk_threshold=90.0,
        check_interval=5,
    )


@pytest.fixture()
def resource_monitor(resource_config: ResourceConfig) -> ResourceMonitor:
    """Create a resource monitor instance for testing."""
    return ResourceMonitor(resource_config)


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
    """Test getting current resource metrics."""
    # Mock resource usage values
    mock_cpu_percent.return_value = 60.0
    mock_virtual_memory.return_value.percent = 70.0
    mock_disk_usage.return_value.percent = 75.0

    metrics = resource_monitor.get_current_metrics()
    assert isinstance(metrics, ResourceMetrics)
    assert metrics.cpu_usage == 60.0
    assert metrics.memory_usage == 70.0
    assert metrics.disk_usage == 75.0


def test_has_exceeded_thresholds(resource_monitor: ResourceMonitor):
    """Test checking if resource usage has exceeded thresholds."""
    # Test when all metrics are below thresholds
    metrics = ResourceMetrics(cpu_percent=70.0, memory_percent=75.0, disk_percent=80.0)
    assert not resource_monitor.has_exceeded_thresholds(metrics)

    # Test when CPU usage exceeds threshold
    metrics = ResourceMetrics(cpu_percent=85.0, memory_percent=75.0, disk_percent=80.0)
    assert resource_monitor.has_exceeded_thresholds(metrics)

    # Test when memory usage exceeds threshold
    metrics = ResourceMetrics(cpu_percent=70.0, memory_percent=90.0, disk_percent=80.0)
    assert resource_monitor.has_exceeded_thresholds(metrics)

    # Test when disk usage exceeds threshold
    metrics = ResourceMetrics(cpu_percent=70.0, memory_percent=75.0, disk_percent=95.0)
    assert resource_monitor.has_exceeded_thresholds(metrics)


def test_get_resource_usage_summary(resource_monitor: ResourceMonitor):
    """Test getting resource usage summary."""
    metrics = ResourceMetrics(cpu_percent=60.0, memory_percent=70.0, disk_percent=80.0)
    summary = resource_monitor.get_resource_usage_summary(metrics)

    assert "CPU Usage" in summary
    assert "Memory Usage" in summary
    assert "Disk Usage" in summary
    assert "60.0%" in summary
    assert "70.0%" in summary
    assert "80.0%" in summary
