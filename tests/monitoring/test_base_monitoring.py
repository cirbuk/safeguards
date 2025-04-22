"""Unit tests for base monitoring system."""

import pytest
from datetime import datetime
from decimal import Decimal

from safeguards.base.monitoring import ResourceMetrics, ResourceThresholds


def test_resource_metrics():
    """Test ResourceMetrics data structure."""
    # Valid metrics
    metrics = ResourceMetrics(
        timestamp=datetime.now(),
        cpu_percent=50.0,
        memory_percent=60.0,
        disk_percent=70.0,
        network_mbps=100.0,
        process_count=10,
        open_files=5,
    )

    assert 0 <= metrics.cpu_percent <= 100
    assert 0 <= metrics.memory_percent <= 100
    assert 0 <= metrics.disk_percent <= 100
    assert metrics.network_mbps >= 0
    assert metrics.process_count >= 0
    assert metrics.open_files >= 0

    # Invalid metrics
    with pytest.raises(ValueError):
        ResourceMetrics(
            timestamp=datetime.now(),
            cpu_percent=150.0,  # Over 100%
            memory_percent=60.0,
            disk_percent=70.0,
            network_mbps=100.0,
            process_count=10,
            open_files=5,
        )

    with pytest.raises(ValueError):
        ResourceMetrics(
            timestamp=datetime.now(),
            cpu_percent=50.0,
            memory_percent=60.0,
            disk_percent=70.0,
            network_mbps=-1.0,  # Negative
            process_count=10,
            open_files=5,
        )


def test_resource_thresholds():
    """Test ResourceThresholds configuration."""
    # Default thresholds
    thresholds = ResourceThresholds()

    assert 0 <= thresholds.cpu_percent <= 100
    assert 0 <= thresholds.memory_percent <= 100
    assert 0 <= thresholds.disk_percent <= 100
    assert thresholds.network_mbps > 0
    assert thresholds.process_count > 0
    assert thresholds.open_files > 0

    # Custom thresholds
    custom = ResourceThresholds(
        cpu_percent=70.0,
        memory_percent=80.0,
        disk_percent=90.0,
        network_mbps=500.0,
        process_count=100,
        open_files=50,
    )

    assert custom.cpu_percent == 70.0
    assert custom.memory_percent == 80.0
    assert custom.disk_percent == 90.0
    assert custom.network_mbps == 500.0
    assert custom.process_count == 100
    assert custom.open_files == 50

    # Invalid thresholds
    with pytest.raises(ValueError):
        ResourceThresholds(cpu_percent=150.0)  # Over 100%

    with pytest.raises(ValueError):
        ResourceThresholds(network_mbps=-1.0)  # Negative

    with pytest.raises(ValueError):
        ResourceThresholds(process_count=-5)  # Negative
