"""Unit tests for system monitor implementation."""

from datetime import datetime, timedelta

import psutil
import pytest

from safeguards.base.monitoring import ResourceThresholds
from safeguards.monitoring.system_monitor import SystemResourceMonitor


class TestSystemResourceMonitor:
    """Test cases for SystemResourceMonitor."""

    @pytest.fixture()
    def monitor(self):
        """Create SystemResourceMonitor instance for testing."""
        return SystemResourceMonitor(
            thresholds=ResourceThresholds(),
            process_filter=None,
        )

    def test_collect_metrics(self, monitor):
        """Test metrics collection."""
        metrics = monitor.collect_metrics()

        # Verify metric ranges
        assert 0 <= metrics.cpu_percent <= 100
        assert 0 <= metrics.memory_percent <= 100
        assert 0 <= metrics.disk_percent <= 100
        assert metrics.network_mbps >= 0
        assert metrics.process_count >= 0
        assert metrics.open_files >= 0

        # Verify timestamp
        assert isinstance(metrics.timestamp, datetime)
        assert metrics.timestamp <= datetime.now()
        assert metrics.timestamp >= datetime.now() - timedelta(minutes=1)

    def test_process_filter(self):
        """Test process filtering."""
        # Create monitor with process filter
        monitor = SystemResourceMonitor(
            thresholds=ResourceThresholds(),
            process_filter="python",
        )

        metrics = monitor.collect_metrics()

        # Get actual Python process count
        python_count = len(
            [p for p in psutil.process_iter(["name"]) if "python" in p.info["name"].lower()],
        )

        assert metrics.process_count == python_count

    def test_check_thresholds(self, monitor):
        """Test threshold checking."""
        metrics = monitor.collect_metrics()
        exceeded = monitor.check_thresholds(metrics)

        # Verify all expected metrics are checked
        assert "cpu_percent" in exceeded
        assert "memory_percent" in exceeded
        assert "disk_percent" in exceeded
        assert "network_mbps" in exceeded
        assert "process_count" in exceeded
        assert "open_files" in exceeded

        # Verify boolean results
        for _metric, is_exceeded in exceeded.items():
            assert isinstance(is_exceeded, bool)

        # Test with custom thresholds
        low_thresholds = ResourceThresholds(
            cpu_percent=1.0,
            memory_percent=1.0,
            disk_percent=1.0,
            network_mbps=1.0,
            process_count=1,
            open_files=1,
        )
        monitor.thresholds = low_thresholds

        exceeded = monitor.check_thresholds(metrics)
        assert any(exceeded.values()), "Some thresholds should be exceeded"

    def test_metrics_history(self, monitor):
        """Test metrics history management."""
        # Collect multiple metrics
        for _ in range(5):
            monitor.collect_metrics()

        # Verify history retention
        assert len(monitor.metrics_history) <= monitor.history_retention_days * 24 * 60

        # Verify history ordering
        for i in range(1, len(monitor.metrics_history)):
            assert monitor.metrics_history[i].timestamp > monitor.metrics_history[i - 1].timestamp

    @pytest.mark.asyncio()
    async def test_async_collection(self, monitor):
        """Test asynchronous metrics collection."""
        metrics = await monitor.collect_metrics_async()

        # Verify metric ranges
        assert 0 <= metrics.cpu_percent <= 100
        assert 0 <= metrics.memory_percent <= 100
        assert 0 <= metrics.disk_percent <= 100
        assert metrics.network_mbps >= 0
        assert metrics.process_count >= 0
        assert metrics.open_files >= 0
