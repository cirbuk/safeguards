"""Tests for pool health monitoring system."""

import pytest
from datetime import datetime, timedelta
from decimal import Decimal

from agent_safety.core.pool_health import (
    PoolHealthMonitor,
    HealthStatus,
    HealthMetricType,
    HealthMetric,
    PoolHealthReport,
)
from agent_safety.testing.mock_implementations import (
    MockNotificationManager,
    MockViolationReporter,
)


@pytest.fixture
def notification_manager():
    """Create mock notification manager."""
    return MockNotificationManager()


@pytest.fixture
def violation_reporter(notification_manager):
    """Create mock violation reporter."""
    return MockViolationReporter(notification_manager)


@pytest.fixture
def health_monitor(notification_manager, violation_reporter):
    """Create health monitor with mock dependencies."""
    return PoolHealthMonitor(
        notification_manager=notification_manager,
        violation_reporter=violation_reporter,
        check_interval=timedelta(seconds=1),  # Short interval for testing
    )


class TestPoolHealthMonitor:
    """Test cases for pool health monitoring system."""

    def test_record_metric(self, health_monitor):
        """Test recording a basic metric."""
        # Arrange
        pool_id = "test_pool"
        metric_type = HealthMetricType.UTILIZATION
        value = 0.5  # 50% utilization

        # Act
        health_monitor.record_metric(pool_id, metric_type, value)

        # Assert
        report = health_monitor.get_pool_health(pool_id)
        assert report is not None
        assert report.status == HealthStatus.HEALTHY
        assert metric_type in report.metrics
        assert report.metrics[metric_type].value == value

    def test_warning_threshold(self, health_monitor):
        """Test warning threshold triggers."""
        # Arrange
        pool_id = "test_pool"
        metric_type = HealthMetricType.UTILIZATION
        value = 0.85  # 85% utilization (above warning threshold)

        # Act
        health_monitor.record_metric(pool_id, metric_type, value)

        # Assert
        report = health_monitor.get_pool_health(pool_id)
        assert report.status == HealthStatus.WARNING
        assert any("Warning" in rec for rec in report.recommendations)

    def test_critical_threshold(self, health_monitor):
        """Test critical threshold triggers."""
        # Arrange
        pool_id = "test_pool"
        metric_type = HealthMetricType.UTILIZATION
        value = 0.96  # 96% utilization (above critical threshold)

        # Act
        health_monitor.record_metric(pool_id, metric_type, value)

        # Assert
        report = health_monitor.get_pool_health(pool_id)
        assert report.status == HealthStatus.CRITICAL
        assert any("Critical" in rec for rec in report.recommendations)

    def test_multiple_metrics(self, health_monitor):
        """Test handling multiple metrics for a pool."""
        # Arrange
        pool_id = "test_pool"
        metrics = [
            (HealthMetricType.UTILIZATION, 0.7),
            (HealthMetricType.RESERVATION_RATE, 0.8),
            (HealthMetricType.EMERGENCY_USAGE, 0.3),
        ]

        # Act
        for metric_type, value in metrics:
            health_monitor.record_metric(pool_id, metric_type, value)

        # Assert
        report = health_monitor.get_pool_health(pool_id)
        assert len(report.metrics) == len(metrics)
        assert report.status == HealthStatus.WARNING  # Due to high reservation rate

    def test_metric_history_cleanup(self, health_monitor):
        """Test cleanup of old metrics."""
        # Arrange
        pool_id = "test_pool"
        metric_type = HealthMetricType.UTILIZATION

        # Set short history window for testing
        health_monitor.history_window = timedelta(seconds=1)

        # Record metric
        health_monitor.record_metric(pool_id, metric_type, 0.5)

        # Wait for metric to expire
        import time

        time.sleep(1.1)

        # Record new metric
        health_monitor.record_metric(pool_id, metric_type, 0.6)

        # Assert
        history = health_monitor._metrics_history[pool_id][metric_type]
        assert len(history) == 1  # Old metric should be cleaned up
        assert history[0].value == 0.6  # Only new metric remains

    def test_recommendations(self, health_monitor):
        """Test recommendation generation."""
        # Arrange
        pool_id = "test_pool"
        metrics = [
            (HealthMetricType.UTILIZATION, 0.85),  # High utilization
            (HealthMetricType.DENIAL_RATE, 0.25),  # High denial rate
        ]

        # Act
        for metric_type, value in metrics:
            health_monitor.record_metric(pool_id, metric_type, value)

        # Assert
        report = health_monitor.get_pool_health(pool_id)
        assert len(report.recommendations) > 0
        assert any("increase" in rec.lower() for rec in report.recommendations)
        assert any("denial" in rec.lower() for rec in report.recommendations)

    def test_status_change_notification(self, health_monitor, notification_manager):
        """Test notifications on status changes."""
        # Arrange
        pool_id = "test_pool"
        metric_type = HealthMetricType.UTILIZATION

        # Act - Record increasingly concerning metrics
        health_monitor.record_metric(pool_id, metric_type, 0.5)  # Healthy
        health_monitor.record_metric(pool_id, metric_type, 0.85)  # Warning
        health_monitor.record_metric(pool_id, metric_type, 0.96)  # Critical

        # Assert
        notifications = notification_manager.get_notifications()
        assert len(notifications) == 2  # Two status changes
        assert any("WARNING" in n.message for n in notifications)
        assert any("CRITICAL" in n.message for n in notifications)

    def test_violation_reporting(self, health_monitor, violation_reporter):
        """Test violation reporting for critical status."""
        # Arrange
        pool_id = "test_pool"
        metric_type = HealthMetricType.UTILIZATION
        value = 0.96  # Critical utilization

        # Act
        health_monitor.record_metric(pool_id, metric_type, value)

        # Assert
        violations = violation_reporter.get_violations()
        assert len(violations) == 1
        assert violations[0].type == "POOL_HEALTH"
        assert violations[0].severity == "HIGH"

    def test_check_interval(self, health_monitor):
        """Test health check interval behavior."""
        # Arrange
        pool_id = "test_pool"
        metric_type = HealthMetricType.UTILIZATION
        health_monitor.check_interval = timedelta(seconds=2)

        # Act - Record metrics in quick succession
        health_monitor.record_metric(pool_id, metric_type, 0.5)
        initial_check = health_monitor._last_check[pool_id]

        health_monitor.record_metric(pool_id, metric_type, 0.6)
        second_check = health_monitor._last_check[pool_id]

        # Assert
        assert initial_check == second_check  # Second check should be skipped

        # Wait for interval
        import time

        time.sleep(2.1)

        # Record another metric
        health_monitor.record_metric(pool_id, metric_type, 0.7)
        final_check = health_monitor._last_check[pool_id]

        # Assert
        assert final_check > second_check  # Check should occur after interval
