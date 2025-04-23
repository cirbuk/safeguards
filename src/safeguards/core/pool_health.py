"""Pool health monitoring system.

This module provides functionality for:
- Monitoring pool utilization and health metrics
- Detecting potential issues and anomalies
- Triggering alerts and notifications
- Providing health status reports
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto

from safeguards.core.notification_manager import NotificationManager
from safeguards.monitoring.violation_reporter import (
    ViolationReporter,
    ViolationSeverity,
    ViolationType,
)


class HealthStatus(Enum):
    """Pool health status levels."""

    HEALTHY = auto()  # Pool is operating normally
    WARNING = auto()  # Some metrics are approaching thresholds
    CRITICAL = auto()  # Critical thresholds exceeded
    DEGRADED = auto()  # Performance is degraded but functional


class HealthMetricType(Enum):
    """Types of health metrics tracked."""

    UTILIZATION = auto()  # Current utilization percentage
    RESERVATION_RATE = auto()  # Rate of new reservations
    DENIAL_RATE = auto()  # Rate of denied requests
    EMERGENCY_USAGE = auto()  # Emergency fund usage
    REBALANCING_FREQUENCY = auto()  # How often rebalancing occurs


@dataclass
class HealthMetric:
    """Individual health metric measurement."""

    type: HealthMetricType
    value: float
    timestamp: datetime = field(default_factory=datetime.now)
    threshold_warning: float = 0.8  # 80% threshold for warnings
    threshold_critical: float = 0.95  # 95% threshold for critical alerts


@dataclass
class PoolHealthReport:
    """Health report for a budget pool."""

    pool_id: str
    status: HealthStatus
    metrics: dict[HealthMetricType, HealthMetric]
    timestamp: datetime = field(default_factory=datetime.now)
    recommendations: list[str] = field(default_factory=list)


class PoolHealthMonitor:
    """Monitors and reports on pool health metrics."""

    def __init__(
        self,
        notification_manager: NotificationManager,
        violation_reporter: ViolationReporter,
        check_interval: timedelta = timedelta(minutes=5),
        history_window: timedelta = timedelta(hours=24),
        utilization_warning: float = 0.8,
        utilization_critical: float = 0.95,
        reservation_rate_warning: float = 0.7,
        emergency_usage_warning: float = 0.5,
    ):
        """Initialize the pool health monitor.

        Args:
            notification_manager: For sending health-related alerts
            violation_reporter: For reporting health violations
            check_interval: How often to check health metrics
            history_window: How long to keep metric history
            utilization_warning: Warning threshold for utilization
            utilization_critical: Critical threshold for utilization
            reservation_rate_warning: Warning threshold for reservation rate
            emergency_usage_warning: Warning threshold for emergency usage
        """
        self.notification_manager = notification_manager
        self.violation_reporter = violation_reporter
        self.check_interval = check_interval
        self.history_window = history_window
        self.utilization_warning = utilization_warning
        self.utilization_critical = utilization_critical
        self.reservation_rate_warning = reservation_rate_warning
        self.emergency_usage_warning = emergency_usage_warning

        # Track metrics history by pool
        self._metrics_history: dict[
            str,
            dict[HealthMetricType, list[HealthMetric]],
        ] = {}
        self._last_check: dict[str, datetime] = {}
        self._pool_status: dict[str, HealthStatus] = {}

    def record_metric(
        self,
        pool_id: str,
        metric_type: HealthMetricType,
        value: float,
    ) -> None:
        """Record a new metric measurement.

        Args:
            pool_id: Pool to record metric for
            metric_type: Type of metric
            value: Metric value
        """
        # Create metric
        metric = HealthMetric(
            type=metric_type,
            value=value,
            threshold_warning=self._get_warning_threshold(metric_type),
            threshold_critical=self._get_critical_threshold(metric_type),
        )

        # Add to history
        pool_metrics = self._metrics_history.setdefault(pool_id, {})
        metric_history = pool_metrics.setdefault(metric_type, [])
        metric_history.append(metric)

        # Cleanup old metrics
        cutoff = datetime.now() - self.history_window
        pool_metrics[metric_type] = [m for m in metric_history if m.timestamp > cutoff]

        # Check if health check is needed
        self._check_health_if_needed(pool_id)

    def get_pool_health(self, pool_id: str) -> PoolHealthReport | None:
        """Get current health report for a pool.

        Args:
            pool_id: Pool to check

        Returns:
            Health report if pool is being monitored, None otherwise
        """
        if pool_id not in self._metrics_history:
            return None

        # Force a health check
        return self._check_pool_health(pool_id)

    def _check_health_if_needed(self, pool_id: str) -> None:
        """Check pool health if enough time has passed since last check.

        Args:
            pool_id: Pool to check
        """
        now = datetime.now()
        last_check = self._last_check.get(pool_id, datetime.min)
        if now - last_check >= self.check_interval:
            self._check_pool_health(pool_id)

    def _check_pool_health(self, pool_id: str) -> PoolHealthReport:
        """Check health metrics and generate a report.

        Args:
            pool_id: Pool to check

        Returns:
            Generated health report
        """
        pool_metrics = self._metrics_history[pool_id]
        current_metrics = {}
        status = HealthStatus.HEALTHY
        recommendations = []

        # Check each metric type
        for metric_type in HealthMetricType:
            if pool_metrics.get(metric_type):
                # Get most recent metric
                metric = pool_metrics[metric_type][-1]
                current_metrics[metric_type] = metric

                # Check thresholds
                if metric.value >= metric.threshold_critical:
                    status = HealthStatus.CRITICAL
                    recommendations.append(
                        f"Critical: {metric_type.name} at {metric.value:.1%}",
                    )
                elif metric.value >= metric.threshold_warning:
                    if status != HealthStatus.CRITICAL:
                        status = HealthStatus.WARNING
                    recommendations.append(
                        f"Warning: {metric_type.name} at {metric.value:.1%}",
                    )

                # Add metric-specific recommendations
                recommendations.extend(
                    self._get_metric_recommendations(metric_type, metric.value),
                )

        # Create report
        report = PoolHealthReport(
            pool_id=pool_id,
            status=status,
            metrics=current_metrics,
            recommendations=recommendations,
        )

        # Update tracking
        self._last_check[pool_id] = datetime.now()
        self._pool_status[pool_id] = status

        # Send notifications if status changed
        self._notify_status_change(pool_id, status)

        return report

    def _get_warning_threshold(self, metric_type: HealthMetricType) -> float:
        """Get warning threshold for a metric type.

        Args:
            metric_type: Type of metric

        Returns:
            Warning threshold value
        """
        thresholds = {
            HealthMetricType.UTILIZATION: self.utilization_warning,
            HealthMetricType.RESERVATION_RATE: self.reservation_rate_warning,
            HealthMetricType.EMERGENCY_USAGE: self.emergency_usage_warning,
            HealthMetricType.DENIAL_RATE: 0.2,  # 20% denial rate
            HealthMetricType.REBALANCING_FREQUENCY: 0.7,  # Rebalancing too often
        }
        return thresholds.get(metric_type, 0.8)  # Default 80%

    def _get_critical_threshold(self, metric_type: HealthMetricType) -> float:
        """Get critical threshold for a metric type.

        Args:
            metric_type: Type of metric

        Returns:
            Critical threshold value
        """
        thresholds = {
            HealthMetricType.UTILIZATION: self.utilization_critical,
            HealthMetricType.RESERVATION_RATE: 0.9,  # 90% reservation rate
            HealthMetricType.EMERGENCY_USAGE: 0.8,  # 80% emergency usage
            HealthMetricType.DENIAL_RATE: 0.4,  # 40% denial rate
            HealthMetricType.REBALANCING_FREQUENCY: 0.9,  # Constant rebalancing
        }
        return thresholds.get(metric_type, 0.95)  # Default 95%

    def _get_metric_recommendations(
        self,
        metric_type: HealthMetricType,
        value: float,
    ) -> list[str]:
        """Get recommendations based on metric type and value.

        Args:
            metric_type: Type of metric
            value: Current metric value

        Returns:
            List of recommendations
        """
        recommendations = []
        if metric_type == HealthMetricType.UTILIZATION and value > 0.8:
            recommendations.extend(
                [
                    "Consider increasing pool budget",
                    "Review agent budget allocations",
                    "Check for unused reservations",
                ],
            )
        elif metric_type == HealthMetricType.RESERVATION_RATE and value > 0.7:
            recommendations.extend(
                [
                    "Review reservation expiry times",
                    "Check for reservation leaks",
                    "Consider implementing request throttling",
                ],
            )
        elif metric_type == HealthMetricType.EMERGENCY_USAGE and value > 0.5:
            recommendations.extend(
                [
                    "Audit emergency fund usage patterns",
                    "Review emergency request criteria",
                    "Consider increasing emergency fund allocation",
                ],
            )
        elif metric_type == HealthMetricType.DENIAL_RATE and value > 0.2:
            recommendations.extend(
                [
                    "Review denial reasons",
                    "Check budget allocation strategy",
                    "Consider adjusting reservation thresholds",
                ],
            )
        return recommendations

    def _notify_status_change(self, pool_id: str, new_status: HealthStatus) -> None:
        """Send notifications when pool health status changes.

        Args:
            pool_id: Pool that changed status
            new_status: New health status
        """
        old_status = self._pool_status.get(pool_id)
        if old_status != new_status:
            severity = (
                "HIGH" if new_status in (HealthStatus.WARNING, HealthStatus.CRITICAL) else "INFO"
            )
            self.notification_manager.send_notification(
                agent_id="system",
                message=f"Pool {pool_id} health changed from {old_status} to {new_status}",
                severity=severity,
            )

            # Report violation for critical status
            if new_status == HealthStatus.CRITICAL:
                self.violation_reporter.report_violation(
                    type=ViolationType.POOL_HEALTH,
                    severity=ViolationSeverity.HIGH,
                    agent_id="system",
                    description=f"Pool {pool_id} health status is CRITICAL",
                )
