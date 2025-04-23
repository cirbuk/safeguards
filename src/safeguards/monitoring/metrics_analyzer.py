"""Advanced metrics analysis for resource and budget monitoring."""

from dataclasses import dataclass
from datetime import datetime

import numpy as np

from safeguards.base.budget import BudgetMetrics
from safeguards.base.monitoring import ResourceMetrics


@dataclass
class TrendAnalysis:
    """Trend analysis results."""

    trend_direction: str  # "increasing", "decreasing", "stable"
    rate_of_change: float  # per hour
    volatility: float  # standard deviation
    forecast_next_hour: float
    forecast_next_day: float
    anomaly_score: float  # 0-1 scale


@dataclass
class ResourceUsagePattern:
    """Resource usage pattern analysis."""

    peak_hours: list[int]  # 0-23 hours
    low_usage_hours: list[int]  # 0-23 hours
    weekly_pattern: dict[str, float]  # day -> average usage
    periodic_spikes: list[datetime]
    correlation_matrix: dict[str, dict[str, float]]


class MetricsAnalyzer:
    """Advanced metrics analysis for resource and budget monitoring."""

    def __init__(self, lookback_days: int = 30):
        """Initialize metrics analyzer.

        Args:
            lookback_days: Number of days of history to analyze
        """
        self.lookback_days = lookback_days

    def analyze_resource_trends(
        self,
        metrics_history: list[ResourceMetrics],
        metric_name: str,
    ) -> TrendAnalysis:
        """Analyze trends for a specific resource metric.

        Args:
            metrics_history: Historical resource metrics
            metric_name: Name of metric to analyze

        Returns:
            Trend analysis results
        """
        if not metrics_history:
            msg = "No metrics history provided"
            raise ValueError(msg)

        # Extract values and timestamps
        values = [float(getattr(m, metric_name)) for m in metrics_history]
        timestamps = [m.timestamp for m in metrics_history]

        # Calculate rate of change (per hour)
        hours_elapsed = (timestamps[-1] - timestamps[0]).total_seconds() / 3600
        total_change = values[-1] - values[0]
        rate_of_change = total_change / hours_elapsed if hours_elapsed > 0 else 0

        # Calculate volatility
        volatility = np.std(values) if len(values) > 1 else 0

        # Simple linear regression for forecasting
        hours_from_start = [(t - timestamps[0]).total_seconds() / 3600 for t in timestamps]
        coefficients = np.polyfit(hours_from_start, values, 1)
        forecast_next_hour = coefficients[0] * (hours_from_start[-1] + 1) + coefficients[1]
        forecast_next_day = coefficients[0] * (hours_from_start[-1] + 24) + coefficients[1]

        # Determine trend direction
        if abs(rate_of_change) < 0.1:  # Threshold for stability
            trend_direction = "stable"
        else:
            trend_direction = "increasing" if rate_of_change > 0 else "decreasing"

        # Calculate anomaly score using z-score based approach
        mean_value = np.mean(values)
        std_value = np.std(values) if len(values) > 1 else 0
        latest_zscore = abs(values[-1] - mean_value) / std_value if std_value > 0 else 0
        anomaly_score = min(1.0, latest_zscore / 3)  # Scale to 0-1

        return TrendAnalysis(
            trend_direction=trend_direction,
            rate_of_change=rate_of_change,
            volatility=volatility,
            forecast_next_hour=forecast_next_hour,
            forecast_next_day=forecast_next_day,
            anomaly_score=anomaly_score,
        )

    def analyze_usage_patterns(
        self,
        metrics_history: list[ResourceMetrics],
    ) -> ResourceUsagePattern:
        """Analyze resource usage patterns.

        Args:
            metrics_history: Historical resource metrics

        Returns:
            Usage pattern analysis
        """
        if not metrics_history:
            msg = "No metrics history provided"
            raise ValueError(msg)

        # Group metrics by hour
        hourly_usage = {i: [] for i in range(24)}
        for metric in metrics_history:
            hour = metric.timestamp.hour
            hourly_usage[hour].append(metric)

        # Find peak and low usage hours
        hourly_averages = {
            hour: np.mean([m.cpu_percent for m in metrics])
            for hour, metrics in hourly_usage.items()
            if metrics
        }

        mean_usage = np.mean(list(hourly_averages.values()))
        peak_hours = [hour for hour, usage in hourly_averages.items() if usage > mean_usage * 1.2]
        low_usage_hours = [
            hour for hour, usage in hourly_averages.items() if usage < mean_usage * 0.8
        ]

        # Analyze weekly patterns
        days = [
            "Monday",
            "Tuesday",
            "Wednesday",
            "Thursday",
            "Friday",
            "Saturday",
            "Sunday",
        ]
        weekly_pattern = dict.fromkeys(days, 0.0)
        day_counts = dict.fromkeys(days, 0)

        for metric in metrics_history:
            day = metric.timestamp.strftime("%A")
            weekly_pattern[day] += metric.cpu_percent
            day_counts[day] += 1

        for day in days:
            if day_counts[day] > 0:
                weekly_pattern[day] /= day_counts[day]

        # Detect periodic spikes
        periodic_spikes = []
        for i in range(1, len(metrics_history)):
            prev = metrics_history[i - 1]
            curr = metrics_history[i]
            if curr.cpu_percent > prev.cpu_percent * 1.5 and curr.cpu_percent > mean_usage * 1.3:
                periodic_spikes.append(curr.timestamp)

        # Calculate correlation matrix
        metrics_attrs = [
            "cpu_percent",
            "memory_percent",
            "disk_percent",
            "network_mbps",
        ]
        correlation_matrix = {}

        for attr1 in metrics_attrs:
            correlation_matrix[attr1] = {}
            values1 = [float(getattr(m, attr1)) for m in metrics_history]

            for attr2 in metrics_attrs:
                values2 = [float(getattr(m, attr2)) for m in metrics_history]
                correlation = np.corrcoef(values1, values2)[0, 1]
                correlation_matrix[attr1][attr2] = correlation

        return ResourceUsagePattern(
            peak_hours=peak_hours,
            low_usage_hours=low_usage_hours,
            weekly_pattern=weekly_pattern,
            periodic_spikes=periodic_spikes,
            correlation_matrix=correlation_matrix,
        )

    def analyze_budget_efficiency(
        self,
        budget_metrics: list[BudgetMetrics],
        resource_metrics: list[ResourceMetrics],
    ) -> dict[str, float]:
        """Analyze budget efficiency relative to resource usage.

        Args:
            budget_metrics: Historical budget metrics
            resource_metrics: Historical resource metrics

        Returns:
            Dictionary of efficiency metrics
        """
        if not budget_metrics or not resource_metrics:
            msg = "Both budget and resource metrics are required"
            raise ValueError(msg)

        # Align timestamps
        budget_times = [m.timestamp for m in budget_metrics]
        resource_times = [m.timestamp for m in resource_metrics]

        start_time = max(min(budget_times), min(resource_times))
        end_time = min(max(budget_times), max(resource_times))

        # Filter metrics within time range
        budget_metrics = [m for m in budget_metrics if start_time <= m.timestamp <= end_time]
        resource_metrics = [m for m in resource_metrics if start_time <= m.timestamp <= end_time]

        # Calculate efficiency metrics
        budget_usage = [float(m.usage_percentage) for m in budget_metrics]
        cpu_usage = [m.cpu_percent / 100 for m in resource_metrics]
        memory_usage = [m.memory_percent / 100 for m in resource_metrics]

        # Budget utilization relative to resource usage
        avg_budget_usage = np.mean(budget_usage)
        avg_cpu_usage = np.mean(cpu_usage)
        avg_memory_usage = np.mean(memory_usage)

        # Calculate efficiency scores
        cpu_efficiency = avg_cpu_usage / avg_budget_usage if avg_budget_usage > 0 else 0
        memory_efficiency = avg_memory_usage / avg_budget_usage if avg_budget_usage > 0 else 0

        # Calculate budget predictability
        budget_volatility = np.std(budget_usage) if len(budget_usage) > 1 else 0

        # Resource usage correlation with budget
        cpu_correlation = np.corrcoef(budget_usage, cpu_usage)[0, 1]
        memory_correlation = np.corrcoef(budget_usage, memory_usage)[0, 1]

        return {
            "cpu_efficiency": cpu_efficiency,
            "memory_efficiency": memory_efficiency,
            "budget_volatility": budget_volatility,
            "cpu_budget_correlation": cpu_correlation,
            "memory_budget_correlation": memory_correlation,
        }
