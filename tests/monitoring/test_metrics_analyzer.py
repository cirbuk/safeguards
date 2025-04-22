"""Tests for the metrics analyzer module."""

import pytest
from datetime import datetime, timedelta
import numpy as np
from decimal import Decimal

from agent_safety.monitoring.metrics_analyzer import MetricsAnalyzer
from agent_safety.base.monitoring import ResourceMetrics
from agent_safety.base.budget import BudgetMetrics


def create_test_resource_metrics(
    start_time: datetime,
    count: int,
    interval_minutes: int = 60,
    cpu_pattern: str = "stable",
) -> list[ResourceMetrics]:
    """Create test resource metrics with specified pattern."""
    metrics = []
    base_cpu = 50.0

    for i in range(count):
        timestamp = start_time + timedelta(minutes=i * interval_minutes)

        if cpu_pattern == "increasing":
            cpu = base_cpu + (i * 2)
        elif cpu_pattern == "decreasing":
            cpu = base_cpu - (i * 2)
        elif cpu_pattern == "volatile":
            cpu = base_cpu + ((-1) ** i * 10)
        else:  # stable
            cpu = base_cpu + np.random.normal(0, 2)

        metrics.append(
            ResourceMetrics(
                timestamp=timestamp,
                cpu_percent=max(0, min(100, cpu)),
                memory_percent=60.0,
                disk_percent=40.0,
                network_mbps=5.0,
                process_count=10,
                open_files=20,
            )
        )
    return metrics


def create_test_budget_metrics(
    start_time: datetime, count: int, interval_minutes: int = 60
) -> list[BudgetMetrics]:
    """Create test budget metrics."""
    metrics = []
    for i in range(count):
        timestamp = start_time + timedelta(minutes=i * interval_minutes)
        usage = Decimal("700")
        total = Decimal("1000")
        period_start = timestamp.replace(hour=0, minute=0, second=0, microsecond=0)
        period_end = period_start + timedelta(days=1)
        metrics.append(
            BudgetMetrics(
                timestamp=timestamp,
                current_usage=usage,
                total_budget=total,
                period_start=period_start,
                period_end=period_end,
                usage_percentage=float(usage / total),
            )
        )
    return metrics


def test_analyze_resource_trends_increasing():
    """Test trend analysis with increasing pattern."""
    start_time = datetime.now()
    metrics = create_test_resource_metrics(
        start_time, count=24, cpu_pattern="increasing"
    )

    analyzer = MetricsAnalyzer()
    analysis = analyzer.analyze_resource_trends(metrics, "cpu_percent")

    assert analysis.trend_direction == "increasing"
    assert analysis.rate_of_change > 0
    assert analysis.forecast_next_hour > metrics[-1].cpu_percent
    assert analysis.forecast_next_day > analysis.forecast_next_hour


def test_analyze_resource_trends_stable():
    """Test trend analysis with stable pattern."""
    start_time = datetime.now()
    # Set a fixed seed for consistent results
    np.random.seed(42)
    metrics = []
    base_cpu = 50.0

    # Create metrics with truly stable values (minimal variation)
    for i in range(24):
        timestamp = start_time + timedelta(minutes=i * 60)
        # Only tiny variations to ensure a stable pattern
        cpu = base_cpu + np.random.normal(0, 0.1)  # Minimal standard deviation
        metrics.append(
            ResourceMetrics(
                timestamp=timestamp,
                cpu_percent=max(0, min(100, cpu)),
                memory_percent=60.0,
                disk_percent=40.0,
                network_mbps=5.0,
                process_count=10,
                open_files=20,
            )
        )

    analyzer = MetricsAnalyzer()
    analysis = analyzer.analyze_resource_trends(metrics, "cpu_percent")

    assert analysis.trend_direction == "stable"
    assert abs(analysis.rate_of_change) < 0.1
    assert analysis.volatility < 5.0


def test_analyze_usage_patterns():
    """Test usage pattern analysis."""
    start_time = datetime.now().replace(hour=0, minute=0)
    metrics = []

    # Create 7 days of metrics with higher usage during business hours
    for day in range(7):
        for hour in range(24):
            timestamp = start_time + timedelta(days=day, hours=hour)

            # Higher usage during business hours on weekdays
            is_weekday = timestamp.weekday() < 5
            is_business_hours = 9 <= hour <= 17

            base_cpu = 80.0 if (is_weekday and is_business_hours) else 30.0
            cpu = base_cpu + np.random.normal(0, 5)

            metrics.append(
                ResourceMetrics(
                    timestamp=timestamp,
                    cpu_percent=max(0, min(100, cpu)),
                    memory_percent=60.0,
                    disk_percent=40.0,
                    network_mbps=5.0,
                    process_count=10,
                    open_files=20,
                )
            )

    analyzer = MetricsAnalyzer()
    patterns = analyzer.analyze_usage_patterns(metrics)

    # Check peak hours are during business hours
    assert all(9 <= hour <= 17 for hour in patterns.peak_hours)

    # Check low usage hours are outside business hours
    assert all(hour < 9 or hour > 17 for hour in patterns.low_usage_hours)

    # Check weekly pattern shows higher weekday usage
    weekday_avg = np.mean(
        [
            patterns.weekly_pattern[day]
            for day in ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
        ]
    )
    weekend_avg = np.mean(
        [patterns.weekly_pattern[day] for day in ["Saturday", "Sunday"]]
    )
    assert weekday_avg > weekend_avg


def test_analyze_budget_efficiency():
    """Test budget efficiency analysis."""
    start_time = datetime.now()

    # Set a fixed seed for consistent results
    np.random.seed(42)

    # Create resource metrics with variation
    resource_metrics = []
    for i in range(24):
        timestamp = start_time + timedelta(minutes=i * 60)
        # Add some variation that correlates with time
        cpu = 50.0 + i * 0.5 + np.random.normal(0, 1)
        resource_metrics.append(
            ResourceMetrics(
                timestamp=timestamp,
                cpu_percent=max(0, min(100, cpu)),
                memory_percent=60.0 + i * 0.2,
                disk_percent=40.0,
                network_mbps=5.0,
                process_count=10,
                open_files=20,
            )
        )

    # Create budget metrics with variation that somewhat correlates with CPU
    budget_metrics = []
    for i in range(24):
        timestamp = start_time + timedelta(minutes=i * 60)
        usage = Decimal(str(650 + i * 5))  # Increase over time like CPU
        total = Decimal("1000")
        period_start = timestamp.replace(hour=0, minute=0, second=0, microsecond=0)
        period_end = period_start + timedelta(days=1)
        budget_metrics.append(
            BudgetMetrics(
                timestamp=timestamp,
                current_usage=usage,
                total_budget=total,
                period_start=period_start,
                period_end=period_end,
                usage_percentage=float(usage / total),
            )
        )

    analyzer = MetricsAnalyzer()
    efficiency = analyzer.analyze_budget_efficiency(budget_metrics, resource_metrics)

    assert 0 <= efficiency["cpu_efficiency"] <= 1
    assert 0 <= efficiency["memory_efficiency"] <= 1
    assert efficiency["budget_volatility"] >= 0

    # Handle potential NaN values in correlations
    cpu_corr = efficiency["cpu_budget_correlation"]
    mem_corr = efficiency["memory_budget_correlation"]

    # Check correlation is valid (either a number between -1 and 1, or NaN)
    if not np.isnan(cpu_corr):
        assert -1 <= cpu_corr <= 1
    if not np.isnan(mem_corr):
        assert -1 <= mem_corr <= 1


def test_empty_metrics():
    """Test handling of empty metrics."""
    analyzer = MetricsAnalyzer()

    with pytest.raises(ValueError):
        analyzer.analyze_resource_trends([], "cpu_percent")

    with pytest.raises(ValueError):
        analyzer.analyze_usage_patterns([])

    with pytest.raises(ValueError):
        analyzer.analyze_budget_efficiency([], [])
