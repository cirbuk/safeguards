# Agent Monitoring Guide

This guide covers the monitoring capabilities of the Safeguards, helping you track agent performance, resource usage, and detect potential safety issues.

## Monitoring Overview

The Safeguards provides several monitoring components:

- **Metrics Collection**: Tracking numeric measurements of agent activity
- **Runtime Monitoring**: Real-time observation of agent behavior
- **Violation Detection**: Identifying when agents break defined constraints
- **Logging**: Recording agent activities for later analysis
- **Visualization**: Displaying monitoring data in dashboards

## Setting Up Basic Monitoring

### Metrics Collector

Start by setting up a metrics collector:

```python
from safeguards.monitoring.metrics_collector import MetricsCollector
from safeguards.types import MetricType

# Create a metrics collector
metrics_collector = MetricsCollector()

# Register an agent to monitor
agent_id = "agent123"
metrics_collector.register_agent(agent_id)

# Record some metrics
metrics_collector.record_metric(
    agent_id=agent_id,
    metric_name="api_calls",
    metric_type=MetricType.COUNTER,
    value=1
)

metrics_collector.record_metric(
    agent_id=agent_id,
    metric_name="response_time_ms",
    metric_type=MetricType.GAUGE,
    value=245.6
)

# Retrieve metrics
api_calls = metrics_collector.get_metric_value(agent_id, "api_calls")
print(f"API calls made by agent: {api_calls}")

# Get all metrics for an agent
all_metrics = metrics_collector.get_agent_metrics(agent_id)
print(f"All metrics: {all_metrics}")
```

### Runtime Monitor

Set up real-time monitoring of agent behavior:

```python
from safeguards.monitoring.runtime_monitor import RuntimeMonitor
from safeguards.types.monitors import ResourceMonitor, ActivityMonitor

# Create a runtime monitor
runtime_monitor = RuntimeMonitor()

# Add a resource monitor for CPU usage
def cpu_usage_monitor(agent_id, metrics):
    """Monitor CPU usage of an agent."""
    cpu_usage = get_agent_cpu_usage(agent_id)  # Your implementation
    if cpu_usage > 80:
        print(f"Warning: Agent {agent_id} CPU usage at {cpu_usage}%")
        return False  # Returning False signals a violation
    return True

# Register the CPU monitor
runtime_monitor.register_monitor(
    agent_id="agent123",
    monitor=ResourceMonitor(
        name="cpu_usage",
        check_function=cpu_usage_monitor,
        check_interval_seconds=30
    )
)

# Add an activity monitor for API call frequency
def api_call_monitor(agent_id, metrics):
    """Monitor API call frequency."""
    recent_calls = metrics.get_metric_value(agent_id, "api_calls",
                                           time_window_seconds=60)
    if recent_calls > 100:
        print(f"Warning: Agent {agent_id} made {recent_calls} API calls in the last minute")
        return False
    return True

# Register the API call monitor
runtime_monitor.register_monitor(
    agent_id="agent123",
    monitor=ActivityMonitor(
        name="api_call_frequency",
        check_function=api_call_monitor,
        check_interval_seconds=60,
        metrics_collector=metrics_collector
    )
)

# Start monitoring
runtime_monitor.start()

# Later, stop monitoring
# runtime_monitor.stop()
```

### Violation Reporter

Set up violation detection and reporting:

```python
from safeguards.monitoring.violation_reporter import ViolationReporter
from safeguards.types import ViolationType, AlertSeverity
from safeguards.core.notification_manager import NotificationManager

# Create a notification manager (for alerting)
notification_manager = NotificationManager()

# Create a violation reporter
violation_reporter = ViolationReporter(notification_manager)

# Report a violation
violation_reporter.report_violation(
    agent_id="agent123",
    violation_type=ViolationType.RESOURCE_LIMIT_EXCEEDED,
    severity=AlertSeverity.HIGH,
    message="Agent exceeded memory usage limit",
    details={
        "limit": "256MB",
        "actual": "312MB",
        "overage_percentage": 21.9
    }
)

# You can also integrate the violation reporter with the runtime monitor
runtime_monitor.set_violation_reporter(violation_reporter)
```

## Comprehensive Monitoring Setup

Here's a complete example that sets up a comprehensive monitoring system:

```python
from safeguards.monitoring import (
    MetricsCollector,
    RuntimeMonitor,
    ViolationReporter,
    LogManager
)
from safeguards.core import NotificationManager, BudgetCoordinator
from safeguards.types import MetricType, ViolationType, AlertSeverity
from safeguards.types.monitors import ResourceMonitor, ActivityMonitor, BudgetMonitor

def setup_monitoring_system():
    """Set up a comprehensive monitoring system."""

    # Core components
    notification_manager = NotificationManager()
    metrics_collector = MetricsCollector()
    violation_reporter = ViolationReporter(notification_manager)
    runtime_monitor = RuntimeMonitor()
    log_manager = LogManager()
    budget_coordinator = BudgetCoordinator(notification_manager)

    # Connect components
    runtime_monitor.set_violation_reporter(violation_reporter)
    runtime_monitor.set_metrics_collector(metrics_collector)

    # Set up logging
    log_manager.set_log_level("DEBUG")
    log_manager.set_log_format("[{timestamp}] {level}: {message}")
    log_manager.set_log_file("agent_monitoring.log")

    # Return the monitoring system components
    return {
        "notification_manager": notification_manager,
        "metrics_collector": metrics_collector,
        "violation_reporter": violation_reporter,
        "runtime_monitor": runtime_monitor,
        "log_manager": log_manager,
        "budget_coordinator": budget_coordinator
    }

def register_agent_for_monitoring(monitoring_system, agent_id, initial_budget=100.0):
    """Register an agent with the monitoring system."""

    # Unpack components
    metrics_collector = monitoring_system["metrics_collector"]
    runtime_monitor = monitoring_system["runtime_monitor"]
    budget_coordinator = monitoring_system["budget_coordinator"]

    # Register with metrics collector
    metrics_collector.register_agent(agent_id)

    # Create a budget pool and register agent
    pool_id = budget_coordinator.create_budget_pool(
        name=f"pool_{agent_id}",
        initial_budget=initial_budget,
        description=f"Budget pool for agent {agent_id}"
    )
    budget_coordinator.register_agent(
        agent_id=agent_id,
        pool_id=pool_id,
        priority=5
    )

    # Set up budget monitoring
    def budget_monitor(agent_id, metrics):
        """Monitor agent budget usage."""
        budget_info = budget_coordinator.get_agent_metrics(agent_id)
        remaining_percentage = (budget_info["remaining_budget"] /
                               budget_info["initial_budget"]) * 100

        metrics_collector.record_metric(
            agent_id=agent_id,
            metric_name="budget_remaining_percentage",
            metric_type=MetricType.GAUGE,
            value=remaining_percentage
        )

        if remaining_percentage < 10:
            monitoring_system["violation_reporter"].report_violation(
                agent_id=agent_id,
                violation_type=ViolationType.BUDGET_DEPLETED,
                severity=AlertSeverity.HIGH,
                message=f"Agent budget nearly depleted ({remaining_percentage:.1f}%)",
                details=budget_info
            )
            return False

        return True

    # Register budget monitor
    runtime_monitor.register_monitor(
        agent_id=agent_id,
        monitor=BudgetMonitor(
            name="budget_usage",
            check_function=budget_monitor,
            check_interval_seconds=60,
            metrics_collector=metrics_collector
        )
    )

    # Set up CPU monitoring
    def cpu_monitor(agent_id, metrics):
        """Monitor CPU usage."""
        # Example implementation - replace with actual CPU monitoring
        cpu_usage = get_agent_cpu_usage(agent_id)  # Your implementation

        metrics_collector.record_metric(
            agent_id=agent_id,
            metric_name="cpu_usage_percent",
            metric_type=MetricType.GAUGE,
            value=cpu_usage
        )

        if cpu_usage > 80:
            monitoring_system["violation_reporter"].report_violation(
                agent_id=agent_id,
                violation_type=ViolationType.RESOURCE_LIMIT_EXCEEDED,
                severity=AlertSeverity.MEDIUM,
                message=f"Agent CPU usage high ({cpu_usage}%)",
                details={"resource": "cpu", "usage": cpu_usage, "limit": 80}
            )
            return False

        return True

    # Register CPU monitor
    runtime_monitor.register_monitor(
        agent_id=agent_id,
        monitor=ResourceMonitor(
            name="cpu_usage",
            check_function=cpu_monitor,
            check_interval_seconds=30,
            metrics_collector=metrics_collector
        )
    )

    # Set up memory monitoring
    def memory_monitor(agent_id, metrics):
        """Monitor memory usage."""
        # Example implementation - replace with actual memory monitoring
        memory_usage = get_agent_memory_usage(agent_id)  # Your implementation

        metrics_collector.record_metric(
            agent_id=agent_id,
            metric_name="memory_usage_mb",
            metric_type=MetricType.GAUGE,
            value=memory_usage
        )

        if memory_usage > 512:  # 512 MB limit
            monitoring_system["violation_reporter"].report_violation(
                agent_id=agent_id,
                violation_type=ViolationType.RESOURCE_LIMIT_EXCEEDED,
                severity=AlertSeverity.MEDIUM,
                message=f"Agent memory usage high ({memory_usage} MB)",
                details={"resource": "memory", "usage": memory_usage, "limit": 512}
            )
            return False

        return True

    # Register memory monitor
    runtime_monitor.register_monitor(
        agent_id=agent_id,
        monitor=ResourceMonitor(
            name="memory_usage",
            check_function=memory_monitor,
            check_interval_seconds=30,
            metrics_collector=metrics_collector
        )
    )

    # Start monitoring if not already started
    if not runtime_monitor.is_running:
        runtime_monitor.start()

    return True

# Example usage of the above functions
def example_usage():
    # Set up monitoring system
    monitoring_system = setup_monitoring_system()

    # Register an agent for monitoring
    agent_id = "agent123"
    register_agent_for_monitoring(monitoring_system, agent_id, initial_budget=200.0)

    # Simulate agent activity
    for i in range(10):
        # Record some metrics
        monitoring_system["metrics_collector"].record_metric(
            agent_id=agent_id,
            metric_name="api_calls",
            metric_type=MetricType.COUNTER,
            value=1
        )

        # Update budget usage
        monitoring_system["budget_coordinator"].update_budget(
            agent_id=agent_id,
            amount=10.0
        )

        # Wait a bit before next activity
        time.sleep(5)

    # Check agent metrics
    metrics = monitoring_system["metrics_collector"].get_agent_metrics(agent_id)
    print(f"Final metrics for {agent_id}: {metrics}")

    # Check budget status
    budget_info = monitoring_system["budget_coordinator"].get_agent_metrics(agent_id)
    print(f"Final budget for {agent_id}: {budget_info}")

    # Stop monitoring
    monitoring_system["runtime_monitor"].stop()

# Helper functions (implement these based on your environment)
def get_agent_cpu_usage(agent_id):
    """Get agent CPU usage percentage."""
    # Implementation depends on your environment and agent implementation
    # This is just a placeholder
    import random
    return random.uniform(10, 90)

def get_agent_memory_usage(agent_id):
    """Get agent memory usage in MB."""
    # Implementation depends on your environment and agent implementation
    # This is just a placeholder
    import random
    return random.uniform(100, 600)
```

## Custom Metrics and Monitors

### Custom Metrics

You can record custom metrics specific to your agents:

```python
# Record a custom metric for tracking task completion
metrics_collector.record_metric(
    agent_id="agent123",
    metric_name="tasks_completed",
    metric_type=MetricType.COUNTER,
    value=1,
    labels={"task_type": "data_processing", "priority": "high"}
)

# Record a timing metric
metrics_collector.record_metric(
    agent_id="agent123",
    metric_name="task_duration_seconds",
    metric_type=MetricType.HISTOGRAM,
    value=3.45,
    labels={"task_type": "data_processing"}
)

# Get aggregated metrics with filters
task_metrics = metrics_collector.get_filtered_metrics(
    agent_id="agent123",
    metric_name="task_duration_seconds",
    filter_labels={"task_type": "data_processing"},
    aggregation="avg",
    time_window_seconds=3600  # Last hour
)
print(f"Average data processing task duration: {task_metrics} seconds")
```

### Custom Monitors

You can create custom monitors for domain-specific behaviors:

```python
from safeguards.types.monitors import CustomMonitor

def data_quality_monitor(agent_id, metrics):
    """Monitor data quality produced by the agent."""
    # Example implementation
    data_quality_score = calculate_data_quality(agent_id)  # Your implementation

    metrics.record_metric(
        agent_id=agent_id,
        metric_name="data_quality_score",
        metric_type=MetricType.GAUGE,
        value=data_quality_score
    )

    if data_quality_score < 0.7:  # Threshold for acceptable quality
        print(f"Warning: Agent {agent_id} produced low quality data: {data_quality_score}")
        return False

    return True

# Register the custom monitor
runtime_monitor.register_monitor(
    agent_id="agent123",
    monitor=CustomMonitor(
        name="data_quality",
        check_function=data_quality_monitor,
        check_interval_seconds=300,  # Check every 5 minutes
        metrics_collector=metrics_collector
    )
)

# Example helper function (implement based on your needs)
def calculate_data_quality(agent_id):
    """Calculate data quality score for agent outputs."""
    # Implementation depends on your specific use case
    # This is just a placeholder
    import random
    return random.uniform(0.5, 1.0)
```

## Working with Monitoring Data

### Querying Metrics

Retrieve and analyze recorded metrics:

```python
from datetime import datetime, timedelta

# Get all metrics for an agent
all_metrics = metrics_collector.get_agent_metrics("agent123")

# Get a specific metric over time
api_calls_over_time = metrics_collector.get_metric_history(
    agent_id="agent123",
    metric_name="api_calls",
    start_time=datetime.now() - timedelta(hours=24),
    end_time=datetime.now(),
    resolution="1h"  # Group by hour
)

# Calculate rate of change
api_call_rate = metrics_collector.get_metric_rate(
    agent_id="agent123",
    metric_name="api_calls",
    time_window_seconds=3600  # Last hour
)
print(f"API calls per second: {api_call_rate}")

# Get aggregated metrics across agents
system_metrics = metrics_collector.get_aggregated_metrics(
    metric_name="memory_usage_mb",
    aggregation="sum",
    filter_labels={"agent_type": "assistant"}
)
print(f"Total memory usage across all assistant agents: {system_metrics} MB")
```

### Visualizing Monitoring Data

Create visualizations of your monitoring data:

```python
from safeguards.visualization import MetricsDashboard

# Create a metrics dashboard
dashboard = MetricsDashboard(metrics_collector)

# Add visualization panels
dashboard.add_panel(
    title="API Calls Over Time",
    metric_name="api_calls",
    panel_type="line_chart",
    time_window_hours=24,
    agent_ids=["agent123", "agent456"]
)

dashboard.add_panel(
    title="Memory Usage",
    metric_name="memory_usage_mb",
    panel_type="gauge",
    agent_id="agent123"
)

dashboard.add_panel(
    title="Budget Remaining",
    metric_name="budget_remaining_percentage",
    panel_type="pie_chart",
    agent_ids=["agent123", "agent456", "agent789"]
)

# Start the dashboard web server
dashboard_url = dashboard.start(host="0.0.0.0", port=8080)
print(f"Dashboard available at: {dashboard_url}")
```

## Agent Health Checks

### Implementing Health Checks

Implement comprehensive health checks for your agents:

```python
from safeguards.monitoring.health import HealthChecker
from safeguards.types.health import HealthStatus, HealthCheck

# Create a health checker
health_checker = HealthChecker()

# Add connectivity check
def check_connectivity(agent_id):
    """Check if agent is reachable."""
    try:
        # Your implementation to ping the agent
        response = ping_agent(agent_id, timeout=2)  # Your implementation
        if response:
            return HealthStatus.HEALTHY, "Agent is reachable"
        else:
            return HealthStatus.UNHEALTHY, "Agent is not responding"
    except Exception as e:
        return HealthStatus.UNHEALTHY, f"Error connecting to agent: {str(e)}"

health_checker.add_check(
    agent_id="agent123",
    check=HealthCheck(
        name="connectivity",
        check_function=check_connectivity,
        check_interval_seconds=60
    )
)

# Add budget health check
def check_budget_health(agent_id):
    """Check if agent has sufficient budget."""
    try:
        budget_info = budget_coordinator.get_agent_metrics(agent_id)
        remaining_percentage = (budget_info["remaining_budget"] /
                               budget_info["initial_budget"]) * 100

        if remaining_percentage > 50:
            return HealthStatus.HEALTHY, f"Budget at {remaining_percentage:.1f}%"
        elif remaining_percentage > 20:
            return HealthStatus.DEGRADED, f"Budget at {remaining_percentage:.1f}%"
        else:
            return HealthStatus.UNHEALTHY, f"Budget depleted: {remaining_percentage:.1f}%"
    except Exception as e:
        return HealthStatus.UNKNOWN, f"Error checking budget: {str(e)}"

health_checker.add_check(
    agent_id="agent123",
    check=HealthCheck(
        name="budget_health",
        check_function=check_budget_health,
        check_interval_seconds=300
    )
)

# Start health checks
health_checker.start()

# Get current health status
health = health_checker.get_health("agent123")
print(f"Overall health: {health.status.name}")
print(f"Health details: {health.details}")

# Example helper function (implement based on your needs)
def ping_agent(agent_id, timeout=2):
    """Check if an agent is responsive."""
    # Implementation depends on your agent architecture
    # This is just a placeholder
    import random
    return random.choice([True, True, True, False])  # 75% chance of success
```

## Auditing and Compliance

### Audit Trail

Implement an audit trail for agent activities:

```python
from safeguards.monitoring.audit import AuditLogger
from safeguards.types.audit import AuditEvent, AuditEventType

# Create an audit logger
audit_logger = AuditLogger(
    log_file="audit_trail.log",
    retention_days=90
)

# Record an audit event
audit_logger.log_event(
    agent_id="agent123",
    event_type=AuditEventType.API_CALL,
    event_details={
        "endpoint": "/api/data",
        "method": "GET",
        "parameters": {"id": "12345"},
        "response_code": 200
    }
)

# Record a state change event
audit_logger.log_event(
    agent_id="agent123",
    event_type=AuditEventType.STATE_CHANGE,
    event_details={
        "previous_state": "IDLE",
        "new_state": "PROCESSING",
        "triggered_by": "user_request"
    }
)

# Query audit events
events = audit_logger.query_events(
    agent_id="agent123",
    event_types=[AuditEventType.API_CALL, AuditEventType.STATE_CHANGE],
    start_time=datetime.now() - timedelta(days=7),
    end_time=datetime.now()
)

print(f"Found {len(events)} audit events")
for event in events[:5]:
    print(f"{event.timestamp} - {event.event_type}: {event.event_details}")
```

### Compliance Reports

Generate compliance reports for agents:

```python
from safeguards.monitoring.compliance import ComplianceReporter
from safeguards.types.compliance import ComplianceCheck, ComplianceStatus

# Create a compliance reporter
compliance_reporter = ComplianceReporter()

# Add compliance checks
compliance_reporter.add_check(
    name="budget_compliance",
    description="Check if agents stay within budget limits",
    check_function=lambda: check_budget_compliance()  # Your implementation
)

compliance_reporter.add_check(
    name="data_privacy",
    description="Check if agents handle sensitive data properly",
    check_function=lambda: check_data_privacy()  # Your implementation
)

compliance_reporter.add_check(
    name="rate_limiting",
    description="Check if agents respect API rate limits",
    check_function=lambda: check_rate_limiting()  # Your implementation
)

# Generate a compliance report
report = compliance_reporter.generate_report()
print(f"Compliance status: {report.overall_status.name}")
print(f"Passed checks: {len([c for c in report.checks if c.status == ComplianceStatus.PASSED])}")
print(f"Failed checks: {len([c for c in report.checks if c.status == ComplianceStatus.FAILED])}")

# Export the report
compliance_reporter.export_report(report, format="pdf", output_file="compliance_report.pdf")

# Example helper functions (implement based on your needs)
def check_budget_compliance():
    """Check if all agents are within budget limits."""
    # Implementation depends on your system
    # This is just a placeholder
    return ComplianceStatus.PASSED, "All agents within budget limits"

def check_data_privacy():
    """Check if agents handle sensitive data properly."""
    # Implementation depends on your system
    # This is just a placeholder
    return ComplianceStatus.PASSED, "No data privacy violations detected"

def check_rate_limiting():
    """Check if agents respect API rate limits."""
    # Implementation depends on your system
    # This is just a placeholder
    return ComplianceStatus.WARNING, "Some agents approaching rate limits"
```

## Best Practices

### Recommended Metrics

Consider tracking these key metrics for agents:

- **Resource Usage**: CPU, memory, network, and disk usage
- **Performance**: Response time, task completion time
- **Activity**: API calls, tasks processed, actions taken
- **Errors**: Count of errors, exceptions, and failures
- **Budget**: Usage rate, remaining budget, budget efficiency
- **Domain-Specific**: Metrics relevant to your specific use case

### Monitoring Recommendations

1. **Monitor at Multiple Levels**:
   - Individual agents
   - Agent pools or groups
   - System-wide metrics

2. **Set Appropriate Thresholds**:
   - Baseline normal behavior before setting alert thresholds
   - Consider dynamic thresholds that adapt to patterns

3. **Balance Detail and Overhead**:
   - Too much monitoring can impact performance
   - Focus on actionable metrics

4. **Implement Graduated Monitoring**:
   - More intensive monitoring for critical agents
   - Less frequent checks for less critical agents

5. **Correlate Metrics and Events**:
   - Look for patterns across different metrics
   - Correlate monitoring data with specific agent actions

## Conclusion

Effective monitoring is critical for maintaining the safety and reliability of agent systems. The Safeguards provides a comprehensive set of tools for metrics collection, runtime monitoring, violation detection, and visualization that allow you to keep track of your agents' behavior and resource usage.

By implementing the patterns shown in this guide, you can gain visibility into your agents' operations, detect potential issues before they become critical, and maintain an audit trail for compliance purposes.

For more information, see:
- [Notifications Guide](notifications.md)
- [Budget Management Guide](budget_management.md)
- [API Reference](../api/monitoring.md)
