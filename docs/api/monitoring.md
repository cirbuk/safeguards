# Monitoring API Reference

This document provides a detailed reference for the monitoring APIs in the Safeguards.

## MetricsCollector

`MetricsCollector` is the core class for gathering and storing metrics about agent performance and resource usage.

```python
from safeguards.monitoring.metrics_collector import MetricsCollector

# Create a metrics collector
metrics_collector = MetricsCollector()
```

### Methods

#### `register_agent`

```python
def register_agent(self, agent_id: str) -> None:
    """
    Register an agent for metrics collection.

    Args:
        agent_id: Unique identifier for the agent
    """
```

#### `record_metric`

```python
def record_metric(
    self,
    agent_id: str,
    metric_name: str,
    value: Union[int, float, Decimal],
    unit: Optional[str] = None,
    context: Optional[Dict[str, Any]] = None
) -> None:
    """
    Record a metric value for an agent.

    Args:
        agent_id: Agent identifier
        metric_name: Name of the metric
        value: Numerical value of the metric
        unit: Optional unit of measurement
        context: Optional contextual information
    """
```

#### `get_agent_metrics`

```python
def get_agent_metrics(
    self,
    agent_id: str,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None
) -> Dict[str, Any]:
    """
    Get all metrics for an agent within the specified time range.

    Args:
        agent_id: Agent identifier
        start_time: ISO-format start time (e.g., "2023-01-01T00:00:00Z")
        end_time: ISO-format end time (e.g., "2023-01-02T00:00:00Z")

    Returns:
        Dictionary containing the agent's metrics
    """
```

#### `get_agent_metrics_by_type`

```python
def get_agent_metrics_by_type(
    self,
    agent_id: str,
    metric_type: str,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None
) -> Dict[str, Any]:
    """
    Get metrics of a specific type for an agent.

    Args:
        agent_id: Agent identifier
        metric_type: Type of metrics to retrieve (e.g., "performance", "budget")
        start_time: ISO-format start time
        end_time: ISO-format end time

    Returns:
        Dictionary containing the requested metrics
    """
```

#### `get_pool_metrics`

```python
def get_pool_metrics(
    self,
    pool_id: str,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None
) -> Dict[str, Any]:
    """
    Get metrics for a budget pool.

    Args:
        pool_id: Pool identifier
        start_time: ISO-format start time
        end_time: ISO-format end time

    Returns:
        Dictionary containing the pool's metrics
    """
```

## ViolationReporter

`ViolationReporter` is responsible for detecting and reporting safety violations.

```python
from safeguards.monitoring.violation_reporter import ViolationReporter
from safeguards.core.notification_manager import NotificationManager

# Create a violation reporter
notification_manager = NotificationManager()
violation_reporter = ViolationReporter(notification_manager)
```

### Methods

#### `report_violation`

```python
def report_violation(
    self,
    agent_id: Optional[str] = None,
    violation_type: ViolationType = ViolationType.UNKNOWN,
    severity: AlertSeverity = AlertSeverity.MEDIUM,
    message: str = "",
    details: Optional[Dict[str, Any]] = None
) -> str:
    """
    Report a safety violation.

    Args:
        agent_id: Optional identifier of the agent involved
        violation_type: Type of violation from ViolationType enum
        severity: Severity level from AlertSeverity enum
        message: Human-readable description of the violation
        details: Additional details about the violation

    Returns:
        Unique identifier for the violation report
    """
```

#### `get_violations`

```python
def get_violations(
    self,
    agent_id: Optional[str] = None,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
    violation_type: Optional[ViolationType] = None,
    min_severity: Optional[AlertSeverity] = None
) -> List[Dict[str, Any]]:
    """
    Retrieve violation reports matching the specified criteria.

    Args:
        agent_id: Optional agent identifier to filter by
        start_time: ISO-format start time
        end_time: ISO-format end time
        violation_type: Optional violation type to filter by
        min_severity: Optional minimum severity level

    Returns:
        List of violation reports matching the criteria
    """
```

## RuntimeMonitor

`RuntimeMonitor` provides real-time monitoring of agent activities.

```python
from safeguards.monitoring.runtime_monitor import RuntimeMonitor

# Create a runtime monitor
runtime_monitor = RuntimeMonitor(metrics_collector)
```

### Methods

#### `register_agent`

```python
def register_agent(self, agent_id: str) -> None:
    """
    Register an agent for runtime monitoring.

    Args:
        agent_id: Agent identifier
    """
```

#### `register_monitor`

```python
def register_monitor(self, monitor: AgentMonitor) -> None:
    """
    Register a custom monitor.

    Args:
        monitor: An AgentMonitor instance to register
    """
```

#### `configure`

```python
def configure(
    self,
    check_interval: int = 60,
    metrics_rollup_interval: int = 300,
    alert_threshold_check_interval: int = 60
) -> None:
    """
    Configure monitoring intervals.

    Args:
        check_interval: Seconds between agent checks
        metrics_rollup_interval: Seconds between metrics aggregation
        alert_threshold_check_interval: Seconds between alert threshold checks
    """
```

#### `start`

```python
def start(self) -> None:
    """
    Start the runtime monitoring process.
    """
```

#### `stop`

```python
def stop(self) -> None:
    """
    Stop the runtime monitoring process.
    """
```

## AgentMonitor

`AgentMonitor` is the base class for implementing custom monitoring logic.

```python
from safeguards.monitoring.agent_monitor import AgentMonitor

class CustomMonitor(AgentMonitor):
    def check(self, agent_id: str, metrics: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        # Custom monitoring logic
        # Return an alert dict if an issue is detected, None otherwise
        pass
```

### Abstract Methods

#### `check`

```python
@abstractmethod
def check(self, agent_id: str, metrics: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Check agent metrics for issues.

    Args:
        agent_id: Agent identifier
        metrics: Current metrics for the agent

    Returns:
        Alert dict if an issue is detected, None otherwise
    """
```

## MetricsAPI

`MetricsAPI` provides a unified interface for metrics operations.

```python
from safeguards.api import APIFactory, APIVersion

# Create a metrics API
api_factory = APIFactory()
metrics_api = api_factory.create_metrics_api(
    version=APIVersion.V1,
    metrics_collector=metrics_collector
)
```

### Methods

#### `register_agent`

```python
def register_agent(self, agent_id: str) -> None:
    """
    Register an agent for metrics collection.

    Args:
        agent_id: Agent identifier
    """
```

#### `record_metric`

```python
def record_metric(
    self,
    agent_id: str,
    metric_name: str,
    value: Union[int, float, Decimal],
    unit: Optional[str] = None,
    context: Optional[Dict[str, Any]] = None
) -> None:
    """
    Record a metric value for an agent.

    Args:
        agent_id: Agent identifier
        metric_name: Name of the metric
        value: Numerical value of the metric
        unit: Optional unit of measurement
        context: Optional contextual information
    """
```

#### `get_agent_metrics`

```python
def get_agent_metrics(
    self,
    agent_id: str,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None
) -> Dict[str, Any]:
    """
    Get all metrics for an agent.

    Args:
        agent_id: Agent identifier
        start_time: ISO-format start time
        end_time: ISO-format end time

    Returns:
        Dictionary containing the agent's metrics
    """
```

#### `get_agent_usage_history`

```python
def get_agent_usage_history(
    self,
    agent_id: str,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Get usage history for an agent.

    Args:
        agent_id: Agent identifier
        start_time: ISO-format start time
        end_time: ISO-format end time

    Returns:
        List of usage records
    """
```

#### `get_violations`

```python
def get_violations(
    self,
    agent_id: Optional[str] = None,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
    violation_type: Optional[ViolationType] = None,
    min_severity: Optional[AlertSeverity] = None
) -> List[Dict[str, Any]]:
    """
    Get violation reports.

    Args:
        agent_id: Optional agent identifier to filter by
        start_time: ISO-format start time
        end_time: ISO-format end time
        violation_type: Optional violation type to filter by
        min_severity: Optional minimum severity level

    Returns:
        List of violation reports matching the criteria
    """
```

## DashboardGenerator

`DashboardGenerator` assists with creating monitoring dashboards.

```python
from safeguards.monitoring.visualization import DashboardGenerator

# Create a dashboard generator
dashboard = DashboardGenerator()
```

### Methods

#### `add_metric_panel`

```python
def add_metric_panel(
    self,
    title: str,
    metric_names: List[str],
    display_type: str = "line",
    timeframe: str = "last_6h",
    agent_ids: Optional[List[str]] = None
) -> None:
    """
    Add a metrics panel to the dashboard.

    Args:
        title: Panel title
        metric_names: List of metrics to display
        display_type: Visualization type (line, gauge, bar, etc.)
        timeframe: Time range to display
        agent_ids: Optional list of agent IDs to include
    """
```

#### `add_alert_panel`

```python
def add_alert_panel(
    self,
    title: str,
    alert_types: Optional[List[str]] = None,
    display_type: str = "list",
    max_items: int = 10
) -> None:
    """
    Add an alerts panel to the dashboard.

    Args:
        title: Panel title
        alert_types: Optional list of alert types to display
        display_type: Visualization type (list, table, etc.)
        max_items: Maximum number of items to display
    """
```

#### `generate_config`

```python
def generate_config(
    self,
    dashboard_title: str,
    refresh_interval: int = 60
) -> Dict[str, Any]:
    """
    Generate dashboard configuration.

    Args:
        dashboard_title: Title of the dashboard
        refresh_interval: Refresh interval in seconds

    Returns:
        Dashboard configuration dictionary
    """
```

## AlertManager

`AlertManager` manages alert rules and triggers notifications.

```python
from safeguards.monitoring.alert_manager import AlertManager

# Create an alert manager
alert_manager = AlertManager(notification_manager)
```

### Methods

#### `add_alert_rule`

```python
def add_alert_rule(
    self,
    name: str,
    condition: Callable[[Dict[str, Any]], bool],
    message: str,
    severity: AlertSeverity = AlertSeverity.MEDIUM,
    throttle_duration: int = 300
) -> None:
    """
    Add an alert rule.

    Args:
        name: Rule name
        condition: Function that takes metrics dict and returns True if alert should trigger
        message: Alert message template
        severity: Alert severity level
        throttle_duration: Minimum seconds between repeated alerts
    """
```

#### `check_metrics`

```python
def check_metrics(
    self,
    agent_id: str,
    metrics: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """
    Check metrics against alert rules.

    Args:
        agent_id: Agent identifier
        metrics: Current metrics

    Returns:
        List of triggered alerts
    """
```

## Examples

### Basic Metrics Collection

```python
from safeguards.api import APIFactory, APIVersion
from safeguards.monitoring.metrics_collector import MetricsCollector
from decimal import Decimal

# Setup
metrics_collector = MetricsCollector()
api_factory = APIFactory()
metrics_api = api_factory.create_metrics_api(
    version=APIVersion.V1,
    metrics_collector=metrics_collector
)

# Register agent
agent_id = "agent123"
metrics_api.register_agent(agent_id)

# Record various metrics
metrics_api.record_metric(
    agent_id=agent_id,
    metric_name="task_completion_time",
    value=2.3,
    unit="seconds",
    context={"task_id": "task456", "task_type": "text_processing"}
)

metrics_api.record_metric(
    agent_id=agent_id,
    metric_name="token_usage",
    value=1250,
    unit="tokens"
)

metrics_api.record_metric(
    agent_id=agent_id,
    metric_name="api_cost",
    value=Decimal("0.025"),
    unit="USD"
)

# Retrieve metrics
agent_metrics = metrics_api.get_agent_metrics(agent_id)
print(f"Metrics for agent {agent_id}:")
print(f"Task completion time: {agent_metrics.get('task_completion_time', 'N/A')}")
print(f"Token usage: {agent_metrics.get('token_usage', 'N/A')}")
print(f"API cost: {agent_metrics.get('api_cost', 'N/A')}")
```

### Custom Monitoring

```python
from safeguards.monitoring.agent_monitor import AgentMonitor
from safeguards.monitoring.runtime_monitor import RuntimeMonitor
from safeguards.types.enums import AlertSeverity
from typing import Dict, Any, Optional

# Create a custom monitor
class ResourceMonitor(AgentMonitor):
    def __init__(self, token_limit: int = 10000, cost_limit: Decimal = Decimal("1.0")):
        self.token_limit = token_limit
        self.cost_limit = cost_limit

    def check(self, agent_id: str, metrics: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        # Check token usage
        token_usage = metrics.get("token_usage", 0)
        if token_usage > self.token_limit:
            return {
                "severity": AlertSeverity.HIGH,
                "message": f"Token usage exceeded for agent {agent_id}",
                "details": {
                    "usage": token_usage,
                    "limit": self.token_limit,
                    "overage": token_usage - self.token_limit
                }
            }

        # Check cost
        api_cost = metrics.get("api_cost", Decimal("0"))
        if api_cost > self.cost_limit:
            return {
                "severity": AlertSeverity.HIGH,
                "message": f"Cost limit exceeded for agent {agent_id}",
                "details": {
                    "cost": str(api_cost),
                    "limit": str(self.cost_limit),
                    "overage": str(api_cost - self.cost_limit)
                }
            }

        return None  # No issues detected

# Set up runtime monitoring
metrics_collector = MetricsCollector()
runtime_monitor = RuntimeMonitor(metrics_collector)

# Register the custom monitor
resource_monitor = ResourceMonitor(token_limit=5000, cost_limit=Decimal("0.5"))
runtime_monitor.register_monitor(resource_monitor)

# Register agent and start monitoring
runtime_monitor.register_agent("agent123")
runtime_monitor.configure(check_interval=30)  # Check every 30 seconds
runtime_monitor.start()
```

### Violation Reporting

```python
from safeguards.monitoring.violation_reporter import ViolationReporter
from safeguards.core.notification_manager import NotificationManager
from safeguards.types.enums import ViolationType, AlertSeverity

# Setup
notification_manager = NotificationManager()
violation_reporter = ViolationReporter(notification_manager)

# Report a violation
violation_id = violation_reporter.report_violation(
    agent_id="agent123",
    violation_type=ViolationType.RESOURCE_LIMIT_EXCEEDED,
    severity=AlertSeverity.HIGH,
    message="Agent exceeded token limit",
    details={
        "limit": 1000,
        "actual_usage": 1500,
        "resource_type": "tokens"
    }
)

print(f"Violation reported with ID: {violation_id}")

# Retrieve violations
violations = violation_reporter.get_violations(
    agent_id="agent123",
    min_severity=AlertSeverity.MEDIUM
)

print(f"Found {len(violations)} violations")
for violation in violations:
    print(f"- {violation['timestamp']}: {violation['message']} ({violation['severity']})")
```

For more information about monitoring in the Safeguards, see:
- [Monitoring Guide](../guides/monitoring.md)
- [Budget Management Guide](../guides/budget_management.md)
- [Agent Safety Guide](../guides/safeguards.md)
