# Safeguards Guide

This guide covers the essential safety features and best practices for the Safeguards.

## Core Safety Concepts

The Safeguards provides several layers of protection to prevent agents from causing harm:

1. **Budget Constraints** - Limit resource usage
2. **Violation Detection** - Identify and report problematic behavior
3. **Guardrails** - Enforce safety boundaries
4. **Monitoring** - Track agent activities in real-time
5. **Notifications** - Alert operators to potential issues

## Setting Up Safety Components

### Notification Manager

The `NotificationManager` is the central component for safety alerts:

```python
from safeguards.core.notification_manager import NotificationManager
from safeguards.types import AlertSeverity

# Create a notification manager
notification_manager = NotificationManager()

# Register a handler for alerts
def alert_handler(alert):
    print(f"Alert received: {alert.message}")
    print(f"Severity: {alert.severity.name}")
    print(f"Agent ID: {alert.agent_id}")
    print(f"Timestamp: {alert.timestamp}")

    # Custom handling logic based on severity
    if alert.severity == AlertSeverity.CRITICAL:
        # Take immediate action (e.g., shutdown agent)
        pass
    elif alert.severity == AlertSeverity.HIGH:
        # Log and notify on-call personnel
        pass

    # Return True to acknowledge handling
    return True

# Add the handler
notification_manager.add_handler(alert_handler)
```

### Violation Reporter

The `ViolationReporter` detects and reports safety violations:

```python
from safeguards.monitoring.violation_reporter import ViolationReporter
from safeguards.types import ViolationType, AlertSeverity

# Create a violation reporter
violation_reporter = ViolationReporter(notification_manager)

# Report a violation
violation_reporter.report_violation(
    agent_id="agent123",
    violation_type=ViolationType.UNAUTHORIZED_ACTION,
    severity=AlertSeverity.HIGH,
    message="Agent attempted unauthorized system access",
    details={
        "action": "file_access",
        "target": "/etc/passwd",
        "timestamp": "2023-05-15T14:30:00Z"
    }
)
```

### Safety API

The Safety API provides a unified interface for safety operations:

```python
from safeguards.api import APIFactory, APIVersion

# Create API factory
api_factory = APIFactory()

# Create safety API
safety_api = api_factory.create_safety_api(
    version=APIVersion.V1,
    notification_manager=notification_manager,
    violation_reporter=violation_reporter
)

# Configure safety policy
safety_api.configure_safety_policy(
    policy_name="strict_policy",
    config={
        "allowed_actions": ["read_file", "write_file", "network_request"],
        "blocked_domains": ["malicious-site.com"],
        "max_token_usage": 10000,
        "require_approval_for": ["system_command", "payment_processing"]
    }
)
```

## Safety Guardrails

### Implementing Action Filters

Action filters prevent agents from performing unsafe actions:

```python
from safeguards.guardrails.action_filter import ActionFilter
from safeguards.types.agent import AgentAction, AgentActionResult

# Create a custom action filter
class NetworkAccessFilter(ActionFilter):
    def __init__(self, allowed_domains=None):
        self.allowed_domains = allowed_domains or []

    def filter(self, agent_id, action):
        if action.action_type == "network_request":
            target_domain = self._extract_domain(action.parameters.get("url", ""))

            if target_domain not in self.allowed_domains:
                return AgentActionResult(
                    success=False,
                    result=None,
                    error="Domain not in allowed list",
                    metadata={
                        "blocked_domain": target_domain,
                        "allowed_domains": self.allowed_domains
                    }
                )

        # Allow action to proceed
        return None

    def _extract_domain(self, url):
        # Simple domain extraction logic
        if url.startswith("http"):
            parts = url.split("/")
            if len(parts) > 2:
                return parts[2]
        return url

# Register the filter with the safety API
network_filter = NetworkAccessFilter(
    allowed_domains=["api.example.com", "data.example.org"]
)
safety_api.register_action_filter(network_filter)
```

### Content Monitoring

Monitor agent-generated content for safety issues:

```python
from safeguards.guardrails.content_monitor import ContentMonitor
from safeguards.types import AlertSeverity

# Create a custom content monitor
class SensitiveDataMonitor(ContentMonitor):
    def __init__(self, patterns=None):
        self.patterns = patterns or [
            r'\b\d{3}-\d{2}-\d{4}\b',  # SSN pattern
            r'\b\d{16}\b',  # Credit card pattern
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'  # Email
        ]

    def analyze(self, agent_id, content):
        import re

        findings = []
        for pattern in self.patterns:
            matches = re.findall(pattern, content)
            if matches:
                findings.append({
                    "pattern": pattern,
                    "match_count": len(matches),
                    "risk": AlertSeverity.HIGH
                })

        if findings:
            # Report violation
            violation_reporter.report_violation(
                agent_id=agent_id,
                violation_type=ViolationType.SENSITIVE_DATA_EXPOSURE,
                severity=AlertSeverity.HIGH,
                message="Agent attempted to output sensitive data",
                details={"findings": findings}
            )

            # Return redacted content
            redacted = content
            for pattern in self.patterns:
                redacted = re.sub(pattern, "[REDACTED]", redacted)
            return redacted

        # Return original content if no issues found
        return content

# Register the monitor
sensitive_data_monitor = SensitiveDataMonitor()
safety_api.register_content_monitor(sensitive_data_monitor)
```

### Resource Constraints

Implement resource constraints to prevent overuse:

```python
from safeguards.guardrails.resource_monitor import ResourceMonitor
from decimal import Decimal

# Create a custom resource monitor
class TokenUsageMonitor(ResourceMonitor):
    def __init__(self, token_limit):
        self.token_limit = token_limit
        self.token_counts = {}

    def check(self, agent_id, usage):
        # Initialize if not present
        if agent_id not in self.token_counts:
            self.token_counts[agent_id] = 0

        # Update count
        self.token_counts[agent_id] += usage
        current_usage = self.token_counts[agent_id]

        # Check if limit exceeded
        if current_usage > self.token_limit:
            # Report violation
            violation_reporter.report_violation(
                agent_id=agent_id,
                violation_type=ViolationType.RESOURCE_LIMIT_EXCEEDED,
                severity=AlertSeverity.MEDIUM,
                message=f"Token usage limit exceeded: {current_usage}/{self.token_limit}",
                details={
                    "limit": self.token_limit,
                    "usage": current_usage,
                    "overage": current_usage - self.token_limit
                }
            )
            return False

        # Return True if within limits
        return True

    def reset(self, agent_id):
        if agent_id in self.token_counts:
            self.token_counts[agent_id] = 0

# Register the monitor
token_monitor = TokenUsageMonitor(token_limit=10000)
safety_api.register_resource_monitor(token_monitor)
```

## Behavioral Safety

### Implementing Safe Agents

Create agents with built-in safety features:

```python
from safeguards.types.agent import Agent
from safeguards.core.budget_coordination import BudgetCoordinator

class SafeAgent(Agent):
    def __init__(self, name, safety_api, **kwargs):
        super().__init__(name, **kwargs)
        self.safety_api = safety_api

    def run(self, task, **kwargs):
        # Safety check before running
        is_safe = self.safety_api.check_task_safety(
            agent_id=self.id,
            task_description=task,
            context=kwargs
        )

        if not is_safe:
            return {
                "status": "blocked",
                "reason": "Task failed safety check",
                "details": "The requested task violates safety policies"
            }

        # Proceed with the task
        try:
            # Task implementation logic
            result = self._process_task(task, **kwargs)

            # Check output for safety issues
            safe_result = self.safety_api.check_content_safety(
                agent_id=self.id,
                content=result
            )

            return safe_result

        except Exception as e:
            # Handle exceptions safely
            self.safety_api.report_exception(
                agent_id=self.id,
                exception=e,
                context={
                    "task": task,
                    "parameters": kwargs
                }
            )

            return {
                "status": "error",
                "reason": "Exception during task execution",
                "details": str(e)
            }

    def _process_task(self, task, **kwargs):
        # Actual task implementation
        # This would be implemented by specific agent subclasses
        pass
```

### Safe Action Execution

Implement safe action execution logic:

```python
def execute_action_safely(agent_id, action, safety_api):
    """Execute an action with safety checks."""
    # Check if action is allowed
    action_result = safety_api.check_action_safety(
        agent_id=agent_id,
        action=action
    )

    if action_result and not action_result.success:
        # Action was blocked
        return action_result

    # Action is allowed, proceed with execution
    try:
        # Actual action execution logic
        result = _execute_action(action)

        # Monitor resource usage
        if hasattr(result, 'resource_usage'):
            safety_api.track_resource_usage(
                agent_id=agent_id,
                resource_type=action.action_type,
                amount=result.resource_usage
            )

        return result

    except Exception as e:
        # Handle and report exception
        safety_api.report_exception(
            agent_id=agent_id,
            exception=e,
            context={"action": action}
        )

        return AgentActionResult(
            success=False,
            result=None,
            error=str(e)
        )

def _execute_action(action):
    # Implement the logic to execute different action types
    # This is a placeholder for the actual implementation
    pass
```

## Safety Monitoring and Alerts

### Setting Up Real-time Monitoring

```python
from safeguards.monitoring.metrics_collector import MetricsCollector
from datetime import datetime, timedelta

# Create a metrics collector
metrics_collector = MetricsCollector()

# Register with safety API
safety_api.register_metrics_collector(metrics_collector)

# Collect and analyze metrics
def analyze_safeguards(agent_id, time_window=24):
    # Get metrics for the last 'time_window' hours
    start_time = datetime.now() - timedelta(hours=time_window)
    metrics = metrics_collector.get_agent_metrics(
        agent_id=agent_id,
        start_time=start_time.isoformat(),
        end_time=datetime.now().isoformat()
    )

    # Analyze violations
    violations = metrics.get("violations", [])
    violation_count = len(violations)

    # Analyze resource usage
    resource_usage = metrics.get("resource_usage", {})
    token_usage = resource_usage.get("token_usage", 0)

    # Check action patterns
    actions = metrics.get("actions", [])
    action_types = {}
    for action in actions:
        action_type = action.get("action_type", "unknown")
        action_types[action_type] = action_types.get(action_type, 0) + 1

    # Generate safety report
    safety_score = calculate_safety_score(
        violation_count=violation_count,
        token_usage=token_usage,
        action_types=action_types
    )

    return {
        "agent_id": agent_id,
        "safety_score": safety_score,
        "time_window_hours": time_window,
        "violation_count": violation_count,
        "token_usage": token_usage,
        "action_patterns": action_types,
        "recommendations": generate_safety_recommendations(safety_score)
    }

def calculate_safety_score(violation_count, token_usage, action_types):
    # Example scoring algorithm
    # This would be more sophisticated in production
    base_score = 100

    # Deduct for violations
    violation_penalty = min(violation_count * 10, 50)

    # Deduct for excessive token usage
    token_penalty = 0
    if token_usage > 10000:
        token_penalty = min((token_usage - 10000) / 1000, 30)

    # Check for suspicious action patterns
    action_penalty = 0
    suspicious_actions = action_types.get("system_command", 0) + action_types.get("network_request", 0)
    if suspicious_actions > 10:
        action_penalty = min(suspicious_actions, 20)

    final_score = max(base_score - violation_penalty - token_penalty - action_penalty, 0)
    return final_score

def generate_safety_recommendations(safety_score):
    if safety_score < 50:
        return [
            "Consider suspending agent activity for review",
            "Implement stricter action filters",
            "Reduce resource allocations"
        ]
    elif safety_score < 70:
        return [
            "Review recent violations",
            "Monitor agent actions more closely",
            "Consider additional guardrails"
        ]
    elif safety_score < 90:
        return [
            "Regular monitoring recommended",
            "Continue with current safety settings"
        ]
    else:
        return ["Agent operating within safe parameters"]
```

### Alert Escalation

Configure alert escalation based on severity:

```python
from safeguards.types import AlertSeverity

def configure_alert_escalation(notification_manager):
    """Configure different handlers for different alert severities."""

    # Handler for low severity alerts
    def handle_low_alerts(alert):
        if alert.severity in [AlertSeverity.LOW, AlertSeverity.INFORMATIONAL]:
            # Log only
            print(f"INFO: {alert.message}")
            return True
        return False

    # Handler for medium severity alerts
    def handle_medium_alerts(alert):
        if alert.severity == AlertSeverity.MEDIUM:
            # Log and send notification
            print(f"WARNING: {alert.message}")
            send_notification("security@example.com", f"Medium Alert: {alert.message}")
            return True
        return False

    # Handler for high severity alerts
    def handle_high_alerts(alert):
        if alert.severity == AlertSeverity.HIGH:
            # Log, send notification, and take action
            print(f"ALERT: {alert.message}")
            send_notification("oncall@example.com", f"High Alert: {alert.message}")

            # Take immediate action
            if alert.agent_id:
                # Example: Reduce agent permissions
                safety_api.restrict_agent_permissions(
                    agent_id=alert.agent_id,
                    restrictions={"block_external_actions": True}
                )
            return True
        return False

    # Handler for critical alerts
    def handle_critical_alerts(alert):
        if alert.severity == AlertSeverity.CRITICAL:
            # Log, send emergency notification, and take immediate action
            print(f"CRITICAL: {alert.message}")
            send_emergency_notification(f"Critical Alert: {alert.message}")

            # Example: Suspend agent
            if alert.agent_id:
                safety_api.suspend_agent(
                    agent_id=alert.agent_id,
                    reason=f"Critical safety alert: {alert.message}"
                )
            return True
        return False

    # Add handlers in order (critical first for fastest response)
    notification_manager.add_handler(handle_critical_alerts)
    notification_manager.add_handler(handle_high_alerts)
    notification_manager.add_handler(handle_medium_alerts)
    notification_manager.add_handler(handle_low_alerts)

def send_notification(recipient, message):
    # Implementation would depend on your notification system
    # Example: email, Slack, etc.
    pass

def send_emergency_notification(message):
    # Implementation for emergency notifications
    # Example: pager duty, SMS alerts, etc.
    pass
```

## Safety Testing

### Implementing Safety Tests

Create comprehensive tests for safety features:

```python
import pytest
from safeguards.core.notification_manager import NotificationManager
from safeguards.monitoring.violation_reporter import ViolationReporter
from safeguards.types import ViolationType, AlertSeverity

# Fixtures for testing
@pytest.fixture
def notification_manager():
    return NotificationManager()

@pytest.fixture
def violation_reporter(notification_manager):
    return ViolationReporter(notification_manager)

@pytest.fixture
def safety_api(notification_manager, violation_reporter):
    api_factory = APIFactory()
    return api_factory.create_safety_api(
        version=APIVersion.V1,
        notification_manager=notification_manager,
        violation_reporter=violation_reporter
    )

@pytest.fixture
def test_agent(safety_api):
    class TestAgent(SafeAgent):
        def _process_task(self, task, **kwargs):
            return f"Processed task: {task}"

    return TestAgent(
        name="test_agent",
        safety_api=safety_api
    )

# Test basic safety checks
def test_basic_safety_check(test_agent, safety_api):
    # Configure a simple safety policy
    safety_api.configure_safety_policy(
        policy_name="test_policy",
        config={
            "blocked_terms": ["malicious", "harmful"]
        }
    )

    # Test with safe task
    result = test_agent.run("normal task")
    assert "Processed task" in result

    # Test with unsafe task
    result = test_agent.run("malicious operation")
    assert "blocked" in result.get("status", "")

# Test action filtering
def test_action_filtering(test_agent, safety_api):
    # Register action filter
    class TestFilter(ActionFilter):
        def filter(self, agent_id, action):
            if action.action_type == "test_action" and "block_me" in action.parameters:
                return AgentActionResult(
                    success=False,
                    error="Blocked action"
                )
            return None

    safety_api.register_action_filter(TestFilter())

    # Test allowed action
    action = AgentAction(
        action_type="test_action",
        parameters={"normal": "value"}
    )
    result = execute_action_safely(test_agent.id, action, safety_api)
    assert result.success

    # Test blocked action
    action = AgentAction(
        action_type="test_action",
        parameters={"block_me": "value"}
    )
    result = execute_action_safely(test_agent.id, action, safety_api)
    assert not result.success

# Test violation reporting
def test_violation_reporting(test_agent, violation_reporter, notification_manager):
    # Set up alert capture
    alerts = []

    def capture_alert(alert):
        alerts.append(alert)
        return True

    notification_manager.add_handler(capture_alert)

    # Report a violation
    violation_reporter.report_violation(
        agent_id=test_agent.id,
        violation_type=ViolationType.UNAUTHORIZED_ACTION,
        severity=AlertSeverity.HIGH,
        message="Test violation"
    )

    # Check alert was captured
    assert len(alerts) == 1
    assert alerts[0].agent_id == test_agent.id
    assert alerts[0].severity == AlertSeverity.HIGH
```

## Emergency Response

### Implementing Kill Switches

Create emergency response mechanisms:

```python
class EmergencyControls:
    def __init__(self, budget_coordinator, notification_manager):
        self.budget_coordinator = budget_coordinator
        self.notification_manager = notification_manager
        self.suspended_agents = set()

    def suspend_agent(self, agent_id, reason=None):
        """Immediately suspend an agent's operations."""
        # Set agent budget to zero
        try:
            self.budget_coordinator.update_budget(
                agent_id=agent_id,
                amount=Decimal("0")
            )

            # Track suspended agent
            self.suspended_agents.add(agent_id)

            # Send notification
            self.notification_manager.send_alert(
                agent_id=agent_id,
                severity=AlertSeverity.HIGH,
                message=f"Agent {agent_id} suspended: {reason or 'Emergency suspension'}"
            )

            return True

        except Exception as e:
            # Log failure
            self.notification_manager.send_alert(
                agent_id=agent_id,
                severity=AlertSeverity.CRITICAL,
                message=f"Failed to suspend agent {agent_id}: {str(e)}"
            )
            return False

    def suspend_all_agents(self, reason=None):
        """Emergency kill switch for all agents."""
        agent_ids = self.budget_coordinator.get_all_agent_ids()
        success_count = 0

        for agent_id in agent_ids:
            if self.suspend_agent(agent_id, reason):
                success_count += 1

        # Send overall notification
        self.notification_manager.send_alert(
            severity=AlertSeverity.CRITICAL,
            message=f"Emergency shutdown triggered: {success_count}/{len(agent_ids)} agents suspended. Reason: {reason or 'Emergency procedure'}"
        )

        return success_count

    def restore_agent(self, agent_id, budget=None):
        """Restore a suspended agent."""
        if agent_id not in self.suspended_agents:
            return False

        try:
            # Set budget to specified amount or default
            if budget is None:
                # Get original budget from metadata or use default
                budget = Decimal("10.0")

            self.budget_coordinator.update_budget(
                agent_id=agent_id,
                amount=budget
            )

            # Remove from suspended list
            self.suspended_agents.remove(agent_id)

            # Send notification
            self.notification_manager.send_alert(
                agent_id=agent_id,
                severity=AlertSeverity.MEDIUM,
                message=f"Agent {agent_id} restored with budget {budget}"
            )

            return True

        except Exception as e:
            # Log failure
            self.notification_manager.send_alert(
                agent_id=agent_id,
                severity=AlertSeverity.HIGH,
                message=f"Failed to restore agent {agent_id}: {str(e)}"
            )
            return False
```

## Safety Best Practices

### General Recommendations

1. **Defense in Depth** - Implement multiple safety layers
2. **Principle of Least Privilege** - Give agents only the access they need
3. **Regular Auditing** - Review agent activities and safety logs frequently
4. **Graceful Degradation** - Plan for safety feature failures
5. **Continuous Testing** - Test safety systems regularly

### Handling User Interactions

When agents interact with users, follow these practices:

```python
def safe_user_interaction(agent_id, user_input, safety_api):
    """Handle user interactions safely."""
    # Sanitize user input
    sanitized_input = safety_api.sanitize_input(
        agent_id=agent_id,
        input_content=user_input
    )

    # Check input for prompt injection or harmful content
    input_analysis = safety_api.analyze_user_input(
        agent_id=agent_id,
        input_content=sanitized_input
    )

    if input_analysis.get("risk_level", "low") == "high":
        # Handle risky input
        return {
            "status": "rejected",
            "reason": "Input poses safety risk",
            "details": input_analysis.get("details", {})
        }

    # Process the input with safety measures
    try:
        # Generate response
        response = generate_agent_response(agent_id, sanitized_input)

        # Check response for safety issues
        safe_response = safety_api.check_content_safety(
            agent_id=agent_id,
            content=response
        )

        return {
            "status": "success",
            "response": safe_response
        }

    except Exception as e:
        # Handle exceptions
        safety_api.report_exception(
            agent_id=agent_id,
            exception=e,
            context={"user_input": sanitized_input}
        )

        return {
            "status": "error",
            "reason": "Error processing request",
            "details": str(e)
        }
```

### Environment Safety

Configure safe execution environments:

```python
def configure_safe_environment(agent_id, safety_api):
    """Configure a safe execution environment for an agent."""
    # Set resource limits
    safety_api.set_resource_limits(
        agent_id=agent_id,
        limits={
            "memory_mb": 512,
            "cpu_time_seconds": 30,
            "storage_mb": 100,
            "network_mb": 50
        }
    )

    # Configure filesystem sandbox
    safety_api.configure_filesystem_access(
        agent_id=agent_id,
        allowed_paths=["/data/agent_workspace"],
        read_only_paths=["/data/shared_resources"],
        blocked_paths=["/", "/etc", "/usr"]
    )

    # Set network restrictions
    safety_api.configure_network_access(
        agent_id=agent_id,
        allowed_domains=["api.example.com"],
        allowed_ports=[443],  # HTTPS only
        block_outbound=True,  # Block all outbound except allowed domains
        log_all_requests=True
    )

    # Configure execution timeouts
    safety_api.set_execution_limits(
        agent_id=agent_id,
        limits={
            "max_execution_time_seconds": 60,
            "max_consecutive_actions": 10,
            "require_confirmation_after": 5
        }
    )

    return "Environment configured with safety restrictions"
```

## Conclusion

Implementing comprehensive safety measures is critical for developing reliable and trustworthy agent systems. The Safeguards provides the tools and infrastructure needed to monitor, control, and secure agent behavior through multiple layers of protection.

For more information, see:
- [API Reference](../api/safety.md)
- [Budget Management Guide](budget_management.md)
- [Monitoring Guide](monitoring.md)
