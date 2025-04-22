# Safety Guardrails Guide

This guide explains how to implement and use safety guardrails in the Safeguards, providing protection mechanisms for your agent systems.

## Introduction to Safety Guardrails

Safety guardrails are preventative mechanisms that constrain agent behavior within safe operational boundaries. They:

- Proactively monitor agent actions before execution
- Enforce predefined safety constraints
- Prevent potentially harmful operations
- Provide fallback mechanisms when violations occur
- Create audit trails of safety-critical decisions

The Safeguards offers several types of guardrails for different safety concerns.

## Core Guardrail Types

### Budget Guardrails

Budget guardrails prevent agents from exceeding their allocated resources:

```python
from decimal import Decimal
from safeguards.guardrails.budget import BudgetGuardrail
from safeguards.types.agent import Agent
from safeguards.budget.manager import BudgetManager

# Create a budget manager
budget_manager = BudgetManager(
    agent_id="agent123",
    initial_budget=Decimal("100.0")
)

# Create a budget guardrail
budget_guardrail = BudgetGuardrail(budget_manager)

# Example of using the guardrail
def process_request(agent, input_data, expected_cost):
    """Process a request with budget safety checks."""
    # Check if sufficient budget
    if not budget_guardrail.validate(expected_cost):
        return {
            "status": "error",
            "message": "Insufficient budget to perform operation",
            "remaining_budget": budget_manager.get_remaining_budget()
        }

    try:
        # Process the request
        result = agent.run(input=input_data)

        # Update budget usage
        actual_cost = result.get("cost", Decimal("0"))
        budget_manager.record_cost(actual_cost)

        return {
            "status": "success",
            "result": result,
            "remaining_budget": budget_manager.get_remaining_budget()
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error during processing: {str(e)}",
            "remaining_budget": budget_manager.get_remaining_budget()
        }
```

### Resource Guardrails

Resource guardrails prevent excessive CPU, memory, or disk usage:

```python
from safeguards.guardrails.resource import ResourceGuardrail
from safeguards.monitoring.resource_monitor import ResourceMonitor

# Create a resource monitor
resource_monitor = ResourceMonitor(
    agent_id="agent123",
    thresholds={
        "cpu_percent": 80,
        "memory_percent": 70,
        "disk_usage_gb": 10
    }
)

# Create a resource guardrail
resource_guardrail = ResourceGuardrail(resource_monitor)

# Example of using the guardrail
def perform_resource_intensive_task(agent, task_params):
    """Perform a resource-intensive task with safety checks."""
    # Validate resource availability before executing
    validation_result = resource_guardrail.validate()

    if not validation_result.is_valid:
        return {
            "status": "error",
            "message": f"Resource check failed: {validation_result.message}",
            "details": validation_result.details
        }

    # Execute the task with guardrail protection
    try:
        result = resource_guardrail.run(
            agent.run,
            input=task_params,
            metadata={"task_type": "resource_intensive"}
        )
        return {
            "status": "success",
            "result": result
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Task execution failed: {str(e)}"
        }
```

### Security Guardrails

Security guardrails prevent unauthorized access and operations:

```python
from safeguards.security.auth import SecurityManager, Permission, Role
from safeguards.types.guardrail import SecurityGuardrail

# Create a security manager
security_manager = SecurityManager()

# Set up roles and permissions
security_manager.add_role("data_processor", [
    Permission.READ_DATA,
    Permission.PROCESS_DATA
])

security_manager.add_role("admin", [
    Permission.READ_DATA,
    Permission.PROCESS_DATA,
    Permission.MODIFY_SYSTEM,
    Permission.MANAGE_USERS
])

# Register an identity
security_manager.register_identity(
    identity_id="agent123",
    roles=["data_processor"]
)

# Create a security guardrail
security_guardrail = SecurityGuardrail(
    security_manager=security_manager,
    required_permissions=[Permission.PROCESS_DATA]
)

# Example of using the guardrail
def protected_operation(agent_id, operation, data):
    """Perform an operation with security guardrail protection."""
    try:
        # Run the operation through the security guardrail
        result = security_guardrail.run(
            lambda: process_data(operation, data),
            identity_id=agent_id,
            metadata={"operation": operation}
        )
        return {
            "status": "success",
            "result": result
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Security constraint violation: {str(e)}"
        }

def process_data(operation, data):
    """Process data based on the operation type."""
    # Implementation would go here
    return {"processed": True, "operation": operation}
```

## Creating Custom Guardrails

You can create custom guardrails for specific needs:

```python
from typing import Any, Dict, Optional, Callable
from safeguards.types.guardrail import Guardrail, ValidationResult

class CustomGuardrail(Guardrail):
    """A custom guardrail implementation."""

    def __init__(self, validation_fn: Callable):
        self.validation_fn = validation_fn

    def validate(self, *args, **kwargs) -> ValidationResult:
        """Validate using the custom validation function."""
        is_valid, message = self.validation_fn(*args, **kwargs)

        return ValidationResult(
            is_valid=is_valid,
            message=message,
            details={"guardrail_type": "custom"}
        )

    def run(self, fn: Callable, *args, **kwargs) -> Any:
        """Run a function with the guardrail applied."""
        # Validate before execution
        validation_result = self.validate(*args, **kwargs)

        if not validation_result.is_valid:
            raise ValueError(f"Guardrail validation failed: {validation_result.message}")

        # Execute the function
        return fn(*args, **kwargs)

# Example custom validation function
def validate_input_complexity(input_text: str) -> tuple[bool, str]:
    """Validate that input text isn't too complex."""
    word_count = len(input_text.split())

    if word_count > 1000:
        return False, f"Input too complex: {word_count} words exceeds 1000 word limit"

    return True, "Input complexity acceptable"

# Create a custom guardrail
complexity_guardrail = CustomGuardrail(validate_input_complexity)

# Use the guardrail
def process_text_safely(agent, text):
    """Process text with complexity guardrail."""
    try:
        result = complexity_guardrail.run(
            agent.run,
            input=text
        )
        return result
    except ValueError as e:
        return {
            "error": str(e),
            "suggestion": "Please simplify your input"
        }
```

## Combining Multiple Guardrails

For comprehensive safety, combine multiple guardrails:

```python
from safeguards.types.guardrail import CompositeGuardrail

# Create a composite guardrail
composite_guardrail = CompositeGuardrail([
    budget_guardrail,
    resource_guardrail,
    security_guardrail,
    complexity_guardrail
])

# Use the composite guardrail
def safe_agent_execution(agent, input_data, agent_id, expected_cost):
    """Execute agent with multiple safety guardrails."""
    try:
        # All guardrails will be checked in sequence
        result = composite_guardrail.run(
            agent.run,
            input=input_data,
            identity_id=agent_id,
            expected_cost=expected_cost
        )
        return {
            "status": "success",
            "result": result
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Guardrail violation: {str(e)}"
        }
```

## Guardrail Policies

Define guardrail policies to apply consistent safety measures:

```python
from safeguards.rules.base import RuleSet, Rule
from safeguards.types.guardrail import PolicyGuardrail

# Define rules
budget_rule = Rule(
    name="budget_limit",
    description="Limit budget usage per request",
    validator=lambda ctx: ctx.get("expected_cost", 0) <= 10.0,
    error_message="Request exceeds maximum budget of 10.0"
)

security_rule = Rule(
    name="authorized_operation",
    description="Ensure operation is authorized",
    validator=lambda ctx: ctx.get("operation") in ["read", "analyze", "summarize"],
    error_message="Operation not permitted"
)

resource_rule = Rule(
    name="resource_limit",
    description="Limit resource usage",
    validator=lambda ctx: ctx.get("resource_intensive", False) is False,
    error_message="Resource intensive operations not allowed"
)

# Create a rule set
rule_set = RuleSet("safeguards_policy")
rule_set.add_rule(budget_rule)
rule_set.add_rule(security_rule)
rule_set.add_rule(resource_rule)

# Create a policy guardrail
policy_guardrail = PolicyGuardrail(rule_set)

# Use the policy guardrail
def execute_with_policy(agent, input_data, context):
    """Execute agent with policy guardrail."""
    try:
        result = policy_guardrail.run(
            agent.run,
            input=input_data,
            context=context
        )
        return {
            "status": "success",
            "result": result
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Policy violation: {str(e)}"
        }
```

## Circuit Breaker Pattern

Implement circuit breakers to prevent repeated failures:

```python
from safeguards.types.guardrail import CircuitBreakerGuardrail

# Create a circuit breaker
circuit_breaker = CircuitBreakerGuardrail(
    failure_threshold=3,
    reset_timeout_seconds=300  # 5 minutes
)

# Use the circuit breaker
def call_service_safely(service_fn, *args, **kwargs):
    """Call a service with circuit breaker protection."""
    try:
        result = circuit_breaker.run(service_fn, *args, **kwargs)
        return result
    except Exception as e:
        if "circuit open" in str(e):
            return {
                "status": "error",
                "message": "Service temporarily unavailable, please try again later",
                "circuit_status": "open"
            }
        else:
            return {
                "status": "error",
                "message": f"Service error: {str(e)}",
                "circuit_status": "closed"
            }
```

## Monitoring Guardrail Activity

Track guardrail interventions:

```python
from safeguards.monitoring.violation_reporter import ViolationReporter
from safeguards.core.notification_manager import NotificationManager
from safeguards.types import ViolationType, AlertSeverity

# Create required components
notification_manager = NotificationManager()
violation_reporter = ViolationReporter(notification_manager)

# Create a monitored guardrail
class MonitoredGuardrail(Guardrail):
    """A guardrail that reports violations."""

    def __init__(self, inner_guardrail, violation_reporter, agent_id):
        self.inner_guardrail = inner_guardrail
        self.violation_reporter = violation_reporter
        self.agent_id = agent_id

    def validate(self, *args, **kwargs) -> ValidationResult:
        """Validate and report if invalid."""
        result = self.inner_guardrail.validate(*args, **kwargs)

        if not result.is_valid:
            # Report the violation
            self.violation_reporter.report_violation(
                agent_id=self.agent_id,
                violation_type=ViolationType.GUARDRAIL_VIOLATION,
                severity=AlertSeverity.HIGH,
                message=result.message,
                details={
                    "guardrail_type": self.inner_guardrail.__class__.__name__,
                    "validation_details": result.details
                }
            )

        return result

    def run(self, fn, *args, **kwargs):
        """Run with validation and violation reporting."""
        try:
            return self.inner_guardrail.run(fn, *args, **kwargs)
        except Exception as e:
            # Report the runtime violation
            self.violation_reporter.report_violation(
                agent_id=self.agent_id,
                violation_type=ViolationType.GUARDRAIL_VIOLATION,
                severity=AlertSeverity.HIGH,
                message=str(e),
                details={
                    "guardrail_type": self.inner_guardrail.__class__.__name__,
                    "exception_type": e.__class__.__name__
                }
            )
            raise

# Wrap an existing guardrail
monitored_guardrail = MonitoredGuardrail(
    budget_guardrail,
    violation_reporter,
    "agent123"
)
```

## Best Practices

### Design Principles

1. **Be Specific**: Define clear, narrow constraints for each guardrail
2. **Defense in Depth**: Use multiple complementary guardrails
3. **Fail Safely**: Ensure guardrails default to conservative behavior
4. **Provide Context**: Include helpful error messages and recovery suggestions
5. **Monitor Interventions**: Track when guardrails prevent actions

### Implementation Tips

1. **Start Simple**: Begin with basic guardrails before adding complexity
2. **Test Thoroughly**: Verify guardrails block problematic scenarios
3. **Minimize Performance Impact**: Optimize validation for fast execution
4. **Include Emergencies**: Create methods to override guardrails in emergencies
5. **Document Expectations**: Make guardrail behaviors clear to users

## Conclusion

Safety guardrails are a powerful way to ensure agent systems operate within defined boundaries. By combining different types of guardrails, you can create comprehensive safety systems that prevent misuse, protect resources, and ensure security.

For more information, see:
- [Budget Management](budget_management.md) for budget constraint details
- [Security Guide](safeguards.md) for security policy information
- [Monitoring Guide](monitoring.md) for tracking guardrail effectiveness
