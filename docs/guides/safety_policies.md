# Safeguards Policies Guide

This guide covers how to implement, enforce, and monitor safety policies for AI agents using the Safeguards.

## Introduction to Safety Policies

Safety policies are rules and constraints that govern agent behavior to prevent harmful, unethical, or unexpected actions. The Safeguards provides a structured approach to defining and enforcing these policies.

## Core Policy Components

The framework's policy system consists of:

- **Policy Rules**: Specific constraints that agents must follow
- **Policy Enforcer**: Component that checks agent actions against rules
- **Policy Violations**: Records of rule breaches
- **Remediation Actions**: Automated responses to violations

## Implementing Basic Safety Policies

### Creating a Policy Set

Start by defining a set of safety policies:

```python
from safeguards.policies import PolicySet, PolicyRule
from safeguards.types import (
    PolicySeverity,
    ViolationType,
    AlertSeverity,
    RemediationActionType
)

# Create a policy set
policy_set = PolicySet(
    name="basic_safety_policies",
    description="Core safety rules for all agents"
)

# Add resource usage policies
policy_set.add_rule(
    PolicyRule(
        name="max_budget_usage",
        description="Limit agent budget consumption",
        check_function=lambda agent_id, context:
            context.get("used_budget", 0) <= context.get("max_budget", 100),
        violation_type=ViolationType.BUDGET_LIMIT_EXCEEDED,
        severity=PolicySeverity.HIGH
    )
)

# Add rate limiting policy
policy_set.add_rule(
    PolicyRule(
        name="api_rate_limit",
        description="Limit API call frequency",
        check_function=lambda agent_id, context:
            context.get("api_calls_per_minute", 0) <= 60,
        violation_type=ViolationType.RATE_LIMIT_EXCEEDED,
        severity=PolicySeverity.MEDIUM
    )
)

# Add content policy
policy_set.add_rule(
    PolicyRule(
        name="content_safety",
        description="Prevent generation of harmful content",
        check_function=lambda agent_id, context:
            not contains_harmful_content(context.get("generated_content", "")),
        violation_type=ViolationType.HARMFUL_CONTENT,
        severity=PolicySeverity.CRITICAL
    )
)

# Helper function for content checking
def contains_harmful_content(content):
    """Check if content contains harmful material."""
    # Implement your content safety checks here
    # This is a placeholder implementation
    harmful_patterns = [
        "how to hack", "illegal activities", "violence against"
    ]
    return any(pattern in content.lower() for pattern in harmful_patterns)
```

### Setting Up Policy Enforcement

Configure the policy enforcer to apply your policies:

```python
from safeguards.policies import PolicyEnforcer
from safeguards.core import NotificationManager, ViolationReporter

# Create dependencies
notification_manager = NotificationManager()
violation_reporter = ViolationReporter(notification_manager)

# Create policy enforcer
policy_enforcer = PolicyEnforcer(
    policy_sets=[policy_set],
    violation_reporter=violation_reporter
)

# Check agent action against policies
action_context = {
    "agent_id": "agent123",
    "action_type": "generate_content",
    "generated_content": "Here's information about machine learning optimization.",
    "used_budget": 75,
    "max_budget": 100,
    "api_calls_per_minute": 30
}

# Enforce policies on the action
result = policy_enforcer.check_action(
    agent_id="agent123",
    action_context=action_context
)

if result.allowed:
    print("Action complies with all policies")
else:
    print(f"Action denied: {result.violation_details['message']}")
    print(f"Violated rule: {result.violation_details['rule_name']}")
```

## Building Comprehensive Policy Systems

Here's a complete example that sets up an advanced policy system:

```python
from safeguards.policies import (
    PolicySet,
    PolicyRule,
    PolicyEnforcer,
    RemediationAction
)
from safeguards.core import (
    NotificationManager,
    ViolationReporter,
    BudgetCoordinator
)
from safeguards.types import (
    PolicySeverity,
    ViolationType,
    AlertSeverity,
    RemediationActionType
)
from safeguards.monitoring import MetricsCollector

def setup_policy_system():
    """Set up a comprehensive policy system."""

    # Create core components
    notification_manager = NotificationManager()
    violation_reporter = ViolationReporter(notification_manager)
    budget_coordinator = BudgetCoordinator(notification_manager)
    metrics_collector = MetricsCollector()

    # Create resource usage policy set
    resource_policies = PolicySet(
        name="resource_policies",
        description="Policies governing resource usage"
    )

    resource_policies.add_rule(
        PolicyRule(
            name="budget_limit",
            description="Agents must stay within allocated budget",
            check_function=lambda agent_id, context:
                check_budget_compliance(agent_id, context, budget_coordinator),
            violation_type=ViolationType.BUDGET_LIMIT_EXCEEDED,
            severity=PolicySeverity.HIGH
        )
    )

    resource_policies.add_rule(
        PolicyRule(
            name="cpu_usage_limit",
            description="Agents must not exceed CPU allocation",
            check_function=lambda agent_id, context:
                context.get("cpu_usage_percent", 0) <= 80,
            violation_type=ViolationType.RESOURCE_LIMIT_EXCEEDED,
            severity=PolicySeverity.MEDIUM
        )
    )

    resource_policies.add_rule(
        PolicyRule(
            name="memory_usage_limit",
            description="Agents must not exceed memory allocation",
            check_function=lambda agent_id, context:
                context.get("memory_usage_mb", 0) <= 512,
            violation_type=ViolationType.RESOURCE_LIMIT_EXCEEDED,
            severity=PolicySeverity.MEDIUM
        )
    )

    # Create operational policy set
    operational_policies = PolicySet(
        name="operational_policies",
        description="Policies governing agent operations"
    )

    operational_policies.add_rule(
        PolicyRule(
            name="api_rate_limiting",
            description="Limit API call frequency",
            check_function=lambda agent_id, context:
                check_api_rate(agent_id, context, metrics_collector),
            violation_type=ViolationType.RATE_LIMIT_EXCEEDED,
            severity=PolicySeverity.MEDIUM
        )
    )

    operational_policies.add_rule(
        PolicyRule(
            name="authorized_actions_only",
            description="Agents can only perform authorized actions",
            check_function=lambda agent_id, context:
                is_action_authorized(agent_id, context),
            violation_type=ViolationType.UNAUTHORIZED_ACTION,
            severity=PolicySeverity.HIGH
        )
    )

    # Create content safety policy set
    content_policies = PolicySet(
        name="content_policies",
        description="Policies governing content generation"
    )

    content_policies.add_rule(
        PolicyRule(
            name="harmful_content_prevention",
            description="Prevent generation of harmful content",
            check_function=lambda agent_id, context:
                not contains_harmful_content(context.get("content", "")),
            violation_type=ViolationType.HARMFUL_CONTENT,
            severity=PolicySeverity.CRITICAL
        )
    )

    content_policies.add_rule(
        PolicyRule(
            name="personal_data_protection",
            description="Prevent exposure of personal data",
            check_function=lambda agent_id, context:
                not contains_personal_data(context.get("content", "")),
            violation_type=ViolationType.PRIVACY_VIOLATION,
            severity=PolicySeverity.CRITICAL
        )
    )

    # Set up remediation actions
    remediation_actions = [
        RemediationAction(
            violation_type=ViolationType.BUDGET_LIMIT_EXCEEDED,
            action_type=RemediationActionType.THROTTLE,
            parameters={"duration_seconds": 300}  # Throttle for 5 minutes
        ),
        RemediationAction(
            violation_type=ViolationType.HARMFUL_CONTENT,
            action_type=RemediationActionType.BLOCK,
            parameters={}  # Block the action completely
        ),
        RemediationAction(
            violation_type=ViolationType.RATE_LIMIT_EXCEEDED,
            action_type=RemediationActionType.DELAY,
            parameters={"delay_seconds": 10}  # Add delay to slow down
        ),
        RemediationAction(
            violation_type=ViolationType.UNAUTHORIZED_ACTION,
            action_type=RemediationActionType.BLOCK,
            parameters={}  # Block unauthorized actions
        ),
        RemediationAction(
            violation_type=ViolationType.PRIVACY_VIOLATION,
            action_type=RemediationActionType.BLOCK,
            parameters={}  # Block privacy violations
        )
    ]

    # Create the policy enforcer with all policy sets
    policy_enforcer = PolicyEnforcer(
        policy_sets=[resource_policies, operational_policies, content_policies],
        violation_reporter=violation_reporter,
        remediation_actions=remediation_actions
    )

    return {
        "policy_enforcer": policy_enforcer,
        "violation_reporter": violation_reporter,
        "notification_manager": notification_manager,
        "budget_coordinator": budget_coordinator,
        "metrics_collector": metrics_collector
    }

# Helper functions for policy checks
def check_budget_compliance(agent_id, context, budget_coordinator):
    """Check if the agent is within budget limits."""
    try:
        budget_info = budget_coordinator.get_agent_metrics(agent_id)
        remaining_percentage = (budget_info["remaining_budget"] /
                               budget_info["initial_budget"]) * 100
        return remaining_percentage >= 10  # At least 10% budget must remain
    except Exception:
        # If we can't get budget info, fail closed
        return False

def check_api_rate(agent_id, context, metrics_collector):
    """Check if the agent is respecting API rate limits."""
    try:
        # Get API calls in the last minute
        api_calls = metrics_collector.get_metric_value(
            agent_id=agent_id,
            metric_name="api_calls",
            time_window_seconds=60
        )
        return api_calls <= context.get("max_api_calls_per_minute", 60)
    except Exception:
        # If we can't get metrics, use context data if available
        return context.get("api_calls_per_minute", 0) <= context.get("max_api_calls_per_minute", 60)

def is_action_authorized(agent_id, context):
    """Check if the action is authorized for the agent."""
    action = context.get("action", "")
    authorized_actions = context.get("authorized_actions", [])
    return action in authorized_actions

def contains_personal_data(content):
    """Check if content contains personal data."""
    # Implement your personal data detection logic
    # This is a placeholder implementation
    import re
    # Check for patterns like email addresses, phone numbers, etc.
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    phone_pattern = r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'
    ssn_pattern = r'\b\d{3}-\d{2}-\d{4}\b'

    return (re.search(email_pattern, content) is not None or
            re.search(phone_pattern, content) is not None or
            re.search(ssn_pattern, content) is not None)

# Example usage
def example_policy_enforcement():
    """Demonstrate policy enforcement."""

    # Set up the policy system
    system = setup_policy_system()
    policy_enforcer = system["policy_enforcer"]

    # Example actions to check
    actions = [
        {
            "agent_id": "agent123",
            "action": "generate_text",
            "content": "Here's how to optimize your machine learning model.",
            "cpu_usage_percent": 45,
            "memory_usage_mb": 256,
            "authorized_actions": ["generate_text", "analyze_data"],
            "max_api_calls_per_minute": 100
        },
        {
            "agent_id": "agent123",
            "action": "access_database",
            "content": "Retrieving user records",
            "cpu_usage_percent": 30,
            "memory_usage_mb": 200,
            "authorized_actions": ["generate_text", "analyze_data"],
            "max_api_calls_per_minute": 100
        },
        {
            "agent_id": "agent123",
            "action": "generate_text",
            "content": "Here's how to hack into a password-protected system and steal data.",
            "cpu_usage_percent": 60,
            "memory_usage_mb": 300,
            "authorized_actions": ["generate_text", "analyze_data"],
            "max_api_calls_per_minute": 100
        }
    ]

    # Check each action against policies
    for i, action_context in enumerate(actions):
        print(f"\nChecking action {i+1}: {action_context['action']}")
        result = policy_enforcer.check_action(
            agent_id=action_context["agent_id"],
            action_context=action_context
        )

        if result.allowed:
            print("✅ Action allowed: Complies with all policies")
        else:
            print(f"❌ Action denied: {result.violation_details['message']}")
            print(f"   Violated rule: {result.violation_details['rule_name']}")
            print(f"   Violation type: {result.violation_details['violation_type']}")
            print(f"   Severity: {result.violation_details['severity']}")
            print(f"   Remediation: {result.remediation.action_type if result.remediation else 'None'}")
```

## Working with Action Contexts

The action context is a key concept in policy enforcement, containing all relevant information about an agent's action:

```python
# Basic action context for text generation
text_generation_context = {
    "action": "generate_text",
    "content": "Here's the information you requested about machine learning.",
    "model": "gpt-4",
    "prompt_tokens": 125,
    "completion_tokens": 350,
    "total_tokens": 475,
    "authorized_actions": ["generate_text", "analyze_data", "summarize"]
}

# Rich action context with additional metadata
rich_context = {
    "action": "data_analysis",
    "content": "Analysis complete. Found 3 anomalies in the dataset.",
    "data_sources": ["customer_database", "transaction_logs"],
    "analysis_type": "anomaly_detection",
    "cpu_usage_percent": 65,
    "memory_usage_mb": 420,
    "execution_time_seconds": 12.3,
    "api_calls": 8,
    "api_calls_per_minute": 40,
    "max_api_calls_per_minute": 100,
    "used_budget": 45.6,
    "max_budget": 100,
    "authorized_actions": ["analyze_data", "generate_report"],
    "user_id": "user_12345",
    "sensitivity_level": "confidential",
    "metadata": {
        "version": "1.0.3",
        "environment": "production",
        "request_id": "req_789012"
    }
}
```

## Policy Rule Templates

Here are templates for common safety policy rules:

### Resource Usage Policies

```python
# Maximum budget policy
budget_rule = PolicyRule(
    name="max_budget_limit",
    description="Limit total budget consumption",
    check_function=lambda agent_id, context:
        context.get("used_budget", 0) <= context.get("max_budget", 100),
    violation_type=ViolationType.BUDGET_LIMIT_EXCEEDED,
    severity=PolicySeverity.HIGH
)

# CPU limit policy
cpu_rule = PolicyRule(
    name="cpu_usage_limit",
    description="Limit CPU usage",
    check_function=lambda agent_id, context:
        context.get("cpu_usage_percent", 0) <= 80,
    violation_type=ViolationType.RESOURCE_LIMIT_EXCEEDED,
    severity=PolicySeverity.MEDIUM
)

# Memory limit policy
memory_rule = PolicyRule(
    name="memory_usage_limit",
    description="Limit memory usage",
    check_function=lambda agent_id, context:
        context.get("memory_usage_mb", 0) <= 512,
    violation_type=ViolationType.RESOURCE_LIMIT_EXCEEDED,
    severity=PolicySeverity.MEDIUM
)

# Token consumption policy
token_rule = PolicyRule(
    name="token_usage_limit",
    description="Limit token usage per request",
    check_function=lambda agent_id, context:
        context.get("total_tokens", 0) <= 4000,
    violation_type=ViolationType.RESOURCE_LIMIT_EXCEEDED,
    severity=PolicySeverity.LOW
)
```

### Access Control Policies

```python
# Authorized actions policy
authorized_actions_rule = PolicyRule(
    name="authorized_actions_only",
    description="Only permit explicitly authorized actions",
    check_function=lambda agent_id, context:
        context.get("action", "") in context.get("authorized_actions", []),
    violation_type=ViolationType.UNAUTHORIZED_ACTION,
    severity=PolicySeverity.HIGH
)

# Data access policy
data_access_rule = PolicyRule(
    name="data_access_control",
    description="Only access authorized data sources",
    check_function=lambda agent_id, context:
        all(source in context.get("authorized_data_sources", [])
            for source in context.get("data_sources", [])),
    violation_type=ViolationType.UNAUTHORIZED_ACCESS,
    severity=PolicySeverity.HIGH
)

# Authentication policy
authentication_rule = PolicyRule(
    name="valid_authentication",
    description="Ensure proper authentication for actions",
    check_function=lambda agent_id, context:
        context.get("authenticated", False) is True,
    violation_type=ViolationType.AUTHENTICATION_FAILURE,
    severity=PolicySeverity.CRITICAL
)
```

### Content Safety Policies

```python
# Harmful content prevention
harmful_content_rule = PolicyRule(
    name="harmful_content_prevention",
    description="Prevent generation of harmful content",
    check_function=lambda agent_id, context:
        not contains_harmful_content(context.get("content", "")),
    violation_type=ViolationType.HARMFUL_CONTENT,
    severity=PolicySeverity.CRITICAL
)

# Personal data protection
personal_data_rule = PolicyRule(
    name="personal_data_protection",
    description="Prevent exposure of personal data",
    check_function=lambda agent_id, context:
        not contains_personal_data(context.get("content", "")),
    violation_type=ViolationType.PRIVACY_VIOLATION,
    severity=PolicySeverity.CRITICAL
)

# Content moderation policy
moderation_rule = PolicyRule(
    name="content_moderation",
    description="Ensure content meets moderation standards",
    check_function=lambda agent_id, context:
        passes_moderation_check(context.get("content", "")),
    violation_type=ViolationType.CONTENT_POLICY_VIOLATION,
    severity=PolicySeverity.HIGH
)

def passes_moderation_check(content):
    """Check if content passes moderation standards."""
    # Implement your moderation logic or call external moderation API
    # This is a placeholder implementation
    return True  # Replace with actual implementation
```

## Implementing Custom Policy Rules

You can implement custom rules for domain-specific requirements:

```python
from safeguards.policies import PolicyRule
from safeguards.types import PolicySeverity, ViolationType

# Create a custom rule for domain-specific requirements
domain_specific_rule = PolicyRule(
    name="financial_advice_policy",
    description="Ensure financial advice includes proper disclaimers",
    check_function=lambda agent_id, context:
        check_financial_advice_compliance(context.get("content", "")),
    violation_type=ViolationType.COMPLIANCE_VIOLATION,
    severity=PolicySeverity.HIGH
)

def check_financial_advice_compliance(content):
    """Check if financial advice includes required disclaimers."""
    if contains_financial_advice(content):
        required_disclaimer = "This is not financial advice. Consult a professional advisor."
        return required_disclaimer.lower() in content.lower()
    return True  # No financial advice detected, so policy doesn't apply

def contains_financial_advice(content):
    """Detect if content contains financial advice."""
    financial_keywords = [
        "invest", "stock", "bond", "portfolio", "retirement",
        "savings", "fund", "return", "dividend", "market"
    ]
    advice_phrases = [
        "you should", "recommend", "consider", "best option",
        "good investment", "bad investment", "opportunity"
    ]

    # Check if content contains both financial keywords and advice phrases
    has_financial_terms = any(keyword in content.lower() for keyword in financial_keywords)
    has_advice = any(phrase in content.lower() for phrase in advice_phrases)

    return has_financial_terms and has_advice
```

## Advanced Policy Features

### Policy Hierarchies and Inheritance

You can organize policies into hierarchies:

```python
from safeguards.policies import PolicyHierarchy

# Create a policy hierarchy
policy_hierarchy = PolicyHierarchy(name="agent_policies")

# Add top-level policy sets
policy_hierarchy.add_policy_set(universal_policies, level=0)  # Apply to all agents

# Add second-level policy sets for different agent types
policy_hierarchy.add_policy_set(
    assistant_policies,
    level=1,
    filter_function=lambda agent_id, context: context.get("agent_type") == "assistant"
)

policy_hierarchy.add_policy_set(
    researcher_policies,
    level=1,
    filter_function=lambda agent_id, context: context.get("agent_type") == "researcher"
)

# Add third-level policies for specific domains
policy_hierarchy.add_policy_set(
    healthcare_policies,
    level=2,
    filter_function=lambda agent_id, context: context.get("domain") == "healthcare"
)

# Create a policy enforcer with the hierarchy
hierarchical_enforcer = PolicyEnforcer(
    policy_hierarchy=policy_hierarchy,
    violation_reporter=violation_reporter
)
```

### Policy Conflict Resolution

Handle conflicts between policies:

```python
from safeguards.policies import ConflictResolutionStrategy

# Create a policy enforcer with conflict resolution
policy_enforcer = PolicyEnforcer(
    policy_sets=[resource_policies, operational_policies, content_policies],
    violation_reporter=violation_reporter,
    conflict_resolution=ConflictResolutionStrategy.MOST_RESTRICTIVE
)

# Available conflict resolution strategies:
# - MOST_RESTRICTIVE: Apply the most restrictive policy (deny if any policy denies)
# - LEAST_RESTRICTIVE: Apply the least restrictive policy (allow if any policy allows)
# - PRIORITY_BASED: Use policy priority to resolve conflicts
# - SEVERITY_BASED: Use violation severity to resolve conflicts
```

### Dynamic Policy Updates

Update policies dynamically:

```python
# Update an existing policy rule
resource_policies.update_rule(
    "budget_limit",
    new_check_function=lambda agent_id, context:
        context.get("used_budget", 0) <= context.get("new_max_budget", 150)
)

# Add a new rule to an existing policy set
resource_policies.add_rule(
    PolicyRule(
        name="network_bandwidth_limit",
        description="Limit network bandwidth usage",
        check_function=lambda agent_id, context:
            context.get("bandwidth_usage_mbps", 0) <= 50,
        violation_type=ViolationType.RESOURCE_LIMIT_EXCEEDED,
        severity=PolicySeverity.MEDIUM
    )
)

# Remove a rule from a policy set
operational_policies.remove_rule("api_rate_limiting")

# Create completely new policy set and add to enforcer
new_policy_set = PolicySet(
    name="experimental_policies",
    description="Experimental policies under evaluation"
)

# Add the new policy set to the enforcer
policy_enforcer.add_policy_set(new_policy_set)
```

## Monitoring Policy Effectiveness

Track and analyze policy enforcement:

```python
from safeguards.monitoring import PolicyMetricsCollector
from safeguards.visualization import PolicyDashboard

# Create a policy metrics collector
policy_metrics = PolicyMetricsCollector()

# Register with the policy enforcer
policy_enforcer.set_metrics_collector(policy_metrics)

# Get policy enforcement statistics
stats = policy_metrics.get_enforcement_stats(
    timeframe="last_24h"
)

print(f"Total actions checked: {stats['total_actions']}")
print(f"Actions allowed: {stats['allowed_actions']} ({stats['allowed_percentage']}%)")
print(f"Actions denied: {stats['denied_actions']} ({stats['denied_percentage']}%)")

# Get violation statistics by policy
violation_stats = policy_metrics.get_violation_stats(
    group_by="policy_rule",
    timeframe="last_7d"
)

print("\nTop violated policies:")
for rule, count in sorted(violation_stats.items(), key=lambda x: x[1], reverse=True)[:5]:
    print(f"- {rule}: {count} violations")

# Create a policy dashboard
dashboard = PolicyDashboard(policy_metrics)

dashboard.add_panel(
    title="Policy Enforcement Overview",
    panel_type="pie_chart",
    data_function=lambda: {
        "Allowed": stats["allowed_actions"],
        "Denied": stats["denied_actions"]
    }
)

dashboard.add_panel(
    title="Violations by Policy",
    panel_type="bar_chart",
    data_function=lambda: violation_stats
)

dashboard.add_panel(
    title="Violations by Severity",
    panel_type="bar_chart",
    data_function=lambda: policy_metrics.get_violation_stats(
        group_by="severity",
        timeframe="last_30d"
    )
)

# Start the dashboard
dashboard_url = dashboard.start(host="0.0.0.0", port=8081)
print(f"Policy dashboard available at: {dashboard_url}")
```

## Best Practices

### Policy Design Principles

1. **Start with Broad Policies**: Begin with fundamental safety constraints that apply to all agents.

2. **Layered Approach**: Implement policies in layers from general to specific:
   - Universal safety policies
   - Domain-specific policies
   - Agent-specific policies

3. **Fail Closed**: Design policies to deny actions when information is incomplete or checks fail.

4. **Regular Updates**: Review and update policies based on new risks and observed behavior.

5. **Test Against Edge Cases**: Validate policies against edge cases and unexpected inputs.

### Policy Implementation Tips

1. **Efficient Check Functions**: Keep policy check functions lightweight and fast.

2. **Appropriate Severity Levels**: Assign severity levels based on potential harm:
   - CRITICAL: Could cause significant harm
   - HIGH: Could cause notable harm or service disruption
   - MEDIUM: Could cause limited harm or service degradation
   - LOW: Minor issues with minimal impact

3. **Clear Documentation**: Document the purpose and expected behavior of each policy.

4. **Logging and Monitoring**: Track policy enforcement to identify patterns and improve policies.

5. **Graceful Failure**: Ensure the policy system fails gracefully if components are unavailable.

## Conclusion

Implementing robust safety policies is essential for responsible AI agent deployment. The Safeguards provides a flexible and comprehensive system for defining, enforcing, and monitoring these policies.

By following the patterns shown in this guide, you can create safety policies that protect against harmful behavior while allowing agents to operate effectively within well-defined constraints.

For more information, see:
- [Monitoring Guide](monitoring.md)
- [Budget Management Guide](budget_management.md)
- [Notifications Guide](notifications.md)
- [API Reference](../api/policies.md)
