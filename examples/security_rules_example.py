"""Example usage of the Agent Safety security rules system."""

from safeguards.rules.base import RuleChain
from safeguards.rules.defaults import (
    PermissionGuardrail,
    RateLimitRule,
    ResourceLimitRule,
    SecurityContextRule,
)


def setup_security_rules():
    """Example of setting up a chain of security rules."""
    # Create a new rule chain
    chain = RuleChain()

    # Define role permissions
    role_permissions = {
        "admin": {"read", "write", "delete", "manage_users"},
        "editor": {"read", "write"},
        "viewer": {"read"},
    }

    # Add permission check
    chain.add_rule(
        PermissionGuardrail(
            required_permissions={"read", "write"},
            role_permissions=role_permissions,
        ),
    )

    # Add security context validation
    chain.add_rule(
        SecurityContextRule(
            required_security_level="medium",
            allowed_environments={"prod", "staging", "dev"},
        ),
    )

    # Add resource limits
    chain.add_rule(
        ResourceLimitRule(
            max_memory_mb=1024,
            max_cpu_percent=80,
        ),
    )

    # Add rate limiting
    chain.add_rule(
        RateLimitRule(
            max_requests=100,
            time_window_seconds=60,
        ),
    )

    return chain


def example_usage():
    """Example of using the security rules."""
    # Set up rules
    security_chain = setup_security_rules()

    # Example 1: Valid request
    result = security_chain.evaluate(
        input_data={
            "user_roles": ["editor"],
            "security_level": "high",
            "environment": "prod",
            "memory_mb": 512,
            "cpu_percent": 50,
            "request_count": 50,
        },
    )
    print("Example 1 (Valid):", result.is_valid)
    if not result.is_valid:
        print("Violations:", [v.message for v in result.violations])

    # Example 2: Invalid permissions
    result = security_chain.evaluate(
        input_data={
            "user_roles": ["viewer"],
            "security_level": "high",
            "environment": "prod",
            "memory_mb": 512,
            "cpu_percent": 50,
            "request_count": 50,
        },
    )
    print("\nExample 2 (Invalid Permissions):", result.is_valid)
    if not result.is_valid:
        print("Violations:", [v.message for v in result.violations])

    # Example 3: Resource limits exceeded
    result = security_chain.evaluate(
        input_data={
            "user_roles": ["admin"],
            "security_level": "high",
            "environment": "prod",
            "memory_mb": 2048,  # Exceeds limit
            "cpu_percent": 90,  # Exceeds limit
            "request_count": 50,
        },
    )
    print("\nExample 3 (Resource Limits):", result.is_valid)
    if not result.is_valid:
        print("Violations:", [v.message for v in result.violations])


if __name__ == "__main__":
    example_usage()
