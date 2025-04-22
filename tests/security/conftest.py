"""Configuration and fixtures for security tests."""

import pytest
from typing import Dict, Set

from agent_safety.rules.base import RuleContext, RulePriority
from agent_safety.rules.defaults import (
    PermissionGuardrail,
    SecurityContextRule,
    RateLimitRule,
)


@pytest.fixture
def test_roles() -> Dict[str, Set[str]]:
    """Common role definitions for testing."""
    return {
        "admin": {
            "read",
            "write",
            "delete",
            "manage_users",
            "manage_roles",
            "manage_system",
            "view_logs",
        },
        "power_user": {
            "read",
            "write",
            "delete",
            "view_logs",
        },
        "editor": {
            "read",
            "write",
        },
        "viewer": {
            "read",
        },
    }


@pytest.fixture
def test_environments() -> Set[str]:
    """Common environment definitions for testing."""
    return {"prod", "staging", "dev", "test"}


@pytest.fixture
def security_rule_chain(test_roles, test_environments):
    """Create a chain of security rules for testing."""
    from agent_safety.rules.base import RuleChain

    chain = RuleChain()

    # Add permission check
    chain.add_rule(
        PermissionGuardrail(
            required_permissions={"read", "write"},
            role_permissions=test_roles,
        )
    )

    # Add security context validation
    chain.add_rule(
        SecurityContextRule(
            required_security_level="medium",
            allowed_environments=test_environments,
        )
    )

    # Add rate limiting
    chain.add_rule(
        RateLimitRule(
            max_requests=100,
            time_window_seconds=60,
        )
    )

    return chain


@pytest.fixture
def mock_context_factory():
    """Factory for creating test contexts with different security settings."""

    def create_context(
        roles: list[str] = None,
        security_level: str = None,
        environment: str = None,
        request_count: int = None,
        operation: str = None,
        metadata: dict = None,
    ) -> RuleContext:
        """Create a test context with the specified security settings.

        Args:
            roles: List of user roles
            security_level: Security level (low, medium, high)
            environment: Execution environment
            request_count: Number of requests for rate limiting
            operation: Operation being performed
            metadata: Additional metadata

        Returns:
            Configured RuleContext instance
        """
        input_data = {}
        if roles is not None:
            input_data["user_roles"] = roles
        if security_level is not None:
            input_data["security_level"] = security_level
        if environment is not None:
            input_data["environment"] = environment
        if request_count is not None:
            input_data["request_count"] = request_count
        if operation is not None:
            input_data["operation"] = operation

        return RuleContext(
            input_data=input_data,
            metadata=metadata or {},
        )

    return create_context
