"""Tests for permission-based access control guardrails."""

import pytest

from safeguards.rules.base import RuleContext
from safeguards.rules.defaults import PermissionGuardrail


@pytest.fixture()
def role_permissions():
    """Test role-permission mappings."""
    return {
        "admin": {"read", "write", "delete", "manage_users"},
        "editor": {"read", "write"},
        "viewer": {"read"},
    }


@pytest.fixture()
def permission_guardrail(role_permissions):
    """Test permission guardrail instance."""
    return PermissionGuardrail(
        required_permissions={"read", "write"},
        role_permissions=role_permissions,
    )


def test_permission_check_success(permission_guardrail):
    """Test successful permission validation."""
    context = RuleContext(
        input_data={
            "user_roles": ["editor"],
            "operation": "edit_document",
        },
    )

    result = permission_guardrail.evaluate(context)
    assert result.is_valid
    assert not result.violations


def test_permission_check_admin_success(permission_guardrail):
    """Test admin role has all permissions."""
    context = RuleContext(
        input_data={
            "user_roles": ["admin"],
            "operation": "edit_document",
        },
    )

    result = permission_guardrail.evaluate(context)
    assert result.is_valid
    assert not result.violations


def test_permission_check_insufficient_permissions(permission_guardrail):
    """Test violation for insufficient permissions."""
    context = RuleContext(
        input_data={
            "user_roles": ["viewer"],
            "operation": "edit_document",
        },
    )

    result = permission_guardrail.evaluate(context)
    assert not result.is_valid
    assert len(result.violations) == 1
    assert "write" in result.violations[0].message


def test_permission_check_invalid_role(permission_guardrail):
    """Test violation for invalid role."""
    context = RuleContext(
        input_data={
            "user_roles": ["invalid_role"],
            "operation": "edit_document",
        },
    )

    result = permission_guardrail.evaluate(context)
    assert not result.is_valid
    assert len(result.violations) == 2  # Missing permissions and invalid role
    assert "Invalid roles" in result.violations[1].message


def test_permission_check_multiple_roles(permission_guardrail):
    """Test combining permissions from multiple roles."""
    context = RuleContext(
        input_data={
            "user_roles": ["viewer", "editor"],
            "operation": "edit_document",
        },
    )

    result = permission_guardrail.evaluate(context)
    assert result.is_valid
    assert not result.violations


def test_permission_check_metadata(permission_guardrail):
    """Test operation context in metadata."""
    context = RuleContext(
        input_data={
            "user_roles": ["editor"],
            "operation": "edit_document",
        },
    )

    permission_guardrail.evaluate(context)
    assert "operation_permissions" in context.metadata
    metadata = context.metadata["operation_permissions"]
    assert set(metadata["required"]) == {"read", "write"}
    assert set(metadata["granted"]) == {"read", "write"}
    assert metadata["roles"] == ["editor"]
