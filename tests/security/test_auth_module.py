"""Tests for the authentication module."""

import pytest

from safeguards.base.auth import AuthConfig, AuthManager
from safeguards.base.security import Permission, Role


def test_auth_config():
    """Test AuthConfig model creation."""
    config = AuthConfig(api_key="test_key", token_expiry_seconds=7200)
    assert config.api_key == "test_key"
    assert config.token_expiry_seconds == 7200


def test_auth_manager_role_management():
    """Test role management in AuthManager."""
    manager = AuthManager(AuthConfig())

    # Test role registration
    role = Role(name="test_role", permissions=[Permission.READ, Permission.WRITE])
    manager.register_role(role)

    # Test role assignment
    manager.assign_role("test_agent", "test_role")

    # Test getting security context
    context = manager.get_security_context("test_agent")
    assert context.agent_id == "test_agent"
    assert len(context.roles) == 1
    assert context.roles[0].name == "test_role"
    assert len(context.roles[0].permissions) == 2


def test_auth_manager_permission_check():
    """Test permission checking in AuthManager."""
    manager = AuthManager(AuthConfig())

    # Register role with specific permissions
    role = Role(name="test_role", permissions=[Permission.READ])
    manager.register_role(role)
    manager.assign_role("test_agent", "test_role")

    # Test permission checks
    assert manager.has_permission("test_agent", Permission.READ) is True
    assert manager.has_permission("test_agent", Permission.WRITE) is False


def test_auth_manager_multiple_roles():
    """Test handling multiple roles in AuthManager."""
    manager = AuthManager(AuthConfig())

    # Register multiple roles
    role1 = Role(name="reader", permissions=[Permission.READ])
    role2 = Role(name="writer", permissions=[Permission.WRITE])

    manager.register_role(role1)
    manager.register_role(role2)

    # Assign multiple roles to agent
    manager.assign_role("test_agent", "reader")
    manager.assign_role("test_agent", "writer")

    # Check combined permissions
    context = manager.get_security_context("test_agent")
    assert len(context.roles) == 2
    assert manager.has_permission("test_agent", Permission.READ) is True
    assert manager.has_permission("test_agent", Permission.WRITE) is True
    assert manager.has_permission("test_agent", Permission.ADMIN) is False


def test_auth_manager_invalid_role():
    """Test handling invalid role assignments."""
    manager = AuthManager(AuthConfig())

    with pytest.raises(KeyError):
        manager.assign_role("test_agent", "nonexistent_role")
