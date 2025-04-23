"""Tests for the security module."""

from safeguards.base.security import (
    Permission,
    Role,
    SecurityContext,
    SecurityValidator,
)


def test_permission_enum():
    """Test Permission enum values."""
    assert Permission.READ == "read"
    assert Permission.WRITE == "write"
    assert Permission.EXECUTE == "execute"
    assert Permission.ADMIN == "admin"


def test_role_creation():
    """Test Role model creation."""
    role = Role(
        name="test_role",
        permissions=[Permission.READ, Permission.WRITE],
        description="Test role",
    )
    assert role.name == "test_role"
    assert len(role.permissions) == 2
    assert Permission.READ in role.permissions
    assert Permission.WRITE in role.permissions
    assert role.description == "Test role"


def test_security_context_creation():
    """Test SecurityContext model creation."""
    role = Role(name="test_role", permissions=[Permission.READ])
    context = SecurityContext(
        agent_id="test_agent",
        roles=[role],
        metadata={"key": "value"},
    )
    assert context.agent_id == "test_agent"
    assert len(context.roles) == 1
    assert context.roles[0].name == "test_role"
    assert context.metadata["key"] == "value"


def test_security_validator():
    """Test SecurityValidator functionality."""
    validator = SecurityValidator()
    role = Role(name="test_role", permissions=[Permission.READ])
    context = SecurityContext(agent_id="test_agent", roles=[role])

    # Test input validation
    assert validator.validate_input("test", context) is True
    assert validator.validate_input(None, context) is False

    # Test output validation
    assert validator.validate_output("test", context) is True
    assert validator.validate_output(None, context) is False

    # Test resource access validation
    assert validator.validate_resource_access("test_resource", Permission.READ, context) is True
    assert validator.validate_resource_access("test_resource", Permission.WRITE, context) is False
