"""Unit tests for authentication and authorization system."""

import pytest
from datetime import datetime
from uuid import UUID

from agent_safety.security.auth import (
    Permission,
    Role,
    AuditLog,
    SecurityManager,
)


def test_permission_enum():
    """Test Permission enum values."""
    assert Permission.BUDGET_VIEW.name == "BUDGET_VIEW"
    assert Permission.AGENT_CREATE.name == "AGENT_CREATE"
    assert Permission.SYSTEM_ADMIN.name == "SYSTEM_ADMIN"


def test_role_enum():
    """Test Role enum values."""
    assert Role.ADMIN.name == "ADMIN"
    assert Role.OPERATOR.name == "OPERATOR"
    assert Role.ANALYST.name == "ANALYST"
    assert Role.AGENT.name == "AGENT"


def test_audit_log():
    """Test AuditLog creation and attributes."""
    log = AuditLog(
        timestamp=datetime.now(),
        action="TEST",
        actor_id="test_user",
        target="test_resource",
        status="SUCCESS",
        details={"key": "value"},
    )

    assert isinstance(log.log_id, UUID)
    assert log.action == "TEST"
    assert log.actor_id == "test_user"
    assert log.target == "test_resource"
    assert log.status == "SUCCESS"
    assert log.details == {"key": "value"}


class TestSecurityManager:
    """Test cases for SecurityManager."""

    @pytest.fixture
    def security_manager(self):
        """Create SecurityManager instance for testing."""
        return SecurityManager()

    def test_register_identity(self, security_manager):
        """Test identity registration."""
        # Register new identity
        security_manager.register_identity(
            "test_user",
            [Role.OPERATOR, Role.ANALYST],
        )

        # Verify roles
        roles = security_manager.get_identity_roles("test_user")
        assert len(roles) == 2
        assert Role.OPERATOR in roles
        assert Role.ANALYST in roles

        # Try registering duplicate
        with pytest.raises(ValueError):
            security_manager.register_identity("test_user", [Role.AGENT])

    def test_has_permission(self, security_manager):
        """Test permission checking."""
        # Register identity with specific roles
        security_manager.register_identity("test_user", [Role.OPERATOR])

        # Check permissions
        assert security_manager.has_permission(
            "test_user",
            Permission.AGENT_CREATE,
        )
        assert security_manager.has_permission(
            "test_user",
            Permission.RESOURCE_VIEW,
        )
        assert not security_manager.has_permission(
            "test_user",
            Permission.SYSTEM_ADMIN,
        )

        # Check unknown identity
        assert not security_manager.has_permission(
            "unknown_user",
            Permission.RESOURCE_VIEW,
        )

    def test_verify_permission(self, security_manager):
        """Test permission verification."""
        # Register identity
        security_manager.register_identity("test_user", [Role.OPERATOR])

        # Verify allowed permission
        security_manager.verify_permission(
            "test_user",
            Permission.AGENT_CREATE,
        )

        # Verify denied permission
        with pytest.raises(PermissionError):
            security_manager.verify_permission(
                "test_user",
                Permission.SYSTEM_ADMIN,
            )

    def test_add_remove_role(self, security_manager):
        """Test role management."""
        # Register identity
        security_manager.register_identity("test_user", [Role.OPERATOR])

        # Add role
        security_manager.add_identity_role("test_user", Role.ANALYST)
        roles = security_manager.get_identity_roles("test_user")
        assert Role.ANALYST in roles

        # Remove role
        security_manager.remove_identity_role("test_user", Role.ANALYST)
        roles = security_manager.get_identity_roles("test_user")
        assert Role.ANALYST not in roles

        # Try removing last role
        with pytest.raises(ValueError):
            security_manager.remove_identity_role("test_user", Role.OPERATOR)

    def test_audit_logging(self, security_manager):
        """Test audit log creation."""
        # Register identity
        security_manager.register_identity("test_user", [Role.OPERATOR])

        # Verify permission (generates audit log)
        security_manager.verify_permission(
            "test_user",
            Permission.AGENT_CREATE,
        )

        # Check audit log
        assert len(security_manager._audit_logs) > 0
        log = security_manager._audit_logs[-1]
        assert log.action == "PERMISSION_CHECK"
        assert log.actor_id == "test_user"
        assert log.status == "ALLOWED"
