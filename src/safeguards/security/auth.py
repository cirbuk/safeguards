"""Authentication and authorization system for agent safety.

This module provides:
- Role-based access control (RBAC)
- Permission management
- Audit logging
- Security context management
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from uuid import UUID, uuid4

from ..core.alert_types import Alert, AlertSeverity


class Permission(Enum):
    """System permissions."""

    # Budget permissions
    BUDGET_VIEW = auto()
    BUDGET_MODIFY = auto()
    BUDGET_TRANSFER = auto()

    # Agent permissions
    AGENT_CREATE = auto()
    AGENT_MODIFY = auto()
    AGENT_DELETE = auto()

    # Resource permissions
    RESOURCE_VIEW = auto()
    RESOURCE_MODIFY = auto()

    # System permissions
    SYSTEM_ADMIN = auto()
    SYSTEM_CONFIG = auto()


class Role(Enum):
    """System roles with predefined permission sets."""

    ADMIN = auto()  # Full system access
    OPERATOR = auto()  # Agent and resource management
    ANALYST = auto()  # Monitoring and analysis
    AGENT = auto()  # Basic agent operations


@dataclass
class AuditLog:
    """Audit log entry."""

    timestamp: datetime
    action: str
    actor_id: str
    target: str
    status: str
    details: dict
    log_id: UUID = field(default_factory=uuid4)


class SecurityManager:
    """Manages authentication, authorization and audit logging."""

    def __init__(self, notification_manager=None):
        """Initialize security manager.

        Args:
            notification_manager: For sending security-related alerts
        """
        self.notification_manager = notification_manager

        # Role -> Permissions mapping
        self._role_permissions: dict[Role, set[Permission]] = {
            Role.ADMIN: set(Permission),  # All permissions
            Role.OPERATOR: {
                Permission.AGENT_CREATE,
                Permission.AGENT_MODIFY,
                Permission.AGENT_DELETE,
                Permission.RESOURCE_VIEW,
                Permission.RESOURCE_MODIFY,
                Permission.BUDGET_VIEW,
                Permission.BUDGET_MODIFY,
            },
            Role.ANALYST: {
                Permission.RESOURCE_VIEW,
                Permission.BUDGET_VIEW,
            },
            Role.AGENT: {
                Permission.RESOURCE_VIEW,
                Permission.BUDGET_VIEW,
            },
        }

        # Identity -> Roles mapping
        self._identity_roles: dict[str, set[Role]] = {}

        # Audit log storage
        self._audit_logs: list[AuditLog] = []

    def register_identity(self, identity: str, roles: list[Role]) -> None:
        """Register an identity with roles.

        Args:
            identity: Identity to register
            roles: Roles to assign

        Raises:
            ValueError: If identity already registered
        """
        if identity in self._identity_roles:
            msg = f"Identity {identity} already registered"
            raise ValueError(msg)

        self._identity_roles[identity] = set(roles)
        self._log_audit(
            action="REGISTER_IDENTITY",
            actor_id="system",
            target=identity,
            status="SUCCESS",
            details={"roles": [r.name for r in roles]},
        )

    def has_permission(self, identity: str, permission: Permission) -> bool:
        """Check if identity has a specific permission.

        Args:
            identity: Identity to check
            permission: Permission to verify

        Returns:
            True if identity has permission
        """
        if identity not in self._identity_roles:
            return False

        roles = self._identity_roles[identity]
        return any(permission in self._role_permissions[role] for role in roles)

    def verify_permission(
        self,
        identity: str,
        permission: Permission,
        target: str | None = None,
    ) -> None:
        """Verify identity has permission or raise error.

        Args:
            identity: Identity to check
            permission: Permission to verify
            target: Optional target of the permission

        Raises:
            PermissionError: If identity lacks permission
        """
        if not self.has_permission(identity, permission):
            self._log_audit(
                action="PERMISSION_CHECK",
                actor_id=identity,
                target=str(permission),
                status="DENIED",
                details={"target": target} if target else {},
            )
            msg = f"Identity {identity} lacks permission {permission.name}"
            raise PermissionError(
                msg,
            )

        self._log_audit(
            action="PERMISSION_CHECK",
            actor_id=identity,
            target=str(permission),
            status="ALLOWED",
            details={"target": target} if target else {},
        )

    def get_identity_roles(self, identity: str) -> set[Role]:
        """Get roles assigned to an identity.

        Args:
            identity: Identity to check

        Returns:
            Set of assigned roles

        Raises:
            ValueError: If identity not found
        """
        if identity not in self._identity_roles:
            msg = f"Identity {identity} not found"
            raise ValueError(msg)

        return self._identity_roles[identity].copy()

    def add_identity_role(self, identity: str, role: Role) -> None:
        """Add a role to an identity.

        Args:
            identity: Identity to modify
            role: Role to add

        Raises:
            ValueError: If identity not found
        """
        if identity not in self._identity_roles:
            msg = f"Identity {identity} not found"
            raise ValueError(msg)

        self._identity_roles[identity].add(role)
        self._log_audit(
            action="ADD_ROLE",
            actor_id="system",
            target=identity,
            status="SUCCESS",
            details={"role": role.name},
        )

    def remove_identity_role(self, identity: str, role: Role) -> None:
        """Remove a role from an identity.

        Args:
            identity: Identity to modify
            role: Role to remove

        Raises:
            ValueError: If identity not found or would have no roles
        """
        if identity not in self._identity_roles:
            msg = f"Identity {identity} not found"
            raise ValueError(msg)

        roles = self._identity_roles[identity]
        if len(roles) == 1 and role in roles:
            msg = f"Cannot remove last role from {identity}"
            raise ValueError(msg)

        roles.discard(role)
        self._log_audit(
            action="REMOVE_ROLE",
            actor_id="system",
            target=identity,
            status="SUCCESS",
            details={"role": role.name},
        )

    def _log_audit(
        self,
        action: str,
        actor_id: str,
        target: str,
        status: str,
        details: dict,
    ) -> None:
        """Create an audit log entry.

        Args:
            action: Action being performed
            actor_id: Identity performing action
            target: Target of the action
            status: Outcome status
            details: Additional details
        """
        log = AuditLog(
            timestamp=datetime.now(),
            action=action,
            actor_id=actor_id,
            target=target,
            status=status,
            details=details,
        )
        self._audit_logs.append(log)

        # Send alert for security events
        if self.notification_manager:
            alert = Alert(
                title=f"Security Event: {action}",
                description=f"Actor: {actor_id}, Target: {target}, Status: {status}",
                severity=AlertSeverity.ERROR if status == "DENIED" else AlertSeverity.INFO,
                metadata={
                    "log_id": str(log.log_id),
                    "action": action,
                    "actor_id": actor_id,
                    "target": target,
                    "status": status,
                    **details,
                },
            )
            self.notification_manager.send_alert(alert)
