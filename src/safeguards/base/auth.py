"""Authentication and authorization module for agent safety framework."""

from pydantic import BaseModel

from safeguards.base.security import Permission, Role, SecurityContext


class AuthConfig(BaseModel):
    """Authentication configuration."""

    api_key: str | None = None
    token_expiry_seconds: int = 3600


class AuthToken(BaseModel):
    """Authentication token."""

    token: str
    expires_at: float


class AuthManager:
    """Manages authentication and authorization for agents."""

    def __init__(self, config: AuthConfig):
        """Initialize auth manager.

        Args:
            config: Authentication configuration
        """
        self._config = config
        self._roles: dict[str, Role] = {}
        self._agent_roles: dict[str, list[str]] = {}

    def register_role(self, role: Role) -> None:
        """Register a new role.

        Args:
            role: Role to register
        """
        self._roles[role.name] = role

    def assign_role(self, agent_id: str, role_name: str) -> None:
        """Assign a role to an agent.

        Args:
            agent_id: Agent identifier
            role_name: Name of role to assign

        Raises:
            KeyError: If role doesn't exist
        """
        if role_name not in self._roles:
            msg = f"Role {role_name} does not exist"
            raise KeyError(msg)

        if agent_id not in self._agent_roles:
            self._agent_roles[agent_id] = []

        if role_name not in self._agent_roles[agent_id]:
            self._agent_roles[agent_id].append(role_name)

    def get_security_context(self, agent_id: str) -> SecurityContext:
        """Get security context for an agent.

        Args:
            agent_id: Agent identifier

        Returns:
            SecurityContext: Security context for the agent
        """
        roles = [self._roles[role_name] for role_name in self._agent_roles.get(agent_id, [])]
        return SecurityContext(agent_id=agent_id, roles=roles)

    def has_permission(self, agent_id: str, permission: Permission) -> bool:
        """Check if agent has a specific permission.

        Args:
            agent_id: Agent identifier
            permission: Permission to check

        Returns:
            bool: True if agent has permission, False otherwise
        """
        context = self.get_security_context(agent_id)
        return any(permission in role.permissions for role in context.roles)
