"""Security module for agent safety framework."""

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class Permission(str, Enum):
    """Permission levels for agent operations."""

    READ = "read"
    WRITE = "write"
    EXECUTE = "execute"
    ADMIN = "admin"


class Role(BaseModel):
    """Role definition with associated permissions."""

    name: str
    permissions: list[Permission]
    description: str | None = None


class SecurityContext(BaseModel):
    """Security context for agent operations."""

    agent_id: str
    roles: list[Role]
    metadata: dict[str, Any] = Field(default_factory=dict)


class SecurityValidator:
    """Validates agent operations against security policies."""

    def validate_input(self, input_data: Any, context: SecurityContext) -> bool:
        """Validate input data against security policies.

        Args:
            input_data: Data to validate
            context: Security context for validation

        Returns:
            bool: True if validation passes, False otherwise
        """
        # Implement input validation logic
        # For now, basic validation - ensure input exists and context is valid
        if input_data is None:
            return False
        return self._validate_context(context)

    def validate_output(self, output_data: Any, context: SecurityContext) -> bool:
        """Validate output data against security policies.

        Args:
            output_data: Data to validate
            context: Security context for validation

        Returns:
            bool: True if validation passes, False otherwise
        """
        # Implement output validation logic
        # For now, basic validation - ensure output exists and context is valid
        if output_data is None:
            return False
        return self._validate_context(context)

    def validate_resource_access(
        self,
        resource: str,
        permission: Permission,
        context: SecurityContext,
    ) -> bool:
        """Validate resource access against security policies.

        Args:
            resource: Resource identifier
            permission: Required permission
            context: Security context for validation

        Returns:
            bool: True if access is allowed, False otherwise
        """
        # Check if context has required permission through any of its roles
        return any(permission in role.permissions for role in context.roles)

    def _validate_context(self, context: SecurityContext) -> bool:
        """Validate security context.

        Args:
            context: Security context to validate

        Returns:
            bool: True if context is valid, False otherwise
        """
        return bool(context.agent_id and context.roles)
