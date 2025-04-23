"""Exception handling for the Agent Safety Framework."""

from dataclasses import dataclass
from datetime import datetime
from typing import Any


@dataclass
class ErrorContext:
    """Context information for errors."""

    agent_id: str | None = None
    pool_id: str | None = None
    resource_type: str | None = None
    threshold: float | None = None
    current_value: float | None = None
    details: dict[str, Any] | None = None


class AgentSafetyError(Exception):
    """Base exception for all agent safety errors."""

    def __init__(
        self,
        message: str,
        agent_id: str | None = None,
        timestamp: datetime | None = None,
        metadata: dict[str, Any] | None = None,
        error_code: str | None = None,
        **kwargs,
    ):
        """Initialize the exception.

        Args:
            message: The error message.
            agent_id: The ID of the agent that caused the error.
            timestamp: The time the error occurred.
            metadata: Additional metadata about the error.
            error_code: Error code for backward compatibility.
            **kwargs: Additional keyword arguments.
        """
        self.message = message
        self.agent_id = agent_id
        self.timestamp = timestamp or datetime.now()
        self.metadata = metadata or {}
        self.error_code = error_code
        super().__init__(message)

    def to_dict(self) -> dict[str, Any]:
        """Convert error to dictionary format."""
        return {
            "error": {
                "code": self.error_code,
                "message": self.message,
                "context": {
                    "agent_id": self.agent_id,
                    "timestamp": self.timestamp.isoformat(),
                    "metadata": self.metadata,
                },
            },
        }


class BudgetError(AgentSafetyError):
    """Base class for budget-related errors."""

    pass


class ResourceError(AgentSafetyError):
    """Base class for resource-related errors."""

    pass


class ConfigurationError(AgentSafetyError):
    """Base class for configuration-related errors."""

    pass


class AuthenticationError(AgentSafetyError):
    """Base class for authentication-related errors."""

    pass


# Budget Errors
class BudgetExceededError(BudgetError):
    """Raised when an agent exceeds its budget allocation."""

    def __init__(
        self,
        message: str = "Budget limit exceeded",
        context: ErrorContext | None = None,
    ):
        super().__init__(
            message=message,
            error_code="BUDGET_EXCEEDED",
            context=context,
            http_status=429,
        )


class InsufficientBudgetError(BudgetError):
    """Raised when there is insufficient budget for an operation."""

    def __init__(
        self,
        message: str = "Insufficient budget",
        context: ErrorContext | None = None,
    ):
        super().__init__(
            message=message,
            error_code="INSUFFICIENT_BUDGET",
            context=context,
            http_status=402,
        )


class BudgetPoolNotFoundError(BudgetError):
    """Raised when a budget pool cannot be found."""

    def __init__(
        self,
        message: str = "Budget pool not found",
        context: ErrorContext | None = None,
    ):
        super().__init__(
            message=message,
            error_code="POOL_NOT_FOUND",
            context=context,
            http_status=404,
        )


# Resource Errors
class ResourceLimitError(ResourceError):
    """Raised when a resource limit is exceeded."""

    def __init__(
        self,
        message: str = "Resource limit exceeded",
        context: ErrorContext | None = None,
    ):
        super().__init__(
            message=message,
            error_code="RESOURCE_LIMIT_EXCEEDED",
            context=context,
            http_status=429,
        )


class ResourceNotAvailableError(ResourceError):
    """Raised when a required resource is not available."""

    def __init__(
        self,
        message: str = "Resource not available",
        context: ErrorContext | None = None,
    ):
        super().__init__(
            message=message,
            error_code="RESOURCE_NOT_AVAILABLE",
            context=context,
            http_status=503,
        )


# Configuration Errors
class InvalidConfigurationError(ConfigurationError):
    """Raised when configuration is invalid."""

    def __init__(
        self,
        message: str = "Invalid configuration",
        context: ErrorContext | None = None,
    ):
        super().__init__(
            message=message,
            error_code="INVALID_CONFIG",
            context=context,
            http_status=400,
        )


class MissingConfigurationError(ConfigurationError):
    """Raised when required configuration is missing."""

    def __init__(
        self,
        message: str = "Missing configuration",
        context: ErrorContext | None = None,
    ):
        super().__init__(
            message=message,
            error_code="MISSING_CONFIG",
            context=context,
            http_status=400,
        )


# Authentication Errors
class InvalidAPIKeyError(AuthenticationError):
    """Raised when API key is invalid."""

    def __init__(
        self,
        message: str = "Invalid API key",
        context: ErrorContext | None = None,
    ):
        super().__init__(
            message=message,
            error_code="INVALID_API_KEY",
            context=context,
            http_status=401,
        )


class UnauthorizedError(AuthenticationError):
    """Raised when operation is unauthorized."""

    def __init__(
        self,
        message: str = "Unauthorized operation",
        context: ErrorContext | None = None,
    ):
        super().__init__(
            message=message,
            error_code="UNAUTHORIZED",
            context=context,
            http_status=403,
        )


def handle_error(error: Exception) -> dict[str, Any]:
    """Convert any error to a standardized format."""
    if isinstance(error, AgentSafetyError):
        return error.to_dict()

    # Handle unexpected errors
    return {
        "error": {
            "code": "INTERNAL_ERROR",
            "message": str(error),
            "context": {"type": type(error).__name__},
        },
    }
