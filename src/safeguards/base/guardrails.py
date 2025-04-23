"""Base interfaces and abstract classes for safety guardrails."""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Generic, TypeVar

T = TypeVar("T")  # Input type
U = TypeVar("U")  # Output type


class GuardrailViolation:
    """Represents a violation of a safety guardrail."""

    def __init__(
        self,
        guardrail_id: str | None = None,
        severity: str = "ERROR",
        message: str = "",
        context: dict[str, Any] | None = None,
        timestamp: datetime | None = None,
        rule_id: str | None = None,
        is_warning: bool = False,
    ):
        """Initialize guardrail violation.

        Args:
            guardrail_id: Unique identifier of the violated guardrail
            severity: Severity level of the violation
            message: Description of the violation
            context: Additional context about the violation
            timestamp: When the violation occurred
            rule_id: Alternative identifier for backward compatibility
            is_warning: Whether this is just a warning
        """
        # For backward compatibility with tests that use rule_id
        self.guardrail_id = guardrail_id if guardrail_id is not None else rule_id
        self.severity = "WARNING" if is_warning else severity
        self.message = message
        self.context = context or {}
        self.timestamp = timestamp or datetime.now()
        # For backward compatibility
        self.rule_id = self.guardrail_id


class ValidationResult:
    """Result of a guardrail validation."""

    def __init__(
        self,
        is_valid: bool,
        violations: list[GuardrailViolation] | None = None,
    ):
        """Initialize validation result.

        Args:
            is_valid: Whether the validation passed
            violations: List of violations if validation failed
        """
        self.is_valid = is_valid
        self.violations = violations or []


class Guardrail(Generic[T, U], ABC):
    """Abstract base class for safety guardrails."""

    def __init__(self, guardrail_id: str):
        """Initialize guardrail.

        Args:
            guardrail_id: Unique identifier for this guardrail
        """
        self.guardrail_id = guardrail_id

    @abstractmethod
    def validate_input(self, input_data: T) -> ValidationResult:
        """Validate input before processing.

        Args:
            input_data: Input data to validate

        Returns:
            Validation result
        """
        ...

    @abstractmethod
    def validate_output(self, output_data: U, input_data: T) -> ValidationResult:
        """Validate output after processing.

        Args:
            output_data: Output data to validate
            input_data: Original input data

        Returns:
            Validation result
        """
        ...

    def __call__(self, input_data: T) -> ValidationResult:
        """Convenience method to validate input.

        Args:
            input_data: Input data to validate

        Returns:
            Validation result
        """
        return self.validate_input(input_data)


class GuardrailRegistry:
    """Registry for managing and executing guardrails."""

    def __init__(self):
        """Initialize guardrail registry."""
        self._guardrails: dict[str, Guardrail] = {}

    def register(self, guardrail: Guardrail) -> None:
        """Register a guardrail.

        Args:
            guardrail: Guardrail to register
        """
        self._guardrails[guardrail.guardrail_id] = guardrail

    def unregister(self, guardrail_id: str) -> None:
        """Unregister a guardrail.

        Args:
            guardrail_id: ID of guardrail to unregister
        """
        self._guardrails.pop(guardrail_id, None)

    def validate_input(self, input_data: Any) -> ValidationResult:
        """Validate input against all registered guardrails.

        Args:
            input_data: Input data to validate

        Returns:
            Combined validation result
        """
        all_violations = []
        for guardrail in self._guardrails.values():
            result = guardrail.validate_input(input_data)
            if not result.is_valid:
                all_violations.extend(result.violations)

        return ValidationResult(
            is_valid=len(all_violations) == 0,
            violations=all_violations,
        )

    def validate_output(self, output_data: Any, input_data: Any) -> ValidationResult:
        """Validate output against all registered guardrails.

        Args:
            output_data: Output data to validate
            input_data: Original input data

        Returns:
            Combined validation result
        """
        all_violations = []
        for guardrail in self._guardrails.values():
            result = guardrail.validate_output(output_data, input_data)
            if not result.is_valid:
                all_violations.extend(result.violations)

        return ValidationResult(
            is_valid=len(all_violations) == 0,
            violations=all_violations,
        )
