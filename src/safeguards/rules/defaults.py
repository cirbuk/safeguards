"""Default safety rule implementations."""

from typing import Any, cast

from ..base.guardrails import GuardrailViolation, ValidationResult
from ..types.guardrail import ResourceUsage
from .base import RuleContext, RulePriority, SafetyRule


class ResourceLimitRule(SafetyRule):
    """Rule to enforce resource usage limits."""

    def __init__(
        self,
        max_memory_mb: float,
        max_cpu_percent: float,
        dependencies: list[type[SafetyRule]] | None = None,
    ):
        """Initialize resource limit rule.

        Args:
            max_memory_mb: Maximum memory usage in MB
            max_cpu_percent: Maximum CPU usage percentage
            dependencies: Other rules that must run before this one
        """
        super().__init__(
            rule_id="resource_limits",
            priority=RulePriority.CRITICAL,
            description="Enforces memory and CPU usage limits",
            dependencies=dependencies,
        )
        self.max_memory_mb = max_memory_mb
        self.max_cpu_percent = max_cpu_percent

    def evaluate(self, context: RuleContext) -> ValidationResult:
        """Evaluate resource usage against limits.

        Args:
            context: Rule evaluation context

        Returns:
            Validation result
        """
        resource_usage = cast(ResourceUsage, context.input_data)
        violations: list[GuardrailViolation] = []

        if resource_usage.memory_mb > self.max_memory_mb:
            violations.append(
                GuardrailViolation(
                    rule_id=self.rule_id,
                    message=f"Memory usage {resource_usage.memory_mb}MB exceeds limit of {self.max_memory_mb}MB",
                ),
            )

        if resource_usage.cpu_percent > self.max_cpu_percent:
            violations.append(
                GuardrailViolation(
                    rule_id=self.rule_id,
                    message=f"CPU usage {resource_usage.cpu_percent}% exceeds limit of {self.max_cpu_percent}%",
                ),
            )

        return ValidationResult(is_valid=len(violations) == 0, violations=violations)


class BudgetLimitRule(SafetyRule):
    """Rule to enforce budget usage limits."""

    def __init__(
        self,
        max_budget: float,
        warn_threshold: float = 0.8,
        dependencies: list[type[SafetyRule]] | None = None,
    ):
        """Initialize budget limit rule.

        Args:
            max_budget: Maximum budget allowed
            warn_threshold: Threshold for warning as fraction of max
            dependencies: Other rules that must run before this one
        """
        super().__init__(
            rule_id="budget_limits",
            priority=RulePriority.HIGH,
            description="Enforces budget usage limits",
            dependencies=dependencies,
        )
        self.max_budget = max_budget
        self.warn_threshold = warn_threshold

    def evaluate(self, context: RuleContext) -> ValidationResult:
        """Evaluate budget usage against limits.

        Args:
            context: Rule evaluation context

        Returns:
            Validation result
        """
        budget_usage = float(context.input_data)
        violations: list[GuardrailViolation] = []

        if budget_usage > self.max_budget:
            violations.append(
                GuardrailViolation(
                    rule_id=self.rule_id,
                    message=f"Budget usage {budget_usage} exceeds limit of {self.max_budget}",
                ),
            )
        elif budget_usage > (self.max_budget * self.warn_threshold):
            violations.append(
                GuardrailViolation(
                    rule_id=self.rule_id,
                    message=f"Budget usage {budget_usage} approaching limit of {self.max_budget}",
                    is_warning=True,
                ),
            )

        return ValidationResult(is_valid=len(violations) == 0, violations=violations)


class InputValidationRule(SafetyRule):
    """Rule to validate input data format and content."""

    def __init__(
        self,
        required_fields: list[str],
        field_types: dict[str, type],
        dependencies: list[type[SafetyRule]] | None = None,
    ):
        """Initialize input validation rule.

        Args:
            required_fields: List of required field names
            field_types: Mapping of field names to expected types
            dependencies: Other rules that must run before this one
        """
        super().__init__(
            rule_id="input_validation",
            priority=RulePriority.CRITICAL,
            description="Validates input data format and content",
            dependencies=dependencies,
        )
        self.required_fields = required_fields
        self.field_types = field_types

    def evaluate(self, context: RuleContext) -> ValidationResult:
        """Evaluate input data against validation rules.

        Args:
            context: Rule evaluation context

        Returns:
            Validation result
        """
        input_data = cast(dict[str, Any], context.input_data)
        violations: list[GuardrailViolation] = []

        # Check required fields
        for field in self.required_fields:
            if field not in input_data:
                violations.append(
                    GuardrailViolation(
                        rule_id=self.rule_id,
                        message=f"Missing required field: {field}",
                    ),
                )

        # Check field types
        for field, expected_type in self.field_types.items():
            if field in input_data and not isinstance(input_data[field], expected_type):
                violations.append(
                    GuardrailViolation(
                        rule_id=self.rule_id,
                        message=f"Invalid type for field {field}: expected {expected_type.__name__}, got {type(input_data[field]).__name__}",
                    ),
                )

        return ValidationResult(is_valid=len(violations) == 0, violations=violations)


class RateLimitRule(SafetyRule):
    """Rule to enforce rate limits on operations."""

    def __init__(
        self,
        max_requests: int,
        time_window_seconds: float,
        dependencies: list[type[SafetyRule]] | None = None,
    ):
        """Initialize rate limit rule.

        Args:
            max_requests: Maximum number of requests allowed
            time_window_seconds: Time window in seconds
            dependencies: Other rules that must run before this one
        """
        super().__init__(
            rule_id="rate_limits",
            priority=RulePriority.HIGH,
            description="Enforces operation rate limits",
            dependencies=dependencies,
        )
        self.max_requests = max_requests
        self.time_window_seconds = time_window_seconds

    def evaluate(self, context: RuleContext) -> ValidationResult:
        """Evaluate request rate against limits.

        Args:
            context: Rule evaluation context

        Returns:
            Validation result
        """
        request_count = cast(int, context.input_data)
        violations: list[GuardrailViolation] = []

        if request_count > self.max_requests:
            violations.append(
                GuardrailViolation(
                    rule_id=self.rule_id,
                    message=f"Request count {request_count} exceeds limit of {self.max_requests} per {self.time_window_seconds} seconds",
                ),
            )

        return ValidationResult(is_valid=len(violations) == 0, violations=violations)


class PermissionGuardrail(SafetyRule):
    """Rule to enforce permission-based access control."""

    def __init__(
        self,
        required_permissions: set[str],
        role_permissions: dict[str, set[str]],
        dependencies: list[type[SafetyRule]] | None = None,
    ):
        """Initialize permission guardrail.

        Args:
            required_permissions: Set of permissions required for the operation
            role_permissions: Mapping of roles to their granted permissions
            dependencies: Other rules that must run before this one
        """
        super().__init__(
            rule_id="permission_check",
            priority=RulePriority.CRITICAL,
            description="Enforces permission-based access control",
            dependencies=dependencies,
        )
        self.required_permissions = required_permissions
        self.role_permissions = role_permissions

    def evaluate(self, context: RuleContext) -> ValidationResult:
        """Evaluate permissions against user roles.

        Args:
            context: Rule evaluation context

        Returns:
            Validation result
        """
        input_data = cast(dict[str, Any], context.input_data)
        violations: list[GuardrailViolation] = []

        # Extract user roles
        user_roles = input_data.get("user_roles", [])
        operation = input_data.get("operation", "unknown")
        granted_permissions: set[str] = set()
        valid_roles: list[str] = []
        invalid_roles: list[str] = []

        # Collect permissions from all roles
        for role in user_roles:
            if role in self.role_permissions:
                valid_roles.append(role)
                granted_permissions.update(self.role_permissions[role])
            else:
                invalid_roles.append(role)

        # Check for missing permissions first (violation[0])
        missing_permissions = self.required_permissions - granted_permissions
        if missing_permissions:
            violations.append(
                GuardrailViolation(
                    guardrail_id=self.rule_id,
                    rule_id=self.rule_id,  # Use both for backward compatibility
                    message=f"Insufficient permissions for operation '{operation}'. Missing permissions: {', '.join(missing_permissions)}",
                ),
            )

        # Check for invalid roles second (violation[1])
        if invalid_roles:
            violations.append(
                GuardrailViolation(
                    guardrail_id=self.rule_id,
                    rule_id=self.rule_id,  # Use both for backward compatibility
                    message=f"Invalid roles: {', '.join(invalid_roles)}",
                ),
            )

        # Store permission details in context metadata
        context.metadata["operation_permissions"] = {
            "required": list(self.required_permissions),
            "granted": list(granted_permissions),
            "roles": valid_roles,
        }

        return ValidationResult(is_valid=len(violations) == 0, violations=violations)


class SecurityContextRule(SafetyRule):
    """Rule to validate security context and environment."""

    def __init__(
        self,
        required_security_level: str,
        allowed_environments: set[str],
        dependencies: list[type[SafetyRule]] | None = None,
    ):
        """Initialize security context rule.

        Args:
            required_security_level: Minimum security level required (low, medium, high)
            allowed_environments: Set of allowed execution environments
            dependencies: Other rules that must run before this one
        """
        super().__init__(
            rule_id="security_context",
            priority=RulePriority.CRITICAL,
            description="Validates security context and environment",
            dependencies=dependencies,
        )
        self.required_security_level = required_security_level.lower()
        self.allowed_environments = {env.lower() for env in allowed_environments}
        self._security_levels = {"low": 0, "medium": 1, "high": 2}

    def evaluate(self, context: RuleContext) -> ValidationResult:
        """Evaluate security context requirements.

        Args:
            context: Rule evaluation context containing:
                    - input_data: Dict with 'security_level' and 'environment'
                    - metadata: Optional additional context

        Returns:
            Validation result
        """
        input_data = cast(dict[str, Any], context.input_data)
        violations: list[GuardrailViolation] = []

        # Check security level
        current_level = input_data.get("security_level", "").lower()
        if not current_level:
            violations.append(
                GuardrailViolation(
                    rule_id=self.rule_id,
                    message="Security level not specified",
                ),
            )
        elif current_level not in self._security_levels:
            violations.append(
                GuardrailViolation(
                    rule_id=self.rule_id,
                    message=f"Invalid security level: {current_level}",
                ),
            )
        elif (
            self._security_levels[current_level]
            < self._security_levels[self.required_security_level]
        ):
            violations.append(
                GuardrailViolation(
                    rule_id=self.rule_id,
                    message=f"Insufficient security level: {current_level} (required: {self.required_security_level})",
                ),
            )

        # Check environment
        current_env = input_data.get("environment", "").lower()
        if not current_env:
            violations.append(
                GuardrailViolation(
                    rule_id=self.rule_id,
                    message="Execution environment not specified",
                ),
            )
        elif current_env not in self.allowed_environments:
            violations.append(
                GuardrailViolation(
                    rule_id=self.rule_id,
                    message=f"Invalid execution environment: {current_env}. Allowed: {', '.join(sorted(self.allowed_environments))}",
                ),
            )

        # Add security context to metadata
        context.metadata["security_context"] = {
            "current_level": current_level,
            "required_level": self.required_security_level,
            "environment": current_env,
            "allowed_environments": sorted(self.allowed_environments),
        }

        return ValidationResult(is_valid=len(violations) == 0, violations=violations)
