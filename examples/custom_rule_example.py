"""Example of implementing a custom security rule."""

from typing import cast

from safeguards.base.guardrails import GuardrailViolation, ValidationResult
from safeguards.rules.base import RuleChain, RuleContext, RulePriority, SafetyRule


class DataPrivacyRule(SafetyRule):
    """Custom rule to enforce data privacy requirements."""

    def __init__(
        self,
        allowed_data_types: list[str],
        pii_fields: list[str],
        required_encryption: bool = True,
        dependencies: list[type[SafetyRule]] | None = None,
    ):
        """Initialize data privacy rule.

        Args:
            allowed_data_types: List of allowed data type identifiers
            pii_fields: List of field names containing personal information
            required_encryption: Whether encryption is required for data
            dependencies: Other rules that must run before this one
        """
        super().__init__(
            rule_id="data_privacy",
            priority=RulePriority.CRITICAL,
            description="Enforces data privacy requirements",
            dependencies=dependencies,
        )
        self.allowed_data_types = set(allowed_data_types)
        self.pii_fields = set(pii_fields)
        self.required_encryption = required_encryption

    def evaluate(self, context: RuleContext) -> ValidationResult:
        """Evaluate data privacy requirements.

        Args:
            context: Rule evaluation context containing:
                    - input_data: Dict with 'data_type', 'fields', 'is_encrypted'
                    - metadata: Optional additional context

        Returns:
            Validation result
        """
        input_data = cast(dict[str, any], context.input_data)
        violations: list[GuardrailViolation] = []

        # Check data type
        data_type = input_data.get("data_type")
        if not data_type:
            violations.append(
                GuardrailViolation(
                    rule_id=self.rule_id,
                    message="Data type not specified",
                ),
            )
        elif data_type not in self.allowed_data_types:
            violations.append(
                GuardrailViolation(
                    rule_id=self.rule_id,
                    message=f"Invalid data type: {data_type}. Allowed: {', '.join(sorted(self.allowed_data_types))}",
                ),
            )

        # Check for PII fields
        fields = set(input_data.get("fields", []))
        pii_fields_present = fields.intersection(self.pii_fields)

        # Check encryption if required
        if pii_fields_present and self.required_encryption:
            is_encrypted = input_data.get("is_encrypted", False)
            if not is_encrypted:
                violations.append(
                    GuardrailViolation(
                        rule_id=self.rule_id,
                        message=f"Encryption required for PII fields: {', '.join(sorted(pii_fields_present))}",
                    ),
                )

        # Add privacy context to metadata
        context.metadata["privacy_context"] = {
            "data_type": data_type,
            "pii_fields_present": sorted(pii_fields_present),
            "encryption_required": self.required_encryption,
            "is_encrypted": input_data.get("is_encrypted", False),
        }

        return ValidationResult(is_valid=len(violations) == 0, violations=violations)


def example_usage():
    """Example of using the custom data privacy rule."""
    # Create privacy rule
    privacy_rule = DataPrivacyRule(
        allowed_data_types=["user_data", "analytics", "logs"],
        pii_fields=["email", "phone", "address", "ssn"],
        required_encryption=True,
    )

    # Create rule chain
    chain = RuleChain()
    chain.add_rule(privacy_rule)

    # Example 1: Valid non-PII data
    result = chain.evaluate(
        input_data={
            "data_type": "analytics",
            "fields": ["page_views", "session_duration"],
            "is_encrypted": False,
        },
    )
    print("Example 1 (Valid non-PII):", result.is_valid)
    if not result.is_valid:
        print("Violations:", [v.message for v in result.violations])

    # Example 2: Unencrypted PII data (should fail)
    result = chain.evaluate(
        input_data={
            "data_type": "user_data",
            "fields": ["username", "email", "phone"],
            "is_encrypted": False,
        },
    )
    print("\nExample 2 (Unencrypted PII):", result.is_valid)
    if not result.is_valid:
        print("Violations:", [v.message for v in result.violations])

    # Example 3: Encrypted PII data (should pass)
    result = chain.evaluate(
        input_data={
            "data_type": "user_data",
            "fields": ["username", "email", "phone"],
            "is_encrypted": True,
        },
    )
    print("\nExample 3 (Encrypted PII):", result.is_valid)
    if not result.is_valid:
        print("Violations:", [v.message for v in result.violations])

    # Example 4: Invalid data type
    result = chain.evaluate(
        input_data={
            "data_type": "financial",
            "fields": ["balance"],
            "is_encrypted": True,
        },
    )
    print("\nExample 4 (Invalid Type):", result.is_valid)
    if not result.is_valid:
        print("Violations:", [v.message for v in result.violations])


if __name__ == "__main__":
    example_usage()
