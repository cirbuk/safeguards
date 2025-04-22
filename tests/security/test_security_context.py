"""Tests for security context validation."""

import pytest

from safeguards.rules.defaults import SecurityContextRule
from safeguards.rules.base import RuleContext


@pytest.fixture
def security_context_rule():
    """Test security context rule instance."""
    return SecurityContextRule(
        required_security_level="medium",
        allowed_environments={"prod", "staging", "dev"},
    )


def test_security_level_sufficient(security_context_rule):
    """Test validation with sufficient security level."""
    context = RuleContext(
        input_data={
            "security_level": "high",
            "environment": "prod",
        }
    )

    result = security_context_rule.evaluate(context)
    assert result.is_valid
    assert not result.violations


def test_security_level_exact_match(security_context_rule):
    """Test validation with exact security level match."""
    context = RuleContext(
        input_data={
            "security_level": "medium",
            "environment": "prod",
        }
    )

    result = security_context_rule.evaluate(context)
    assert result.is_valid
    assert not result.violations


def test_security_level_insufficient(security_context_rule):
    """Test violation for insufficient security level."""
    context = RuleContext(
        input_data={
            "security_level": "low",
            "environment": "prod",
        }
    )

    result = security_context_rule.evaluate(context)
    assert not result.is_valid
    assert len(result.violations) == 1
    assert "Insufficient security level" in result.violations[0].message


def test_security_level_invalid(security_context_rule):
    """Test violation for invalid security level."""
    context = RuleContext(
        input_data={
            "security_level": "ultra",
            "environment": "prod",
        }
    )

    result = security_context_rule.evaluate(context)
    assert not result.is_valid
    assert len(result.violations) == 1
    assert "Invalid security level" in result.violations[0].message


def test_environment_valid(security_context_rule):
    """Test validation with valid environment."""
    for env in ["prod", "staging", "dev"]:
        context = RuleContext(
            input_data={
                "security_level": "high",
                "environment": env,
            }
        )

        result = security_context_rule.evaluate(context)
        assert result.is_valid
        assert not result.violations


def test_environment_invalid(security_context_rule):
    """Test violation for invalid environment."""
    context = RuleContext(
        input_data={
            "security_level": "high",
            "environment": "test",
        }
    )

    result = security_context_rule.evaluate(context)
    assert not result.is_valid
    assert len(result.violations) == 1
    assert "Invalid execution environment" in result.violations[0].message


def test_missing_security_level(security_context_rule):
    """Test violation for missing security level."""
    context = RuleContext(
        input_data={
            "environment": "prod",
        }
    )

    result = security_context_rule.evaluate(context)
    assert not result.is_valid
    assert len(result.violations) == 1
    assert "Security level not specified" in result.violations[0].message


def test_missing_environment(security_context_rule):
    """Test violation for missing environment."""
    context = RuleContext(
        input_data={
            "security_level": "high",
        }
    )

    result = security_context_rule.evaluate(context)
    assert not result.is_valid
    assert len(result.violations) == 1
    assert "Execution environment not specified" in result.violations[0].message


def test_security_context_metadata(security_context_rule):
    """Test security context in metadata."""
    context = RuleContext(
        input_data={
            "security_level": "high",
            "environment": "prod",
        }
    )

    security_context_rule.evaluate(context)
    assert "security_context" in context.metadata
    metadata = context.metadata["security_context"]
    assert metadata["current_level"] == "high"
    assert metadata["required_level"] == "medium"
    assert metadata["environment"] == "prod"
    assert set(metadata["allowed_environments"]) == {"prod", "staging", "dev"}
