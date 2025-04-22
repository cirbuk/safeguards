"""Tests for the token tracking module."""

from datetime import datetime, timedelta
from decimal import Decimal
import pytest

from agent_safety.base.budget import BudgetPeriod
from agent_safety.budget.token_tracker import TokenTracker, TokenUsage


@pytest.fixture
def token_tracker():
    """Create a token tracker instance for testing."""
    model_costs = {
        "gpt-4": {
            "input": Decimal("0.03"),  # $0.03 per 1K input tokens
            "output": Decimal("0.06"),  # $0.06 per 1K output tokens
        },
        "gpt-3.5": {
            "input": Decimal("0.002"),  # $0.002 per 1K input tokens
            "output": Decimal("0.004"),  # $0.004 per 1K output tokens
        },
    }
    return TokenTracker(
        model_costs=model_costs,
        token_budget=1000000,  # 1M tokens
        cost_budget=Decimal("100.0"),  # $100
        period=BudgetPeriod.DAILY,
    )


def test_record_usage(token_tracker):
    """Test recording token usage."""
    usage = token_tracker.record_usage(
        model_id="gpt-4",
        input_tokens=1000,
        output_tokens=200,
    )

    assert isinstance(usage, TokenUsage)
    assert usage.model_id == "gpt-4"
    assert usage.input_tokens == 1000
    assert usage.output_tokens == 200
    assert usage.cost > Decimal("0")

    # Test unknown model
    with pytest.raises(ValueError):
        token_tracker.record_usage(
            model_id="unknown-model",
            input_tokens=1000,
            output_tokens=200,
        )


def test_get_usage_in_period(token_tracker):
    """Test getting usage for a specific period."""
    # Record some usage
    now = datetime.now()
    yesterday = now - timedelta(days=1)
    tomorrow = now + timedelta(days=1)

    # Record usage for different times
    token_tracker.record_usage(
        model_id="gpt-4",
        input_tokens=1000,
        output_tokens=200,
        timestamp=yesterday,
    )
    token_tracker.record_usage(
        model_id="gpt-4",
        input_tokens=1000,
        output_tokens=200,
        timestamp=now,
    )
    token_tracker.record_usage(
        model_id="gpt-4",
        input_tokens=1000,
        output_tokens=200,
        timestamp=tomorrow,
    )

    # Get usage for today
    today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    today_end = today_start + timedelta(days=1)
    usage = token_tracker.get_usage_in_period(today_start, today_end)

    assert len(usage) == 1  # Should only include today's usage


def test_get_total_tokens(token_tracker):
    """Test getting total token counts."""
    # Record usage for different models
    token_tracker.record_usage(
        model_id="gpt-4",
        input_tokens=1000,
        output_tokens=200,
    )
    token_tracker.record_usage(
        model_id="gpt-3.5",
        input_tokens=2000,
        output_tokens=400,
    )

    # Test total tokens
    assert token_tracker.get_total_tokens() == 3600  # All tokens

    # Test model-specific tokens
    assert token_tracker.get_total_tokens("gpt-4") == 1200
    assert token_tracker.get_total_tokens("gpt-3.5") == 2400


def test_get_total_cost(token_tracker):
    """Test getting total costs."""
    # Record usage for different models
    token_tracker.record_usage(
        model_id="gpt-4",
        input_tokens=1000,
        output_tokens=200,
    )
    token_tracker.record_usage(
        model_id="gpt-3.5",
        input_tokens=2000,
        output_tokens=400,
    )

    # Test total cost
    total_cost = token_tracker.get_total_cost()
    assert total_cost > Decimal("0")

    # Test model-specific costs
    gpt4_cost = token_tracker.get_total_cost("gpt-4")
    gpt35_cost = token_tracker.get_total_cost("gpt-3.5")
    assert gpt4_cost > gpt35_cost  # GPT-4 should be more expensive


def test_check_budget_available(token_tracker):
    """Test budget availability checking."""
    # Test within token budget
    assert token_tracker.check_budget_available(
        input_tokens=1000,
        output_tokens=200,
        model_id="gpt-4",
    )

    # Test exceeding token budget
    assert not token_tracker.check_budget_available(
        input_tokens=2000000,  # 2M tokens
        output_tokens=400000,
        model_id="gpt-4",
    )

    # Test within cost budget
    assert token_tracker.check_budget_available(
        input_tokens=1000,
        output_tokens=200,
        model_id="gpt-4",
    )

    # Test exceeding cost budget
    assert not token_tracker.check_budget_available(
        input_tokens=1000000,  # Large number of tokens
        output_tokens=200000,
        model_id="gpt-4",
    )


def test_get_usage_stats(token_tracker):
    """Test getting usage statistics."""
    # Record usage for different models
    token_tracker.record_usage(
        model_id="gpt-4",
        input_tokens=1000,
        output_tokens=200,
    )
    token_tracker.record_usage(
        model_id="gpt-3.5",
        input_tokens=2000,
        output_tokens=400,
    )

    stats = token_tracker.get_usage_stats()

    # Check stats for each model
    assert "gpt-4" in stats
    assert "gpt-3.5" in stats
    assert stats["gpt-4"]["total_tokens"] == 1200
    assert stats["gpt-3.5"]["total_tokens"] == 2400
    assert stats["gpt-4"]["input_tokens"] == 1000
    assert stats["gpt-3.5"]["input_tokens"] == 2000


def test_reset_tracking(token_tracker):
    """Test resetting tracking data."""
    # Record some usage
    token_tracker.record_usage(
        model_id="gpt-4",
        input_tokens=1000,
        output_tokens=200,
    )

    # Verify usage is recorded
    assert token_tracker.get_total_tokens() > 0
    assert token_tracker.get_total_cost() > Decimal("0")

    # Reset tracking
    token_tracker.reset_tracking()

    # Verify everything is reset
    assert token_tracker.get_total_tokens() == 0
    assert token_tracker.get_total_cost() == Decimal("0")
    assert len(token_tracker.usage_history) == 0
