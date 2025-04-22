"""Pytest configuration and shared fixtures."""

from decimal import Decimal
import pytest

from safeguards.base.budget import BudgetPeriod
from safeguards.budget.api_tracker import APITracker
from safeguards.budget.token_tracker import TokenTracker


@pytest.fixture
def model_costs():
    """Common model costs for testing."""
    return {
        "gpt-4": {
            "input": Decimal("0.03"),  # $0.03 per 1K input tokens
            "output": Decimal("0.06"),  # $0.06 per 1K output tokens
        },
        "gpt-3.5": {
            "input": Decimal("0.002"),  # $0.002 per 1K input tokens
            "output": Decimal("0.004"),  # $0.004 per 1K output tokens
        },
    }


@pytest.fixture
def api_costs():
    """Common API costs for testing."""
    return {
        "search": Decimal("0.01"),  # $0.01 per call
        "embedding": Decimal("0.0004"),  # $0.0004 per call
        "image-gen": Decimal("0.02"),  # $0.02 per call
    }


@pytest.fixture
def token_tracker(model_costs):
    """Shared token tracker instance."""
    return TokenTracker(
        model_costs=model_costs,
        token_budget=1000000,  # 1M tokens
        cost_budget=Decimal("100.0"),  # $100
        period=BudgetPeriod.DAILY,
    )


@pytest.fixture
def api_tracker(api_costs):
    """Shared API tracker instance."""
    return APITracker(
        api_costs=api_costs,
        call_budget=10000,  # 10K calls
        cost_budget=Decimal("100.0"),  # $100
        period=BudgetPeriod.DAILY,
        data_transfer_cost_per_gb=Decimal("0.10"),  # $0.10 per GB
    )
