"""Tests for the budget manager module."""

from decimal import Decimal
import pytest
from unittest.mock import MagicMock

from safeguards.budget.manager import BudgetManager
from safeguards.types import BudgetConfig


@pytest.fixture
def budget_config() -> BudgetConfig:
    """Create a test budget configuration."""
    return BudgetConfig(
        total_budget=Decimal("1000"),
        hourly_limit=Decimal("100"),
        daily_limit=Decimal("500"),
        warning_threshold=75.0,
    )


@pytest.fixture
def budget_manager(budget_config: BudgetConfig) -> BudgetManager:
    """Create a budget manager instance for testing."""
    return BudgetManager(budget_config)


def test_budget_manager_initialization(budget_manager: BudgetManager):
    """Test budget manager initialization."""
    assert budget_manager.total_budget == Decimal("1000")
    assert budget_manager.hourly_limit == Decimal("100")
    assert budget_manager.daily_limit == Decimal("500")
    assert budget_manager.warning_threshold == 75.0


def test_has_sufficient_budget(budget_manager: BudgetManager):
    """Test checking if there is sufficient budget."""
    assert budget_manager.has_sufficient_budget(Decimal("50"))
    assert not budget_manager.has_sufficient_budget(Decimal("2000"))


def test_has_exceeded_budget(budget_manager: BudgetManager):
    """Test checking if budget has been exceeded."""
    budget_manager.record_cost(Decimal("600"))
    assert not budget_manager.has_exceeded_budget()

    budget_manager.record_cost(Decimal("500"))
    assert budget_manager.has_exceeded_budget()


def test_record_cost(budget_manager: BudgetManager):
    """Test recording costs."""
    budget_manager.record_cost(Decimal("50"))
    assert budget_manager.total_spent == Decimal("50")

    budget_manager.record_cost(Decimal("25"))
    assert budget_manager.total_spent == Decimal("75")


def test_get_remaining_budget(budget_manager: BudgetManager):
    """Test getting remaining budget."""
    budget_manager.record_cost(Decimal("300"))
    assert budget_manager.get_remaining_budget() == Decimal("700")


def test_get_budget_usage_percent(budget_manager: BudgetManager):
    """Test getting budget usage percentage."""
    budget_manager.record_cost(Decimal("250"))
    assert budget_manager.get_budget_usage_percent() == 25.0


def test_reset_budget(budget_manager: BudgetManager):
    """Test resetting the budget."""
    budget_manager.record_cost(Decimal("500"))
    budget_manager.reset_budget()
    assert budget_manager.total_spent == Decimal("0")
    assert budget_manager.get_remaining_budget() == budget_manager.total_budget
