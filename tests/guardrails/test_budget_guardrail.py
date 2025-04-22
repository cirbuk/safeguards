"""Tests for budget guardrail."""

from decimal import Decimal
import pytest
from unittest.mock import MagicMock

from safeguards.budget.manager import BudgetManager, BudgetOverride
from safeguards.guardrails.budget import BudgetGuardrail
from safeguards.types import Agent, RunContext


class TestAgent(Agent):
    """Test agent implementation."""

    def run(self, **kwargs):
        """Mock implementation."""
        return {"status": "success"}


@pytest.fixture
def test_agent() -> Agent:
    """Create a test agent."""
    return TestAgent(name="test_agent")


@pytest.fixture
def budget_manager() -> BudgetManager:
    """Create a mock budget manager."""
    manager = MagicMock(spec=BudgetManager)
    manager.has_sufficient_budget.return_value = True
    manager.has_exceeded_budget.return_value = False
    manager.get_minimum_required.return_value = Decimal("10.0")
    return manager


@pytest.fixture
def budget_guardrail(budget_manager: BudgetManager) -> BudgetGuardrail:
    """Create a budget guardrail instance."""
    return BudgetGuardrail(budget_manager)


@pytest.fixture
def run_context(test_agent: Agent) -> RunContext:
    """Create a test run context."""
    return RunContext(
        agent=test_agent,
        inputs={"test": "input"},
        metadata={"test": "metadata"},
    )


@pytest.mark.asyncio
async def test_run_sufficient_budget(
    budget_guardrail: BudgetGuardrail,
    run_context: RunContext,
):
    """Test run with sufficient budget."""
    result = await budget_guardrail.run(run_context)
    assert result is None


@pytest.mark.asyncio
async def test_run_insufficient_budget(
    budget_guardrail: BudgetGuardrail,
    budget_manager: BudgetManager,
    run_context: RunContext,
):
    """Test run with insufficient budget."""
    budget_manager.has_sufficient_budget.return_value = False
    budget_manager.request_override.return_value = BudgetOverride(
        status="REJECTED",
        amount=Decimal("10.0"),
        reason="Test rejection",
    )

    result = await budget_guardrail.run(run_context)
    assert "Budget exceeded" in result
    assert "Override request rejected" in result


@pytest.mark.asyncio
async def test_run_override_pending(
    budget_guardrail: BudgetGuardrail,
    budget_manager: BudgetManager,
    run_context: RunContext,
):
    """Test run with pending override."""
    budget_manager.has_sufficient_budget.return_value = False
    budget_manager.request_override.return_value = BudgetOverride(
        status="PENDING",
        amount=Decimal("10.0"),
        reason="Test pending",
    )

    result = await budget_guardrail.run(run_context)
    assert "Budget exceeded" in result
    assert "Override request pending" in result


@pytest.mark.asyncio
async def test_validate_within_budget(
    budget_guardrail: BudgetGuardrail,
    run_context: RunContext,
):
    """Test validation when within budget."""
    result = await budget_guardrail.validate(run_context, {"test": "result"})
    assert result is None


@pytest.mark.asyncio
async def test_validate_exceeded_budget(
    budget_guardrail: BudgetGuardrail,
    budget_manager: BudgetManager,
    run_context: RunContext,
):
    """Test validation when budget exceeded."""
    budget_manager.has_exceeded_budget.return_value = True

    result = await budget_guardrail.validate(run_context, {"test": "result"})
    assert "exceeded allocated budget" in result
