"""Tests for the budget coordination system."""

from decimal import Decimal

import pytest

from safeguards.base.budget import BudgetConfig, BudgetPeriod
from safeguards.base.budget_coordinator import BudgetCoordinator


def test_budget_coordinator_registration():
    """Test agent registration with budget coordinator."""
    coordinator = BudgetCoordinator()
    config = BudgetConfig(total_budget=Decimal(1000), period=BudgetPeriod.DAILY)

    # Test registration
    coordinator.register_agent("test_agent", config, priority=1)

    # Test getting remaining budget for registered agent
    remaining = coordinator.get_remaining_budget("test_agent")
    assert remaining == 1000

    # Test getting remaining budget for unregistered agent
    with pytest.raises(KeyError):
        coordinator.get_remaining_budget("unknown_agent")


def test_budget_coordinator_usage_tracking():
    """Test usage tracking in budget coordinator."""
    coordinator = BudgetCoordinator()
    config = BudgetConfig(total_budget=Decimal(1000), period=BudgetPeriod.DAILY)
    coordinator.register_agent("test_agent", config)

    # Track some usage
    coordinator.track_usage("test_agent", 100)

    # Check remaining budget
    remaining = coordinator.get_remaining_budget("test_agent")
    assert remaining == 900

    # Check metrics
    metrics = coordinator.get_metrics("test_agent")
    assert metrics.current_usage == Decimal(100)
    assert metrics.usage_percentage == 0.1  # 10%


def test_budget_coordinator_multiple_agents():
    """Test handling multiple agents in budget coordinator."""
    coordinator = BudgetCoordinator()

    # Register multiple agents with different priorities
    config1 = BudgetConfig(total_budget=Decimal(1000), period=BudgetPeriod.DAILY)
    config2 = BudgetConfig(total_budget=Decimal(500), period=BudgetPeriod.DAILY)

    coordinator.register_agent("agent1", config1, priority=1)
    coordinator.register_agent("agent2", config2, priority=2)

    # Track usage for both agents
    coordinator.track_usage("agent1", 800)  # 80% usage
    coordinator.track_usage("agent2", 100)  # 20% usage

    # Check metrics for both agents
    metrics1 = coordinator.get_metrics("agent1")
    metrics2 = coordinator.get_metrics("agent2")

    assert metrics1.usage_percentage == 0.8
    assert metrics2.usage_percentage == 0.2


def test_budget_coordinator_reallocation():
    """Test budget reallocation based on usage."""
    coordinator = BudgetCoordinator()
    config = BudgetConfig(total_budget=Decimal(1000), period=BudgetPeriod.DAILY)
    coordinator.register_agent("test_agent", config)

    # Simulate high usage
    coordinator.track_usage("test_agent", 950)  # 95% usage

    # Trigger reallocation
    coordinator.reallocate_budgets()

    # Check if budget was increased
    new_remaining = coordinator.get_remaining_budget("test_agent")
    assert new_remaining > 50  # Original remaining was 50
