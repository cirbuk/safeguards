"""Tests for the cost estimation module."""

from decimal import Decimal

import pytest

from safeguards.base.budget import BudgetPeriod
from safeguards.budget.cost_estimation import CostEstimate, CostEstimator


@pytest.fixture()
def cost_estimator():
    """Create a cost estimator instance for testing."""
    llm_costs = {
        "gpt-4": Decimal("0.03"),  # $0.03 per 1K tokens
        "gpt-3.5": Decimal("0.002"),  # $0.002 per 1K tokens
    }
    api_costs = {
        "search": Decimal("0.01"),  # $0.01 per call
        "embedding": Decimal("0.0004"),  # $0.0004 per call
    }
    storage_cost_per_gb = Decimal("0.02")  # $0.02 per GB
    compute_cost_per_hour = Decimal("0.10")  # $0.10 per hour

    return CostEstimator(
        llm_costs=llm_costs,
        api_costs=api_costs,
        storage_cost_per_gb=storage_cost_per_gb,
        compute_cost_per_hour=compute_cost_per_hour,
    )


def test_estimate_llm_cost(cost_estimator):
    """Test LLM cost estimation."""
    # Test input tokens
    cost = cost_estimator.estimate_llm_cost("gpt-4", 1000)
    assert cost == Decimal("0.03")

    # Test output tokens (should be more expensive)
    cost = cost_estimator.estimate_llm_cost("gpt-4", 1000, is_output=True)
    assert cost == Decimal("0.039")  # 1.3x multiplier

    # Test unknown model
    cost = cost_estimator.estimate_llm_cost("unknown-model", 1000)
    assert cost == Decimal("0")


def test_estimate_api_cost(cost_estimator):
    """Test API cost estimation."""
    # Test basic API call cost
    cost = cost_estimator.estimate_api_cost("search", 100)
    assert cost == Decimal("1.0")  # 100 * $0.01

    # Test with data transfer
    cost = cost_estimator.estimate_api_cost("search", 100, data_size_mb=1000)
    assert cost > Decimal("1.0")  # Should include data transfer cost

    # Test unknown endpoint
    cost = cost_estimator.estimate_api_cost("unknown-endpoint", 100)
    assert cost == Decimal("0")


def test_estimate_storage_cost(cost_estimator):
    """Test storage cost estimation."""
    cost = cost_estimator.estimate_storage_cost(10.0, 24.0)
    assert cost == Decimal("0.2")  # 10 GB * $0.02 per GB * 1 day


def test_estimate_compute_cost(cost_estimator):
    """Test compute cost estimation."""
    # Test CPU only
    cost = cost_estimator.estimate_compute_cost(10.0)
    assert cost == Decimal("1.0")  # 10 hours * $0.10 per hour

    # Test with GPU
    cost = cost_estimator.estimate_compute_cost(10.0, 5.0)
    assert cost == Decimal("6.0")  # CPU: 1.0, GPU: 5.0


def test_create_total_estimate(cost_estimator):
    """Test comprehensive cost estimation."""
    llm_usage = {"gpt-4": 1000, "gpt-3.5": 5000}
    api_calls = {"search": 100, "embedding": 1000}
    storage_gb = 10.0
    cpu_hours = 24.0
    gpu_hours = 2.0

    estimate = cost_estimator.create_total_estimate(
        llm_usage=llm_usage,
        api_calls=api_calls,
        storage_gb=storage_gb,
        cpu_hours=cpu_hours,
        gpu_hours=gpu_hours,
        period=BudgetPeriod.DAILY,
    )

    assert isinstance(estimate, CostEstimate)
    assert estimate.total_cost > Decimal("0")
    assert len(estimate.breakdown) > 0
    assert 0 < estimate.confidence_score <= 1
    assert estimate.margin_of_error >= 0


def test_confidence_score_calculation(cost_estimator):
    """Test confidence score adjustments."""
    # Test base case
    score = cost_estimator._calculate_confidence_score(
        llm_usage={"gpt-4": 100},
        api_calls={"search": 10},
        storage_gb=1.0,
        cpu_hours=1.0,
        gpu_hours=0.0,
    )
    assert score == 0.8  # Base confidence

    # Test with large values
    score = cost_estimator._calculate_confidence_score(
        llm_usage={"gpt-4": 2000000},  # Large token count
        api_calls={"search": 2000},  # Large API usage
        storage_gb=200.0,  # Large storage
        cpu_hours=48.0,  # Long compute time
        gpu_hours=48.0,
    )
    assert score < 0.8  # Should be lower due to large values
    assert score >= 0.5  # Should not go below minimum
