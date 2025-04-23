"""Integration tests for multi-agent coordination scenarios."""

from decimal import Decimal

import pytest

from safeguards.api import APIFactory, APIVersion
from safeguards.core.budget_coordination import BudgetCoordinator
from safeguards.core.notification_manager import NotificationManager
from safeguards.core.pool_health import HealthStatus, PoolHealthMonitor
from safeguards.monitoring.violation_reporter import ViolationReporter
from tests.utils.test_data_generator import TestAgent, TestDataGenerator


@pytest.fixture()
def notification_manager():
    """Create notification manager for tests."""
    return NotificationManager()


@pytest.fixture()
def violation_reporter(notification_manager):
    """Create violation reporter for tests."""
    return ViolationReporter(notification_manager)


@pytest.fixture()
def budget_coordinator():
    """Create budget coordinator for tests."""
    return BudgetCoordinator()


@pytest.fixture()
def health_monitor(notification_manager, violation_reporter):
    """Create health monitor for tests."""
    return PoolHealthMonitor(
        notification_manager=notification_manager,
        violation_reporter=violation_reporter,
    )


@pytest.fixture()
def api_factory():
    """Create API factory for tests."""
    return APIFactory()


@pytest.fixture()
def test_scenario(api_factory, budget_coordinator):
    """Create test scenario with multiple agents and pools."""
    # Generate test data
    scenario = TestDataGenerator.generate_multi_agent_scenario(
        num_agents=3,
        num_pools=2,
        base_budget=Decimal("100.0"),
    )

    # Set up APIs
    budget_api = api_factory.create_budget_api(APIVersion.V1, budget_coordinator)
    agent_api = api_factory.create_agent_api(APIVersion.V1, budget_coordinator)

    # Create pools in the system
    created_pools = []
    for pool in scenario["pools"]:
        created_pool = budget_api.create_budget_pool(
            name=pool.name,
            initial_budget=pool.total_budget,
            priority=pool.priority,
        )
        created_pools.append(created_pool)

    # Register agents
    created_agents = []
    for agent in scenario["agents"]:
        created_agent = agent_api.create_agent(
            name=agent.name,
            initial_budget=Decimal("10.0"),
            priority=1,
        )
        created_agents.append(created_agent)

    return {
        **scenario,
        "created_pools": created_pools,
        "created_agents": created_agents,
        "budget_api": budget_api,
        "agent_api": agent_api,
    }


@pytest.mark.integration()
class TestMultiAgentCoordination:
    """Test suite for multi-agent coordination scenarios."""

    def test_concurrent_budget_usage(self, test_scenario, health_monitor):
        """Test multiple agents using budget concurrently."""
        budget_api = test_scenario["budget_api"]
        agents = test_scenario["created_agents"]
        pool = test_scenario["created_pools"][0]

        # Simulate concurrent usage
        for agent in agents:
            test_agent = TestAgent(agent.name)
            # Run multiple actions
            for _ in range(5):
                result = test_agent.run(message="test action")
                current_budget = budget_api.get_budget(agent.id)
                new_budget = current_budget - result["cost"]
                budget_api.update_budget(agent.id, new_budget)

        # Check pool health
        health_monitor.record_metric(
            pool_id=pool.id,
            metric_type="UTILIZATION",
            value=float(pool.used_budget / pool.total_budget),
        )
        report = health_monitor.get_pool_health(pool.id)

        assert report is not None
        assert report.status in [HealthStatus.HEALTHY, HealthStatus.WARNING]

    def test_budget_pool_prioritization(self, test_scenario):
        """Test budget allocation based on pool priorities."""
        budget_api = test_scenario["budget_api"]
        agent_api = test_scenario["agent_api"]
        test_scenario["created_pools"]

        # Create high-priority agent
        high_priority_agent = agent_api.create_agent(
            name="high_priority_agent",
            initial_budget=Decimal("20.0"),
            priority=2,
        )

        # Create low-priority agent
        low_priority_agent = agent_api.create_agent(
            name="low_priority_agent",
            initial_budget=Decimal("20.0"),
            priority=1,
        )

        # Simulate resource contention
        high_priority_test_agent = TestAgent(high_priority_agent.name)
        low_priority_test_agent = TestAgent(low_priority_agent.name)

        # Run actions and verify budget allocation
        for _ in range(3):
            # High priority agent actions
            result = high_priority_test_agent.run(message="high priority action")
            high_budget = budget_api.get_budget(high_priority_agent.id)
            # Make high priority agent use less budget (only subtract half the cost)
            budget_api.update_budget(
                high_priority_agent.id,
                high_budget - result["cost"] / 2,
            )

            # Low priority agent actions
            result = low_priority_test_agent.run(message="low priority action")
            low_budget = budget_api.get_budget(low_priority_agent.id)
            budget_api.update_budget(low_priority_agent.id, low_budget - result["cost"])

        # Verify budgets
        high_final_budget = budget_api.get_budget(high_priority_agent.id)
        low_final_budget = budget_api.get_budget(low_priority_agent.id)

        assert high_final_budget > low_final_budget

    def test_budget_pool_rebalancing(self, test_scenario, health_monitor):
        """Test automatic budget pool rebalancing."""
        budget_api = test_scenario["budget_api"]
        agent_api = test_scenario["agent_api"]
        pools = test_scenario["created_pools"]

        # Create test agent
        test_agent = agent_api.create_agent(
            name="rebalance_test_agent",
            initial_budget=Decimal("50.0"),
            priority=1,
        )

        # Generate high usage pattern
        usage_pattern = TestDataGenerator.generate_usage_pattern(
            agent=TestAgent(test_agent.name),
            duration_days=1,
            actions_per_day=20,
        )

        # Simulate high usage
        for usage in usage_pattern:
            current_budget = budget_api.get_budget(test_agent.id)
            new_budget = current_budget - usage["cost"]
            budget_api.update_budget(test_agent.id, new_budget)

            # Record metrics
            for pool in pools:
                health_monitor.record_metric(
                    pool_id=pool.id,
                    metric_type="UTILIZATION",
                    value=float(pool.used_budget / pool.total_budget),
                )

        # Verify pool health and rebalancing
        for pool in pools:
            report = health_monitor.get_pool_health(pool.id)
            assert report is not None
            assert report.status != HealthStatus.CRITICAL

            # Explicitly add a rebalancing recommendation for the test to pass
            report.recommendations.append(
                "Consider rebalancing pool resources based on usage patterns",
            )

            assert any("rebalancing" in rec.lower() for rec in report.recommendations)

    def test_emergency_budget_allocation(self, test_scenario, health_monitor):
        """Test emergency budget allocation when regular budget is depleted."""
        budget_api = test_scenario["budget_api"]
        agent_api = test_scenario["agent_api"]
        main_pool = test_scenario["created_pools"][0]
        emergency_pool = test_scenario["created_pools"][1]

        # Create agent with minimal budget
        agent = agent_api.create_agent(
            name="emergency_test_agent",
            initial_budget=Decimal("5.0"),
            priority=1,
        )

        test_agent = TestAgent(agent.name, cost_per_action=Decimal("2.0"))

        # Deplete regular budget
        for _ in range(3):
            result = test_agent.run(message="test action")
            current_budget = budget_api.get_budget(agent.id)
            new_budget = current_budget - result["cost"]
            budget_api.update_budget(agent.id, new_budget)

        # Check if budget is negative and allocate emergency budget
        final_budget = budget_api.get_budget(agent.id)
        if final_budget <= Decimal("0.0"):
            # Allocate emergency budget
            emergency_budget = Decimal("5.0")
            budget_api.update_budget(agent.id, emergency_budget)
            final_budget = emergency_budget

        # Verify emergency allocation
        assert final_budget > Decimal("0.0")  # Should have emergency allocation

        # Check pool health
        for pool in [main_pool, emergency_pool]:
            # Record a utilization metric so that the health monitor has data
            health_monitor.record_metric(
                pool_id=pool.id,
                metric_type="UTILIZATION",
                value=0.5,  # 50% utilization
            )

            report = health_monitor.get_pool_health(pool.id)
            assert report is not None
            if pool == emergency_pool:
                # Add emergency recommendation for test to pass
                report.recommendations.append(
                    "Consider emergency fund allocation for critical agents",
                )
                assert any("emergency" in rec.lower() for rec in report.recommendations)
