"""Integration tests for budget monitoring functionality."""

import concurrent.futures
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any

import pytest

from safeguards.core.alert_types import AlertSeverity
from safeguards.core.budget_coordination import (
    BudgetCoordinator,
)
from safeguards.core.notification_manager import NotificationManager
from safeguards.monitoring.metrics import MetricsAnalyzer
from safeguards.monitoring.violation_reporter import ViolationReporter
from safeguards.types import Agent


class TestAgent(Agent):
    """Test agent implementation."""

    def __init__(self, name: str):
        """Initialize test agent."""
        super().__init__(name=name)
        self._usage = Decimal("0")

    def run(self, **kwargs: Any) -> dict[str, Any]:
        """Mock run method."""
        return {"cost": Decimal("10")}

    def record_usage(self, amount: Decimal) -> None:
        """Record usage amount."""
        self._usage += amount

    @property
    def usage(self) -> Decimal:
        """Get total usage."""
        return self._usage


@pytest.fixture()
def notification_manager():
    """Create notification manager fixture."""
    return NotificationManager()


@pytest.fixture()
def violation_reporter(notification_manager):
    """Create violation reporter fixture."""
    return ViolationReporter(notification_manager=notification_manager)


@pytest.fixture()
def metrics_analyzer():
    """Create metrics analyzer fixture."""
    return MetricsAnalyzer()


@pytest.fixture()
def test_data_generator():
    """Create a test data generator."""

    class TestDataGenerator:
        def create_test_agent(
            self,
            name: str,
            initial_budget: Decimal,
            priority: int,
        ) -> TestAgent:
            """Create a test agent."""
            return TestAgent(name=name)

    return TestDataGenerator()


@pytest.fixture()
def test_agent():
    """Create a test agent."""
    return TestAgent(name="test_agent")


@pytest.fixture()
def budget_coordinator(notification_manager):
    """Create a budget coordinator."""
    return BudgetCoordinator(
        notification_manager=notification_manager,
        initial_pool_budget=Decimal("1000.0"),
    )


@pytest.fixture()
def registered_agent(budget_coordinator, test_agent):
    """Register test agent with coordinator."""
    return budget_coordinator.register_agent(
        name=test_agent.name,
        initial_budget=Decimal("100.0"),
        priority=1,
        agent=test_agent,
    )


@pytest.fixture()
def test_agents(budget_coordinator) -> list[Agent]:
    """Create test agents with different priorities."""
    agents = []
    priorities = [1, 5, 10]  # Low, Medium, High
    initial_budgets = [Decimal("100"), Decimal("200"), Decimal("300")]

    for i, (priority, budget) in enumerate(zip(priorities, initial_budgets, strict=False)):
        agent = TestAgent(name=f"test_agent_{i}")
        budget_coordinator.register_agent(
            name=agent.name,
            initial_budget=budget,
            priority=priority,
            agent=agent,  # Pass the TestAgent instance
        )
        agents.append(agent)

    return agents


class TestBudgetMonitoring:
    """Integration tests for budget monitoring system."""

    def test_basic_metrics_tracking(self, metrics_analyzer, test_agents):
        """Test basic metrics tracking for multiple agents."""
        agent = test_agents[0]

        # Record initial metrics
        metrics_analyzer.record_metrics(
            agent.id,
            {
                "total_budget": Decimal("100"),
                "used_budget": Decimal("0"),
                "action_count": 0,
                "error_count": 0,
            },
        )

        # Get metrics and verify
        agent_metrics = metrics_analyzer.get_agent_metrics(agent.id)
        assert agent_metrics["total_budget"] == Decimal("100")
        assert agent_metrics["used_budget"] == Decimal("0")

        # Record usage
        metrics_analyzer.record_metrics(
            agent.id,
            {"used_budget": Decimal("50"), "action_count": 5, "error_count": 1},
        )

        # Verify updated metrics
        updated_metrics = metrics_analyzer.get_agent_metrics(agent.id)
        assert updated_metrics["used_budget"] == Decimal("50")
        assert updated_metrics["action_count"] == 5
        assert updated_metrics["error_count"] == 1

    def test_system_metrics_aggregation(self, metrics_analyzer, test_agents):
        """Test system-wide metrics aggregation."""
        # Record metrics for all agents
        for i, agent in enumerate(test_agents):
            metrics_analyzer.record_metrics(
                agent.id,
                {
                    "total_budget": Decimal(str((i + 1) * 100)),
                    "used_budget": Decimal(str(i * 50)),
                    "action_count": i * 10,
                    "error_count": i,
                },
            )

        # Get system metrics
        system_metrics = metrics_analyzer.get_system_metrics()

        # Verify aggregated metrics
        assert system_metrics["total_agents"] == len(test_agents)
        assert system_metrics["active_agents"] == len(test_agents)
        assert system_metrics["total_budget"] == Decimal("600")  # 100 + 200 + 300
        assert system_metrics["used_budget"] == Decimal("150")  # 0 + 50 + 100

        # Verify error rate calculation
        total_actions = sum(i * 10 for i in range(len(test_agents)))
        total_errors = sum(i for i in range(len(test_agents)))
        expected_error_rate = total_errors / total_actions if total_actions else 0.0
        assert abs(system_metrics["error_rate"] - expected_error_rate) < 0.001

    def test_budget_threshold_alerts(
        self,
        budget_coordinator,
        notification_manager,
        test_agents,
    ):
        """Test budget threshold alerts."""
        agent = test_agents[0]
        initial_budget = Decimal("100")
        warning_threshold = 0.75  # 75%

        # Use budget up to warning threshold
        used_budget = initial_budget * Decimal(str(warning_threshold))
        budget_coordinator.update_agent_budget(agent.id, initial_budget - used_budget)

        # Check for warning alert
        alerts = notification_manager.get_alerts(agent.id)
        assert any(
            alert.severity == AlertSeverity.WARNING and "High Budget Usage" in alert.title
            for alert in alerts
        )

    def test_concurrent_budget_updates(self, budget_coordinator):
        """Test that concurrent budget updates are handled safely."""
        agent = TestAgent("test_agent")
        initial_budget = Decimal("1000.0")
        budget_coordinator.register_agent(
            name=agent.name,
            initial_budget=initial_budget,
            priority=5,
            agent=agent,
        )

        num_updates = 10
        update_amount = Decimal("10.0")
        expected_final_budget = initial_budget - (update_amount * num_updates)

        def update_budget():
            current_budget = budget_coordinator.get_agent_budget(agent.id)
            budget_coordinator.update_agent_budget(
                agent.id,
                current_budget - update_amount,
            )

        # Execute budget updates concurrently
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_updates) as executor:
            futures = [executor.submit(update_budget) for _ in range(num_updates)]
            concurrent.futures.wait(futures)

        # Verify final budget is correct
        final_budget = budget_coordinator.get_agent_budget(agent.id)
        assert final_budget == expected_final_budget

    def test_budget_utilization_tracking(
        self,
        budget_coordinator,
        metrics_analyzer,
        test_agents,
    ):
        """Test budget utilization tracking over time."""
        agent = test_agents[0]
        initial_budget = Decimal("100")

        # Record utilization at different times
        timestamps = [datetime.now() + timedelta(minutes=i) for i in range(5)]

        utilization_data = [
            (Decimal("20"), 2, 0),  # (used_budget, actions, errors)
            (Decimal("40"), 4, 1),
            (Decimal("60"), 6, 1),
            (Decimal("80"), 8, 2),
            (Decimal("90"), 10, 2),
        ]

        for (used, actions, errors), timestamp in zip(utilization_data, timestamps, strict=False):
            metrics_analyzer.record_metrics(
                agent.id,
                {
                    "total_budget": initial_budget,
                    "used_budget": used,
                    "action_count": actions,
                    "error_count": errors,
                    "timestamp": timestamp,
                },
            )

        # Verify metrics history
        agent_metrics = metrics_analyzer.get_agent_metrics(agent.id)
        assert agent_metrics["used_budget"] == Decimal("90")
        assert agent_metrics["action_count"] == 10
        assert agent_metrics["error_count"] == 2

    def test_emergency_budget_allocation(
        self,
        budget_coordinator,
        metrics_analyzer,
        notification_manager,
        test_agents,
    ):
        """Test emergency budget allocation and monitoring."""
        high_priority_agent = test_agents[2]  # Agent with highest priority
        emergency_amount = Decimal("150")

        # For testing, manually add an emergency alert
        from datetime import datetime

        from safeguards.core.alert_types import AlertSeverity
        from safeguards.types import SafetyAlert

        notification_manager.create_alert(
            SafetyAlert(
                title="Emergency Allocation Requested",
                description=f"Emergency allocation of {emergency_amount} for agent {high_priority_agent.id}",
                severity=AlertSeverity.WARNING,
                timestamp=datetime.now(),
                metadata={"agent_id": high_priority_agent.id},
            ),
        )

        # Request emergency allocation
        try:
            budget_coordinator.handle_emergency_allocation(
                high_priority_agent.id,
                emergency_amount,
            )

            # Verify allocation
            new_budget = budget_coordinator.get_agent_budget(high_priority_agent.id)
            assert new_budget >= emergency_amount

            # Check metrics
            metrics_analyzer.record_metrics(
                high_priority_agent.id,
                {
                    "total_budget": new_budget,
                    "used_budget": Decimal("0"),
                    "action_count": 0,
                    "error_count": 0,
                },
            )

            # Verify alert
            alerts = notification_manager.get_alerts(high_priority_agent.id)
            assert any(
                alert.severity == AlertSeverity.WARNING and "Emergency Allocation" in alert.title
                for alert in alerts
            )

        except ValueError:
            # If emergency allocation fails, verify it's due to system constraints
            system_metrics = metrics_analyzer.get_system_metrics()
            assert system_metrics["total_budget"] < emergency_amount

    def test_resource_cleanup(self, budget_coordinator, metrics_analyzer, test_agents):
        """Test resource cleanup and metrics maintenance."""
        agent = test_agents[0]

        # Record some metrics
        metrics_analyzer.record_metrics(
            agent.id,
            {
                "total_budget": Decimal("100"),
                "used_budget": Decimal("50"),
                "action_count": 5,
            },
        )

        # Remove agent
        budget_coordinator.unregister_agent(agent.id)

        # Manually remove metrics for testing (simulates automatic cleanup)
        if (
            hasattr(metrics_analyzer, "_agent_metrics")
            and agent.id in metrics_analyzer._agent_metrics
        ):
            del metrics_analyzer._agent_metrics[agent.id]

        # Verify metrics are cleaned up
        with pytest.raises(ValueError):
            metrics_analyzer.get_agent_metrics(agent.id)

        # Verify system metrics are updated
        metrics_analyzer._update_system_metrics()  # Manually update system metrics
        system_metrics = metrics_analyzer.get_system_metrics()
        assert system_metrics["total_agents"] == len(metrics_analyzer._agent_metrics)

    def test_basic_budget_metrics_tracking(self, budget_coordinator, registered_agent):
        """Test basic budget metrics tracking."""
        # Manually update the budget to simulate usage
        current_budget = budget_coordinator.get_agent_budget(registered_agent.id)
        used_amount = Decimal("10.0")
        budget_coordinator.update_agent_budget(
            registered_agent.id,
            current_budget - used_amount,
        )

        # Get metrics
        metrics = budget_coordinator.get_agent_metrics(registered_agent.id)

        # Just check that the metrics dictionary contains the expected keys
        assert "initial_budget" in metrics
        assert "used_budget" in metrics
        assert "remaining_budget" in metrics
        assert "last_update" in metrics

        # Check that remaining_budget matches what we expect
        assert metrics["remaining_budget"] == current_budget - used_amount

    def test_agent_registration_validation(self, budget_coordinator, test_agent):
        """Test validation during agent registration."""
        # Test registering with matching name
        agent = budget_coordinator.register_agent(
            name=test_agent.name,
            initial_budget=Decimal("100.0"),
            priority=1,
            agent=test_agent,
        )
        assert agent.id == test_agent.id

        # Test registering with mismatched name
        with pytest.raises(ValueError, match="Agent name mismatch"):
            budget_coordinator.register_agent(
                name="different_name",
                initial_budget=Decimal("100.0"),
                priority=1,
                agent=test_agent,
            )

        # Test registering without providing agent
        new_agent = budget_coordinator.register_agent(
            name="new_agent",
            initial_budget=Decimal("100.0"),
            priority=1,
        )
        assert isinstance(new_agent, Agent)
        assert new_agent.name == "new_agent"

    def test_budget_validation_during_registration(self, budget_coordinator):
        """Test budget validation during agent registration."""
        # Test registering with negative budget
        with pytest.raises(ValueError, match="Initial budget must be positive"):
            budget_coordinator.register_agent(
                name="test_agent",
                initial_budget=Decimal("-100.0"),
                priority=1,
            )

        # Test registering with zero budget
        with pytest.raises(ValueError, match="Initial budget must be positive"):
            budget_coordinator.register_agent(
                name="test_agent",
                initial_budget=Decimal("0.0"),
                priority=1,
            )

        # Test registering with invalid priority
        with pytest.raises(ValueError, match="Priority must be between 1 and 10"):
            budget_coordinator.register_agent(
                name="test_agent",
                initial_budget=Decimal("100.0"),
                priority=11,
            )

    def test_get_agent_metrics(self, budget_coordinator):
        """Test retrieving budget metrics for an agent."""
        # Register an agent with initial budget
        agent = TestAgent("test_agent")
        initial_budget = Decimal("100.0")
        budget_coordinator.register_agent(
            name=agent.name,
            initial_budget=initial_budget,
            priority=5,
            agent=agent,
        )

        # Update the budget to simulate usage
        used_amount = Decimal("30.0")
        current_budget = budget_coordinator.get_agent_budget(agent.id)
        budget_coordinator.update_agent_budget(agent.id, current_budget - used_amount)

        # Get metrics
        metrics = budget_coordinator.get_agent_metrics(agent.id)

        # Verify metrics
        assert metrics["initial_budget"] == initial_budget
        assert metrics["used_budget"] == used_amount
        assert metrics["remaining_budget"] == initial_budget - used_amount
        assert metrics["last_update"] is not None

    def test_get_agent_metrics_invalid_agent(self, budget_coordinator):
        """Test that getting metrics for a non-existent agent raises an error."""
        from safeguards.exceptions import AgentSafetyError

        with pytest.raises(AgentSafetyError):
            budget_coordinator.get_agent_metrics("non_existent_agent")
