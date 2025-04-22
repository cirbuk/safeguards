"""Test fixtures for common testing scenarios."""

from decimal import Decimal
from typing import Dict, List, Optional, Tuple

from agent_safety.core.dynamic_budget import AgentPriority
from agent_safety.monitoring.violation_reporter import (
    ViolationType,
    ViolationSeverity,
)
from agent_safety.testing.mock_implementations import (
    TestScenarioBuilder,
    MockBudgetState,
)


class TestScenarios:
    """Common test scenarios."""

    @staticmethod
    def create_basic_scenario() -> TestScenarioBuilder:
        """Create a basic test scenario with two agents and one pool.

        Returns:
            Test scenario builder
        """
        builder = TestScenarioBuilder()

        # Add agents
        builder.add_agent("agent1", Decimal("100"), AgentPriority.HIGH)
        builder.add_agent("agent2", Decimal("50"), AgentPriority.MEDIUM)

        # Add pool
        builder.add_pool(
            "pool1",
            total_budget=Decimal("1000"),
            min_balance=Decimal("100"),
            priority=AgentPriority.HIGH,
        )

        # Add agents to pool
        builder.add_agent_to_pool("agent1", "pool1", Decimal("200"))
        builder.add_agent_to_pool("agent2", "pool1", Decimal("100"))

        return builder

    @staticmethod
    def create_multi_pool_scenario() -> TestScenarioBuilder:
        """Create a scenario with multiple pools and agents.

        Returns:
            Test scenario builder
        """
        builder = TestScenarioBuilder()

        # Add agents with different priorities
        builder.add_agent("high_priority", Decimal("200"), AgentPriority.HIGH)
        builder.add_agent("medium_priority", Decimal("150"), AgentPriority.MEDIUM)
        builder.add_agent("low_priority", Decimal("100"), AgentPriority.LOW)

        # Add pools with different priorities
        builder.add_pool(
            "high_pool",
            total_budget=Decimal("1000"),
            min_balance=Decimal("200"),
            priority=AgentPriority.HIGH,
        )
        builder.add_pool(
            "medium_pool",
            total_budget=Decimal("500"),
            min_balance=Decimal("100"),
            priority=AgentPriority.MEDIUM,
        )
        builder.add_pool(
            "low_pool",
            total_budget=Decimal("300"),
            min_balance=Decimal("50"),
            priority=AgentPriority.LOW,
        )

        # Distribute agents across pools
        builder.add_agent_to_pool("high_priority", "high_pool", Decimal("300"))
        builder.add_agent_to_pool("medium_priority", "medium_pool", Decimal("200"))
        builder.add_agent_to_pool("low_priority", "low_pool", Decimal("100"))

        return builder

    @staticmethod
    def create_violation_scenario() -> Tuple[TestScenarioBuilder, List[Dict]]:
        """Create a scenario with various budget violations.

        Returns:
            Tuple of (scenario builder, list of violation details)
        """
        builder = TestScenarioBuilder()

        # Add agents and pools
        builder.add_agent("agent1", Decimal("100"))
        builder.add_agent("agent2", Decimal("50"))
        builder.add_pool(
            "pool1",
            total_budget=Decimal("1000"),
            min_balance=Decimal("200"),
        )
        builder.add_agent_to_pool("agent1", "pool1", Decimal("300"))
        builder.add_agent_to_pool("agent2", "pool1", Decimal("200"))

        # Create various violations
        violations = [
            {
                "agent_id": "agent1",
                "type": ViolationType.OVERSPEND,
                "severity": ViolationSeverity.HIGH,
                "amount": Decimal("150"),
            },
            {
                "agent_id": "agent2",
                "type": ViolationType.POOL_BREACH,
                "severity": ViolationSeverity.CRITICAL,
                "amount": Decimal("250"),
                "pool_id": "pool1",
            },
            {
                "agent_id": "agent1",
                "type": ViolationType.RATE_LIMIT,
                "severity": ViolationSeverity.MEDIUM,
                "amount": Decimal("50"),
            },
        ]

        # Create the violations
        for violation in violations:
            builder.create_violation(
                agent_id=violation["agent_id"],
                violation_type=violation["type"],
                severity=violation["severity"],
                amount=violation["amount"],
                pool_id=violation.get("pool_id"),
            )

        return builder, violations

    @staticmethod
    def create_transfer_scenario() -> TestScenarioBuilder:
        """Create a scenario for testing transfers.

        Returns:
            Test scenario builder
        """
        builder = TestScenarioBuilder()

        # Add agents with initial balances
        builder.add_agent("sender", Decimal("500"))
        builder.add_agent("receiver", Decimal("100"))

        # Add pools with different priorities and balances
        builder.add_pool(
            "high_pool",
            total_budget=Decimal("2000"),
            min_balance=Decimal("500"),
            priority=AgentPriority.HIGH,
        )
        builder.add_pool(
            "low_pool",
            total_budget=Decimal("1000"),
            min_balance=Decimal("200"),
            priority=AgentPriority.LOW,
        )

        # Add agents to pools
        builder.add_agent_to_pool("sender", "high_pool", Decimal("1000"))
        builder.add_agent_to_pool("receiver", "low_pool", Decimal("300"))

        return builder

    @staticmethod
    def create_emergency_scenario() -> TestScenarioBuilder:
        """Create a scenario for testing emergency budget allocation.

        Returns:
            Test scenario builder
        """
        builder = TestScenarioBuilder()

        # Add agents with different priorities
        builder.add_agent("critical_agent", Decimal("100"), AgentPriority.HIGH)
        builder.add_agent("normal_agent1", Decimal("200"), AgentPriority.MEDIUM)
        builder.add_agent("normal_agent2", Decimal("150"), AgentPriority.MEDIUM)

        # Add pools
        builder.add_pool(
            "emergency_pool",
            total_budget=Decimal("2000"),
            min_balance=Decimal("500"),
            priority=AgentPriority.HIGH,
        )
        builder.add_pool(
            "regular_pool",
            total_budget=Decimal("1000"),
            min_balance=Decimal("200"),
            priority=AgentPriority.MEDIUM,
        )

        # Distribute agents
        builder.add_agent_to_pool("critical_agent", "emergency_pool", Decimal("500"))
        builder.add_agent_to_pool("normal_agent1", "regular_pool", Decimal("400"))
        builder.add_agent_to_pool("normal_agent2", "regular_pool", Decimal("300"))

        return builder

    @staticmethod
    def create_pool_rebalancing_scenario() -> TestScenarioBuilder:
        """Create a scenario for testing pool rebalancing.

        Returns:
            Test scenario builder
        """
        builder = TestScenarioBuilder()

        # Add agents
        agents = [
            ("agent1", Decimal("100"), AgentPriority.HIGH),
            ("agent2", Decimal("150"), AgentPriority.HIGH),
            ("agent3", Decimal("200"), AgentPriority.MEDIUM),
            ("agent4", Decimal("120"), AgentPriority.MEDIUM),
            ("agent5", Decimal("80"), AgentPriority.LOW),
        ]

        for agent_id, balance, priority in agents:
            builder.add_agent(agent_id, balance, priority)

        # Add pools with varying utilization
        pools = [
            ("high_util_pool", Decimal("1000"), Decimal("100"), AgentPriority.HIGH),
            ("med_util_pool", Decimal("800"), Decimal("100"), AgentPriority.MEDIUM),
            ("low_util_pool", Decimal("500"), Decimal("50"), AgentPriority.LOW),
        ]

        for pool_id, budget, min_balance, priority in pools:
            builder.add_pool(pool_id, budget, min_balance, priority)

        # Distribute agents to create imbalance
        allocations = [
            ("agent1", "high_util_pool", Decimal("400")),
            ("agent2", "high_util_pool", Decimal("300")),
            ("agent3", "med_util_pool", Decimal("300")),
            ("agent4", "med_util_pool", Decimal("200")),
            ("agent5", "low_util_pool", Decimal("100")),
        ]

        for agent_id, pool_id, allocation in allocations:
            builder.add_agent_to_pool(agent_id, pool_id, allocation)

        return builder
