"""Test fixtures for budget coordination system."""

from dataclasses import dataclass, field
from decimal import Decimal

from safeguards.core.dynamic_budget import AgentPriority


@dataclass
class AgentFixture:
    """Test fixture for agents."""

    id: str
    name: str
    balance: Decimal


@dataclass
class PoolFixture:
    """Test fixture for budget pools."""

    id: str
    total_budget: Decimal
    agents: set[str] = field(default_factory=set)
    min_balance: Decimal = Decimal("0")
    priority: AgentPriority = AgentPriority.MEDIUM


class TestScenarios:
    """Common test scenarios for coordination tests."""

    @staticmethod
    def basic_transfer_scenario() -> dict[str, any]:
        """Create a scenario for basic transfer testing.

        Returns:
            Dict with agents, pools and expected results
        """
        return {
            "agents": {
                "agent1": AgentFixture(
                    id="agent1",
                    name="Agent 1",
                    balance=Decimal("100.00"),
                ),
                "agent2": AgentFixture(
                    id="agent2",
                    name="Agent 2",
                    balance=Decimal("50.00"),
                ),
            },
            "pools": {
                "pool1": PoolFixture(
                    id="pool1",
                    total_budget=Decimal("200.00"),
                    agents={"agent1", "agent2"},
                ),
            },
            "transfer_amount": Decimal("25.00"),
            "expected_balance_agent1": Decimal("75.00"),
            "expected_balance_agent2": Decimal("75.00"),
        }

    @staticmethod
    def multi_pool_scenario() -> dict[str, any]:
        """Create a scenario with multiple pools.

        Returns:
            Dict with multiple agents and pools
        """
        return {
            "agents": {
                "agent1": AgentFixture(
                    id="agent1",
                    name="Agent 1",
                    balance=Decimal("100.00"),
                ),
                "agent2": AgentFixture(
                    id="agent2",
                    name="Agent 2",
                    balance=Decimal("150.00"),
                ),
                "agent3": AgentFixture(
                    id="agent3",
                    name="Agent 3",
                    balance=Decimal("200.00"),
                ),
            },
            "pools": {
                "pool1": PoolFixture(
                    id="pool1",
                    total_budget=Decimal("300.00"),
                    agents={"agent1", "agent2"},
                    priority=AgentPriority.HIGH,
                ),
                "pool2": PoolFixture(
                    id="pool2",
                    total_budget=Decimal("400.00"),
                    agents={"agent3"},
                    priority=AgentPriority.MEDIUM,
                ),
            },
        }

    @staticmethod
    def emergency_scenario() -> dict[str, any]:
        """Create a scenario for emergency allocation testing.

        Returns:
            Dict with agents, pools and emergency parameters
        """
        return {
            "agents": {
                "agent1": AgentFixture(
                    id="agent1",
                    name="Agent 1",
                    balance=Decimal("50.00"),
                ),
                "agent2": AgentFixture(
                    id="agent2",
                    name="Agent 2",
                    balance=Decimal("25.00"),
                ),
                "critical_agent": AgentFixture(
                    id="critical_agent",
                    name="Critical Agent",
                    balance=Decimal("10.00"),
                ),
            },
            "pools": {
                "main_pool": PoolFixture(
                    id="main_pool",
                    total_budget=Decimal("200.00"),
                    agents={"agent1", "agent2", "critical_agent"},
                    min_balance=Decimal("20.00"),  # Reserved for emergencies
                ),
            },
            "emergency_amount": Decimal("30.00"),
            "emergency_reason": "Critical system operation",
        }
