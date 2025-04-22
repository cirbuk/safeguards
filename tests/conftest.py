"""Common test fixtures for agent safety tests."""

from decimal import Decimal
from typing import Generator
import pytest
from unittest.mock import MagicMock

from safeguards.types.agent import Agent
from safeguards import (
    SafetyController,
    SafetyConfig,
    BudgetManager,
    ResourceMonitor,
    NotificationManager,
)


class TestAgent(Agent):
    """Test implementation of Agent interface."""

    def run(self, **kwargs):
        """Mock implementation of run method."""
        return {"status": "success", "output": "test output"}


@pytest.fixture
def mock_agent() -> Agent:
    """Create a mock agent for testing."""
    return TestAgent(name="test_agent", instructions="Test agent for unit tests.")


@pytest.fixture
def mock_budget_manager() -> BudgetManager:
    """Create a mock budget manager."""
    manager = MagicMock(spec=BudgetManager)
    manager.has_sufficient_budget.return_value = True
    manager.has_exceeded_budget.return_value = False
    manager.get_minimum_required.return_value = Decimal("10.0")
    return manager


@pytest.fixture
def mock_resource_monitor() -> ResourceMonitor:
    """Create a mock resource monitor."""
    monitor = MagicMock(spec=ResourceMonitor)
    monitor.get_metrics.return_value = {
        "cpu_percent": 50.0,
        "memory_percent": 60.0,
        "disk_percent": 70.0,
    }
    return monitor


@pytest.fixture
def mock_notification_manager() -> NotificationManager:
    """Create a mock notification manager."""
    return MagicMock(spec=NotificationManager)


@pytest.fixture
def safety_config() -> SafetyConfig:
    """Create a test safety configuration."""
    return SafetyConfig(
        total_budget=Decimal("1000"),
        hourly_limit=Decimal("100"),
        daily_limit=Decimal("500"),
        cpu_threshold=80.0,
        memory_threshold=80.0,
        budget_warning_threshold=75.0,
    )


@pytest.fixture
def safety_controller(
    safety_config: SafetyConfig,
    mock_budget_manager: BudgetManager,
    mock_resource_monitor: ResourceMonitor,
    mock_notification_manager: NotificationManager,
) -> Generator[SafetyController, None, None]:
    """Create a safety controller with mock components."""
    controller = SafetyController(safety_config)
    controller.budget_manager = mock_budget_manager
    controller.resource_monitor = mock_resource_monitor
    controller.notification_manager = mock_notification_manager
    yield controller
