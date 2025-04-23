"""Tests for the budget monitoring functionality."""

from decimal import Decimal

import pytest

from safeguards.core.alert_types import AlertSeverity
from safeguards.core.notification_manager import NotificationManager
from safeguards.monitoring.budget_monitor import BudgetMonitor


@pytest.fixture()
def notification_manager():
    """Create a notification manager for testing."""
    return NotificationManager()


@pytest.fixture()
def budget_monitor(notification_manager):
    """Create a budget monitor with a notification manager."""
    return BudgetMonitor(
        notification_manager,
        warning_threshold=0.75,
        critical_threshold=0.9,
    )


class TestBudgetMonitor:
    """Test cases for the BudgetMonitor class."""

    def test_initialization(self, budget_monitor):
        """Test that the budget monitor initializes correctly."""
        assert budget_monitor.warning_threshold == Decimal("0.75")
        assert budget_monitor.critical_threshold == Decimal("0.9")
        assert isinstance(budget_monitor._alerted_agents, dict)
        assert "warning" in budget_monitor._alerted_agents
        assert "critical" in budget_monitor._alerted_agents

    def test_check_budget_usage_warning(self, budget_monitor, notification_manager):
        """Test that a warning alert is generated when usage exceeds the warning threshold."""
        agent_id = "test_agent"
        total_budget = Decimal("100")
        used_budget = Decimal("80")  # 80% usage, above warning threshold (75%)

        budget_monitor.check_budget_usage(agent_id, used_budget, total_budget)

        # Verify alert was created
        alerts = notification_manager.get_alerts(agent_id)
        assert len(alerts) == 1
        assert alerts[0].severity == AlertSeverity.WARNING
        assert "High Budget Usage" in alerts[0].title
        assert agent_id in budget_monitor._alerted_agents["warning"]

    def test_check_budget_usage_critical(self, budget_monitor, notification_manager):
        """Test that a critical alert is generated when usage exceeds the critical threshold."""
        agent_id = "test_agent"
        total_budget = Decimal("100")
        used_budget = Decimal("95")  # 95% usage, above critical threshold (90%)

        budget_monitor.check_budget_usage(agent_id, used_budget, total_budget)

        # Verify alert was created
        alerts = notification_manager.get_alerts(agent_id)
        assert len(alerts) == 1
        assert alerts[0].severity == AlertSeverity.CRITICAL
        assert "Critical Budget Usage" in alerts[0].title
        assert agent_id in budget_monitor._alerted_agents["critical"]

    def test_check_budget_usage_below_thresholds(
        self,
        budget_monitor,
        notification_manager,
    ):
        """Test that no alerts are generated when usage is below thresholds."""
        agent_id = "test_agent"
        total_budget = Decimal("100")
        used_budget = Decimal("50")  # 50% usage, below thresholds

        budget_monitor.check_budget_usage(agent_id, used_budget, total_budget)

        # Verify no alerts
        alerts = notification_manager.get_alerts(agent_id)
        assert len(alerts) == 0
        assert agent_id not in budget_monitor._alerted_agents["warning"]
        assert agent_id not in budget_monitor._alerted_agents["critical"]

    def test_alert_clearing(self, budget_monitor, notification_manager):
        """Test that alerts are cleared when usage drops below thresholds."""
        agent_id = "test_agent"
        total_budget = Decimal("100")

        # First trigger a warning
        used_budget = Decimal("80")  # 80% usage
        budget_monitor.check_budget_usage(agent_id, used_budget, total_budget)
        assert agent_id in budget_monitor._alerted_agents["warning"]

        # Then drop below threshold
        used_budget = Decimal("70")  # 70% usage
        budget_monitor.check_budget_usage(agent_id, used_budget, total_budget)
        assert agent_id not in budget_monitor._alerted_agents["warning"]

    def test_get_budget_status(self, budget_monitor):
        """Test retrieving budget status for an agent."""
        agent_id = "test_agent"
        total_budget = Decimal("100")
        used_budget = Decimal("75")  # 75% usage

        # Record budget usage
        budget_monitor.check_budget_usage(agent_id, used_budget, total_budget)

        # Get status
        status = budget_monitor.get_budget_status(agent_id)
        assert status["total_budget"] == total_budget
        assert status["used_budget"] == used_budget
        assert status["usage_ratio"] == Decimal("0.75")
        assert status["warning_alert"] == True
        assert status["critical_alert"] == False

    def test_reset_agent_alerts(self, budget_monitor):
        """Test resetting alerts for an agent."""
        agent_id = "test_agent"
        total_budget = Decimal("100")
        used_budget = Decimal("95")  # 95% usage, triggers critical alert

        # Record budget usage to trigger alerts
        budget_monitor.check_budget_usage(agent_id, used_budget, total_budget)
        assert agent_id in budget_monitor._alerted_agents["critical"]

        # Reset alerts
        budget_monitor.reset_agent_alerts(agent_id)
        assert agent_id not in budget_monitor._alerted_agents["warning"]
        assert agent_id not in budget_monitor._alerted_agents["critical"]

    def test_get_all_budget_statuses(self, budget_monitor):
        """Test retrieving budget statuses for all tracked agents."""
        # Add data for two agents
        budget_monitor.check_budget_usage("agent1", Decimal("50"), Decimal("100"))
        budget_monitor.check_budget_usage("agent2", Decimal("80"), Decimal("100"))

        # Get all statuses
        statuses = budget_monitor.get_all_budget_statuses()
        assert len(statuses) == 2
        assert "agent1" in statuses
        assert "agent2" in statuses
        assert statuses["agent1"]["usage_ratio"] == Decimal("0.5")
        assert statuses["agent2"]["usage_ratio"] == Decimal("0.8")

    def test_clear_all_alerts(self, budget_monitor):
        """Test clearing all alerts."""
        # Add data for two agents to trigger alerts
        budget_monitor.check_budget_usage("agent1", Decimal("80"), Decimal("100"))
        budget_monitor.check_budget_usage("agent2", Decimal("95"), Decimal("100"))

        # Verify alerts were created
        assert "agent1" in budget_monitor._alerted_agents["warning"]
        assert "agent2" in budget_monitor._alerted_agents["critical"]

        # Clear all alerts
        budget_monitor.clear_all_alerts()
        assert len(budget_monitor._alerted_agents["warning"]) == 0
        assert len(budget_monitor._alerted_agents["critical"]) == 0
