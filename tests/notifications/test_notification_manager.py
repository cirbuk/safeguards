"""Tests for notification manager."""

from unittest.mock import patch

import pytest

from safeguards.notifications.manager import (
    NotificationLevel,
    NotificationManager,
)


@pytest.fixture()
def notification_manager() -> NotificationManager:
    """Create a notification manager instance."""
    return NotificationManager()


def test_notify(notification_manager: NotificationManager):
    """Test sending notifications."""
    notification_manager.notify(
        level=NotificationLevel.INFO,
        message="Test message",
        agent_id="test_agent",
        metadata={"test": "data"},
    )

    assert len(notification_manager.notifications) == 1
    notification = notification_manager.notifications[0]
    assert notification.level == NotificationLevel.INFO
    assert notification.message == "Test message"
    assert notification.agent_id == "test_agent"
    assert notification.metadata == {"test": "data"}


def test_get_notifications_all(notification_manager: NotificationManager):
    """Test getting all notifications."""
    notification_manager.notify(level=NotificationLevel.INFO, message="Info message")
    notification_manager.notify(
        level=NotificationLevel.WARNING,
        message="Warning message",
    )
    notification_manager.notify(level=NotificationLevel.ERROR, message="Error message")

    notifications = notification_manager.get_notifications()
    assert len(notifications) == 3


def test_get_notifications_filtered_by_level(notification_manager: NotificationManager):
    """Test getting notifications filtered by level."""
    notification_manager.notify(level=NotificationLevel.INFO, message="Info message")
    notification_manager.notify(
        level=NotificationLevel.WARNING,
        message="Warning message",
    )
    notification_manager.notify(level=NotificationLevel.ERROR, message="Error message")

    notifications = notification_manager.get_notifications(
        level=NotificationLevel.WARNING,
    )
    assert len(notifications) == 1
    assert notifications[0].message == "Warning message"


def test_get_notifications_filtered_by_agent(notification_manager: NotificationManager):
    """Test getting notifications filtered by agent."""
    notification_manager.notify(
        level=NotificationLevel.INFO,
        message="Agent 1 message",
        agent_id="agent1",
    )
    notification_manager.notify(
        level=NotificationLevel.WARNING,
        message="Agent 2 message",
        agent_id="agent2",
    )

    notifications = notification_manager.get_notifications(agent_id="agent1")
    assert len(notifications) == 1
    assert notifications[0].agent_id == "agent1"


def test_clear_notifications_all(notification_manager: NotificationManager):
    """Test clearing all notifications."""
    notification_manager.notify(level=NotificationLevel.INFO, message="Info message")
    notification_manager.notify(
        level=NotificationLevel.WARNING,
        message="Warning message",
    )

    notification_manager.clear_notifications()
    assert len(notification_manager.notifications) == 0


def test_clear_notifications_by_level(notification_manager: NotificationManager):
    """Test clearing notifications by level."""
    notification_manager.notify(level=NotificationLevel.INFO, message="Info message")
    notification_manager.notify(
        level=NotificationLevel.WARNING,
        message="Warning message",
    )

    notification_manager.clear_notifications(level=NotificationLevel.INFO)
    notifications = notification_manager.get_notifications()
    assert len(notifications) == 1
    assert notifications[0].level == NotificationLevel.WARNING


def test_clear_notifications_by_agent(notification_manager: NotificationManager):
    """Test clearing notifications by agent."""
    notification_manager.notify(
        level=NotificationLevel.INFO,
        message="Agent 1 message",
        agent_id="agent1",
    )
    notification_manager.notify(
        level=NotificationLevel.WARNING,
        message="Agent 2 message",
        agent_id="agent2",
    )

    notification_manager.clear_notifications(agent_id="agent1")
    notifications = notification_manager.get_notifications()
    assert len(notifications) == 1
    assert notifications[0].agent_id == "agent2"


@patch("builtins.print")
def test_notification_handlers(mock_print, notification_manager: NotificationManager):
    """Test notification level handlers."""
    # Test INFO handler
    notification_manager.notify(level=NotificationLevel.INFO, message="Info message")
    mock_print.assert_called_with("INFO: Info message")

    # Test WARNING handler
    notification_manager.notify(
        level=NotificationLevel.WARNING,
        message="Warning message",
    )
    mock_print.assert_called_with("WARNING: Warning message")

    # Test ERROR handler
    notification_manager.notify(level=NotificationLevel.ERROR, message="Error message")
    mock_print.assert_called_with("ERROR: Error message")

    # Test CRITICAL handler
    notification_manager.notify(
        level=NotificationLevel.CRITICAL,
        message="Critical message",
    )
    mock_print.assert_called_with("CRITICAL: Critical message")
