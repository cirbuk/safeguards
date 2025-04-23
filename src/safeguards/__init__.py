"""Safeguards - Core package for AI agent safety."""

from .base.budget import BudgetManager
from .base.monitoring import ResourceMonitor
from .core.notification_manager import NotificationManager
from .core.safety_controller import SafetyController
from .types import SafetyAlert, SafetyConfig, SafetyMetrics

# Version information
__version__ = "0.1.0"  # This should match the version in pyproject.toml

__all__ = [
    "BudgetManager",
    "NotificationManager",
    "ResourceMonitor",
    "SafetyAlert",
    "SafetyConfig",
    "SafetyController",
    "SafetyMetrics",
]
