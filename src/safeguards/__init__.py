"""Safeguards - Core package for AI agent safety."""

import os

from .core.safety_controller import SafetyController
from .types import SafetyConfig, SafetyMetrics, SafetyAlert
from .base.budget import BudgetManager
from .base.monitoring import ResourceMonitor
from .core.notification_manager import NotificationManager

# Version information
__version__ = "0.1.0"  # This should match the version in pyproject.toml

__all__ = [
    "SafetyController",
    "SafetyConfig",
    "SafetyMetrics",
    "SafetyAlert",
    "BudgetManager",
    "ResourceMonitor",
    "NotificationManager",
]
