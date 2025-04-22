"""
Swarm management for coordinated agent safety controls.

WARNING: EXPERIMENTAL MODULE
This module is experimental and not ready for production use. The API is subject to
change without notice. Use at your own risk.

For multi-agent coordination in production, please use BudgetCoordinator directly
as shown in examples/multi_agent.py.
"""

from .config import SwarmConfig
from .controller import SwarmController

__all__ = ["SwarmConfig", "SwarmController"]
