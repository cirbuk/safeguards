"""Base Guardrail interface and related types for agent safety module."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional
from datetime import datetime

from .agent import Agent


@dataclass
class RunContext:
    """Context for a single agent run."""

    agent: Agent
    inputs: Dict[str, Any]
    metadata: Dict[str, Any]


@dataclass
class ResourceUsage:
    """Resource usage data for agent operations."""

    cpu_usage: float
    memory_usage: float
    disk_usage: float
    network_usage: float
    timestamp: datetime = field(default_factory=datetime.now)
    agent_id: Optional[str] = None
    duration: Optional[float] = None


class Guardrail(ABC):
    """Base interface for agent guardrails."""

    @abstractmethod
    async def run(self, context: RunContext) -> Optional[str]:
        """Run guardrail checks before agent execution.

        Args:
            context: Run context containing agent and execution info

        Returns:
            Error message if checks fail, None otherwise
        """
        pass

    @abstractmethod
    async def validate(self, context: RunContext, result: Any) -> Optional[str]:
        """Validate results after agent execution.

        Args:
            context: Run context containing agent and execution info
            result: Result from agent execution

        Returns:
            Error message if validation fails, None otherwise
        """
        pass
