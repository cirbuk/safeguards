"""Base Guardrail interface and related types for agent safety module."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from .agent import Agent


@dataclass
class RunContext:
    """Context for a single agent run."""

    agent: Agent
    inputs: dict[str, Any]
    metadata: dict[str, Any]


@dataclass
class ResourceUsage:
    """Resource usage data for agent operations."""

    cpu_usage: float
    memory_usage: float
    disk_usage: float
    network_usage: float
    timestamp: datetime = field(default_factory=datetime.now)
    agent_id: str | None = None
    duration: float | None = None


class Guardrail(ABC):
    """Base interface for agent guardrails."""

    @abstractmethod
    async def run(self, context: RunContext) -> str | None:
        """Run guardrail checks before agent execution.

        Args:
            context: Run context containing agent and execution info

        Returns:
            Error message if checks fail, None otherwise
        """
        pass

    @abstractmethod
    async def validate(self, context: RunContext, result: Any) -> str | None:
        """Validate results after agent execution.

        Args:
            context: Run context containing agent and execution info
            result: Result from agent execution

        Returns:
            Error message if validation fails, None otherwise
        """
        pass
