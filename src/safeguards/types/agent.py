"""Base Agent interface for agent safety module."""

from abc import ABC, abstractmethod
from typing import Any


class Agent(ABC):
    """Base interface for agents that can be monitored by safety controllers."""

    def __init__(self, name: str, instructions: str | None = None):
        """Initialize the agent.

        Args:
            name: The name of the agent
            instructions: Optional instructions for the agent
        """
        self.name = name
        self.instructions = instructions or ""

    @abstractmethod
    def run(self, **kwargs: Any) -> dict[str, Any]:
        """Run the agent with the given inputs.

        Args:
            **kwargs: Arbitrary keyword arguments for agent execution

        Returns:
            Dict containing the agent's output
        """
        pass

    @property
    def id(self) -> str:
        """Get the unique identifier for this agent.

        By default, just use the agent's name for simplicity and compatibility
        with tests. Subclasses can override this if needed.

        Returns:
            The agent's name as its ID
        """
        return self.name
