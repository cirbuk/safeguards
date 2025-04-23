"""Default agent implementation for internal use."""

from typing import Any

from safeguards.types.agent import Agent


class DefaultAgent(Agent):
    """Concrete implementation of Agent for internal use.

    This simple agent implementation is used as a default or placeholder
    when a specific agent implementation is not required.
    """

    def __init__(self, name: str, instructions: str | None = None):
        """Initialize the default agent.

        Args:
            name: The name of the agent
            instructions: Optional instructions for the agent
        """
        super().__init__(name=name, instructions=instructions)

    def run(self, **kwargs: Any) -> dict[str, Any]:
        """Simple implementation of the abstract run method.

        Args:
            **kwargs: Arbitrary keyword arguments for agent execution

        Returns:
            Dict containing a simple success response
        """
        return {"success": True}
