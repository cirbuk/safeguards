"""Budget guardrail implementation."""

from typing import Any

from ..budget import BudgetManager
from ..types.guardrail import Guardrail, RunContext


class BudgetGuardrail(Guardrail):
    """Guardrail for monitoring and enforcing budget limits.

    This guardrail:
    1. Tracks budget usage during agent execution
    2. Enforces budget limits
    3. Requests overrides when needed
    4. Alerts on budget thresholds

    Example:
        ```python
        from safeguards import BudgetManager, BudgetGuardrail
        from safeguards.types import Agent

        budget_manager = BudgetManager(total_budget=1000)
        agent = Agent(
            name="Assistant",
            instructions="You are a helpful assistant",
            guardrails=[BudgetGuardrail(budget_manager)]
        )
        ```
    """

    def __init__(self, budget_manager: BudgetManager):
        """Initialize the budget guardrail.

        Args:
            budget_manager: Budget manager instance
        """
        self.budget_manager = budget_manager

    async def run(self, context: RunContext) -> str | None:
        """Run budget checks before agent execution.

        Args:
            context: Run context with agent info

        Returns:
            Error message if budget exceeded, None otherwise
        """
        agent_id = context.agent.id

        # Check if agent has sufficient budget
        if not self.budget_manager.has_sufficient_budget(agent_id):
            # Try to request override
            override = self.budget_manager.request_override(
                agent_id,
                self.budget_manager.get_minimum_required(agent_id),
                f"Insufficient budget for agent {agent_id} to continue execution",
            )

            if override.status == "REJECTED":
                return f"Budget exceeded for agent {agent_id}. Override request rejected."

            if override.status == "PENDING":
                return f"Budget exceeded for agent {agent_id}. Override request pending approval."

        return None

    async def validate(self, context: RunContext, result: Any) -> str | None:
        """Validate budget usage after agent execution.

        Args:
            context: Run context with agent info
            result: Result from agent execution

        Returns:
            Error message if validation fails, None otherwise
        """
        agent_id = context.agent.id

        # Check if execution exceeded budget
        if self.budget_manager.has_exceeded_budget(agent_id):
            return f"Agent {agent_id} exceeded allocated budget during execution."

        return None
