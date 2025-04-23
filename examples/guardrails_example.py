"""Example of implementing and using safety guardrails."""

import asyncio
import inspect
import time
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from safeguards.budget.manager import BudgetManager as BaseManager
from safeguards.core.notification_manager import NotificationManager
from safeguards.guardrails.budget import BudgetGuardrail
from safeguards.guardrails.resource import ResourceGuardrail
from safeguards.monitoring.resource_monitor import ResourceMonitor
from safeguards.types import BudgetConfig, ResourceThresholds
from safeguards.types.agent import Agent


# Create a customized BudgetManager that's compatible with our example
class BudgetManager(BaseManager):
    """Extended BudgetManager for the example."""

    def has_sufficient_budget(self, agent_id=None) -> bool:
        """Check if there is sufficient budget for a given cost.

        Args:
            agent_id: Optional agent ID, not used in this simple example

        Returns:
            bool: True if there is sufficient budget, False otherwise
        """
        return self.get_remaining_budget() > Decimal("0")

    def has_exceeded_budget(self, agent_id=None) -> bool:
        """Check if total spending has exceeded the budget.

        Args:
            agent_id: Optional agent ID, not used in this simple example

        Returns:
            bool: True if budget is exceeded, False otherwise
        """
        return self.get_remaining_budget() <= Decimal("0")

    def get_minimum_required(self, agent_id=None) -> Decimal:
        """Get minimum required budget for an agent.

        Args:
            agent_id: Optional agent ID

        Returns:
            Decimal: Minimum budget required
        """
        return Decimal("10.0")


class ExampleAgent(Agent):
    """Example agent implementation."""

    def __init__(self, name: str, cost_per_action: Decimal = Decimal("1.0")):
        super().__init__(name)
        self.cost_per_action = cost_per_action
        self.action_count = 0

    def run(self, **kwargs: Any) -> dict[str, Any]:
        """Run the agent with the given input."""
        # Simulate agent work
        self.action_count += 1
        input_text = kwargs.get("input", "")

        # Simulate processing
        time.sleep(0.5)  # Simulate work

        return {
            "result": f"Processed: {input_text}",
            "action_count": self.action_count,
            "cost": self.cost_per_action,
        }


@dataclass
class ValidationResult:
    """Simple validation result class for custom guardrails."""

    is_valid: bool
    message: str
    details: dict[str, Any] = None


class ContentGuardrail:
    """Example custom guardrail that checks content safety."""

    def __init__(self, forbidden_words: list[str]):
        self.forbidden_words = [word.lower() for word in forbidden_words]

    def validate_input(
        self,
        input_text: str | None = None,
        **kwargs,
    ) -> ValidationResult:
        """Validate that input doesn't contain forbidden words."""
        if not input_text:
            input_text = kwargs.get("input", "")

        if not input_text:
            return ValidationResult(
                is_valid=True,
                message="No input text to validate",
                details={},
            )

        lower_input = input_text.lower()

        # Check for forbidden words
        found_words = []
        for word in self.forbidden_words:
            if word in lower_input:
                found_words.append(word)

        if found_words:
            return ValidationResult(
                is_valid=False,
                message=f"Input contains forbidden words: {', '.join(found_words)}",
                details={"forbidden_words": found_words},
            )

        return ValidationResult(
            is_valid=True,
            message="Input content is safe",
            details={},
        )

    async def run(self, context) -> str | None:
        """Run function with content safety checks.

        Args:
            context: Either a RunContext object or a dictionary with inputs

        Returns:
            Error message or None
        """
        # Extract input text
        if hasattr(context, "inputs"):
            # It's a RunContext
            input_text = context.inputs.get("input", "")
        else:
            # It's a dictionary
            input_text = context.get("input", "")

        # Validate content
        validation_result = self.validate_input(input_text)

        if not validation_result.is_valid:
            return f"Content safety violation: {validation_result.message}"

        # No issues found
        return None

    async def validate(self, context, result) -> str | None:
        """Framework-compatible validation method.

        Validates results to ensure they don't contain any forbidden words.

        Args:
            context: RunContext object
            result: Result from agent execution

        Returns:
            Error message if validation fails, None otherwise
        """
        # In a real scenario, we might want to check the agent's output too
        if isinstance(result, dict) and "result" in result:
            # Validate the result string
            validation_result = self.validate_input(result["result"])
            if not validation_result.is_valid:
                return f"Output content violation: {validation_result.message}"

        return None


class CompositeGuardrail:
    """Combines multiple guardrails into one."""

    def __init__(self, guardrails: list[Any]):
        self.guardrails = guardrails

    async def run(self, fn, **kwargs):
        """Run all guardrails in sequence."""
        # Create context for guardrails
        kwargs.copy()

        # For framework guardrails, we need to create a proper RunContext
        # Create a mock agent for demonstration purposes
        from safeguards.types.guardrail import RunContext

        # Get the agent from kwargs
        agent = kwargs.get("agent")

        # Make sure we have a valid agent
        if agent is None:
            msg = "An agent instance must be provided to run guardrails"
            raise ValueError(msg)

        # Create a RunContext for framework guardrails
        run_context = RunContext(agent=agent, inputs=kwargs, metadata={})

        # Check each guardrail
        for guardrail in self.guardrails:
            # For custom guardrails with only validate_input method (not framework-compatible)
            if hasattr(guardrail, "validate_input") and not hasattr(guardrail, "run"):
                validation_result = guardrail.validate_input(**kwargs)
                if hasattr(validation_result, "is_valid") and not validation_result.is_valid:
                    msg = f"Guardrail violation: {validation_result.message}"
                    raise ValueError(
                        msg,
                    )

            # For framework-compatible guardrails
            elif hasattr(guardrail, "run"):
                error = await guardrail.run(run_context)
                if error:
                    msg = f"Guardrail violation: {error}"
                    raise ValueError(msg)

        # All guardrails passed, run the function
        # Check if the function is async
        if inspect.iscoroutinefunction(fn):
            result = await fn(**kwargs)
        else:
            result = fn(**kwargs)

        # Validate result with framework guardrails
        for guardrail in self.guardrails:
            if hasattr(guardrail, "validate") and hasattr(guardrail, "run"):
                error = await guardrail.validate(run_context, result)
                if error:
                    msg = f"Result validation failed: {error}"
                    raise ValueError(msg)

        return result


async def main():
    """Run the guardrails example."""
    print("=== Safety Guardrails Example ===")

    # Create notification manager for alerts
    NotificationManager()

    # Set up notification callback
    def alert_callback(agent_id, alert_type, severity, message):
        print(f"ALERT: [{severity.name}] {message}")

    # Create an agent
    agent = ExampleAgent("example_agent", cost_per_action=Decimal("10.0"))

    # Create budget components
    budget_config = BudgetConfig(
        total_budget=Decimal("100.0"),
        hourly_limit=Decimal("50.0"),
        daily_limit=Decimal("100.0"),
        warning_threshold=75.0,
    )
    budget_manager = BudgetManager(config=budget_config)

    # Create resource monitoring
    resource_thresholds = ResourceThresholds(cpu_percent=80.0, memory_percent=70.0)
    resource_monitor = ResourceMonitor(thresholds=resource_thresholds)

    # Create content guardrail
    content_guardrail = ContentGuardrail(
        forbidden_words=["dangerous", "harmful", "illegal"],
    )

    # Create budget guardrail
    budget_guardrail = BudgetGuardrail(budget_manager)

    # Create resource guardrail
    resource_guardrail = ResourceGuardrail(resource_monitor)

    # Create composite guardrail
    composite_guardrail = CompositeGuardrail(
        [content_guardrail, budget_guardrail, resource_guardrail],
    )

    # Test valid input
    try:
        print("\nTest 1: Valid input")
        result = await composite_guardrail.run(
            agent.run,
            agent=agent,
            input="Process this normal text safely",
        )
        print(f"Result: {result}")

        # Update budget
        budget_manager.record_cost(result["cost"])
        print(f"Remaining budget: {budget_manager.get_remaining_budget()}")

    except Exception as e:
        print(f"Error: {e!s}")

    # Test content violation
    try:
        print("\nTest 2: Content violation")
        result = await composite_guardrail.run(
            agent.run,
            agent=agent,
            input="Process this dangerous and harmful content",
        )
        print(f"Result: {result}")  # Should not reach here

    except Exception as e:
        print(f"Error: {e!s}")

    # Test budget violation
    try:
        print("\nTest 3: Budget violation (simulated)")
        # Deplete the budget
        budget_manager.record_cost(budget_manager.get_remaining_budget())
        print(
            f"Remaining budget after depletion: {budget_manager.get_remaining_budget()}",
        )

        # This should now fail due to budget constraint
        result = await composite_guardrail.run(
            agent.run,
            agent=agent,
            input="This should fail due to budget constraints",
        )
        print(f"Result: {result}")  # Should not reach here

    except Exception as e:
        print(f"Error: {e!s}")

    # Print a summary
    print("\n=== Guardrails Demonstration Summary ===")
    print("1. Content Guardrail: Prevents processing of unsafe content")
    print("2. Budget Guardrail: Ensures operations stay within budget limits")
    print("3. Resource Guardrail: Monitors system resource usage")
    print("\nAll guardrails worked together to provide a safety framework.")


if __name__ == "__main__":
    asyncio.run(main())
