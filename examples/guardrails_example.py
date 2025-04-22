"""Example of implementing and using safety guardrails."""

from decimal import Decimal
from typing import Dict, Any, List, Optional
import asyncio
import time
from dataclasses import dataclass

from agent_safety.types.agent import Agent
from agent_safety.guardrails.budget import BudgetGuardrail
from agent_safety.guardrails.resource import ResourceGuardrail
from agent_safety.budget.manager import BudgetManager
from agent_safety.monitoring.resource_monitor import ResourceMonitor
from agent_safety.core.notification_manager import NotificationManager
from agent_safety.types.enums import AlertSeverity


class ExampleAgent(Agent):
    """Example agent implementation."""

    def __init__(self, name: str, cost_per_action: Decimal = Decimal("1.0")):
        super().__init__(name)
        self.cost_per_action = cost_per_action
        self.action_count = 0

    def run(self, **kwargs: Any) -> Dict[str, Any]:
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
    details: Dict[str, Any] = None


class ContentGuardrail:
    """Example custom guardrail that checks content safety."""

    def __init__(self, forbidden_words: List[str]):
        self.forbidden_words = [word.lower() for word in forbidden_words]

    def validate(self, input_text: Optional[str] = None, **kwargs) -> ValidationResult:
        """Validate that input doesn't contain forbidden words."""
        if not input_text:
            input_text = kwargs.get("input", "")

        if not input_text:
            return ValidationResult(
                is_valid=True, message="No input text to validate", details={}
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
            is_valid=True, message="Input content is safe", details={}
        )

    async def run(self, context: Dict[str, Any]) -> Optional[str]:
        """Run function with content safety checks."""
        # Extract input text
        input_text = context.get("input", "")

        # Validate content
        validation_result = self.validate(input_text)

        if not validation_result.is_valid:
            return f"Content safety violation: {validation_result.message}"

        # No issues found
        return None


class CompositeGuardrail:
    """Combines multiple guardrails into one."""

    def __init__(self, guardrails: List[Any]):
        self.guardrails = guardrails

    async def run(self, fn, **kwargs):
        """Run all guardrails in sequence."""
        # Create context for guardrails
        context = kwargs.copy()

        # Check each guardrail
        for guardrail in self.guardrails:
            # For custom guardrails
            if hasattr(guardrail, "validate"):
                validation_result = guardrail.validate(**kwargs)
                if (
                    hasattr(validation_result, "is_valid")
                    and not validation_result.is_valid
                ):
                    raise ValueError(
                        f"Guardrail violation: {validation_result.message}"
                    )

            # For framework guardrails
            elif hasattr(guardrail, "run"):
                error = await guardrail.run(context)
                if error:
                    raise ValueError(f"Guardrail violation: {error}")

        # All guardrails passed, run the function
        return fn(**kwargs)


async def main():
    """Run the guardrails example."""
    print("=== Safety Guardrails Example ===")

    # Create notification manager for alerts
    notification_manager = NotificationManager()

    # Set up notification callback
    def alert_callback(agent_id, alert_type, severity, message):
        print(f"ALERT: [{severity.name}] {message}")

    notification_manager.register_callback("alerts", alert_callback)

    # Create an agent
    agent = ExampleAgent("example_agent", cost_per_action=Decimal("10.0"))

    # Create budget components
    budget_manager = BudgetManager(agent_id=agent.id, initial_budget=Decimal("100.0"))

    # Create resource monitoring
    resource_monitor = ResourceMonitor(
        agent_id=agent.id, thresholds={"cpu_percent": 80, "memory_percent": 70}
    )

    # Create content guardrail
    content_guardrail = ContentGuardrail(
        forbidden_words=["dangerous", "harmful", "illegal"]
    )

    # Create budget guardrail
    budget_guardrail = BudgetGuardrail(budget_manager)

    # Create resource guardrail
    resource_guardrail = ResourceGuardrail(resource_monitor)

    # Create composite guardrail
    composite_guardrail = CompositeGuardrail(
        [content_guardrail, budget_guardrail, resource_guardrail]
    )

    # Test valid input
    try:
        print("\nTest 1: Valid input")
        result = await composite_guardrail.run(
            agent.run, input="Process this normal text safely"
        )
        print(f"Result: {result}")

        # Update budget
        budget_manager.record_cost(result["cost"])
        print(f"Remaining budget: {budget_manager.get_remaining_budget()}")

    except Exception as e:
        print(f"Error: {str(e)}")

    # Test content violation
    try:
        print("\nTest 2: Content violation")
        result = await composite_guardrail.run(
            agent.run, input="Process this dangerous and harmful content"
        )
        print(f"Result: {result}")  # Should not reach here

    except Exception as e:
        print(f"Error: {str(e)}")

    # Test budget violation
    try:
        print("\nTest 3: Budget violation")
        # Reduce budget to force violation
        for _ in range(8):
            budget_manager.record_cost(Decimal("10.0"))

        print(f"Remaining budget: {budget_manager.get_remaining_budget()}")

        result = await composite_guardrail.run(
            agent.run, input="Process this after budget depletion"
        )
        print(f"Result: {result}")  # Should not reach here

    except Exception as e:
        print(f"Error: {str(e)}")

    # Report how guardrails helped
    print("\n=== Guardrails Protection Summary ===")
    print("1. Content Guardrail: Prevented processing of unsafe content")
    print("2. Budget Guardrail: Prevented exceeding allocated budget")
    print("3. Resource Guardrail: Monitored system resource usage")
    print("\nAll guardrails worked together to ensure safe operation.")


if __name__ == "__main__":
    asyncio.run(main())
