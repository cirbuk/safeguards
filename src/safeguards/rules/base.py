"""Base classes and interfaces for the safety rules system."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

from ..base.guardrails import GuardrailViolation, ValidationResult


class RulePriority(Enum):
    """Priority levels for safety rules."""

    CRITICAL = auto()  # Must pass, blocks execution
    HIGH = auto()  # Should pass, may block based on config
    MEDIUM = auto()  # Warning if fails
    LOW = auto()  # Informational only


@dataclass
class RuleContext:
    """Context for rule execution."""

    input_data: Any
    metadata: dict[str, Any] = field(default_factory=dict)
    parent_rules: set[str] = field(default_factory=set)


class SafetyRule(ABC):
    """Base class for safety rules."""

    def __init__(
        self,
        rule_id: str,
        priority: RulePriority,
        description: str,
        dependencies: list[type["SafetyRule"]] | None = None,
    ):
        """Initialize safety rule.

        Args:
            rule_id: Unique identifier for this rule
            priority: Rule priority level
            description: Human readable description
            dependencies: Other rules that must run before this one
        """
        self.rule_id = rule_id
        self.priority = priority
        self.description = description
        self.dependencies = dependencies or []

    @abstractmethod
    def evaluate(self, context: RuleContext) -> ValidationResult:
        """Evaluate this rule against the provided context.

        Args:
            context: Rule evaluation context

        Returns:
            Validation result with any violations
        """
        pass

    def get_dependency_ids(self) -> set[str]:
        """Get IDs of all dependency rules.

        Returns:
            Set of rule IDs this rule depends on
        """
        return {dep().rule_id for dep in self.dependencies}


class RuleChain:
    """Manages execution of multiple rules in order."""

    def __init__(self):
        """Initialize rule chain."""
        self._rules: dict[str, SafetyRule] = {}
        self._sorted_rules: list[SafetyRule] | None = None

    def add_rule(self, rule: SafetyRule) -> None:
        """Add a rule to the chain.

        Args:
            rule: Rule to add
        """
        self._rules[rule.rule_id] = rule
        self._sorted_rules = None  # Invalidate cached order

    def remove_rule(self, rule_id: str) -> None:
        """Remove a rule from the chain.

        Args:
            rule_id: ID of rule to remove
        """
        self._rules.pop(rule_id, None)
        self._sorted_rules = None

    def _sort_rules(self) -> list[SafetyRule]:
        """Sort rules by priority and dependencies.

        Returns:
            Sorted list of rules
        """
        if self._sorted_rules is not None:
            return self._sorted_rules

        # Build dependency graph
        graph: dict[str, set[str]] = {}
        for rule in self._rules.values():
            graph[rule.rule_id] = rule.get_dependency_ids()

        # Topological sort
        sorted_ids: list[str] = []
        visited: set[str] = set()
        temp_visited: set[str] = set()

        def visit(rule_id: str) -> None:
            if rule_id in temp_visited:
                msg = f"Circular dependency detected involving rule {rule_id}"
                raise ValueError(
                    msg,
                )
            if rule_id in visited:
                return

            temp_visited.add(rule_id)
            for dep_id in graph[rule_id]:
                if dep_id not in self._rules:
                    msg = f"Missing dependency {dep_id} for rule {rule_id}"
                    raise ValueError(msg)
                visit(dep_id)
            temp_visited.remove(rule_id)
            visited.add(rule_id)
            sorted_ids.append(rule_id)

        # Sort by dependencies
        for rule_id in self._rules:
            if rule_id not in visited:
                visit(rule_id)

        # Sort by priority within dependency constraints
        self._sorted_rules = sorted(
            [self._rules[rule_id] for rule_id in sorted_ids],
            key=lambda r: r.priority.value,
        )
        return self._sorted_rules

    def evaluate(
        self,
        input_data: Any,
        metadata: dict[str, Any] | None = None,
    ) -> ValidationResult:
        """Evaluate all rules in chain.

        Args:
            input_data: Input data to validate
            metadata: Optional metadata for rule context

        Returns:
            Combined validation result
        """
        sorted_rules = self._sort_rules()
        all_violations: list[GuardrailViolation] = []
        executed_rules: set[str] = set()

        for rule in sorted_rules:
            context = RuleContext(
                input_data=input_data,
                metadata=metadata or {},
                parent_rules=executed_rules.copy(),
            )

            result = rule.evaluate(context)
            executed_rules.add(rule.rule_id)

            if not result.is_valid:
                all_violations.extend(result.violations)
                # Stop on critical rule failure
                if rule.priority == RulePriority.CRITICAL:
                    break

        return ValidationResult(
            is_valid=len(all_violations) == 0,
            violations=all_violations,
        )
