"""Configuration for swarm management."""

from dataclasses import dataclass, field
from decimal import Decimal

from agents.guardrails import Guardrail

from ..types import SafetyConfig


@dataclass
class SwarmConfig(SafetyConfig):
    """Configuration for swarm management extending safety config."""

    # Swarm-specific settings
    max_concurrent_agents: int = 10
    min_agent_budget: Decimal = Decimal("10.0")
    budget_rebalance_interval: int = 300  # seconds

    # Dynamic scaling
    enable_auto_scaling: bool = True
    scale_up_threshold: float = 80.0  # Percentage of budget/resource usage
    scale_down_threshold: float = 20.0

    # Coordination
    coordination_strategy: str = "COOPERATIVE"  # COOPERATIVE, COMPETITIVE, HIERARCHICAL
    resource_allocation_strategy: str = "FAIR"  # FAIR, PRIORITY_BASED, DYNAMIC

    # Failover settings
    enable_failover: bool = True
    failover_retry_limit: int = 3

    # Swarm guardrails
    total_cpu_limit: float = 90.0  # Percentage
    total_memory_limit: float = 90.0
    enable_load_balancing: bool = True

    # Additional guardrails
    custom_guardrails: list[type[Guardrail]] = field(default_factory=list)
