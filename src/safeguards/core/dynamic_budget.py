"""Dynamic budget management system for FounderX.

This module provides functionality for dynamic budget allocation and management:
- Usage-based budget allocation
- Priority-based distribution
- Automatic budget reallocation
- Budget adjustment triggers
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum, auto

from .alert_types import Alert, AlertSeverity
from .notification_manager import NotificationManager


class AgentPriority(Enum):
    """Priority levels for agents."""

    CRITICAL = auto()  # Mission-critical agents (e.g., core monitoring)
    HIGH = auto()  # High-priority business functions
    MEDIUM = auto()  # Standard operations
    LOW = auto()  # Non-critical background tasks
    MINIMAL = auto()  # Debug/testing agents


class BudgetAdjustmentTrigger(Enum):
    """Triggers for budget adjustments."""

    USAGE_THRESHOLD = auto()  # Usage reaches certain threshold
    TIME_WINDOW = auto()  # Regular time-based adjustment
    PRIORITY_CHANGE = auto()  # Agent priority changes
    MANUAL = auto()  # Manual adjustment request
    EMERGENCY = auto()  # Emergency reallocation


@dataclass
class AgentBudgetProfile:
    """Budget profile for an individual agent."""

    agent_id: str
    priority: AgentPriority
    base_allocation: Decimal
    min_allocation: Decimal
    max_allocation: Decimal | None = None
    current_allocation: Decimal | None = None
    usage_history: list[Decimal] = field(default_factory=list)
    last_adjustment: datetime = field(default_factory=datetime.now)
    adjustment_cooldown: timedelta = field(default=timedelta(hours=1))


@dataclass
class BudgetPool:
    """Shared budget pool for multiple agents."""

    pool_id: str
    total_budget: Decimal
    priority: int = 0
    allocated_budget: Decimal = field(default=Decimal("0"))
    reserved_budget: Decimal = field(default=Decimal("0"))
    agent_allocations: dict[str, Decimal] = field(default_factory=dict)
    agents: set[str] = field(default_factory=set)
    priority_weights: dict[AgentPriority, float] = field(
        default_factory=lambda: {
            AgentPriority.CRITICAL: 1.0,
            AgentPriority.HIGH: 0.8,
            AgentPriority.MEDIUM: 0.6,
            AgentPriority.LOW: 0.4,
            AgentPriority.MINIMAL: 0.2,
        },
    )
    used_budget: Decimal = field(default=Decimal("0"))
    last_update: datetime = field(default_factory=datetime.now)

    def __post_init__(self):
        """Initialize attributes after the dataclass constructor."""
        # For backward compatibility with code that expects a name attribute
        self._name = self.pool_id

    @property
    def name(self) -> str:
        """Get pool name (alias for pool_id for compatibility)."""
        return self.pool_id

    @property
    def id(self) -> str:
        """Get pool ID (alias for pool_id for compatibility)."""
        return self.pool_id

    @property
    def remaining_budget(self) -> Decimal:
        """Get remaining budget in pool."""
        return self.total_budget - self.used_budget

    @property
    def initial_budget(self) -> Decimal:
        """For backward compatibility - Return total budget."""
        return self.total_budget


class DynamicBudgetManager:
    """Manages dynamic budget allocation across agents."""

    def __init__(
        self,
        notification_manager: NotificationManager,
        default_pool_budget: Decimal,
        adjustment_interval: timedelta = timedelta(hours=1),
        usage_threshold: float = 0.8,
        reserve_ratio: float = 0.1,
    ):
        """Initialize the budget manager.

        Args:
            notification_manager: For sending budget-related alerts
            default_pool_budget: Total budget for the default pool
            adjustment_interval: How often to check for adjustments
            usage_threshold: Usage % that triggers reallocation
            reserve_ratio: Ratio of budget to keep in reserve
        """
        self.notification_manager = notification_manager
        self.adjustment_interval = adjustment_interval
        self.usage_threshold = usage_threshold
        self.reserve_ratio = reserve_ratio

        # Initialize budget pools
        self.budget_pools: dict[str, BudgetPool] = {
            "default": BudgetPool(
                pool_id="default",
                total_budget=default_pool_budget,
                reserved_budget=default_pool_budget * Decimal(str(reserve_ratio)),
            ),
        }

        # Track agent profiles
        self.agent_profiles: dict[str, AgentBudgetProfile] = {}

        # Track pool memberships
        self.agent_pool_mapping: dict[str, str] = {}

        # Last adjustment timestamp
        self.last_global_adjustment = datetime.now()

    def register_agent(
        self,
        agent_id: str,
        priority: AgentPriority,
        base_allocation: Decimal,
        min_allocation: Decimal,
        max_allocation: Decimal | None = None,
        pool_id: str = "default",
    ) -> None:
        """Register a new agent for budget management.

        Args:
            agent_id: Unique identifier for the agent
            priority: Agent's priority level
            base_allocation: Initial budget allocation
            min_allocation: Minimum required budget
            max_allocation: Maximum allowed budget (optional)
            pool_id: Budget pool to assign agent to
        """
        if pool_id not in self.budget_pools:
            msg = f"Budget pool {pool_id} does not exist"
            raise ValueError(msg)

        if agent_id in self.agent_profiles:
            msg = f"Agent {agent_id} is already registered"
            raise ValueError(msg)

        # Create agent profile
        profile = AgentBudgetProfile(
            agent_id=agent_id,
            priority=priority,
            base_allocation=base_allocation,
            min_allocation=min_allocation,
            max_allocation=max_allocation,
            current_allocation=base_allocation,
        )

        # Update tracking
        self.agent_profiles[agent_id] = profile
        self.agent_pool_mapping[agent_id] = pool_id

        # Update pool allocation
        pool = self.budget_pools[pool_id]
        pool.agent_allocations[agent_id] = base_allocation
        pool.allocated_budget += base_allocation

        # Validate pool budget
        if pool.allocated_budget > (pool.total_budget - pool.reserved_budget):
            self._handle_overallocation(pool)

    def update_agent_usage(self, agent_id: str, usage: Decimal) -> None:
        """Update an agent's usage history and trigger adjustments if needed.

        Args:
            agent_id: Agent to update
            usage: Current usage amount
        """
        if agent_id not in self.agent_profiles:
            msg = f"Unknown agent {agent_id}"
            raise ValueError(msg)

        profile = self.agent_profiles[agent_id]
        profile.usage_history.append(usage)

        # Keep last 24 hours of history
        datetime.now() - timedelta(hours=24)
        profile.usage_history = profile.usage_history[-24:]

        # Check if adjustment is needed
        if self._should_adjust_budget(profile, usage):
            self._adjust_agent_budget(
                agent_id,
                trigger=BudgetAdjustmentTrigger.USAGE_THRESHOLD,
            )

    def adjust_pool_budget(
        self,
        pool_id: str,
        new_total: Decimal,
        trigger: BudgetAdjustmentTrigger = BudgetAdjustmentTrigger.MANUAL,
    ) -> None:
        """Adjust the total budget for a pool.

        Args:
            pool_id: Pool to adjust
            new_total: New total budget
            trigger: Reason for adjustment
        """
        if pool_id not in self.budget_pools:
            msg = f"Unknown pool {pool_id}"
            raise ValueError(msg)

        pool = self.budget_pools[pool_id]
        old_total = pool.total_budget
        pool.total_budget = new_total
        pool.reserved_budget = new_total * Decimal(str(self.reserve_ratio))

        # Reallocate if budget decreased
        if new_total < old_total:
            self._reallocate_pool_budget(pool, trigger)

    def _should_adjust_budget(
        self,
        profile: AgentBudgetProfile,
        current_usage: Decimal,
    ) -> bool:
        """Check if an agent's budget should be adjusted.

        Args:
            profile: Agent profile to check
            current_usage: Latest usage amount

        Returns:
            True if budget adjustment is needed
        """
        # Check cooldown period
        if datetime.now() - profile.last_adjustment < profile.adjustment_cooldown:
            return False

        # Check usage threshold
        if current_usage >= profile.current_allocation * Decimal(
            str(self.usage_threshold),
        ):
            return True

        # Check if significantly under-utilizing
        avg_usage = sum(profile.usage_history) / len(profile.usage_history)
        return avg_usage <= profile.current_allocation * Decimal("0.5")

    def _adjust_agent_budget(
        self,
        agent_id: str,
        trigger: BudgetAdjustmentTrigger,
    ) -> None:
        """Adjust an individual agent's budget allocation.

        Args:
            agent_id: Agent to adjust
            trigger: Reason for adjustment
        """
        profile = self.agent_profiles[agent_id]
        pool_id = self.agent_pool_mapping[agent_id]
        pool = self.budget_pools[pool_id]

        # Calculate new allocation based on usage and priority
        avg_usage = sum(profile.usage_history) / len(profile.usage_history)
        pool.priority_weights[profile.priority]

        # Adjust allocation up or down based on usage
        if avg_usage >= profile.current_allocation * Decimal(str(self.usage_threshold)):
            # Increase allocation
            requested_increase = avg_usage * Decimal("1.2") - profile.current_allocation
            available_budget = pool.total_budget - pool.allocated_budget

            if available_budget >= requested_increase:
                new_allocation = min(
                    profile.current_allocation + requested_increase,
                    profile.max_allocation or Decimal("inf"),
                )
            else:
                # Not enough budget - try to reallocate
                self._reallocate_pool_budget(pool, trigger)
                # Recalculate available budget
                available_budget = pool.total_budget - pool.allocated_budget
                new_allocation = min(
                    profile.current_allocation + available_budget,
                    profile.max_allocation or Decimal("inf"),
                )
        else:
            # Decrease allocation if significantly under-utilizing
            new_allocation = max(avg_usage * Decimal("1.2"), profile.min_allocation)

        # Update allocations
        old_allocation = profile.current_allocation
        profile.current_allocation = new_allocation
        pool.agent_allocations[agent_id] = new_allocation
        pool.allocated_budget += new_allocation - old_allocation

        # Update adjustment timestamp
        profile.last_adjustment = datetime.now()

        # Send notification
        self._send_adjustment_alert(
            agent_id=agent_id,
            old_allocation=old_allocation,
            new_allocation=new_allocation,
            trigger=trigger,
        )

    def _reallocate_pool_budget(
        self,
        pool: BudgetPool,
        trigger: BudgetAdjustmentTrigger,
    ) -> None:
        """Reallocate budget within a pool based on priorities.

        Args:
            pool: Pool to reallocate
            trigger: Reason for reallocation
        """
        # Get agents in this pool
        pool_agents = [
            agent_id
            for agent_id, pool_id in self.agent_pool_mapping.items()
            if pool_id == pool.pool_id
        ]

        # Calculate priority-weighted allocations
        total_weight = sum(
            pool.priority_weights[self.agent_profiles[agent_id].priority]
            for agent_id in pool_agents
        )

        available_budget = pool.total_budget - pool.reserved_budget

        # Allocate based on priority weights
        for agent_id in pool_agents:
            profile = self.agent_profiles[agent_id]
            weight = pool.priority_weights[profile.priority]

            # Calculate new allocation
            new_allocation = Decimal(str(weight / total_weight)) * available_budget

            # Apply min/max constraints
            new_allocation = max(new_allocation, profile.min_allocation)
            if profile.max_allocation:
                new_allocation = min(new_allocation, profile.max_allocation)

            # Update allocations
            old_allocation = profile.current_allocation
            profile.current_allocation = new_allocation
            pool.agent_allocations[agent_id] = new_allocation

            # Send notification
            self._send_adjustment_alert(
                agent_id=agent_id,
                old_allocation=old_allocation,
                new_allocation=new_allocation,
                trigger=trigger,
            )

        # Update pool allocated budget
        pool.allocated_budget = sum(pool.agent_allocations.values())

    def _handle_overallocation(self, pool: BudgetPool) -> None:
        """Handle pool overallocation by forcing reallocation.

        Args:
            pool: Overallocated pool
        """
        self._send_alert(
            title=f"Budget Pool {pool.pool_id} Overallocated",
            description=(
                f"Pool budget: {pool.total_budget}\n"
                f"Allocated: {pool.allocated_budget}\n"
                f"Reserved: {pool.reserved_budget}\n"
                "Forcing reallocation..."
            ),
            severity=AlertSeverity.ERROR,
        )

        self._reallocate_pool_budget(pool, BudgetAdjustmentTrigger.EMERGENCY)

    def _send_adjustment_alert(
        self,
        agent_id: str,
        old_allocation: Decimal,
        new_allocation: Decimal,
        trigger: BudgetAdjustmentTrigger,
    ) -> None:
        """Send alert about budget adjustment.

        Args:
            agent_id: Adjusted agent
            old_allocation: Previous budget
            new_allocation: New budget
            trigger: Reason for adjustment
        """
        change_pct = ((new_allocation - old_allocation) / old_allocation) * 100

        self._send_alert(
            title=f"Agent {agent_id} Budget Adjusted",
            description=(
                f"Trigger: {trigger.name}\n"
                f"Old allocation: {old_allocation}\n"
                f"New allocation: {new_allocation}\n"
                f"Change: {change_pct:+.1f}%"
            ),
            severity=AlertSeverity.INFO if change_pct >= 0 else AlertSeverity.WARNING,
            metadata={
                "agent_id": agent_id,
                "trigger": trigger.name,
                "old_allocation": str(old_allocation),
                "new_allocation": str(new_allocation),
                "change_percent": float(change_pct),
            },
        )

    def _send_alert(
        self,
        title: str,
        description: str,
        severity: AlertSeverity,
        metadata: dict | None = None,
    ) -> None:
        """Send an alert via notification manager.

        Args:
            title: Alert title
            description: Alert description
            severity: Alert severity level
            metadata: Optional metadata dict
        """
        alert = Alert(
            title=title,
            description=description,
            severity=severity,
            metadata=metadata or {},
        )
        self.notification_manager.send_alert(alert)
