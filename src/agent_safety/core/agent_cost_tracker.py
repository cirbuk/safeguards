"""Agent-specific cost tracking and allocation.

This module provides functionality for:
- Tracking costs per agent
- Managing agent-specific budgets
- Allocating costs across different operations
- Monitoring agent resource utilization
"""

from dataclasses import dataclass
from decimal import Decimal
from typing import Dict, List, Optional

from founderx.core.cost_management import CostType
from founderx.core.cost_tracker import CostTracker
from founderx.core.token_tracker import TokenUsageTracker
from founderx.core.api_tracker import APIUsageTracker


@dataclass
class AgentBudget:
    """Budget configuration for an agent."""

    total_budget: Decimal
    daily_budget: Optional[Decimal] = None
    hourly_budget: Optional[Decimal] = None
    cost_types: Optional[List[CostType]] = None  # If None, all cost types allowed


class AgentCostTracker:
    """Tracks and manages costs for individual agents."""

    def __init__(
        self,
        agent_id: str,
        cost_tracker: CostTracker,
        token_tracker: TokenUsageTracker,
        api_tracker: APIUsageTracker,
        budget: Optional[AgentBudget] = None,
    ):
        """Initialize agent cost tracker.

        Args:
            agent_id: Unique identifier for the agent
            cost_tracker: Global cost tracker instance
            token_tracker: Token usage tracker instance
            api_tracker: API usage tracker instance
            budget: Optional budget configuration for the agent
        """
        self.agent_id = agent_id
        self.cost_tracker = cost_tracker
        self.token_tracker = token_tracker
        self.api_tracker = api_tracker
        self.budget = budget

        # Initialize cost tracking
        self.costs: Dict[CostType, Decimal] = {ct: Decimal(0) for ct in CostType}
        self.total_cost = Decimal(0)

        # Initialize operation tracking
        self.operation_costs: Dict[str, Dict[CostType, Decimal]] = {}

    def start_operation(self, operation_id: str) -> None:
        """Start tracking costs for a new operation.

        Args:
            operation_id: Unique identifier for the operation
        """
        if operation_id in self.operation_costs:
            raise ValueError(f"Operation {operation_id} already exists")

        self.operation_costs[operation_id] = {ct: Decimal(0) for ct in CostType}

    def end_operation(self, operation_id: str) -> Dict[CostType, Decimal]:
        """End tracking costs for an operation.

        Args:
            operation_id: Unique identifier for the operation

        Returns:
            Dictionary of costs by type for the operation

        Raises:
            ValueError: If operation_id doesn't exist
        """
        if operation_id not in self.operation_costs:
            raise ValueError(f"Operation {operation_id} not found")

        costs = self.operation_costs.pop(operation_id)
        return costs

    def add_cost(
        self,
        cost_type: CostType,
        amount: Decimal,
        operation_id: Optional[str] = None,
    ) -> bool:
        """Add a cost for the agent.

        Args:
            cost_type: Type of cost being added
            amount: Cost amount
            operation_id: Optional operation to attribute cost to

        Returns:
            True if cost was added successfully, False if budget would be exceeded

        Raises:
            ValueError: If operation_id provided but not started
        """
        # Check if cost type is allowed for this agent
        if (
            self.budget
            and self.budget.cost_types
            and cost_type not in self.budget.cost_types
        ):
            return False

        # Check budget limits
        if not self._check_budget_limits(amount):
            return False

        # Add to global cost tracker
        self.cost_tracker.add_cost(cost_type, amount)

        # Update agent-specific tracking
        self.costs[cost_type] += amount
        self.total_cost += amount

        # Update operation tracking if applicable
        if operation_id:
            if operation_id not in self.operation_costs:
                raise ValueError(f"Operation {operation_id} not started")
            self.operation_costs[operation_id][cost_type] += amount

        return True

    def track_completion_tokens(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        operation_id: Optional[str] = None,
    ) -> bool:
        """Track completion token usage and costs.

        Args:
            model: Name of the model used
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            operation_id: Optional operation to attribute cost to

        Returns:
            True if tracking successful, False if budget would be exceeded
        """
        cost = self.token_tracker.track_completion(
            model, input_tokens, output_tokens, record_cost=False
        )
        return self.add_cost(CostType.COMPUTE, cost, operation_id)

    def track_embedding_tokens(
        self,
        model: str,
        num_tokens: int,
        operation_id: Optional[str] = None,
    ) -> bool:
        """Track embedding token usage and costs.

        Args:
            model: Name of the model used
            num_tokens: Number of tokens embedded
            operation_id: Optional operation to attribute cost to

        Returns:
            True if tracking successful, False if budget would be exceeded
        """
        cost = self.token_tracker.track_embedding(model, num_tokens, record_cost=False)
        return self.add_cost(CostType.VECTOR_STORE, cost, operation_id)

    def track_api_call(
        self,
        api: str,
        success: bool = True,
        operation_id: Optional[str] = None,
    ) -> bool:
        """Track API call and costs.

        Args:
            api: Name of the API service
            success: Whether the call was successful
            operation_id: Optional operation to attribute cost to

        Returns:
            True if tracking successful, False if budget would be exceeded
        """
        # First check if call is allowed by rate limits
        can_call, _ = self.api_tracker.can_make_call(api)
        if not can_call:
            return False

        # Get cost per call
        cost = self.api_tracker.api_configs[api].cost_per_call

        # Check if we can afford it
        if not self._check_budget_limits(cost):
            return False

        # Record the call without cost tracking (we'll do it ourselves)
        success, _ = self.api_tracker.record_call(api, success, record_cost=False)
        if not success:
            return False

        # Add the cost through our tracking
        cost_type = getattr(CostType, f"{api.upper()}_API")
        return self.add_cost(cost_type, cost, operation_id)

    def get_costs(self, operation_id: Optional[str] = None) -> Dict[CostType, Decimal]:
        """Get costs by type.

        Args:
            operation_id: Optional operation to get costs for

        Returns:
            Dictionary of costs by type

        Raises:
            ValueError: If operation_id provided but not found
        """
        if operation_id:
            if operation_id not in self.operation_costs:
                raise ValueError(f"Operation {operation_id} not found")
            return self.operation_costs[operation_id]
        return self.costs

    def get_total_cost(self, operation_id: Optional[str] = None) -> Decimal:
        """Get total cost.

        Args:
            operation_id: Optional operation to get total cost for

        Returns:
            Total cost

        Raises:
            ValueError: If operation_id provided but not found
        """
        if operation_id:
            if operation_id not in self.operation_costs:
                raise ValueError(f"Operation {operation_id} not found")
            return sum(self.operation_costs[operation_id].values())
        return self.total_cost

    def _check_budget_limits(self, amount: Decimal) -> bool:
        """Check if a cost would exceed budget limits.

        Args:
            amount: Cost amount to check

        Returns:
            True if cost is within budget, False otherwise
        """
        if not self.budget:
            return True

        # Check total budget
        if self.total_cost + amount > self.budget.total_budget:
            return False

        # Check daily budget
        if self.budget.daily_budget:
            daily_costs = self.cost_tracker.get_costs_in_window("daily")
            if sum(daily_costs.values()) + amount > self.budget.daily_budget:
                return False

        # Check hourly budget
        if self.budget.hourly_budget:
            hourly_costs = self.cost_tracker.get_costs_in_window("hourly")
            if sum(hourly_costs.values()) + amount > self.budget.hourly_budget:
                return False

        return True
