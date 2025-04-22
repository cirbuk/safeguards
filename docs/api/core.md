# Core API Reference

This document provides detailed information about the core API components of the Safeguards.

## Agent

The `Agent` class is the base abstraction for all agent implementations.

### Agent Base Class

```python
from safeguards.types.agent import Agent
```

#### Constructor

```python
def __init__(self, name: str) -> None:
    """Initialize an agent.

    Args:
        name: Agent name
    """
```

#### Properties

```python
@property
def id(self) -> str:
    """Get agent ID.

    Returns:
        Unique ID for this agent
    """
```

#### Methods

```python
@abstractmethod
def run(self, **kwargs: Any) -> Dict[str, Any]:
    """Execute agent functionality.

    Args:
        **kwargs: Arbitrary keyword arguments for agent execution

    Returns:
        Dictionary containing execution results
    """
```

### Implementation Example

```python
from decimal import Decimal
from typing import Dict, Any
from safeguards.types.agent import Agent

class MyAgent(Agent):
    def __init__(self, name: str, cost_per_action: Decimal = Decimal("0.1")):
        super().__init__(name)
        self.cost_per_action = cost_per_action
        self.action_count = 0

    def run(self, **kwargs: Any) -> Dict[str, Any]:
        """Execute a task and track cost."""
        self.action_count += 1

        # Perform your agent logic here
        result = {"status": "success", "message": "Task completed"}

        # Return result with cost information
        return {
            "result": result,
            "action_count": self.action_count,
            "cost": self.cost_per_action,
        }
```

## BudgetCoordinator

The `BudgetCoordinator` class manages resource allocation and budget tracking.

```python
from safeguards.core.budget_coordination import BudgetCoordinator
```

### Constructor

```python
def __init__(
    self,
    notification_manager=None,
    initial_pool_budget: Decimal = Decimal("0")
) -> None:
    """Initialize the budget coordinator.

    Args:
        notification_manager: For sending coordination-related alerts
        initial_pool_budget: Initial budget in the pool
    """
```

### Agent Management Methods

```python
def register_agent(
    self,
    name: str,
    initial_budget: Decimal,
    priority: int = 0,
    agent: Optional[Agent] = None,
) -> Agent:
    """Register a new agent.

    Args:
        name: Agent name
        initial_budget: Initial budget allocation
        priority: Agent priority level (1-10)
        agent: Optional existing agent instance to register

    Returns:
        Registered agent

    Raises:
        ValueError: If validation fails
    """

def create_agent(
    self,
    name: str,
    initial_budget: Decimal,
    priority: int = 0
) -> Agent:
    """Create and register a new agent.

    Args:
        name: Name of the agent
        initial_budget: Initial budget allocation
        priority: Priority level (default 0)

    Returns:
        Newly created agent

    Raises:
        ValueError: If validation fails
    """

def get_agent(self, agent_id: str) -> Agent:
    """Get an agent.

    Args:
        agent_id: ID of agent to get

    Returns:
        Agent instance
    """

def unregister_agent(self, agent_id: str) -> None:
    """Unregister an agent.

    Args:
        agent_id: ID of agent to unregister

    Raises:
        AgentSafetyError: If agent not found
    """
```

### Budget Management Methods

```python
def get_agent_budget(self, agent_id: str) -> Decimal:
    """Get current budget for an agent.

    Args:
        agent_id: Agent to get budget for

    Returns:
        Current budget amount
    """

def update_agent_budget(self, agent_id: str, new_budget: Decimal) -> None:
    """Update budget for an agent.

    Args:
        agent_id: Agent to update budget for
        new_budget: New budget amount

    Raises:
        BudgetError: If update fails
    """

def get_agent_metrics(self, agent_id: str) -> Dict:
    """Get budget metrics for a specific agent.

    Args:
        agent_id: The ID of the agent to get metrics for

    Returns:
        Dict containing budget metrics including:
        - initial_budget: Initial budget allocated
        - used_budget: Amount of budget used
        - remaining_budget: Current remaining budget
        - last_update: Timestamp of last budget update
    """
```

### Pool Management Methods

```python
def create_pool(
    self,
    pool_id: str,
    total_budget: Decimal,
    priority: int = 0,
) -> BudgetPool:
    """Create a new budget pool.

    Args:
        pool_id: Unique identifier for the pool
        total_budget: Total budget allocation for the pool
        priority: Pool priority level

    Returns:
        Created budget pool

    Raises:
        ValueError: If pool ID already exists
    """

def get_pool(self, pool_id: str) -> BudgetPool:
    """Get a budget pool.

    Args:
        pool_id: ID of pool to get

    Returns:
        Budget pool
    """

def get_agent_pool(self, agent_id: str) -> BudgetPool:
    """Get pool for an agent.

    Args:
        agent_id: Agent to get pool for

    Returns:
        Agent's budget pool
    """

def get_pools(self) -> List[BudgetPool]:
    """Get all budget pools.

    Returns:
        List of all budget pools
    """
```

### Transfer Methods

```python
def request_transfer(
    self,
    source_id: str,
    target_id: str,
    amount: Decimal,
    transfer_type: TransferType,
    justification: str,
    requester: str,
    priority: AgentPriority = AgentPriority.MEDIUM,
    metadata: Optional[Dict] = None,
) -> UUID:
    """Request a budget transfer.

    Args:
        source_id: Source agent/pool ID
        target_id: Target agent/pool ID
        amount: Amount to transfer
        transfer_type: Type of transfer
        justification: Reason for transfer
        requester: Identity of requester
        priority: Transfer priority
        metadata: Additional transfer metadata

    Returns:
        Transfer request ID

    Raises:
        ValueError: If invalid source/target or insufficient funds
    """

def approve_transfer(
    self,
    request_id: UUID,
    approver: str,
    execute: bool = True,
) -> None:
    """Approve a budget transfer request.

    Args:
        request_id: Transfer to approve
        approver: Identity of approver
        execute: Whether to execute transfer immediately

    Raises:
        ValueError: If request not found or invalid status
    """

def execute_transfer(self, request_id: UUID) -> None:
    """Execute an approved transfer.

    Args:
        request_id: Transfer to execute

    Raises:
        ValueError: If request not found or not approved
    """

def reject_transfer(self, request_id: UUID, rejector: str, reason: str) -> None:
    """Reject a transfer request.

    Args:
        request_id: Transfer to reject
        rejector: Identity of rejector
        reason: Reason for rejection

    Raises:
        ValueError: If request not found or invalid status
    """
```

### Advanced Methods

```python
def select_optimal_pool(self, agent_id: str, required_budget: Decimal) -> str:
    """Select the optimal pool for an agent based on budget requirements and priority.

    Args:
        agent_id: Agent needing allocation
        required_budget: Minimum budget needed

    Returns:
        ID of the selected pool

    Raises:
        ValueError: If no suitable pool is found
    """

def optimize_pool_allocations(self) -> None:
    """Optimize budget allocations across all pools based on priorities and usage."""

def handle_emergency_allocation(
    self, agent_id: str, required_budget: Decimal
) -> None:
    """Handle emergency budget allocation for critical operations.

    Args:
        agent_id: Agent requiring emergency allocation
        required_budget: Budget amount needed

    Raises:
        ValueError: If emergency allocation cannot be satisfied
    """

def auto_scale_pools(self) -> None:
    """Automatically scale pools based on utilization and demand."""
```

## NotificationManager

The `NotificationManager` manages notifications and alerts.

```python
from safeguards.core.notification_manager import NotificationManager
```

### Constructor

```python
def __init__(self):
    """Initialize the notification manager."""
```

### Methods

```python
def send_alert(self, alert: Alert) -> None:
    """Send an alert.

    Args:
        alert: Alert to send
    """

def send_notification(
    self, agent_id: str, message: str, severity: str = "INFO"
) -> None:
    """Send a notification.

    Args:
        agent_id: ID of agent notification is for
        message: Notification message
        severity: Notification severity level
    """

def add_handler(self, handler: Callable[[Any], None]) -> None:
    """Add a notification handler.

    Args:
        handler: Handler function to call for notifications
    """

def remove_handler(self, handler: Callable[[Any], None]) -> None:
    """Remove a notification handler.

    Args:
        handler: Handler to remove
    """
```

## ViolationReporter

The `ViolationReporter` reports and tracks safety violations.

```python
from safeguards.monitoring.violation_reporter import ViolationReporter
```

### Constructor

```python
def __init__(self, notification_manager: NotificationManager = None):
    """Initialize the violation reporter.

    Args:
        notification_manager: For sending violation notifications
    """
```

### Methods

```python
def report_violation(
    self,
    violation_type: ViolationType,
    severity: ViolationSeverity,
    context: ViolationContext,
    description: str,
    agent_id: Optional[str] = None,
    pool_id: Optional[str] = None,
) -> UUID:
    """Report a violation.

    Args:
        violation_type: Type of violation
        severity: Severity level
        context: Context information
        description: Detailed description
        agent_id: Optional agent ID
        pool_id: Optional pool ID

    Returns:
        Violation report ID
    """

def get_violation(self, violation_id: UUID) -> Optional[ViolationReport]:
    """Get a violation report.

    Args:
        violation_id: ID of violation to get

    Returns:
        Violation report if found, None otherwise
    """

def get_violations_by_agent(self, agent_id: str) -> List[ViolationReport]:
    """Get violations for an agent.

    Args:
        agent_id: Agent to get violations for

    Returns:
        List of violation reports
    """

def get_violations_by_pool(self, pool_id: str) -> List[ViolationReport]:
    """Get violations for a pool.

    Args:
        pool_id: Pool to get violations for

    Returns:
        List of violation reports
    """
```

## TransactionManager

The `TransactionManager` ensures atomic operations.

```python
from safeguards.core.transaction import TransactionManager
```

### Methods

```python
def begin_transaction(self) -> None:
    """Begin a new transaction."""

def commit(self) -> None:
    """Commit the current transaction."""

def rollback(self) -> None:
    """Roll back the current transaction."""

def get_state(self) -> Optional[T]:
    """Get the current transaction state.

    Returns:
        Current state or None if no transaction is active
    """

def set_state(self, state: T) -> None:
    """Set the current transaction state.

    Args:
        state: New state to set
    """
```

## BudgetPool

The `BudgetPool` class represents a container for shared resources.

```python
from safeguards.core.dynamic_budget import BudgetPool
```

### Constructor

```python
@dataclass
class BudgetPool:
    """Shared budget pool for multiple agents."""

    pool_id: str
    total_budget: Decimal
    priority: int = 0
    allocated_budget: Decimal = field(default=Decimal("0"))
    reserved_budget: Decimal = field(default=Decimal("0"))
    agent_allocations: Dict[str, Decimal] = field(default_factory=dict)
    agents: Set[str] = field(default_factory=set)
    priority_weights: Dict[AgentPriority, float] = field(...)
    used_budget: Decimal = field(default=Decimal("0"))
    last_update: datetime = field(default_factory=datetime.now)
```

### Properties

```python
@property
def name(self) -> str:
    """Get pool name (alias for pool_id for compatibility)."""

@property
def id(self) -> str:
    """Get pool ID (alias for pool_id for compatibility)."""

@property
def remaining_budget(self) -> Decimal:
    """Get remaining budget in pool."""

@property
def initial_budget(self) -> Decimal:
    """For backward compatibility - Return total budget."""
```

## Alert Types

```python
from safeguards.core.alert_types import Alert, AlertSeverity
```

### Alert Class

```python
@dataclass
class Alert:
    """Alert notification."""

    title: str
    description: str
    severity: AlertSeverity
    metadata: Dict = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
```

### AlertSeverity Enum

```python
class AlertSeverity(Enum):
    """Alert severity levels."""

    INFO = auto()
    LOW = auto()
    MEDIUM = auto()
    HIGH = auto()
    CRITICAL = auto()
```

## Violation Types

```python
from safeguards.monitoring.violation_reporter import ViolationType, ViolationSeverity
```

### ViolationType Enum

```python
class ViolationType(Enum):
    """Types of budget violations."""

    OVERSPEND = auto()        # Spent more than allocated
    UNAUTHORIZED = auto()     # Unauthorized budget operation
    RATE_LIMIT = auto()       # Too many operations in time window
    RESOURCE_BREACH = auto()  # Exceeded resource allocation
    POOL_BREACH = auto()      # Pool depleted below minimum
    POOL_HEALTH = auto()      # Pool health issues detected
```

### ViolationSeverity Enum

```python
class ViolationSeverity(Enum):
    """Severity levels for violations."""

    LOW = auto()
    MEDIUM = auto()
    HIGH = auto()
    CRITICAL = auto()
```

## TransferType Enum

```python
from safeguards.core.budget_coordination import TransferType
```

```python
class TransferType(Enum):
    """Types of budget transfers."""

    DIRECT = auto()       # Direct transfer between agents
    POOL_DEPOSIT = auto() # Agent depositing to shared pool
    POOL_WITHDRAW = auto() # Agent withdrawing from shared pool
    ALLOCATION = auto()   # Initial allocation from pool
    REALLOCATION = auto() # Moving budget between agents
    RETURN = auto()       # Return unused budget to pool
```

## AgentPriority Enum

```python
from safeguards.core.dynamic_budget import AgentPriority
```

```python
class AgentPriority(Enum):
    """Priority levels for agents."""

    CRITICAL = auto()  # Mission-critical agents (e.g., core monitoring)
    HIGH = auto()      # High-priority business functions
    MEDIUM = auto()    # Standard operations
    LOW = auto()       # Non-critical background tasks
    MINIMAL = auto()   # Debug/testing agents
```

## Exceptions

```python
from safeguards.exceptions import (
    AgentSafetyError,
    BudgetError,
    ResourceError,
    ErrorContext,
)
```

### Base Exception

```python
class AgentSafetyError(Exception):
    """Base exception for Agent Safety errors."""

    def __init__(
        self,
        message: str,
        code: Optional[str] = None,
        context: Optional[ErrorContext] = None
    ):
        """Initialize the exception.

        Args:
            message: Error message
            code: Error code
            context: Error context
        """
        super().__init__(message)
        self.message = message
        self.code = code
        self.context = context or ErrorContext()
```

### Specific Exceptions

```python
class BudgetError(AgentSafetyError):
    """Exception for budget-related errors."""

class ResourceError(AgentSafetyError):
    """Exception for resource-related errors."""
```

## Next Steps

- Explore the [Budget API Reference](budget.md) for budget-specific APIs
- Check the [Agent API Reference](agent.md) for agent-specific APIs
- Review the [Monitoring API Reference](monitoring.md) for monitoring capabilities
