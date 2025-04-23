"""Mock implementations for testing."""

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum, auto


# Mock enums for testing
class MockViolationType(Enum):
    """Mock violation types."""

    OVERSPEND = auto()
    POOL_BREACH = auto()
    RATE_LIMIT = auto()
    UNAUTHORIZED = auto()
    POLICY_BREACH = auto()
    POOL_HEALTH = auto()


class MockViolationSeverity(Enum):
    """Mock violation severity levels."""

    CRITICAL = auto()
    HIGH = auto()
    MEDIUM = auto()
    LOW = auto()


class MockTransferType(Enum):
    """Mock transfer types."""

    DIRECT = auto()
    POOL_DEPOSIT = auto()
    POOL_WITHDRAW = auto()
    ALLOCATION = auto()
    REALLOCATION = auto()
    RETURN = auto()


class MockTransferStatus(Enum):
    """Mock transfer status."""

    PENDING = auto()
    APPROVED = auto()
    EXECUTED = auto()
    FAILED = auto()
    REJECTED = auto()
    ROLLED_BACK = auto()


class MockAgentPriority(Enum):
    """Mock agent priority levels."""

    HIGH = auto()
    MEDIUM = auto()
    LOW = auto()


@dataclass
class MockNotification:
    """Mock notification for testing."""

    agent_id: str
    message: str
    timestamp: datetime = field(default_factory=datetime.now)
    severity: str = "INFO"


class MockNotificationManager:
    """Mock notification manager for testing."""

    def __init__(self):
        """Initialize mock notification manager."""
        self.notifications: list[MockNotification] = []
        self.alerts: list[dict] = []

    def send_notification(self, agent_id: str, message: str, severity: str = "INFO"):
        """Send a mock notification."""
        notification = MockNotification(agent_id, message, severity=severity)
        self.notifications.append(notification)

    def create_alert(self, alert):
        """Create a mock alert."""
        self.alerts.append(
            {
                "title": getattr(alert, "title", "Alert"),
                "description": getattr(alert, "description", ""),
                "severity": getattr(alert, "severity", "INFO"),
                "timestamp": getattr(alert, "timestamp", datetime.now()),
                "metadata": getattr(alert, "metadata", {}),
            },
        )

    def send_alert(self, alert):
        """Send a mock alert."""
        self.create_alert(alert)

    def get_notifications(self):
        """Get all notifications."""
        return self.notifications

    def get_alerts(self, agent_id=None):
        """Get alerts for agent."""
        if agent_id:
            return [a for a in self.alerts if a.get("metadata", {}).get("agent_id") == agent_id]
        return self.alerts


class MockViolationReporter:
    """Mock violation reporter for testing."""

    def __init__(self, notification_manager=None):
        """Initialize mock violation reporter."""
        self.notification_manager = notification_manager
        self.violations = []

    def report_violation(self, **kwargs):
        """Report a mock violation."""
        violation = {
            "type": kwargs.get("type", MockViolationType.OVERSPEND),
            "severity": kwargs.get("severity", MockViolationSeverity.HIGH),
            "agent_id": kwargs.get("agent_id", ""),
            "description": kwargs.get("description", ""),
            "context": kwargs.get("context"),
            "timestamp": datetime.now(),
        }
        self.violations.append(violation)
        return violation

    def get_violations(self):
        """Get all violations."""
        return self.violations

    def get_active_violations(self, agent_id=None):
        """Get active violations for agent."""
        violations = self.violations
        if agent_id:
            violations = [v for v in violations if v.get("agent_id") == agent_id]
        return violations


@dataclass
class MockBudgetState:
    """Mock budget state for testing."""

    balances: dict[str, Decimal] = field(default_factory=dict)
    transfers: dict[str, dict] = field(default_factory=dict)
    pools: dict[str, dict] = field(default_factory=dict)
    agent_pools: dict[str, set[str]] = field(default_factory=dict)


# Base class for agent mocks
class MockAgent:
    """Mock agent class."""

    def __init__(self, name):
        """Initialize with a name."""
        self.name = name
        self.id = name  # For simplicity, use name as ID

    def run(self, **kwargs):
        """Run method implementation."""
        return {"result": "success"}


class MockBudgetCoordinator:
    """Mock budget coordinator for testing."""

    def __init__(self, notification_manager=None):
        """Initialize mock budget coordinator."""
        self.notification_manager = notification_manager or MockNotificationManager()
        self.violation_reporter = MockViolationReporter(self.notification_manager)
        self.state_history = []

        # Initialize storage
        self._balances = {}
        self._pools = {}
        self._agent_pools = {}
        self._transfer_requests = {}
        self._agents = {}

    def _save_state(self):
        """Save the current state."""
        state = MockBudgetState(
            balances=self._balances.copy(),
            transfers=self._transfer_requests.copy(),
            pools=self._pools.copy(),
            agent_pools=self._agent_pools.copy(),
        )
        self.state_history.append(state)

    def clear(self):
        """Clear all state."""
        self._balances = {}
        self._transfer_requests = {}
        self._pools = {}
        self._agent_pools = {}
        self.state_history = []

    def register_agent(
        self,
        agent_id,
        initial_budget=Decimal("0"),
        priority=0,
        agent=None,
    ):
        """Register an agent."""
        if not agent:
            agent = MockAgent(agent_id)
        self._agents[agent_id] = agent
        self._balances[agent_id] = initial_budget
        return agent

    def get_agent(self, agent_id):
        """Get an agent by ID."""
        return self._agents.get(agent_id)

    def get_agent_budget(self, agent_id):
        """Get agent budget."""
        return self._balances.get(agent_id, Decimal("0"))

    def update_agent_budget(self, agent_id, new_budget):
        """Update agent budget."""
        self._balances[agent_id] = new_budget
