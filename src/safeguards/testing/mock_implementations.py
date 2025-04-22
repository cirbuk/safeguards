"""Mock implementations of core components for testing."""

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Dict, List, Optional, Set
from uuid import UUID

from safeguards.core.alert_types import Alert, AlertSeverity
from safeguards.core.budget_coordination import (
    BudgetCoordinator,
    SharedPool,
    TransferRequest,
    TransferStatus,
    TransferType,
)
from safeguards.core.dynamic_budget import AgentBudgetProfile, AgentPriority
from safeguards.monitoring.violation_reporter import (
    ViolationContext,
    ViolationReport,
    ViolationReporter,
    ViolationType,
    ViolationSeverity,
    Violation,
)


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
        self.notifications: List[MockNotification] = []
        self.active_alerts: Set[str] = set()

    def send_notification(self, agent_id: str, message: str, severity: str = "INFO"):
        """Send a mock notification."""
        notification = MockNotification(agent_id, message, severity=severity)
        self.notifications.append(notification)

    def create_alert(self, alert_id: str, message: str):
        """Create a mock alert."""
        self.active_alerts.add(alert_id)
        self.send_notification("system", f"Alert created: {message}", "ALERT")

    def resolve_alert(self, alert_id: str):
        """Resolve a mock alert."""
        if alert_id in self.active_alerts:
            self.active_alerts.remove(alert_id)
            self.send_notification("system", f"Alert resolved: {alert_id}", "INFO")

    def get_active_alerts(self) -> Set[str]:
        """Get active alerts."""
        return self.active_alerts

    def get_notifications(
        self, agent_id: Optional[str] = None
    ) -> List[MockNotification]:
        """Get notifications, optionally filtered by agent ID."""
        if agent_id:
            return [n for n in self.notifications if n.agent_id == agent_id]
        return self.notifications


@dataclass
class MockViolation(Violation):
    """Mock violation for testing."""

    type: ViolationType
    message: str
    agent_id: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict = field(default_factory=dict)
    severity: ViolationSeverity = ViolationSeverity.HIGH
    id: str = ""
    resolved: bool = False
    resolution_time: Optional[datetime] = None


class MockViolationReporter:
    """Mock violation reporter for testing."""

    def __init__(self, notification_manager: MockNotificationManager):
        """Initialize mock violation reporter."""
        self.notification_manager = notification_manager
        self.violations: Dict[str, MockViolation] = {}
        self._violation_counter = 0

    def _generate_violation_id(self) -> str:
        """Generate a unique violation ID."""
        self._violation_counter += 1
        return f"violation_{self._violation_counter}"

    def report_violation(
        self,
        type: ViolationType,
        severity: ViolationSeverity,
        agent_id: str,
        message: str,
    ) -> str:
        """Report a mock violation."""
        violation_id = self._generate_violation_id()
        violation = MockViolation(
            id=violation_id,
            type=type,
            severity=severity,
            agent_id=agent_id,
            message=message,
        )
        self.violations[violation_id] = violation

        # Create alert for high severity violations
        if severity in [ViolationSeverity.HIGH, ViolationSeverity.CRITICAL]:
            self.notification_manager.create_alert(
                violation_id, f"High severity violation: {message}"
            )

        return violation_id

    def resolve_violation(self, violation_id: str):
        """Resolve a mock violation."""
        if violation_id in self.violations:
            violation = self.violations[violation_id]
            violation.resolved = True
            violation.resolution_time = datetime.now()

            # Resolve associated alert if exists
            if violation.severity in [
                ViolationSeverity.HIGH,
                ViolationSeverity.CRITICAL,
            ]:
                self.notification_manager.resolve_alert(violation_id)

    def get_violation(self, violation_id: str) -> Optional[MockViolation]:
        """Get a specific violation by ID."""
        return self.violations.get(violation_id)

    def get_active_violations(
        self, agent_id: Optional[str] = None
    ) -> List[MockViolation]:
        """Get active violations, optionally filtered by agent ID."""
        violations = [v for v in self.violations.values() if not v.resolved]
        if agent_id:
            violations = [v for v in violations if v.agent_id == agent_id]
        return violations

    def get_violation_history(
        self,
        agent_id: Optional[str] = None,
        type: Optional[ViolationType] = None,
    ) -> List[MockViolation]:
        """Get violation history with optional filters."""
        violations = list(self.violations.values())

        if agent_id:
            violations = [v for v in violations if v.agent_id == agent_id]
        if type:
            violations = [v for v in violations if v.type == type]

        return sorted(violations, key=lambda v: v.timestamp)


@dataclass
class MockBudgetState:
    """Mock budget state for testing."""

    balances: Dict[str, Decimal] = field(default_factory=dict)
    transfers: Dict[UUID, TransferRequest] = field(default_factory=dict)
    pools: Dict[str, SharedPool] = field(default_factory=dict)
    agent_pools: Dict[str, Set[str]] = field(default_factory=dict)


class MockBudgetCoordinator(BudgetCoordinator):
    """Mock implementation of budget coordinator for testing."""

    def __init__(self, initial_state: Optional[MockBudgetState] = None):
        """Initialize mock budget coordinator.

        Args:
            initial_state: Optional initial state
        """
        super().__init__(MockNotificationManager())
        self.state_history: List[MockBudgetState] = []
        self.transfer_history: List[TransferRequest] = []
        self.violation_reporter = MockViolationReporter(MockNotificationManager())

        if initial_state:
            self._balances = initial_state.balances.copy()
            self._transfer_requests = initial_state.transfers.copy()
            self._shared_pools = initial_state.pools.copy()
            self._agent_pools = initial_state.agent_pools.copy()

        self._save_state()

    def _save_state(self) -> None:
        """Save current state to history."""
        current_state = MockBudgetState(
            balances=self._balances.copy(),
            transfers=self._transfer_requests.copy(),
            pools=self._shared_pools.copy(),
            agent_pools=self._agent_pools.copy(),
        )
        self.state_history.append(current_state)

    async def request_transfer(
        self,
        source_id: str,
        target_id: str,
        amount: Decimal,
        transfer_type: TransferType,
        notes: str = "",
    ) -> TransferRequest:
        """Record transfer request.

        Args:
            source_id: Source agent/pool ID
            target_id: Target agent/pool ID
            amount: Amount to transfer
            transfer_type: Type of transfer
            notes: Optional notes

        Returns:
            Created transfer request
        """
        request = await super().request_transfer(
            source_id=source_id,
            target_id=target_id,
            amount=amount,
            transfer_type=transfer_type,
            notes=notes,
        )
        self.transfer_history.append(request)
        self._save_state()
        return request

    def clear(self) -> None:
        """Clear all recorded history."""
        self.state_history = []
        self.transfer_history = []
        self.violation_reporter.clear()
        self._balances = {}
        self._transfer_requests = {}
        self._shared_pools = {}
        self._agent_pools = {}
        self._save_state()


class TestScenarioBuilder:
    """Helper class for building test scenarios."""

    def __init__(self):
        """Initialize scenario builder."""
        self.coordinator = MockBudgetCoordinator()
        self.agents: Dict[str, AgentBudgetProfile] = {}
        self.pools: Dict[str, SharedPool] = {}

    def add_agent(
        self,
        agent_id: str,
        initial_balance: Decimal = Decimal("0"),
        priority: AgentPriority = AgentPriority.MEDIUM,
    ) -> None:
        """Add an agent to the scenario.

        Args:
            agent_id: Agent ID
            initial_balance: Initial balance
            priority: Agent priority
        """
        self.coordinator.register_agent(agent_id)
        self.coordinator._balances[agent_id] = initial_balance
        self.agents[agent_id] = AgentBudgetProfile(
            agent_id=agent_id,
            priority=priority,
            current_balance=initial_balance,
        )

    def add_pool(
        self,
        pool_id: str,
        total_budget: Decimal,
        min_balance: Decimal = Decimal("0"),
        priority: AgentPriority = AgentPriority.MEDIUM,
    ) -> None:
        """Add a pool to the scenario.

        Args:
            pool_id: Pool ID
            total_budget: Total budget
            min_balance: Minimum balance
            priority: Pool priority
        """
        pool = self.coordinator.create_shared_pool(
            pool_id=pool_id,
            total_budget=total_budget,
            min_balance=min_balance,
            priority=priority,
        )
        self.pools[pool_id] = pool

    def add_agent_to_pool(
        self,
        agent_id: str,
        pool_id: str,
        allocation: Optional[Decimal] = None,
    ) -> None:
        """Add an agent to a pool.

        Args:
            agent_id: Agent ID
            pool_id: Pool ID
            allocation: Optional initial allocation
        """
        self.coordinator.add_agent_to_pool(
            agent_id=agent_id,
            pool_id=pool_id,
            initial_allocation=allocation,
        )

    def create_violation(
        self,
        agent_id: str,
        violation_type: ViolationType,
        severity: ViolationSeverity,
        amount: Decimal,
        pool_id: Optional[str] = None,
    ) -> ViolationReport:
        """Create a test violation.

        Args:
            agent_id: Agent ID
            violation_type: Type of violation
            severity: Violation severity
            amount: Violation amount
            pool_id: Optional pool ID

        Returns:
            Created violation report
        """
        context = ViolationContext(
            agent_id=agent_id,
            pool_id=pool_id,
            current_balance=self.coordinator._balances.get(agent_id, Decimal("0")),
            violation_amount=amount,
        )
        return self.coordinator.violation_reporter.report_violation(
            violation_type=violation_type,
            severity=severity,
            context=context,
            message=f"Test violation for agent {agent_id}",
        )

    def clear(self) -> None:
        """Clear the test scenario."""
        self.coordinator.clear()
        self.agents = {}
        self.pools = {}
