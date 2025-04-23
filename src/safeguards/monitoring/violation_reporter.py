"""Budget violation reporting and tracking system."""

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum, auto
from uuid import UUID, uuid4

from ..core.alert_types import AlertSeverity
from ..types import SafetyAlert


class ViolationType(Enum):
    """Types of budget violations."""

    OVERSPEND = auto()  # Exceeded allocated budget
    RATE_LIMIT = auto()  # Exceeded spending rate limit
    UNAUTHORIZED = auto()  # Unauthorized budget access
    POOL_BREACH = auto()  # Pool minimum balance breach
    POLICY_BREACH = auto()  # Budget policy violation


class ViolationSeverity(Enum):
    """Severity levels for budget violations."""

    CRITICAL = auto()  # Immediate action required
    HIGH = auto()  # Urgent attention needed
    MEDIUM = auto()  # Important but not urgent
    LOW = auto()  # Minor violation


# Add Violation class for backward compatibility with tests
@dataclass
class Violation:
    """Backward compatibility class for tests."""

    type: str
    message: str
    agent_id: str | None = None
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: dict = field(default_factory=dict)
    severity: str = "HIGH"


@dataclass
class ViolationContext:
    """Context information for a budget violation."""

    agent_id: str
    pool_id: str | None
    current_balance: Decimal
    violation_amount: Decimal
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: dict = field(default_factory=dict)


@dataclass
class ViolationReport:
    """Detailed report of a budget violation."""

    violation_type: ViolationType
    severity: ViolationSeverity
    context: ViolationContext
    description: str
    report_id: UUID = field(default_factory=uuid4)
    created_at: datetime = field(default_factory=datetime.now)
    resolved_at: datetime | None = None
    resolution_notes: str = ""
    escalation_level: int = 0
    related_reports: set[UUID] = field(default_factory=set)


class ViolationReporter:
    """Handles reporting and tracking of budget violations."""

    def __init__(self, notification_manager=None):
        """Initialize the violation reporter.

        Args:
            notification_manager: Optional notification manager for alerts
        """
        self.notification_manager = notification_manager
        self._active_violations: dict[UUID, ViolationReport] = {}
        self._resolved_violations: dict[UUID, ViolationReport] = {}
        self._agent_violations: dict[str, set[UUID]] = {}
        self._pool_violations: dict[str, set[UUID]] = {}

    def report_violation(
        self,
        violation_type: ViolationType,
        severity: ViolationSeverity,
        context: ViolationContext,
        description: str,
    ) -> ViolationReport:
        """Report a new budget violation.

        Args:
            violation_type: Type of violation
            severity: Violation severity
            context: Violation context
            description: Detailed description

        Returns:
            Created violation report
        """
        report = ViolationReport(
            violation_type=violation_type,
            severity=severity,
            context=context,
            description=description,
        )

        # Store violation
        self._active_violations[report.report_id] = report

        # Track by agent
        if context.agent_id not in self._agent_violations:
            self._agent_violations[context.agent_id] = set()
        self._agent_violations[context.agent_id].add(report.report_id)

        # Track by pool if applicable
        if context.pool_id:
            if context.pool_id not in self._pool_violations:
                self._pool_violations[context.pool_id] = set()
            self._pool_violations[context.pool_id].add(report.report_id)

        # Send notification
        self._send_violation_alert(report)

        # Auto-escalate critical violations
        if severity == ViolationSeverity.CRITICAL:
            self.escalate_violation(report.report_id)

        return report

    def resolve_violation(self, report_id: UUID, resolution_notes: str) -> None:
        """Mark a violation as resolved.

        Args:
            report_id: ID of violation to resolve
            resolution_notes: Notes on how violation was resolved

        Raises:
            ValueError: If violation not found
        """
        if report_id not in self._active_violations:
            msg = f"Violation {report_id} not found"
            raise ValueError(msg)

        report = self._active_violations[report_id]
        report.resolved_at = datetime.now()
        report.resolution_notes = resolution_notes

        # Move to resolved violations
        self._resolved_violations[report_id] = report
        del self._active_violations[report_id]

        # Update tracking
        if report.context.agent_id in self._agent_violations:
            self._agent_violations[report.context.agent_id].remove(report_id)

        if report.context.pool_id and report.context.pool_id in self._pool_violations:
            self._pool_violations[report.context.pool_id].remove(report_id)

        # Send resolution notification
        if self.notification_manager:
            alert = SafetyAlert(
                title="Budget Violation Resolved",
                description=f"Violation {report_id} has been resolved: {resolution_notes}",
                severity=AlertSeverity.INFO,
                metadata={
                    "report_id": str(report_id),
                    "agent_id": report.context.agent_id,
                    "violation_type": report.violation_type.name,
                    "resolution_time": str(datetime.now() - report.created_at),
                },
            )
            self.notification_manager.send_alert(alert)

    def escalate_violation(self, report_id: UUID) -> None:
        """Escalate a violation to the next level.

        Args:
            report_id: ID of violation to escalate

        Raises:
            ValueError: If violation not found
        """
        if report_id not in self._active_violations:
            msg = f"Violation {report_id} not found"
            raise ValueError(msg)

        report = self._active_violations[report_id]
        report.escalation_level += 1

        # Send escalation notification
        if self.notification_manager:
            alert = SafetyAlert(
                title="Budget Violation Escalated",
                description=(f"Violation {report_id} escalated to level {report.escalation_level}"),
                severity=AlertSeverity.ERROR,
                metadata={
                    "report_id": str(report_id),
                    "agent_id": report.context.agent_id,
                    "violation_type": report.violation_type.name,
                    "escalation_level": report.escalation_level,
                },
            )
            self.notification_manager.send_alert(alert)

    def link_violations(self, report_ids: list[UUID]) -> None:
        """Link related violations together.

        Args:
            report_ids: IDs of violations to link

        Raises:
            ValueError: If any violation not found
        """
        # Validate all reports exist
        reports = []
        for report_id in report_ids:
            if report_id in self._active_violations:
                reports.append(self._active_violations[report_id])
            elif report_id in self._resolved_violations:
                reports.append(self._resolved_violations[report_id])
            else:
                msg = f"Violation {report_id} not found"
                raise ValueError(msg)

        # Link reports together
        for report in reports:
            report.related_reports.update(rid for rid in report_ids if rid != report.report_id)

    def get_active_violations(
        self,
        agent_id: str | None = None,
        pool_id: str | None = None,
        min_severity: ViolationSeverity | None = None,
    ) -> list[ViolationReport]:
        """Get active violations with optional filtering.

        Args:
            agent_id: Filter by agent
            pool_id: Filter by pool
            min_severity: Filter by minimum severity

        Returns:
            List of matching violation reports
        """
        violations = self._active_violations.values()

        if agent_id:
            violations = [v for v in violations if v.context.agent_id == agent_id]

        if pool_id:
            violations = [v for v in violations if v.context.pool_id == pool_id]

        if min_severity:
            violations = [v for v in violations if v.severity.value >= min_severity.value]

        return sorted(violations, key=lambda v: v.created_at, reverse=True)

    def get_violation_history(
        self,
        agent_id: str | None = None,
        pool_id: str | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
    ) -> list[ViolationReport]:
        """Get historical violation reports with optional filtering.

        Args:
            agent_id: Filter by agent
            pool_id: Filter by pool
            start_time: Filter by start time
            end_time: Filter by end time

        Returns:
            List of matching violation reports
        """
        violations = list(self._resolved_violations.values())

        if agent_id:
            violations = [v for v in violations if v.context.agent_id == agent_id]

        if pool_id:
            violations = [v for v in violations if v.context.pool_id == pool_id]

        if start_time:
            violations = [v for v in violations if v.created_at >= start_time]

        if end_time:
            violations = [v for v in violations if v.created_at <= end_time]

        return sorted(violations, key=lambda v: v.created_at, reverse=True)

    def get_violation_stats(
        self,
        agent_id: str | None = None,
        pool_id: str | None = None,
    ) -> dict:
        """Get violation statistics.

        Args:
            agent_id: Filter by agent
            pool_id: Filter by pool

        Returns:
            Dictionary of violation statistics
        """
        violations = list(self._active_violations.values()) + list(
            self._resolved_violations.values(),
        )

        if agent_id:
            violations = [v for v in violations if v.context.agent_id == agent_id]

        if pool_id:
            violations = [v for v in violations if v.context.pool_id == pool_id]

        stats = {
            "total_violations": len(violations),
            "active_violations": len([v for v in violations if v.resolved_at is None]),
            "resolved_violations": len(
                [v for v in violations if v.resolved_at is not None],
            ),
            "by_type": {},
            "by_severity": {},
            "avg_resolution_time": None,
            "escalation_rate": 0,
        }

        # Count by type and severity
        for v in violations:
            stats["by_type"][v.violation_type.name] = (
                stats["by_type"].get(v.violation_type.name, 0) + 1
            )
            stats["by_severity"][v.severity.name] = stats["by_severity"].get(v.severity.name, 0) + 1

        # Calculate average resolution time
        resolved = [v for v in violations if v.resolved_at is not None]
        if resolved:
            total_time = sum((v.resolved_at - v.created_at).total_seconds() for v in resolved)
            stats["avg_resolution_time"] = total_time / len(resolved)

        # Calculate escalation rate
        escalated = len([v for v in violations if v.escalation_level > 0])
        if violations:
            stats["escalation_rate"] = escalated / len(violations)

        return stats

    def _send_violation_alert(self, report: ViolationReport) -> None:
        """Send alert for a violation.

        Args:
            report: Violation report to send alert for
        """
        if not self.notification_manager:
            return

        # Map violation severity to alert severity
        severity_map = {
            ViolationSeverity.LOW: AlertSeverity.INFO,
            ViolationSeverity.MEDIUM: AlertSeverity.WARNING,
            ViolationSeverity.HIGH: AlertSeverity.ERROR,
            ViolationSeverity.CRITICAL: AlertSeverity.CRITICAL,
        }

        alert = SafetyAlert(
            title=f"Budget Violation: {report.violation_type.name}",
            description=report.description,
            severity=severity_map[report.severity],
            metadata={
                "report_id": str(report.report_id),
                "agent_id": report.context.agent_id,
                "pool_id": report.context.pool_id,
                "violation_type": report.violation_type.name,
                "violation_amount": str(report.context.violation_amount),
                "current_balance": str(report.context.current_balance),
            },
        )
        self.notification_manager.send_alert(alert)
