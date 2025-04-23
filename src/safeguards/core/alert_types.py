"""Alert type definitions for the Agent Safety Framework."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto


class AlertSeverity(Enum):
    """Alert severity levels."""

    INFO = auto()
    WARNING = auto()
    ERROR = auto()
    CRITICAL = auto()


@dataclass
class Alert:
    """Alert notification."""

    title: str
    description: str
    severity: AlertSeverity
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: dict = field(default_factory=dict)
    alert_id: str | None = None
    source: str | None = None
    resolved: bool = False
    resolution_time: datetime | None = None
    resolution_notes: str | None = None
